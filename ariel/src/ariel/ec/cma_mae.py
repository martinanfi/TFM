"""CMA-MAE outer EA (pyribs) glued into ariel TrackedEA.

Note: this file uses a dash in its name. Rename to ``cma_mae.py`` before
importing it as a module.
"""

# Standard library
import math
import sys
print(sys.executable)
# Third-party libraries
import numpy as np
import mujoco
import networkx as nx
from joblib import Parallel, delayed

# pyribs
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

# ariel
from ariel.ec.individual import Individual
from ariel.ec.population import Population
from ariel.simulation.full_robot_ribs import decode_robot, full_robot
from ariel.body_phenotypes.robogen_lite.config import ModuleType


#---------------------------------------------------------#
# Global constants
#---------------------------------------------------------#

# Simulation
dt = 0.01
steps = 1800

# Contact-check window for early-rejection of robots that fall over
BUFFER_FALL = 350
CONTACT_CHECK_START = BUFFER_FALL
CONTACT_CHECK_END = 700
MAX_EARLY_CONTACTS = 1700

# pyribs rejects non-finite objectives in scheduler.tell(). Failed robots
# (controller is None, early-contact reject, NaN) return -inf from
# _evaluate_one. We substitute a very-negative finite sentinel so:
#   * scheduler.tell() doesn't crash on the batch,
#   * the main archive's threshold-drift logic rejects them (sentinel is
#     well below any sane threshold_min),
#   * result_archive cells holding the sentinel are filtered out when we
#     rebuild Population.
PYRIBS_OBJ_SENTINEL = -1e30


#---------------------------------------------------------#
# Morphological descriptors (CMA-MAE behaviour descriptors)
#---------------------------------------------------------#

def _real_module_subgraph(robot: full_robot) -> nx.DiGraph:
    """Restrict ``robot.body_phenotype`` to non-NONE modules.

    NDE + HPD keeps NMODULES nodes in the graph but many are placeholder
    NONE slots. Descriptors should ignore those.
    """
    keep = [
        n for n, attrs in robot.body_phenotype.nodes(data=True)
        if attrs.get("type") != ModuleType.NONE.name
    ]
    return robot.body_phenotype.subgraph(keep)


def descriptor_branching(robot: full_robot) -> float:
    """Fraction of internal (non-leaf) real modules whose out-degree is >= 2.

    Continuous in [0, 1]. 0 means a pure chain (snake), 1 means every
    internal joint forks.
    """
    g = _real_module_subgraph(robot)
    internal = [n for n in g.nodes() if g.out_degree(n) >= 1]
    if not internal:
        return 0.0
    forks = sum(1 for n in internal if g.out_degree(n) >= 2)
    return forks / len(internal)


def descriptor_limb_lengths(robot: full_robot) -> dict:
    """Limb length distribution.

    A limb is a path from the core (node 0) to a leaf in the real-module
    subgraph. Length is measured in edges (modules between core and leaf).

    Returns
    -------
    dict
        ``n_limbs``     : number of leaves reachable from the core
        ``max_length``  : longest core-to-leaf distance
        ``mean_length`` : mean core-to-leaf distance
        ``std_length``  : std of core-to-leaf distances
    """
    empty = {"n_limbs": 0, "max_length": 0, "mean_length": 0.0, "std_length": 0.0}
    g = _real_module_subgraph(robot)
    if 0 not in g.nodes():
        return empty

    depths = nx.single_source_shortest_path_length(g, 0)
    leaves = [
        n for n in g.nodes()
        if n != 0 and n in depths and g.out_degree(n) == 0
    ]
    if not leaves:
        return empty

    lengths = np.array([depths[n] for n in leaves], dtype=float)
    return {
        "n_limbs":     int(lengths.size),
        "max_length":  int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "std_length":  float(lengths.std()),
    }


def evaluate_individual(individual, performance_fun, NDE, DECODER, world_cls, n_steps=steps, seed=0):
    robot = decode_robot(individual, NDE=NDE, DECODER=DECODER, world_cls=world_cls, seed=seed)
    if robot.controller is None:
        return -float('inf'), np.array([0.0, 0.0])
    fitness = performance_fun(robot, n_steps=n_steps)
    measures = get_measures(robot)
    return fitness, measures


def evaluate_pop(population, performance_fun, NDE, DECODER, world_cls, n_steps=steps, seed=0):
    to_eval = [ind for ind in population if ind.requires_eval]
    if not to_eval:
        return population

    out = Parallel(n_jobs=-1)(
        delayed(evaluate_individual)(ind, performance_fun, NDE, DECODER, world_cls, n_steps=n_steps, seed=seed)
        for ind in to_eval
    )
    fitnesses, measures_list = zip(*out)

    for ind, fit, meas in zip(to_eval, fitnesses, measures_list):
        ind.fitness = fit
        ind.tags = ind.tags or {}
        ind.tags["measures"] = list(meas)
        ind.requires_eval = False

    return population

fitness_function_w = 0.1
epsilon = 10e-10
target_directions_deg = [0, 120, 240] # in degrees

# L1 imbalance penalty for performance_3p_tl: penalises spread between the
# per-direction fitnesses so a one-trick robot (e.g. only goes forward) can't
# match a true steerer. Set BALANCE_LAMBDA = 0.0 to disable.
# Penalty form:  -λ · Σ_i |f_i − mean(f)|
# At λ = 1/N (where N = len(target_directions_deg)) a perfectly one-sided
# result [Nμ, 0, ..., 0] gets ~halved. Start small and tune.
BALANCE_LAMBDA = 1.0 / len(target_directions_deg) / 2  # ~0.17 for 3 dirs

# ----------------------------- #

def get_path_length(positions):
    l = 0
    for i in range(len(positions)-1):
        l += math.dist(positions[i], positions[i+1])
    return l

def get_params_fitness_tl(positions, target_direction):
    origin = positions[0]
    end_point = positions[-1]
    dx = end_point[0] - origin[0]
    dy = end_point[1] - origin[1]
    delta = np.arctan2(dy, dx)
    # signed angular difference, wrapped to (-π, π]
    diff = delta - target_direction
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    # unsigned error angle θ
    if np.abs(diff) > np.pi:
        theta = 2 * np.pi - np.abs(diff)
    else:
        theta = np.abs(diff)
    gamma = math.dist(origin, end_point)
    alpha = gamma * np.sin(theta)     # lateral error magnitude
    beta  = gamma * np.cos(diff)      # signed forward/backward progress
    l = get_path_length(positions)
    return beta, alpha, theta, l

def formula_tl(beta, alpha, theta, l, w = fitness_function_w, epsilon = epsilon):
    return (np.abs(beta) / (l + epsilon)) * ((beta / (theta +1)) - (w * alpha))


def performance_targeted_loc(robot, target_direction_deg = 0, n_steps = steps):
    robot.model.opt.timestep = dt
    mujoco.mj_resetData(robot.model, robot.data)
    mujoco.mj_forward(robot.model, robot.data)
    robot.controller.reset()

    n_act = len(robot.actuators)
    positions = []
    total_early_contacts = 0
    for t in range(n_steps):
        mujoco.mj_step(robot.model, robot.data)
        if CONTACT_CHECK_START <= t < CONTACT_CHECK_END:
            total_early_contacts += robot.data.ncon
            if total_early_contacts > MAX_EARLY_CONTACTS:
                return -float('inf')
        if t >= BUFFER_FALL:
            robot.set_forward_vec()
            alpha_angle = robot.get_alpha_angle(target_direction_deg)
            # NaCPG runs every physics step using the latest params.
            cpg_t = (t - BUFFER_FALL) * dt
            angles = robot.controller.forward(alpha_angle=alpha_angle, time=cpg_t)
            for i in range(n_act):
                robot.data.ctrl[i] = float(angles[i])

            core_pos = robot.data.xpos[robot.core_id].copy()
            positions.append(core_pos[0:2])

    beta, alpha, theta, l = get_params_fitness_tl(positions, np.deg2rad(target_direction_deg))
    return formula_tl(beta, alpha, theta, l)

def performance_3p_tl(robot, n_steps = steps):
    if robot.controller is None:
        return - float("inf")

    fitnesses = [
        performance_targeted_loc(robot, target_direction_deg=deg, n_steps=n_steps)
        for deg in target_directions_deg
    ]

    # Propagate the contact-reject sentinel: if any direction was rejected, the
    # whole evaluation is invalid (mean over -inf is meaningless and the L1
    # term explodes).
    if any(not math.isfinite(f) for f in fitnesses):
        return -float("inf")

    total = sum(fitnesses)
    if BALANCE_LAMBDA > 0.0 and len(fitnesses) > 1:
        mean_f = total / len(fitnesses)
        l1_dev = sum(abs(f - mean_f) for f in fitnesses)
        total -= BALANCE_LAMBDA * l1_dev

    return float(total)


#---------------------------------------------------------#
# CMA-MAE outer EA (pyribs)
#---------------------------------------------------------#


def get_measures(robot) -> np.ndarray:
    """CMA-MAE behaviour descriptors: (branching, limb_std_length).

    ``limb_std_length`` is the within-robot std of root-to-leaf depths — a
    "balance vs lopsidedness" measure. Decoupled from branching: two robots
    with identical branching can have very different std_length depending on
    whether their forks resolve to a symmetric tree or to one long tendril.
    """
    if robot.controller is None:
        return np.array([0.0, 0.0], dtype=np.float64)
    return np.array(
        [
            descriptor_branching(robot),
            float(descriptor_limb_lengths(robot)["std_length"]),
        ],
        dtype=np.float64,
    )


class CMAMAEOperation:
    """Stateful EAOperation: one ``__call__`` = one pyribs ask / eval / tell cycle.

    Returned ``Population`` is the *current* ``result_archive`` snapshot, so
    survivors reappear every generation in the CSV log without extra work.

    The objective fed to pyribs is **post-inner-CMA fitness**: each candidate
    is decoded, the inner learner is run, and the learned fitness is what the
    archive sees. The genotype that goes back to ``tell`` is the one that was
    ``ask``-ed (non-Lamarckian); learned CPG params are stashed on the
    returned ``Individual`` for external logging only.

    Notes
    -----
    - ``solution_dim`` is inferred from a single ``initializer_fn`` draw.
    - Each emitter gets its own random ``x0`` from ``initializer_fn``.
    - ``self._sol_aux`` is hashed by ``solution.tobytes()`` so we can recover
      per-elite extras (CPG_params, learning_curve, fitness_pre) when
      rebuilding the Population from the archive.
    """

    def __init__(
        self,
        *,
        performance_fun,
        learner_fn,                # signature: (ind, perf_fn, learn_gens, SEED, NDE, DECODER, world_cls) -> ind
        initializer_fn,            # signature: (RNG) -> Individual
        NDE,
        DECODER,
        world_cls,
        RNG,
        SEED,
        learning_generations: int,
        measure_fn=get_measures,
        # archive
        archive_dims: tuple[int, int] = (10, 10),
        measure_ranges: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 12.0)),
        learning_rate: float = 0.01,
        threshold_min: float = -np.inf,
        # emitters
        num_emitters: int = 5,
        batch_size: int = 36,
        sigma0: float = 0.5,
        es: str = "sep_cma_es",
        selection: str = 'fitness',   # 'fitness' | 'ldelta' | 'mixed'
        # exec
        n_jobs: int = -1,
    ) -> None:
        self.performance_fun = performance_fun
        self.learner_fn = learner_fn
        self.initializer_fn = initializer_fn
        self.NDE = NDE
        self.DECODER = DECODER
        self.world_cls = world_cls
        self.RNG = RNG
        self.SEED = SEED
        self.learning_generations = learning_generations
        self.measure_fn = measure_fn
        self.n_jobs = n_jobs
        self.selection = selection

        # solution_dim from a sample draw of create_individual
        sample = initializer_fn(RNG)
        self.solution_dim = len(sample.genotype)

        # per-emitter random x0
        x0s = [
            np.asarray(initializer_fn(RNG).genotype, dtype=np.float64)
            for _ in range(num_emitters)
        ]

        self.archive = GridArchive(
            solution_dim=self.solution_dim,
            dims=list(archive_dims),
            ranges=[tuple(r) for r in measure_ranges],
            learning_rate=learning_rate,
            threshold_min=threshold_min,
        )
        self.result_archive = GridArchive(
            solution_dim=self.solution_dim,
            dims=list(archive_dims),
            ranges=[tuple(r) for r in measure_ranges],
        )
        self.emitters = [
            EvolutionStrategyEmitter(
                archive=self.archive,
                x0=x0,
                sigma0=sigma0,
                ranker="imp",
                es=es,
                selection_rule="mu",
                restart_rule="basic",
                batch_size=batch_size,
            )
            for x0 in x0s
        ]
        self.scheduler = Scheduler(
            archive=self.archive,
            emitters=self.emitters,
            result_archive=self.result_archive,
        )

        # solution-bytes -> {cpg_params, learning_curve, fitness_pre}
        self._sol_aux: dict[bytes, dict] = {}
        self._seeded = False

    # ---- evaluation ------------------------------------------------------- #
    # _evaluate_one lives at module scope (see bottom of file) so joblib only
    # pickles the small callables/configs it needs, not the whole operation
    # (which holds pyribs archive/emitters/scheduler).

    def _seed_from_population(self, population) -> None:
        """Inject externally-evaluated init pop into archive + result_archive.

        Must match _evaluate_one's objective convention: insert ΔL into the
        archive when ldelta mode is on, otherwise post-learning fitness.
        Without this the first generation's archive is scored under one
        convention and subsequent generations under another.
        """
        sols, objs, meas, kept, post_fits = [], [], [], [], []
        for ind in population:
            try:
                post_fit = float(ind.fitness)
            except Exception:
                continue
            if not math.isfinite(post_fit):
                continue
            try:
                pre_fit = float(ind.fitness_pre)
            except Exception:
                pre_fit = float("nan")
            if self.selection == 'ldelta':
                if math.isfinite(pre_fit):
                    obj = post_fit - pre_fit
                else:
                    continue
            elif self.selection == 'mixed':
                if math.isfinite(pre_fit):
                    obj = post_fit - 0.5 * pre_fit
                else:
                    continue
            elif self.selection == 'auc':
                lc = (ind.tags or {}).get("learning_curve", [])
                auc_val = _compute_raw_auc(lc, pre_fit)
                if math.isfinite(auc_val) and math.isfinite(pre_fit):
                    obj = 0.5 * post_fit + 0.5 * AUC_FITNESS_SCALE * auc_val
                else:
                    continue
            elif self.selection == 'rawAUC':
                lc = (ind.tags or {}).get("learning_curve", [])
                auc_val = _compute_raw_auc(lc, pre_fit)
                if math.isfinite(auc_val):
                    obj = auc_val
                else:
                    continue
            else:  # 'fitness'
                obj = post_fit
            sols.append(np.asarray(ind.genotype, dtype=np.float64))
            objs.append(obj)
            post_fits.append(post_fit)
            m = (ind.tags or {}).get("measures")
            if m is None:
                # Recompute measures if init pipeline didn't store them
                robot = decode_robot(
                    ind, NDE=self.NDE, DECODER=self.DECODER,
                    world_cls=self.world_cls, seed=self.SEED,
                )
                m = self.measure_fn(robot)
            meas.append(np.asarray(m, dtype=np.float64))
            kept.append(ind)

        if not sols:
            return
        sols_arr = np.stack(sols)
        objs_arr = np.array(objs, dtype=np.float64)
        meas_arr = np.stack(meas)
        self.archive.add(sols_arr, objs_arr, meas_arr)
        self.result_archive.add(sols_arr, objs_arr, meas_arr)

        # zip against `kept` (filtered), not `population` (unfiltered) — the two
        # lists would be misaligned. Read CPG_params_ raw: the ind.CPG_params
        # property raises when unset (learn_cma skips it for failed robots).
        for sol, ind, post_fit in zip(sols_arr, kept, post_fits):
            self._sol_aux[sol.tobytes()] = {
                "cpg_params": ind.CPG_params_ if ind.CPG_params_ is not None else [],
                "learning_curve": (ind.tags or {}).get("learning_curve", []),
                "fitness_pre": float(ind.fitness_pre),
                "post_fit": post_fit,
            }

    # ---- archive state transfer ------------------------------------------ #

    def seed_archive_directly(
        self,
        solutions: np.ndarray,
        objectives: np.ndarray,
        measures: np.ndarray,
        sol_aux: dict,
    ) -> None:
        """Pre-populate archives from a prior phase without running _seed_from_population.

        Call this on a freshly constructed operation to branch from an existing
        phase's result_archive. Both the main and result archives are seeded;
        _sol_aux is copied so the Population snapshot lookups keep working.
        _seeded is set to True so __call__ skips its own seeding on first call.
        """
        sols = np.asarray(solutions, dtype=np.float64)
        objs = np.asarray(objectives, dtype=np.float64)
        meas = np.asarray(measures, dtype=np.float64)
        # Filter sentinel-substituted failures before inserting.
        valid = objs > PYRIBS_OBJ_SENTINEL / 2
        if valid.any():
            self.archive.add(sols[valid], objs[valid], meas[valid])
            self.result_archive.add(sols[valid], objs[valid], meas[valid])
        self._sol_aux = dict(sol_aux)
        self._seeded = True

    def get_population_snapshot(self) -> list:
        """Return a fresh list of Individuals from the current result_archive.

        Used to hand the archive state to a new TrackedEA as its starting
        population when branching into phase-2 variants.
        """
        snapshot = []
        data = self.result_archive.data(return_type="dict")
        for sol, obj, m in zip(data["solution"], data["objective"], data["measures"]):
            if float(obj) <= PYRIBS_OBJ_SENTINEL / 2:
                continue
            aux = self._sol_aux.get(np.asarray(sol).tobytes(), {})
            ind = Individual()
            ind.genotype = np.asarray(sol).tolist()
            ind.fitness = float(aux.get("post_fit", float(obj)))
            ind.fitness_pre = aux.get("fitness_pre", ind.fitness_)
            ind.CPG_params = aux.get("cpg_params", [])
            ind.tags = {
                "learning_curve": aux.get("learning_curve", []),
                "measures": np.asarray(m).tolist(),
            }
            ind.requires_eval = False
            snapshot.append(ind)
        return snapshot

    # ---- operation interface --------------------------------------------- #

    def __call__(self, population, **_) -> Population:
        # First call: prime the archive with the initial population.
        if not self._seeded:
            self._seed_from_population(list(population))
            self._seeded = True

        # TrackedEA.fetch_population() filters by alive=True each generation.
        # The previous generation's archive snapshot is still in the DB with
        # alive=True, so unless we flip these we'll re-fetch them next gen and
        # the alive set grows unbounded (same pattern as RevDE_step). The new
        # archive-snapshot individuals we return below are inserted as fresh
        # rows; the dead ones get committed with alive=False and disappear
        # from subsequent fetches.
        dead = []
        for ind in population:
            ind.alive = False
            dead.append(ind)

        # Ask: pyribs returns (num_emitters * batch_size, solution_dim).
        solutions = self.scheduler.ask()

        # Evaluate (each eval includes the inner CMA loop).
        # Module-level function => joblib pickles only the args we pass.
        records = Parallel(n_jobs=self.n_jobs)(
            delayed(_evaluate_one)(
                s,
                NDE=self.NDE,
                DECODER=self.DECODER,
                world_cls=self.world_cls,
                SEED=self.SEED,
                performance_fun=self.performance_fun,
                learner_fn=self.learner_fn,
                learning_generations=self.learning_generations,
                measure_fn=self.measure_fn,
                selection=self.selection,
            )
            for s in solutions
        )
        objs = np.array([r["obj"] for r in records], dtype=np.float64)
        meas = np.stack([r["meas"] for r in records])
        # pyribs requires finite objectives; failed robots return -inf.
        objs = np.where(np.isfinite(objs), objs, PYRIBS_OBJ_SENTINEL)

        # Cache extras keyed by exact solution bytes so we can recover them
        # when iterating the archive later (pyribs only stores sol/obj/meas).
        # ``post_fit`` is the actual post-learning fitness — distinct from
        # ``obj`` which may be ΔL in ldelta mode. The CSV logs post_fit so the
        # ``fitness`` column has consistent meaning across ldelta/learning runs.
        for sol, rec in zip(solutions, records):
            self._sol_aux[sol.tobytes()] = {
                "cpg_params": rec["cpg_params"],
                "learning_curve": rec["learning_curve"],
                "fitness_pre": rec["fitness_pre"],
                "post_fit": rec["post_fit"],
            }

        # Tell: scheduler handles archive.add() + emitter.tell() for both
        # the main and result archives.
        self.scheduler.tell(objs, meas)

        # Build the returned Population = result_archive snapshot.
        # result_archive == best-ever per cell, no threshold drift filtering.
        new_pop = []
        data = self.result_archive.data(return_type="dict")
        for sol, obj, m in zip(data["solution"], data["objective"], data["measures"]):
            # Skip cells whose only occupant is a sentinel-substituted failure.
            # (result_archive has no threshold, so it accepts the sentinel into
            # empty cells; the main archive's threshold_min rejects them.)
            if float(obj) <= PYRIBS_OBJ_SENTINEL / 2:
                continue
            aux = self._sol_aux.get(np.asarray(sol).tobytes(), {})
            ind = Individual()
            ind.genotype = np.asarray(sol).tolist()
            # Log post-learning fitness, not the archive objective. In ldelta
            # mode ``obj`` is ΔL = post − pre; the CSV's ``fitness`` column
            # always means post-learning fitness so downstream plots that do
            # ``df["fitness"] − df["fitness_pre"]`` keep working.
            ind.fitness = float(aux.get("post_fit", float(obj)))
            ind.fitness_pre = aux.get("fitness_pre", float("nan"))
            ind.CPG_params = aux.get("cpg_params", [])
            ind.tags = {
                "learning_curve": aux.get("learning_curve", []),
                "measures": np.asarray(m).tolist(),
            }
            ind.requires_eval = False
            new_pop.append(ind)

        return Population(new_pop + dead)


#---------------------------------------------------------#
# Worker function (module-level so joblib pickling is cheap)
#---------------------------------------------------------#

AUC_FITNESS_SCALE = 1.0 / 20.0   # brings raw AUC (~20× fitness) down to fitness scale


def _compute_raw_auc(curve, fitness_pre) -> float:
    """Sum of the cumulative-max learning curve (raw, no subtraction of fitness_pre)."""
    if not curve:
        return float("nan")
    arr = np.array(curve, dtype=float)
    arr = np.where((arr <= -1e29) | ~np.isfinite(arr), -np.inf, arr)
    arr = np.maximum.accumulate(arr)
    pre = float(fitness_pre) if math.isfinite(float(fitness_pre)) else 0.0
    arr = np.where(np.isfinite(arr), arr, pre)
    return float(np.sum(arr))


def _evaluate_one(
    sol_array: np.ndarray,
    *,
    NDE,
    DECODER,
    world_cls,
    SEED,
    performance_fun,
    learner_fn,
    learning_generations: int,
    measure_fn,
    selection: str,
) -> dict:
    """Decode -> measures -> inner CMA -> objective.

    Returns a dict with the post-learning ``obj``, the morphology ``meas``,
    and the extras needed for the CSV log. Defined at module scope so
    ``joblib.delayed`` doesn't drag the pyribs archive/emitters along with
    every worker invocation.
    """
    ind = Individual()
    ind.genotype = sol_array.tolist()
    robot = decode_robot(
        ind, NDE=NDE, DECODER=DECODER, world_cls=world_cls, seed=SEED,
    )
    if robot.controller is None:
        return {
            "obj": -float("inf"),
            "post_fit": -float("inf"),
            "meas": np.array([0.0, 0.0], dtype=np.float64),
            "cpg_params": [],
            "learning_curve": [],
            "fitness_pre": -float("inf"),
        }

    meas = np.asarray(measure_fn(robot), dtype=np.float64)
    # Pre-learning fitness (so we can report a learning delta).
    pre_fit = performance_fun(robot)
    ind.fitness = pre_fit
    # Inner CMA — non-Lamarckian: only fitness / CPG_params get written.
    ind = learner_fn(
        ind, performance_fun, learning_generations, SEED,
        NDE, DECODER, world_cls,
    )
    learning_curve = (ind.tags or {}).get("learning_curve", [])
    post_fit = float(ind.fitness)
    # learn_cma short-circuits without setting fitness_pre when pre_fit is
    # non-finite (controller None / early-contact reject), so derive ΔL from
    # pre_fit/post_fit directly instead of reading ind.fitness_pre.
    if selection == 'ldelta':
        if math.isfinite(pre_fit) and math.isfinite(post_fit):
            obj = post_fit - pre_fit
        else:
            obj = -float("inf")
    elif selection == 'mixed':
        # 0.5*ldelta + 0.5*post = post - 0.5*pre; rewards learning AND final quality
        if math.isfinite(pre_fit) and math.isfinite(post_fit):
            obj = post_fit - 0.5 * pre_fit
        else:
            obj = -float("inf")
    elif selection == 'auc':
        # 0.5 * post_fit + 0.5 * (1/20) * AUC — equal variance contribution
        auc_val = _compute_raw_auc(learning_curve, pre_fit)
        if math.isfinite(auc_val) and math.isfinite(post_fit):
            obj = 0.5 * post_fit + 0.5 * AUC_FITNESS_SCALE * auc_val
        else:
            obj = -float("inf")
    elif selection == 'rawAUC':
        # Pure AUC as archive objective (no fitness term)
        auc_val = _compute_raw_auc(learning_curve, pre_fit)
        obj = auc_val if math.isfinite(auc_val) else -float("inf")
    else:  # 'fitness'
        obj = post_fit
    return {
        "obj": float(obj),
        "post_fit": post_fit,
        "meas": meas,
        "cpg_params": ind.CPG_params_ if ind.CPG_params_ is not None else [],
        "learning_curve": list(learning_curve),
        "fitness_pre": float(pre_fit) if math.isfinite(pre_fit) else -float("inf"),
    }
