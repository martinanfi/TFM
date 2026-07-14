"""Generational (mu+lambda) genetic algorithm driver for the ARIEL
morphology+controller task, mimicking the EA of Luo et al. (2022),
"The Effects of Learning in Morphologically Evolving Robot Systems".

This is the third sibling alongside the sep-CMA-ES driver (``EA_FINAL_es.py``)
and the CMA-MAE quality-diversity driver (``EA_FINAL_ribs.py``). It reuses the
*same* ARIEL decode -> (Baldwinian inner-learn) -> fitness pipeline, the same
genome layout, the same geometry-aware collision pruning and the same
TrackedEA/CSV logging as the sep-CMA-ES file; only the OUTER optimiser changes:
a classic crossover+mutation genetic algorithm instead of a single search
distribution.

GA design (faithful to Luo et al., 2022, Table 2 + Section 3.5):
* Population-based, generational (mu + lambda):
      mu      = POP_SIZE        (100 in the paper)  -> retained parents
      lambda  = OFFSPRING_SIZE  (50  in the paper)  -> offspring per generation
  Each generation:
    - parent selection : binary tournament (size 2) WITH replacement,
                         lambda pairs -> one child per pair;
    - variation        : per-pair crossover with prob CROSSOVER_PROB (0.8),
                         then per-individual mutation with prob MUTATION_PROB
                         (0.8);
    - survivor selection: (mu + lambda) elitist truncation -- keep the best mu
                         of (parents UNION offspring).  [The paper's prose says
                         "top 50 parents plus the 50 offspring"; we implement the
                         canonical (mu+lambda) = best mu of the size-(mu+lambda)
                         union, which is the standard reading of the named
                         "(mu+lambda) selection mechanism".]

* Genome = the existing flat real vector, worked on DIRECTLY in true bounds
  (no CMA-style normalised space, no clip-and-penalise):
      [type_p(64), conn_p(64), rot_p(64), w_extend(_W), controller(N)]
  with per-block bounds (morphology 3x64 -> [-32,32], everything else -> [-1,1]).
  Genes are CLIPPED to bounds after variation.

* Crossover = uniform per-gene recombination (one child: each gene copied from
  parent A or B with p=0.5). Mutation = per-gene Gaussian step, sigma a fraction
  of each gene's range, clipped to bounds. These are the real-vector analogs of
  MultiNEAT's operators referenced by the paper; the *per-individual* rates,
  tournament size, pop/offspring sizes and (mu+lambda) selection are taken
  verbatim from Table 2.

* Controller learning = Baldwinian (as in the sep-CMA-ES sibling): every
  evaluation runs an inner CMA local search (nevergrad ``CMA``) over the CPG
  phase/amplitude columns purely to SCORE the individual; the learned params are
  NOT written back into the heritable genome (logged for replay only). This is
  the Triangle-of-Life "Infancy" stage; the paper used RevDEknn, here we keep the
  project's nevergrad-CMA learner for consistency with the other drivers.

* Selection objective handed to the GA (ranking for tournaments + survivors):
    - 'fitness' : post-learning performance  -> faithful Luo "Evolution+Learning"
    - 'ldelta'  : learning gain, post - pre  -> plasticity-selection extension
    - 'nsga2'   : NSGA-II Pareto selection on BOTH (fitness, ldelta) at once
                  (Deb et al., 2002) -> evolves the performance/plasticity
                  trade-off front in a single run, instead of one scalar branch
  The logged ``fitness`` column is always post-learning fitness, and
  ``fitness_pre`` is logged too, so the learning delta (the paper's central
  measure) is recoverable from the CSV regardless of the selection objective.

Run structure (mirrors the sep-CMA-ES sibling so the analysis pipeline is shared):
    Phase 1 : GA with learning OFF  -> morphology search (== "Evolution-Only").
    Phase 2 : per selection objective, WARM-START from the Phase-1 final
              population and continue with Baldwinian learning ON
              (== "Evolution+Learning"); each branch logs its inherited
              population as generation 0.
No CMA-MAE / pyribs / archive anywhere.
"""

import os
import sys
import copy
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

import math
import torch
import numpy as np
import networkx as nx
import mujoco
import nevergrad as ng
from joblib import Parallel, delayed

from rich.console import Console
from rich.traceback import install

from ariel.simulation.controllers.cpg_decoder import CPGDecoder
from ariel.body_phenotypes.robogen_lite.decoders.weighted_decoding import (
    WeightedHighProbabilityDecoder as whpd,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding as nde
from ariel.simulation.environments import SimpleFlatWorld, RuggedTerrainWorld
import ariel.simulation.full_robot_ribs as frr
from ariel.simulation.full_robot_ribs import decode_robot, full_robot, N_STATIC_FEATURES
from ariel.body_phenotypes.robogen_lite.config import ModuleType
from ariel.ec import Individual, Population, TrackedEA, log_alive_population_csv
from ariel.ec import config

LOG_DIR = _HERE.parent / "experiment_data"

# ---------------------------------------------------------#
config.is_maximisation = True
config.db_handling = "delete"
install()
console = Console()

dt = 0.01
steps = 1800

# ---------------------------------------------------------#
# HYPERPARAMS
# ---------------------------------------------------------#
# Simulation
BUFFER_FALL = 350
CONTACT_CHECK_START = BUFFER_FALL
CONTACT_CHECK_END = 700
MAX_EARLY_CONTACTS = 500000

# Body / NDE
NMODULES = 30
GENE_SIZE_NDE = 64
NUMELS_CONTROLLER = CPGDecoder(in_dim=N_STATIC_FEATURES, out_dim=3).num_params

# w_extend low-rank factorisation (see full_robot_ribs.W_EXTEND_RANK). Storing
# the NMODULES**2 branching matrix as W = U @ V.T (rank r) cuts 900 dims to
# 2*NMODULES*r without banning any topology. Set to None for the legacy matrix.
W_EXTEND_RANK = 2
frr.W_EXTEND_RANK = W_EXTEND_RANK   # configure the decoder for the genome size below

# ---- Genetic algorithm (Luo et al., 2022, Table 2 + Section 3.5) ---------- #
POP_SIZE = 96            # mu : retained population (= 2*lambda, keeps the paper's lambda/mu = 0.5)
OFFSPRING_SIZE = 48      # lambda : offspring/gen AND the per-generation parallel batch.
                         # Useful CPUs == this value: set it to your core count (or run on
                         # OFFSPRING_SIZE / k cores for k balanced waves). 48 -> 48 cores (1 wave).
TOURNAMENT_SIZE = 2      # binary tournament, with replacement      (Table 2 "Tournament size")
CROSSOVER_PROB = 0.8     # per-pair crossover probability           (Table 2 "Crossover")
MUTATION_PROB = 0.8      # per-individual mutation probability      (Table 2 "Mutation")
# Real-vector analogs of MultiNEAT's operators (NOT specified by the paper;
# tune freely). MUT_GENE_PROB = per-gene chance of being perturbed once an
# individual is selected for mutation; MUT_SIGMA = Gaussian step size as a
# fraction of each gene's [lo, hi] range.
MUT_GENE_PROB = 0.1
MUT_SIGMA = 0.1

# No identical individuals in a population. When True, every generation is kept
# genotype-distinct: a child that lands byte-identical to one already in the
# population (or to a sibling) is re-mutated until distinct, BEFORE it is
# evaluated. This costs no extra evaluations (still lambda offspring/gen) and
# injects no foreign genomes -- a collision just gets a real mutation step.
# Applied in BOTH phases, so Phase 1 never accumulates clones and its warm-start
# handoff (Phase-2 gen 0) is honestly distinct rather than scrubbed; the seed is
# never modified. Mild departure from Luo et al. (which lets clones through).
# DEDUP_DECIMALS = genes rounded to this many decimals when testing identity, so
# only true clones collide, not genuine mutants.
DEDUP_POPULATION = True
DEDUP_DECIMALS = 9

# Logging granularity. True -> log the FULL mu population each generation (the
# "fitness over the entire population" Luo et al. report). False -> log only the
# top mu//2. The GA update is identical either way.
LOG_WHOLE_POPULATION = True

# Collision handling. The decoder's occupancy check is an integer unit grid that
# ignores per-module rotation and unequal sizes, so decoded bodies can have
# interpenetrating branches. With this on, after decoding we drop any module
# whose oriented bounding box overlaps a NON-parent module (and its subtree) --
# the Luo et al. "if occupied, don't place, branch ends" rule with real geometry.
ENFORCE_NO_OVERLAP = True
OVERLAP_PEN_THR = 0.005          # metres; ignore mere face-touching, flag real overlap

# Dynamic self-collision penalty. The static prune above only removes rest-pose
# overlaps; once the hinges swing, non-adjacent modules can still intersect and
# their contact impulses inflate displacement (the fitness-inflation exploit).
# With this on, each rollout's fitness is multiplied by (1 - duty), where duty is
# the fraction of scored steps (t >= BUFFER_FALL) with >=1 non-adjacent
# self-contact penetrating deeper than OVERLAP_PEN_THR -- a parameter-free,
# scale-invariant discount that keeps only the reward earned while NOT
# self-colliding. Applied to positive fitness only (never improves a bad robot).
PENALIZE_SELF_COLLISION = True

PHASE1_GENERATIONS = 80          # morphology search, learning OFF (match the ES sibling)
PHASE2_GENERATIONS = 50          # Baldwinian GA, learning ON       (Table 2 "Generations" = 30)
# Inner learning budget per evaluation (inner CMA assessments per individual).
# With the single-direction objective each eval is ONE rollout (not 3), so this
# is now the dominant cost knob. Effort vs the original 3-direction ES-64 run,
# at mu=96 / lambda=48 / 50+30 gens, single direction, both Phase-2 branches:
#     learn = 48  ->  ~0.41x ES
#     learn = 60  ->  ~0.51x ES   ("half")
#     learn = 100 ->  ~0.83x ES   (the paper's budget; still cheaper than the ES)
# Slide it to trade learning depth for compute.
LEARNING_GENERATIONS = 30

# Phase-2 selection objectives. One branch is run per entry, all warm-started
# from the SAME Phase-1 final population (see run_experiment). 'fitness' =
# faithful Luo Evolution+Learning; 'ldelta' = plasticity-selection extension;
# 'nsga2' = NSGA-II Pareto selection on both objectives at once. The two scalar
# entries stay as the single-objective reference conditions for 'nsga2'.
PHASE2_SELECTIONS = ["fitness", "ldelta", "nsga2"]

# ---------------------------------------------------------#
# Genome layout: object genes only.
#   block (name, start, end, low, high)
# ---------------------------------------------------------#
_W = frr.w_extend_numels(NMODULES)   # 2*NMODULES*r (low-rank) or 900 (full)
BLOCKS = [
    ("type",       0,                  GENE_SIZE_NDE,        -32.0, 32.0),
    ("conn",       GENE_SIZE_NDE,      2 * GENE_SIZE_NDE,    -32.0, 32.0),
    ("rot",        2 * GENE_SIZE_NDE,  3 * GENE_SIZE_NDE,    -32.0, 32.0),
    ("w_extend",   3 * GENE_SIZE_NDE,  3 * GENE_SIZE_NDE + _W, -1.0, 1.0),
    ("controller", 3 * GENE_SIZE_NDE + _W,
                   3 * GENE_SIZE_NDE + _W + NUMELS_CONTROLLER, -1.0, 1.0),
]
GENOME_LEN = BLOCKS[-1][2]          # total number of object genes

# Per-coordinate bound vectors. The GA works directly in this true space; genes
# are clipped to [LO, HI] after variation. SPAN scales the mutation step.
LO = np.empty(GENOME_LEN, dtype=np.float64)
HI = np.empty(GENOME_LEN, dtype=np.float64)
for _name, _s, _e, _lo, _hi in BLOCKS:
    LO[_s:_e] = _lo
    HI[_s:_e] = _hi
SPAN = HI - LO


# ---------------------------------------------------------#
# Performance functions (copied from the QD/ES drivers so this file is
# standalone and pulls no pyribs dependency)
# ---------------------------------------------------------#

fitness_function_w = 0.1
epsilon = 10e-10
target_directions_deg = [0, 120, 240]
BALANCE_LAMBDA = 1.0 / len(target_directions_deg) / 2


def get_path_length(positions):
    l = 0
    for i in range(len(positions) - 1):
        l += math.dist(positions[i], positions[i + 1])
    return l


def get_params_fitness_tl(positions, target_direction):
    origin = positions[0]
    end_point = positions[-1]
    dx = end_point[0] - origin[0]
    dy = end_point[1] - origin[1]
    delta = np.arctan2(dy, dx)
    diff = delta - target_direction
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    if np.abs(diff) > np.pi:
        theta = 2 * np.pi - np.abs(diff)
    else:
        theta = np.abs(diff)
    gamma = math.dist(origin, end_point)
    alpha = gamma * np.sin(theta)
    beta = gamma * np.cos(diff)
    l = get_path_length(positions)
    return beta, alpha, theta, l


def formula_tl(beta, alpha, theta, l, w=fitness_function_w, epsilon=epsilon):
    return (np.abs(beta) / (l + epsilon)) * ((beta / (theta + 1)) - (w * alpha))


def performance_targeted_loc(robot, target_direction_deg=0, n_steps=steps):
    robot.model.opt.timestep = dt
    mujoco.mj_resetData(robot.model, robot.data)
    mujoco.mj_forward(robot.model, robot.data)
    robot.controller.reset()

    geom2mod, parent = _self_collision_caches(robot)
    n_act = len(robot.actuators)
    positions = []
    collide_steps = 0
    scored_steps = 0
    for t in range(n_steps):
        mujoco.mj_step(robot.model, robot.data)
        if t >= BUFFER_FALL:
            scored_steps += 1
            if PENALIZE_SELF_COLLISION and _step_self_collides(robot.data, geom2mod, parent):
                collide_steps += 1
            robot.set_forward_vec()
            alpha_angle = robot.get_alpha_angle(target_direction_deg)
            cpg_t = (t - BUFFER_FALL) * dt
            angles = robot.controller.forward(alpha_angle=alpha_angle, time=cpg_t)
            for i in range(n_act):
                robot.data.ctrl[i] = float(angles[i])
            core_pos = robot.data.xpos[robot.core_id].copy()
            positions.append(core_pos[0:2])

    beta, alpha, theta, l = get_params_fitness_tl(
        positions, np.deg2rad(target_direction_deg)
    )
    fit = formula_tl(beta, alpha, theta, l)
    return _finalize_fitness(robot, fit, collide_steps, scored_steps)


def performance_3p_tl(robot, n_steps=steps):
    if robot.controller is None:
        robot._last_raw_fit = -float("inf")
        robot._last_duty = 0.0
        return -float("inf")
    fitnesses, raws, duties = [], [], []
    for deg in target_directions_deg:
        f = performance_targeted_loc(robot, target_direction_deg=deg, n_steps=n_steps)
        fitnesses.append(f)
        raws.append(getattr(robot, "_last_raw_fit", f))
        duties.append(getattr(robot, "_last_duty", 0.0))
    if any(not math.isfinite(f) for f in fitnesses):
        robot._last_raw_fit = -float("inf")
        robot._last_duty = 0.0
        return -float("inf")
    total = sum(fitnesses)
    if BALANCE_LAMBDA > 0.0 and len(fitnesses) > 1:
        mean_f = total / len(fitnesses)
        l1_dev = sum(abs(f - mean_f) for f in fitnesses)
        total -= BALANCE_LAMBDA * l1_dev
    # Aggregate the per-direction logging stash: raw = summed undiscounted score
    # (parallel to total = summed shaped score), duty = mean collision fraction.
    robot._last_raw_fit = float(sum(raws))
    robot._last_duty = float(sum(duties) / len(duties))
    return float(total)


def performance_initial_heading(robot, n_steps=steps):
    """Single-direction targeted locomotion, one rollout instead of three.

    The target direction is the robot's OWN forward heading, captured once at the
    end of the buffer-fall settling period (the first step with t == BUFFER_FALL)
    and then held fixed for the rest of the rollout -- "walk straight in whatever
    direction you're facing after you've settled". Both the steering error
    (``get_alpha_angle``) and the fitness scoring (``get_params_fitness_tl``) use
    this same frozen heading, and the trajectory origin is the core position at
    that same instant, so the objective is internally consistent.

    Deterministic: the settled pose -- and therefore the captured heading -- is a
    pure function of the decoded body and the fixed reset state; there is no RNG
    in the rollout. This replaces the world-frame 3-direction ``performance_3p_tl``
    (and so drops the per-evaluation cost ~3x; no BALANCE_LAMBDA term, since
    there is a single direction).
    """
    if robot.controller is None:
        return -float("inf")

    robot.model.opt.timestep = dt
    mujoco.mj_resetData(robot.model, robot.data)
    mujoco.mj_forward(robot.model, robot.data)
    robot.controller.reset()

    geom2mod, parent = _self_collision_caches(robot)
    n_act = len(robot.actuators)
    positions = []
    target_dir_deg = None            # frozen once, at t == BUFFER_FALL
    collide_steps = 0
    scored_steps = 0
    for t in range(n_steps):
        mujoco.mj_step(robot.model, robot.data)
        if t >= BUFFER_FALL:
            scored_steps += 1
            if PENALIZE_SELF_COLLISION and _step_self_collides(robot.data, geom2mod, parent):
                collide_steps += 1
            robot.set_forward_vec()
            if target_dir_deg is None:
                # capture the settled heading at the first post-fall step and hold it
                fwd = robot.forward_vec
                target_dir_deg = float(np.rad2deg(np.arctan2(fwd[1], fwd[0])))
            alpha_angle = robot.get_alpha_angle(target_dir_deg)
            cpg_t = (t - BUFFER_FALL) * dt
            angles = robot.controller.forward(alpha_angle=alpha_angle, time=cpg_t)
            for i in range(n_act):
                robot.data.ctrl[i] = float(angles[i])
            core_pos = robot.data.xpos[robot.core_id].copy()
            positions.append(core_pos[0:2])

    if not positions or target_dir_deg is None:
        robot._last_raw_fit = -float("inf")
        robot._last_duty = 0.0
        return -float("inf")

    beta, alpha, theta, l = get_params_fitness_tl(
        positions, np.deg2rad(target_dir_deg)
    )
    fit = formula_tl(beta, alpha, theta, l)
    return _finalize_fitness(robot, fit, collide_steps, scored_steps)


# ---------------------------------------------------------#
# Morphological descriptors (logged only)
# ---------------------------------------------------------#

def _real_module_subgraph(robot: full_robot) -> nx.DiGraph:
    keep = [
        n for n, attrs in robot.body_phenotype.nodes(data=True)
        if attrs.get("type") != ModuleType.NONE.name
    ]
    return robot.body_phenotype.subgraph(keep)


def descriptor_branching(robot: full_robot) -> float:
    g = _real_module_subgraph(robot)
    internal = [n for n in g.nodes() if g.out_degree(n) >= 1]
    if not internal:
        return 0.0
    forks = sum(1 for n in internal if g.out_degree(n) >= 2)
    return forks / len(internal)


def descriptor_limb_lengths(robot: full_robot) -> dict:
    empty = {"n_limbs": 0, "max_length": 0, "mean_length": 0.0, "std_length": 0.0}
    g = _real_module_subgraph(robot)
    if 0 not in g.nodes():
        return empty
    depths = nx.single_source_shortest_path_length(g, 0)
    leaves = [n for n in g.nodes() if n != 0 and n in depths and g.out_degree(n) == 0]
    if not leaves:
        return empty
    lengths = np.array([depths[n] for n in leaves], dtype=float)
    return {
        "n_limbs": int(lengths.size),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "std_length": float(lengths.std()),
    }


def get_measures(robot) -> np.ndarray:
    """Descriptors (branching, limb_std_length) -- logged for diversity analysis."""
    if robot.controller is None:
        return np.array([0.0, 0.0], dtype=np.float64)
    return np.array(
        [
            descriptor_branching(robot),
            float(descriptor_limb_lengths(robot)["std_length"]),
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------#
# Inner Baldwinian controller learning (nevergrad CMA)
# ---------------------------------------------------------#

# base_params columns: 0=phase, 1=amplitudes, 2=w. We only learn 0 and 1.
LEARN_BASE_COLS = (0, 1)
LEARN_BASE_COL_NAMES = ("phase", "amplitudes")

# Bounds for the learned CPG groups, enforced in set_learn_params and the inner
# CMA search. Amplitude is capped at the servo output range (±π/2 = CPG_AMP_SCALE):
# above it the NaCPG output saturates its ±π/2 hard clamp and degenerates into a
# snapping square wave -> impulsive servo torque (the "jumping" exploit). Phase is
# an angle, so ±π covers its full range (values outside wrap and are redundant).
CPG_PHASE_LO, CPG_PHASE_HI = -math.pi, math.pi
CPG_AMP_LO,   CPG_AMP_HI   = -math.pi / 2, math.pi / 2


def get_learn_params(controller) -> np.ndarray:
    return (
        controller.base_params[:, list(LEARN_BASE_COLS)]
        .detach().flatten().cpu().numpy()
    )


def set_learn_params(controller, x: np.ndarray) -> None:
    n = controller.base_params.shape[0]
    x_t = torch.as_tensor(x, dtype=torch.float32).view(n, len(LEARN_BASE_COLS))
    new_base = controller.base_params.detach().clone()
    for i, col in enumerate(LEARN_BASE_COLS):
        new_base[:, col] = x_t[:, i]
    # Enforce CPG bounds (cols 0=phase, 1=amplitude) on the executed AND logged
    # params, so a learner step can never push amplitude past the ±π/2 servo range.
    new_base[:, 0].clamp_(CPG_PHASE_LO, CPG_PHASE_HI)
    new_base[:, 1].clamp_(CPG_AMP_LO, CPG_AMP_HI)
    controller.base_params = new_base
    for col, name in zip(LEARN_BASE_COLS, LEARN_BASE_COL_NAMES):
        controller.set_params_by_group(name, new_base[:, col])


def _inner_learn_cma(robot, performance_fun, budget, seed):
    """Run an inner CMA local search on the controller; return learned best.

    Baldwinian: the caller uses the returned post-learning fitness to SCORE the
    individual but discards the learned params w.r.t. the heritable genome
    (they are only logged for replay).

    Returns
    -------
    (post_fit, learned_params_list, learning_curve)
    """
    init_params = get_learn_params(robot.controller).copy()  # rollback snapshot
    # The decoder does NOT clamp phase (only amplitude), so a decoded phase can sit
    # outside [-pi, pi]. WRAP the phase entries into [-pi, pi] instead of clipping
    # them: the CPG uses phase only via cos/sin of phase DIFFERENCES (2*pi-periodic),
    # so wrapping preserves the decoded controller's dynamics exactly, whereas a raw
    # clip (e.g. phase 4.0 -> pi rather than its equivalent -2.28) would silently
    # start the search from a DIFFERENT controller than the one pre_fit scored,
    # biasing the learning delta (post - pre). Phase is col 0 of each [phase, amp]
    # pair, i.e. the even indices of the flat vector.
    init_params[0::2] = (init_params[0::2] + math.pi) % (2.0 * math.pi) - math.pi
    # Per-element bounds for the flat [phase, amp, phase, amp, ...] vector, so the
    # inner CMA samples within the CPG range instead of wandering into the
    # amplitude-saturation regime and being silently clipped by set_learn_params.
    n_joints = robot.controller.base_params.shape[0]
    lower = np.tile([CPG_PHASE_LO, CPG_AMP_LO], n_joints)
    upper = np.tile([CPG_PHASE_HI, CPG_AMP_HI], n_joints)
    init_params = np.clip(init_params, lower, upper)   # amp clamp + phase safety net
    param = ng.p.Array(init=init_params)
    param.set_bounds(lower, upper, method="clipping")
    optimizer = ng.optimizers.CMA(parametrization=param, budget=budget)
    optimizer.parametrization.random_state.seed(seed)
    curve = []

    def objective(x):
        set_learn_params(robot.controller, x)
        f = performance_fun(robot)
        curve.append(float(f) if math.isfinite(f) else -1e30)
        return -f

    recommendation = optimizer.minimize(objective)
    set_learn_params(robot.controller, recommendation.value)
    post = performance_fun(robot)
    if not math.isfinite(post):
        # roll back to the un-learned controller
        set_learn_params(robot.controller, init_params)
        post = performance_fun(robot)
    return post, get_learn_params(robot.controller).tolist(), curve


# ---------------------------------------------------------#
# Geometry-aware collision pruning (see ENFORCE_NO_OVERLAP)
# ---------------------------------------------------------#
import re as _re
_ROBOT_PREFIX = _re.compile(r"^robot\d+_")


def _geom_module(name: str | None) -> int | None:
    """Module index a geom belongs to, parsed from its MuJoCo name.

    Names look like ``robot1_<from>-<to>-<face>-...-<type>``; the module is the
    ``to`` of the last triple. ``robot<N>_core`` -> 0 (IDX_OF_CORE).
    """
    if not name:
        return None
    m = _ROBOT_PREFIX.match(name)
    if not m:
        return None
    body = name[m.end():]
    if body == "core":
        return 0
    ints = [int(p) for p in body.split("-") if p.isdigit()]
    return ints[-2] if len(ints) >= 2 else None


# ---------------------------------------------------------#
# Dynamic self-collision detection + penalty (see PENALIZE_SELF_COLLISION)
# ---------------------------------------------------------#

def _self_collision_caches(robot):
    """Build & cache (geom_id -> module array, child -> parent map) on the robot
    once, for cheap per-step self-collision classification. ``geom2mod[gid]`` is
    the module index for robot geoms and -1 for world/ground geoms. Recomputing
    this per rollout would cost ~30 rebuilds/individual inside the learner."""
    cache = getattr(robot, "_sc_cache", None)
    if cache is not None:
        return cache
    m = robot.model
    geom2mod = np.full(m.ngeom, -1, dtype=np.int64)
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] == 0:
            continue
        mod = _geom_module(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid))
        if mod is not None:
            geom2mod[gid] = mod
    parent = {c: p for p, c in robot.body_phenotype.edges()}
    robot._sc_cache = (geom2mod, parent)
    return robot._sc_cache


def _step_self_collides(data, geom2mod, parent) -> bool:
    """True if the CURRENT contact set has >=1 real self-collision: a contact
    between two DIFFERENT, non parent-child robot modules, penetrating deeper
    than OVERLAP_PEN_THR. Ground contacts, adjacent-module touching and mere
    face-grazing are all ignored."""
    for c in range(data.ncon):
        con = data.contact[c]
        if con.dist >= -OVERLAP_PEN_THR:              # grazing / not real overlap
            continue
        m1 = int(geom2mod[con.geom1])
        m2 = int(geom2mod[con.geom2])
        if m1 < 0 or m2 < 0 or m1 == m2:              # world/ground or same module
            continue
        if parent.get(m1) == m2 or parent.get(m2) == m1:   # adjacent parent-child
            continue
        return True
    return False


def _apply_self_collision_penalty(fit, collide_steps, scored_steps) -> float:
    """Multiplicative discount: keep only the share of a POSITIVE reward earned
    while not self-colliding -- ``fit * (1 - duty)``. Dead/negative fitness is
    returned unchanged so the discount can never *improve* a bad robot."""
    if not math.isfinite(fit):
        return -float("inf")
    if PENALIZE_SELF_COLLISION and fit > 0.0 and scored_steps > 0:
        duty = collide_steps / scored_steps
        fit = fit * (1.0 - duty)
    return float(fit)


def _finalize_fitness(robot, raw_fit, collide_steps, scored_steps) -> float:
    """Return the SHAPED fitness (selection/learning signal) exactly as before,
    but also stash on the robot the RAW undiscounted task score and the collision
    duty so both stay recoverable in the logs. ``fitness`` (shaped) is what the GA
    ranks and the inner CMA optimises; ``_last_raw_fit`` is the true locomotion
    performance (no self-collision discount) and ``_last_duty`` = fraction of
    scored steps that self-collided (how hard the penalty bit). The stash is a
    pure add-on: it never changes the returned value."""
    shaped = _apply_self_collision_penalty(raw_fit, collide_steps, scored_steps)
    robot._last_raw_fit = float(raw_fit) if math.isfinite(raw_fit) else -float("inf")
    robot._last_duty = (collide_steps / scored_steps) if scored_steps > 0 else 0.0
    return shaped


def _module_obbs(robot) -> dict:
    """{module_idx: [(center, half_extents, R3x3), ...]} from the live geoms."""
    m, d = robot.model, robot.data
    mujoco.mj_forward(m, d)
    obbs: dict = {}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        mod = _geom_module(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
        if mod is None:
            continue
        obbs.setdefault(mod, []).append(
            (d.geom_xpos[g].copy(), m.geom_size[g].copy(),
             d.geom_xmat[g].reshape(3, 3).copy())
        )
    return obbs


def _obb_penetration(A, B) -> float:
    """OBB-OBB penetration depth via the separating-axis theorem (0 = separated)."""
    cA, hA, RA = A
    cB, hB, RB = B
    aA = [RA[:, i] for i in range(3)]
    aB = [RB[:, i] for i in range(3)]
    cands = list(aA) + list(aB)
    for a in aA:
        for b in aB:
            x = np.cross(a, b)
            n = np.linalg.norm(x)
            if n > 1e-6:
                cands.append(x / n)
    t = cB - cA
    min_ov = np.inf
    for L in cands:
        L = L / (np.linalg.norm(L) + 1e-12)
        rA = sum(abs(np.dot(L, aA[i])) * hA[i] for i in range(3))
        rB = sum(abs(np.dot(L, aB[i])) * hB[i] for i in range(3))
        ov = rA + rB - abs(np.dot(L, t))
        if ov <= 0.0:
            return 0.0
        min_ov = min(min_ov, ov)
    return float(min_ov)


def _modules_overlap(obbs, i, j) -> bool:
    return any(_obb_penetration(A, B) > OVERLAP_PEN_THR
               for A in obbs.get(i, []) for B in obbs.get(j, []))


class _ShimDecoder:
    """Decoder whose ``probability_matrices_to_graph`` returns a fixed graph."""
    def __init__(self, graph, num_modules):
        self.graph = graph
        self.num_modules = num_modules

    def probability_matrices_to_graph(self, *_a, **_k):
        return self.graph


def _decode_robot_no_overlap(ind, *, NDE, DECODER, world_cls, SEED):
    """Decode ``ind``; if ENFORCE_NO_OVERLAP, drop cross-branch-colliding modules
    (and their subtrees) and rebuild the pruned body. Returns a full_robot."""
    robot = decode_robot(ind, NDE=NDE, DECODER=DECODER, world_cls=world_cls, seed=SEED)
    if not ENFORCE_NO_OVERLAP or robot.controller is None:
        return robot

    g = robot.body_phenotype
    obbs = _module_obbs(robot)
    if not obbs:
        return robot
    parent = {c: p for p, c in g.edges()}
    order = [n for n in nx.bfs_tree(g, 0).nodes() if n in obbs] if 0 in obbs else list(obbs)

    kept: list[int] = []
    for mod in order:
        if mod == 0:
            kept.append(mod)
            continue
        par = parent.get(mod)
        if par is not None and par not in kept:
            continue                                  # parent was dropped -> drop subtree
        if any(k != par and _modules_overlap(obbs, mod, k) for k in kept):
            continue                                  # collides with a non-parent module
        kept.append(mod)

    if len(kept) == len(obbs):
        return robot                                  # nothing pruned

    pruned = g.subgraph(set(kept)).copy()
    shim = _ShimDecoder(pruned, getattr(DECODER, "num_modules", NMODULES))
    return decode_robot(ind, NDE=NDE, DECODER=shim, world_cls=world_cls, seed=SEED)


# ---------------------------------------------------------#
# Worker: decode -> (optional inner learn) -> fitness + descriptors
# Module-level so joblib only pickles the small args we pass.
# `geno` is the true-range genome (already clipped to bounds by the GA).
# ---------------------------------------------------------#

def _evaluate_one(
    geno,
    *,
    NDE,
    DECODER,
    world_cls,
    SEED,
    performance_fun,
    learn: bool,
    learning_generations: int,
) -> dict:
    # fitness      = SHAPED score (self-collision-discounted); GA selects on this.
    # fitness_raw  = RAW task performance (no discount); for reporting/plots only.
    # collision_duty = fraction of scored steps that self-collided. The *_pre[_raw]
    # variants are the same quantities measured BEFORE inner learning.
    fail = {
        "geno": list(geno),
        "fitness": -float("inf"),
        "fitness_pre": -float("inf"),
        "fitness_raw": -float("inf"),
        "fitness_pre_raw": -float("inf"),
        "collision_duty": 0.0,
        "collision_duty_pre": 0.0,
        "cpg_params": [],
        "learning_curve": [],
        "measures": [0.0, 0.0],
    }
    ind = Individual()
    ind.genotype = list(geno)
    robot = _decode_robot_no_overlap(ind, NDE=NDE, DECODER=DECODER, world_cls=world_cls, SEED=SEED)
    if robot.controller is None:
        return fail

    pre_fit = performance_fun(robot)
    pre_raw = getattr(robot, "_last_raw_fit", pre_fit)
    pre_duty = float(getattr(robot, "_last_duty", 0.0))
    measures = np.asarray(get_measures(robot), dtype=np.float64).tolist()

    if not learn:
        f = float(pre_fit) if math.isfinite(pre_fit) else -float("inf")
        r = float(pre_raw) if math.isfinite(pre_raw) else -float("inf")
        return {
            "geno": list(geno),
            "fitness": f,
            "fitness_pre": f,
            "fitness_raw": r,
            "fitness_pre_raw": r,
            "collision_duty": pre_duty,
            "collision_duty_pre": pre_duty,
            "cpg_params": [],
            "learning_curve": [],
            "measures": measures,
        }

    if not math.isfinite(pre_fit):
        # nothing to learn from a dead-on-arrival robot
        return {**fail, "measures": measures}

    post_fit, learned_params, curve = _inner_learn_cma(
        robot, performance_fun, learning_generations, SEED
    )
    post_raw = getattr(robot, "_last_raw_fit", post_fit)
    post_duty = float(getattr(robot, "_last_duty", 0.0))
    return {
        "geno": list(geno),
        "fitness": float(post_fit) if math.isfinite(post_fit) else -float("inf"),
        "fitness_pre": float(pre_fit),
        "fitness_raw": float(post_raw) if math.isfinite(post_raw) else -float("inf"),
        "fitness_pre_raw": float(pre_raw) if math.isfinite(pre_raw) else -float("inf"),
        "collision_duty": post_duty,
        "collision_duty_pre": pre_duty,
        "cpg_params": learned_params,
        "learning_curve": list(curve),
        "measures": measures,
    }


# ---------------------------------------------------------#
# Individual <-> evaluation-record helpers
# ---------------------------------------------------------#

def _record_to_individual(rec: dict) -> Individual:
    """Build a fresh (logged) Individual from an evaluation record."""
    ind = Individual()
    ind.genotype = list(rec["geno"])           # true-range object genome
    ind.fitness = rec["fitness"]
    ind.fitness_pre = rec["fitness_pre"]
    ind.CPG_params = rec.get("cpg_params", [])
    ind.tags = {
        "learning_curve": rec.get("learning_curve", []),
        "measures": rec.get("measures", []),
        # Raw (undiscounted) task performance + self-collision duty, so the true
        # locomotion score stays recoverable alongside the shaped `fitness`.
        "fitness_raw": rec.get("fitness_raw"),
        "fitness_pre_raw": rec.get("fitness_pre_raw"),
        "collision_duty": rec.get("collision_duty"),
        "collision_duty_pre": rec.get("collision_duty_pre"),
    }
    ind.requires_eval = False
    return ind


# ---------------------------------------------------------#
# Genetic algorithm operation: one ``__call__`` == one generation
# (select -> crossover -> mutate -> evaluate -> (mu+lambda) survive)
# ---------------------------------------------------------#

class GeneticAlgorithm:
    """Stateful (mu+lambda) GA operation, mirroring SepCMAES's interface.

    The actual evolving state -- the mu evaluated parents (a list of evaluation
    record dicts) -- lives on the operation (``self.population``), exactly as
    SepCMAES owns its ``self.es``. TrackedEA's per-generation DB/CSV round-trip is
    used only for logging: incoming rows are flagged dead and the freshly evolved
    mu population is returned as alive rows.
    """

    def __init__(
        self,
        *,
        performance_fun,
        NDE,
        DECODER,
        world_cls,
        SEED,
        learn: bool,
        learning_generations: int = LEARNING_GENERATIONS,
        selection: str = "fitness",   # 'fitness' | 'ldelta' | 'nsga2'
        rng: np.random.Generator,
        n_jobs: int = -1,
    ) -> None:
        self.performance_fun = performance_fun
        self.NDE = NDE
        self.DECODER = DECODER
        self.world_cls = world_cls
        self.SEED = SEED
        self.learn = learn
        self.learning_generations = learning_generations
        self.selection = selection
        self.nsga2 = (selection == "nsga2")   # Pareto path vs scalar `_value`
        self.rng = rng
        self.n_jobs = n_jobs
        self.population: list[dict] = []   # mu evaluated records, sorted best-first
        self.best_record: dict | None = None

    # ---- selection objective --------------------------------------------- #

    def _value(self, rec: dict) -> float:
        """Quantity to MAXIMISE for ranking. 'fitness' = post-learning score,
        'ldelta' = post - pre. Non-finite fitness ranks last."""
        f = rec["fitness"]
        if not math.isfinite(f):
            return -float("inf")
        if self.selection == "ldelta":
            return f - rec["fitness_pre"]
        return f

    # ---- NSGA-II multi-objective selection (selection == 'nsga2') --------- #
    # Objectives, both MAXIMISED: (post-learning fitness, ldelta = post - pre).
    # The scalar path above is untouched; these run only on the 'nsga2' branch.
    # Dead/non-finite individuals are sunk to a sentinel worst objective so they
    # never dominate a live robot and always fall into the last front.

    def _objectives(self, rec: dict) -> tuple:
        f = rec["fitness"]
        pre = rec["fitness_pre"]
        if not (math.isfinite(f) and math.isfinite(pre)):
            return (-math.inf, -math.inf)
        return (f, f - pre)

    @staticmethod
    def _dominates(a: tuple, b: tuple) -> bool:
        """True if objective vector `a` Pareto-dominates `b` (maximisation):
        no worse in every objective and strictly better in at least one."""
        return (all(x >= y for x, y in zip(a, b))
                and any(x > y for x, y in zip(a, b)))

    def _fast_nondominated_sort(self, objs: list) -> list:
        """Deb et al. (2002) fast non-dominated sort. Returns a list of fronts
        (each a list of indices into `objs`), best (non-dominated) front first."""
        n = len(objs)
        dominated = [[] for _ in range(n)]     # solutions each i dominates
        n_dom = [0] * n                        # how many solutions dominate i
        fronts: list[list[int]] = [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates(objs[p], objs[q]):
                    dominated[p].append(q)
                elif self._dominates(objs[q], objs[p]):
                    n_dom[p] += 1
            if n_dom[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt: list[int] = []
            for p in fronts[i]:
                for q in dominated[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        fronts.pop()                           # drop the trailing empty front
        return fronts

    @staticmethod
    def _crowding(objs: list, front: list) -> dict:
        """Crowding distance within one front (Deb et al., 2002). Boundary points
        get +inf; degenerate or non-finite objective ranges are skipped so the
        dead front (all -inf) never produces NaNs."""
        dist = {i: 0.0 for i in front}
        n = len(front)
        if n <= 2:
            for i in front:
                dist[i] = math.inf
            return dist
        m = len(objs[front[0]])
        for k in range(m):
            order = sorted(front, key=lambda i: objs[i][k])
            dist[order[0]] = math.inf
            dist[order[-1]] = math.inf
            rng = objs[order[-1]][k] - objs[order[0]][k]
            if not math.isfinite(rng) or rng <= 0.0:
                continue
            for j in range(1, n - 1):
                i = order[j]
                if math.isinf(dist[i]):
                    continue
                dist[i] += (objs[order[j + 1]][k] - objs[order[j - 1]][k]) / rng
        return dist

    def _nsga2_annotate(self, records: list) -> list:
        """Write `_rank` (front index) and `_crowd` (crowding distance) onto every
        record in `records`; return the fronts."""
        objs = [self._objectives(r) for r in records]
        fronts = self._fast_nondominated_sort(objs)
        for rank, front in enumerate(fronts):
            cd = self._crowding(objs, front)
            for i in front:
                records[i]["_rank"] = rank
                records[i]["_crowd"] = cd[i]
        return fronts

    @staticmethod
    def _crowded_better(a: dict, b: dict) -> bool:
        """Crowded-comparison operator: `a` beats `b` if it sits on a better
        (lower-index) front, or the same front but a less crowded region."""
        if a["_rank"] != b["_rank"]:
            return a["_rank"] < b["_rank"]
        return a["_crowd"] > b["_crowd"]

    @staticmethod
    def _nsga2_best(pop: list) -> dict | None:
        """Reporting representative under two objectives (there is no single
        'best'): the highest-fitness member of the first Pareto front."""
        if not pop:
            return None
        front0 = [r for r in pop if r.get("_rank", 0) == 0] or pop
        return max(front0, key=lambda r: r["fitness"]
                   if math.isfinite(r["fitness"]) else -math.inf)

    def _survive_nsga2(self, union: list) -> None:
        """(mu+lambda) environmental selection, NSGA-II style: fill the next
        population front by front, truncating the first overflowing front by
        crowding distance (least-crowded kept)."""
        fronts = self._nsga2_annotate(union)
        survivors: list[int] = []
        for front in fronts:
            if len(survivors) + len(front) <= POP_SIZE:
                survivors.extend(front)
            else:
                room = POP_SIZE - len(survivors)
                front.sort(key=lambda i: union[i]["_crowd"], reverse=True)
                survivors.extend(front[:room])
                break
        self.population = [union[i] for i in survivors]
        # Re-rank the mu survivors so `_rank`/`_crowd` are self-consistent for the
        # next generation's mating tournament and for `best_record`.
        self._nsga2_annotate(self.population)
        self.population.sort(key=lambda r: (r["_rank"], -r["_crowd"]))
        self.best_record = self._nsga2_best(self.population)

    # ---- evaluation ------------------------------------------------------ #

    def _evaluate(self, genos: list) -> list[dict]:
        return Parallel(n_jobs=self.n_jobs)(
            delayed(_evaluate_one)(
                g,
                NDE=self.NDE,
                DECODER=self.DECODER,
                world_cls=self.world_cls,
                SEED=self.SEED,
                performance_fun=self.performance_fun,
                learn=self.learn,
                learning_generations=self.learning_generations,
            )
            for g in genos
        )

    def _resort(self) -> None:
        if self.nsga2:
            self._nsga2_annotate(self.population)
            self.population.sort(key=lambda r: (r["_rank"], -r["_crowd"]))
            self.best_record = self._nsga2_best(self.population)
            return
        self.population.sort(key=self._value, reverse=True)
        self.best_record = self.population[0] if self.population else None

    def seed_population(self, genomes: list) -> list[dict]:
        """Evaluate `genomes` (under this op's learn flag) and install them as the
        current mu population. Used to build generation 0 (random for Phase 1, or
        the warm-started Phase-1 survivors for Phase 2)."""
        genos = [list(np.asarray(g, dtype=np.float64)) for g in genomes]
        self.population = self._evaluate(genos)
        self._resort()
        return self.population

    # ---- variation ------------------------------------------------------- #

    def _tournament(self, pool: list[dict]) -> dict:
        """Binary (size-k) tournament WITH replacement: pick k random, keep best.
        Ranking is the scalar `_value` for 'fitness'/'ldelta', or the NSGA-II
        crowded-comparison (front, then crowding) for 'nsga2'."""
        idx = self.rng.integers(0, len(pool), size=TOURNAMENT_SIZE)
        best = pool[idx[0]]
        if self.nsga2:
            for i in idx[1:]:
                if self._crowded_better(pool[i], best):
                    best = pool[i]
            return best
        best_v = self._value(best)
        for i in idx[1:]:
            v = self._value(pool[i])
            if v > best_v:
                best, best_v = pool[i], v
        return best

    def _crossover(self, ga, gb) -> np.ndarray:
        """Uniform per-gene crossover -> one child. With prob (1-CROSSOVER_PROB)
        the child is a clone of parent A (no recombination)."""
        a = np.asarray(ga, dtype=np.float64)
        b = np.asarray(gb, dtype=np.float64)
        if self.rng.random() >= CROSSOVER_PROB:
            return a.copy()
        take_a = self.rng.random(a.shape) < 0.5
        return np.where(take_a, a, b)

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        """Per-individual Gaussian mutation (prob MUTATION_PROB), per-gene step
        scaled by each gene's range, then clip to bounds."""
        if self.rng.random() < MUTATION_PROB:
            gene_mask = self.rng.random(x.shape) < MUT_GENE_PROB
            noise = self.rng.normal(0.0, MUT_SIGMA, size=x.shape) * SPAN
            x = x + gene_mask * noise
        return np.clip(x, LO, HI)

    def _geno_key(self, x) -> tuple:
        """Hashable identity key for a genome: genes rounded to DEDUP_DECIMALS so
        byte-identical clones collide but genuine mutants (which differ by
        ~MUT_SIGMA*SPAN) do not."""
        return tuple(np.round(np.asarray(x, dtype=np.float64), DEDUP_DECIMALS))

    def _force_mutate(self, x: np.ndarray) -> np.ndarray:
        """Like _mutate but ALWAYS perturbs >=1 gene (bypasses the MUTATION_PROB
        gate). Used only to break an exact duplicate."""
        x = np.asarray(x, dtype=np.float64).copy()
        gene_mask = self.rng.random(x.shape) < MUT_GENE_PROB
        if not gene_mask.any():
            gene_mask[self.rng.integers(x.shape[0])] = True
        noise = self.rng.normal(0.0, MUT_SIGMA, size=x.shape) * SPAN
        return np.clip(x + gene_mask * noise, LO, HI)

    def _distinct(self, x: np.ndarray, seen: set) -> np.ndarray:
        """Return a genome whose key isn't already in `seen` (re-mutating on
        collision), then register it. Repairs only true clones; genuine variants
        pass straight through. No-op when DEDUP_POPULATION is off."""
        x = np.asarray(x, dtype=np.float64)
        if not DEDUP_POPULATION:
            return x
        key = self._geno_key(x)
        tries = 0
        while key in seen and tries < 20:
            x = self._force_mutate(x)
            key = self._geno_key(x)
            tries += 1
        seen.add(key)
        return x

    # ---- one generation -------------------------------------------------- #

    def step(self) -> None:
        parents = self.population
        # No identical individuals: force each offspring genotype-distinct from
        # the parents and from already-created siblings, BEFORE evaluation.
        seen = {self._geno_key(p["geno"]) for p in parents} if DEDUP_POPULATION else set()
        child_genos = []
        for _ in range(OFFSPRING_SIZE):
            pa = self._tournament(parents)
            pb = self._tournament(parents)
            child = self._mutate(self._crossover(pa["geno"], pb["geno"]))
            child = self._distinct(child, seen)
            child_genos.append(child.tolist())

        offspring = self._evaluate(child_genos)

        # (mu + lambda) elitist survivor selection: best mu of the union. For
        # 'nsga2' this is Pareto rank + crowding (Deb et al., 2002); otherwise
        # scalar `_value` truncation.
        union = parents + offspring
        if self.nsga2:
            self._survive_nsga2(union)
        else:
            union.sort(key=self._value, reverse=True)
            self.population = union[:POP_SIZE]
            self.best_record = self.population[0]

    # ---- operation interface --------------------------------------------- #

    def __call__(self, population, **_) -> Population:
        # Flag the previous generation's snapshot rows dead so TrackedEA's
        # alive-filtered fetch doesn't accumulate them.
        dead = []
        for ind in population:
            ind.alive = False
            dead.append(ind)

        self.step()

        to_log = (self.population if LOG_WHOLE_POPULATION
                  else self.population[: max(1, POP_SIZE // 2)])
        logged = [_record_to_individual(r) for r in to_log]
        return Population(logged + dead)


# ---------------------------------------------------------#
# Driver
# ---------------------------------------------------------#

def _run_ga_phase(op: GeneticAlgorithm, num_steps, *, learning, tag, seed):
    """Log the generation-0 population (op.population), then run ``num_steps`` GA
    generations. ``op`` must already be seeded (op.seed_population(...))."""
    ea = TrackedEA(
        population=Population([_record_to_individual(r) for r in op.population]),
        operations=[op],
        pop_size=POP_SIZE,
        learning=learning,
        ldelta=False,
        learning_epochs=op.learning_generations if learning else 0,
        tag=tag,
        seed=seed,
        num_steps=num_steps,
        log_dir=LOG_DIR,
    )
    # Constructor (db_handling="delete") has truncated the CSV; this append starts
    # a clean file before ea.run() writes gen 1+. Fresh Individuals avoid the
    # DetachedInstanceError from re-logging DB-bound rows.
    log_alive_population_csv(
        Population([_record_to_individual(r) for r in op.population]), 0, ea.csv_path
    )
    ea.run()
    return op


def _world_tag(world_cls) -> str:
    """Short terrain tag for the CSV filename, derived from the world class, so
    each terrain lands in its own file even when LOG_DIR is shared."""
    name = world_cls.__name__
    known = {
        "SimpleFlatWorld": "flat",
        "RuggedTerrainWorld": "rugged",
        "RuggedTiltedWorld": "ruggedtilted",
        "SimpleTiltedWorld": "tilted",
    }
    if name in known:
        return known[name]
    return name.replace("World", "").replace("Terrain", "").lower() or name.lower()


def run_experiment(seed, world_cls, performance_fun):
    SEED = seed
    WTAG = _world_tag(world_cls)   # terrain tag -> keeps each world's CSVs separate
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # SQLite tracking DB on node-local scratch, unique per process. Avoids the
    # GPFS readonly/locking failures and cross-seed collisions from sharing a
    # single ./__data__/database.db (CSV results still go to LOG_DIR on GPFS).
    config.output_folder = (
        Path(os.environ.get("TMPDIR", "/tmp")) / f"ea_db_seed{SEED}_{os.getpid()}"
    )
    config.output_folder.mkdir(parents=True, exist_ok=True)

    NDE = nde(number_of_modules=NMODULES, genotype_size=GENE_SIZE_NDE)
    DECODER = whpd(num_modules=NMODULES)

    # ---------------------------------------------------------------- #
    # Phase 1: GA, learning OFF -- morphology search (== "Evolution-Only").
    # ---------------------------------------------------------------- #
    print(f"[seed={SEED}] Phase 1 -- GA, learning OFF ({PHASE1_GENERATIONS} gens)")
    init_rng = np.random.default_rng(SEED)
    init_genomes = [init_rng.uniform(LO, HI) for _ in range(POP_SIZE)]
    op1 = GeneticAlgorithm(
        performance_fun=performance_fun, NDE=NDE, DECODER=DECODER,
        world_cls=world_cls, SEED=SEED, learn=False,
        learning_generations=0, selection="fitness",
        rng=np.random.default_rng(SEED),
    )
    op1.seed_population(init_genomes)
    _run_ga_phase(op1, PHASE1_GENERATIONS,
                  learning=False, tag=f"ga_p1_{WTAG}", seed=SEED)
    best1 = op1.best_record
    print(f"[seed={SEED}] Phase 1 done -- best fitness {best1['fitness']:.4f}")

    # ---------------------------------------------------------------- #
    # Phase 2: one branch per selection objective, each WARM-STARTED from the
    # SAME Phase-1 final population. The expensive Phase-1 search is shared;
    # branches differ only in the selection objective driving the GA. The
    # inherited population is re-evaluated WITH learning ON to get a fair gen-0.
    # ---------------------------------------------------------------- #
    warm_genomes = [r["geno"] for r in op1.population]
    for sel in PHASE2_SELECTIONS:
        op2 = GeneticAlgorithm(
            performance_fun=performance_fun, NDE=NDE, DECODER=DECODER,
            world_cls=world_cls, SEED=SEED, learn=True,
            learning_generations=LEARNING_GENERATIONS, selection=sel,
            # Same RNG seed for every Phase-2 branch, so the branches differ ONLY
            # by their selection objective (not by the random draws feeding
            # tournament/crossover/mutation). The inner CMA learner is already
            # seeded identically across branches (SEED), so the whole Phase-2
            # comparison is now a controlled A/B/C on the selection rule alone.
            rng=np.random.default_rng(SEED + 1),
        )
        op2.seed_population(warm_genomes)
        print(f"[seed={SEED}] Phase 2 [{sel}] running ({PHASE2_GENERATIONS} gens)")
        _run_ga_phase(op2, PHASE2_GENERATIONS,
                      learning=True, tag=f"ga_p2_{sel}_{WTAG}", seed=SEED)
        best = op2.best_record
        print(f"[seed={SEED}] Phase 2 [{sel}] done -- "
              f"best fitness {best['fitness']:.4f} "
              f"(delta {best['fitness'] - best['fitness_pre']:.4f})")


if __name__ == "__main__":
    seed = 3 #int(sys.argv[1])
    print(f"Running seed={seed}")
    # Single-direction objective (deterministic, ~3x cheaper than performance_3p_tl).
    # Swap to performance_3p_tl for the original 3-direction (0/120/240 deg) task.
    run_experiment(seed, SimpleFlatWorld, performance_initial_heading)
