"""Render the best individual(s) of the GA driver's CSVs into MP4s.

Faithful to EA_FINAL_ga.py: it reuses that driver's overlap-pruning decode
(``_decode_robot_no_overlap``), its controller-param setter, and its
*frozen own-heading* rollout (``performance_initial_heading``) so the video
shows the exact behaviour that was scored.

Each seed N produces three Phase-2 branches (``PHASE2_SELECTIONS``):
``learning{N}_{config}_ga_p2_{fitness,ldelta,nsga2}_flat.csv``. Run with no
arguments to render, in one go, the best ``DEFAULT_RANKS`` (top 2 by last-gen
fitness) of *every* variant for *every* seed in ``DEFAULT_SEEDS``. The
reproduction seed is taken from the ``learning{N}`` filename (``run_experiment``
seeds torch/np with exactly that N), so seed 2's files decode with seed 2 — not
a hardcoded 1.

Usage:
    python render_ga_best.py                         # run-once: DEFAULT_SEEDS, best 2 of every variant
    python render_ga_best.py --seed 7                # all variants of seed 7, best 2 each
    python render_ga_best.py --seeds 7 9 12 --ranks 1 2 3
    python render_ga_best.py --seed 7 --variants fitness ldelta
    python render_ga_best.py --dry-run               # list what would render, no MuJoCo/ffmpeg
    python render_ga_best.py experiment_data/learning7_96-50_30_ga_p2_ldelta_flat.csv
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

# genotype / learning_curve cells can exceed the default 128 KB csv field cap.
csv.field_size_limit(sys.maxsize)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "ariel" / "src"))
sys.path.insert(0, str(_HERE / "ariel"))

import numpy as np
import torch
import mujoco

# Reuse the driver verbatim (import-safe: its run code is under __main__).
import EA_FINAL_ga as ga
from ariel.ec import Individual
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding as nde
from ariel.body_phenotypes.robogen_lite.decoders.weighted_decoding import (
    WeightedHighProbabilityDecoder as whpd,
)
from ariel.simulation.environments import SimpleFlatWorld

# Capture one frame every STEPS_PER_FRAME sim steps; the output framerate is
# derived from the driver's dt so the video duration equals the real evaluated
# time exactly (steps * dt), i.e. real-time playback with no rounding drift.
STEPS_PER_FRAME = 2
WIDTH, HEIGHT = 640, 480
DATA_DIR = _HERE / "experiment_data"
OUT_DIR = _HERE / "videos_ga"
# The three Phase-2 selection branches EA_FINAL_ga.py writes per seed
# (PHASE2_SELECTIONS). Each seed N produces one CSV per variant.
VARIANTS = ("fitness", "ldelta", "nsga2")
DEFAULT_CONFIG = "96-50_30"   # POP-GENS_MODULES token in the filename
# Seeds rendered in run-once mode (no CSV / --seed / --seeds given).
DEFAULT_SEEDS = (7, 9, 10, 11, 12, 13)
# Ranks rendered per CSV by default: the best two of the last generation.
DEFAULT_RANKS = (1, 2)


def seed_from_name(path):
    """Reproduction seed = the number in 'learning{N}'. run_experiment() seeds
    torch/np with exactly this N before building the NDE/DECODER, so the decode
    is only reproducible with the matching seed. None if the name has no tag."""
    m = re.search(r"learning(\d+)", Path(path).stem)
    return int(m.group(1)) if m else None


def discover_csvs(seed, config, variants):
    """learning{seed}_{config}_ga_p2_{variant}_flat.csv for each variant, in
    order. Missing variants (e.g. a seed with only 'fitness') are reported and
    skipped rather than raising."""
    found = []
    for v in variants:
        matches = sorted(DATA_DIR.glob(f"learning{seed}_{config}_ga_p2_{v}_flat.csv"))
        if matches:
            found.extend(matches)
        else:
            print(f"  (no '{v}' file for seed {seed} config {config} — skipping)")
    return found


def load_rank(csv_path, rank):
    """Load the rank-th best individual (1 = best) of the last generation,
    ranked by fitness. Returns None if the last gen has fewer than ``rank``
    individuals, so callers can skip rather than crash."""
    rows = list(csv.DictReader(open(csv_path, newline="")))
    last_gen = max(int(r["generation"]) for r in rows)
    last = [r for r in rows if int(r["generation"]) == last_gen]
    ranked = sorted(last, key=lambda r: float(r["fitness"]), reverse=True)
    if rank < 1 or rank > len(ranked):
        return None
    r = ranked[rank - 1]
    ind = Individual()
    ind.genotype = json.loads(r["genotype"])
    ind.fitness = float(r["fitness"])
    ind.fitness_pre = float(r["fitness_pre"])
    ind.CPG_params = json.loads(r["cpg_params"]) if r.get("cpg_params") else []
    return ind, last_gen, len(last)


def render(ind, out_path, seed):
    # Match the driver's init order: seed BEFORE building NDE/DECODER.
    torch.manual_seed(seed)
    np.random.seed(seed)
    NDE = nde(number_of_modules=ga.NMODULES, genotype_size=ga.GENE_SIZE_NDE)
    DECODER = whpd(num_modules=ga.NMODULES)

    robot = ga._decode_robot_no_overlap(
        ind, NDE=NDE, DECODER=DECODER, world_cls=SimpleFlatWorld, SEED=seed
    )
    if robot.controller is None:
        raise SystemExit("decoded robot has no controller (empty body) — nothing to render")

    if ind.CPG_params:
        cpg = np.asarray(ind.CPG_params, dtype="float32")
        expected = robot.controller.base_params.shape[0] * len(ga.LEARN_BASE_COLS)
        if len(cpg) == expected:
            ga.set_learn_params(robot.controller, cpg)
        else:
            print(f"  WARN CPG_params size {len(cpg)} != {expected}; using genome defaults")

    robot.model.opt.timestep = ga.dt
    mujoco.mj_resetData(robot.model, robot.data)
    mujoco.mj_forward(robot.model, robot.data)
    robot.controller.reset()

    steps_per_frame = STEPS_PER_FRAME
    out_fps = 1.0 / (ga.dt * steps_per_frame)  # real-time: duration == steps*dt
    n_act = len(robot.actuators)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = subprocess.Popen([
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", f"{out_fps:.6f}",
        "-i", "pipe:0",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out_path),
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    target_dir_deg = None  # frozen at t == BUFFER_FALL, exactly like the driver
    with mujoco.Renderer(robot.model, width=WIDTH, height=HEIGHT) as renderer:
        for t in range(ga.steps):
            mujoco.mj_step(robot.model, robot.data)
            if t >= ga.BUFFER_FALL:
                robot.set_forward_vec()
                if target_dir_deg is None:
                    fwd = robot.forward_vec
                    target_dir_deg = float(np.rad2deg(np.arctan2(fwd[1], fwd[0])))
                alpha = robot.get_alpha_angle(target_dir_deg)
                cpg_t = (t - ga.BUFFER_FALL) * ga.dt
                angles = robot.controller.forward(alpha_angle=alpha, time=cpg_t)
                for i in range(n_act):
                    robot.data.ctrl[i] = float(angles[i])
            if t % steps_per_frame == 0:
                cam = mujoco.MjvCamera()
                cam.lookat[:] = robot.data.xpos[robot.core_id].copy()
                cam.distance = 2.0
                cam.azimuth = 135
                cam.elevation = -30
                renderer.update_scene(robot.data, camera=cam)
                ffmpeg.stdin.write(renderer.render().tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csv", nargs="?", default=None,
                   help="single CSV to render; omit to use --seed/--seeds (or the default seed set)")
    p.add_argument("--seed", type=int, default=None,
                   help=f"one experiment seed N -> render every learning{{N}} variant ({'/'.join(VARIANTS)})")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help=f"several seeds at once (default when nothing is given: {' '.join(map(str, DEFAULT_SEEDS))})")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS),
                   help=f"selection variants to render (default: {' '.join(VARIANTS)})")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"POP-GENS_MODULES token in the filename (default {DEFAULT_CONFIG})")
    p.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS),
                   help=f"ranks to render, by last-gen fitness (default best {len(DEFAULT_RANKS)}: "
                        f"{' '.join(map(str, DEFAULT_RANKS))})")
    p.add_argument("--rank", type=int, default=None,
                   help="render only this single rank; overrides --ranks (back-compat)")
    p.add_argument("--run-seed", type=int, default=None,
                   help="override reproduction seed (default: parsed from the filename)")
    p.add_argument("--out-dir", default=str(OUT_DIR),
                   help=f"output root (default {OUT_DIR.name}/)")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would render (seed/variant/rank/fitness) without running MuJoCo/ffmpeg")
    args = p.parse_args()

    if args.csv:
        csv_list = [Path(args.csv)]
    else:
        if args.seed is not None:
            seeds = [args.seed]
        elif args.seeds:
            seeds = args.seeds
        else:
            seeds = list(DEFAULT_SEEDS)
            print(f"no CSV/--seed given -> default seed set: {seeds}")
        csv_list = []
        for s in seeds:
            print(f"seed {s}: discovering variants in {DATA_DIR.name}/ ...")
            csv_list.extend(discover_csvs(s, args.config, args.variants))
        if not csv_list:
            raise SystemExit(
                f"No variant CSVs found for seeds {seeds} (config {args.config}) in {DATA_DIR}"
            )

    # --rank (single) overrides --ranks (best-N) for back-compat.
    ranks = [args.rank] if args.rank is not None else args.ranks

    out_root = Path(args.out_dir)
    n_done = 0
    for csv_path in csv_list:
        if not csv_path.exists():
            print(f"SKIP {csv_path} — file not found")
            continue
        run_seed = args.run_seed if args.run_seed is not None else seed_from_name(csv_path)
        if run_seed is None:
            run_seed = 1
            print(f"  WARN could not parse seed from {csv_path.name}; using run seed 1")
        for rank in ranks:
            loaded = load_rank(csv_path, rank)
            if loaded is None:
                print(f"  SKIP {csv_path.name} rank {rank} — last gen has fewer than {rank} individuals")
                continue
            ind, gen, n = loaded
            print(f"{csv_path.name}: seed={run_seed} last gen={gen} ({n} individuals); "
                  f"rank {rank} fitness={ind.fitness:.4f} (pre={ind.fitness_pre:.4f})")
            out = out_root / csv_path.stem / f"rank{rank:02d}.mp4"
            if args.dry_run:
                print(f"  [dry-run] -> {out}")
                continue
            render(ind, out, run_seed)
            n_done += 1
    if args.dry_run:
        print("\n[dry-run] nothing rendered.")
    else:
        print(f"\nDone: {n_done} video(s) written under {out_root}/")


if __name__ == "__main__":
    main()
