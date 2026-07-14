"""One-shot aggregate plots for the current ga_p2 (96-50_30) batch.

Run once (``python plot_all.py``) to produce the full comparison figure set,
pooled across all ``SEEDS`` below. For each condition every seed's CSV is
concatenated, so the std bands reflect across-seed + within-generation spread.

  learning{S}_96-50_30_ga_p2_fitness_flat.csv   -> Baseline   (df_a)
  learning{S}_96-50_30_ga_p2_ldelta_flat.csv    -> ΔL-aware   (df_b)

To re-run on a different batch, just edit ``SEEDS``. The middle filename token
(96-50_30 etc.) is globbed per seed, so it doesn't have to be hardcoded.

Produces (all to ``plots/``, tagged ``ga_p2_seeds{seeds}``):
  performance_per_gen / _mean / _mean_nostd, learning_delta_per_gen,
  mean_fitness_decomposition, scatter_fitness_vs_ldelta (+ top20),
  learning_curves, archive_heatmap, archive_dl_heatmap, archive_max_dl_heatmap.

Note: the old CMA-MAE (``*_cmamae.csv``) discovery this file used to do is gone
— that batch's data is no longer present. See ``plot_seeds.py`` for the sibling
this was consolidated from.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- config ----
DATA_DIR = Path("experiment_data")
PLOTS_ROOT = Path("plots")
#SEEDS = [7, 9, 10, 11, 12, 13,14]  
SEEDS = [15]      # current batch: seeds with full fitness+ldelta
# Tag: "1-8" for a contiguous range, else the seeds joined.
_seed_tag = (f"{SEEDS[0]}-{SEEDS[-1]}"
             if SEEDS == list(range(SEEDS[0], SEEDS[-1] + 1))
             else "_".join(map(str, SEEDS)))
TAG = f"ga_p2_seeds{_seed_tag}"
# Each seed combination writes into its own subfolder, so different seed
# sets stay side by side for per-seed comparison (plots/<TAG>/...).
OUT_DIR = PLOTS_ROOT / TAG
# Filename token (96-30_30 vs 96-50_30) varies per seed, so glob it (see _path).
GENERATIONS = [1, 50, 100]
# Only plot up to the last generation of interest; data beyond it is dropped.
MAX_GEN = max(GENERATIONS)
TOP_K_CURVES = 10
LABEL_A = "Baseline"   # fitness selection
LABEL_B = "ΔL-aware"   # ldelta selection
LABEL_C = "NSGA-II"    # nsga2 selection
COLOR_A = "green"
COLOR_B = "blueviolet"
COLOR_C = "darkorange"
SENTINEL_CUTOFF = -1e29
# ----------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Tag:   {TAG}")
print(f"Seeds: {SEEDS}")


def _path(seed, cond):
    """Locate a seed's CSV by globbing, so the middle token (96-30_30 vs
    96-50_30, etc.) doesn't have to be hardcoded. Returns a non-existent path
    if nothing matches (callers check .exists())."""
    matches = sorted(DATA_DIR.glob(f"learning{seed}_*_ga_p2_{cond}_flat.csv"))
    return matches[0] if matches else DATA_DIR / f"learning{seed}_NONE_ga_p2_{cond}_flat.csv"


def load_concat(cond):
    frames = []
    for seed in SEEDS:
        path = _path(seed, cond)
        if not path.exists():
            print(f"  (missing) {path.name}")
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    mask = (np.isfinite(df["fitness"]) & np.isfinite(df["fitness_pre"])
            & (df["generation"] <= MAX_GEN))
    return df.loc[mask].reset_index(drop=True)


df_a = load_concat("fitness")
df_b = load_concat("ldelta")
df_c = load_concat("nsga2")

# ============================================================
# FIGURE 1 / 1b: performance per generation
# FIGURE 2: learning delta per generation
# ============================================================

def perf_stats(df):
    perf = (df.groupby("generation")["fitness"]
              .agg(mean="mean", std="std")
              .sort_index())
    perf["max"] = (df.groupby(["generation", "seed"])["fitness"].max()
                     .groupby("generation").mean())
    return perf

def delta_stats(df):
    d = df.assign(ld=df["fitness"] - df["fitness_pre"])
    return (d.groupby("generation")["ld"]
             .agg(mean="mean", std="std")
             .sort_index())


perf_a = perf_stats(df_a)
perf_b = perf_stats(df_b)
perf_c = perf_stats(df_c)
delta_a = delta_stats(df_a)
delta_b = delta_stats(df_b)
delta_c = delta_stats(df_c)

# --- Figure 1: performance (mean + max) ---
plt.figure()
for perf, color, label in [(perf_a, COLOR_A, LABEL_A), (perf_b, COLOR_B, LABEL_B), (perf_c, COLOR_C, LABEL_C)]:
    if perf.empty:
        continue
    plt.plot(perf.index, perf["mean"], color=color, label=f"{label} mean")
    plt.plot(perf.index, perf["max"], color=color, linestyle="--", label=f"{label} max")
    plt.fill_between(perf.index,
                     perf["mean"] - perf["std"],
                     perf["mean"] + perf["std"],
                     color=color, alpha=0.25)
plt.xlabel("Generation")
plt.ylabel("Performance")
plt.title(f"Performance per generation ({TAG})")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / f"performance_per_gen_{TAG}.pdf", format="pdf")
plt.close()

# --- Figure 1b: performance mean-only ---
plt.figure()
for perf, color, label in [(perf_a, COLOR_A, LABEL_A), (perf_b, COLOR_B, LABEL_B), (perf_c, COLOR_C, LABEL_C)]:
    if perf.empty:
        continue
    plt.plot(perf.index, perf["mean"], color=color, label=f"{label} mean")
    plt.fill_between(perf.index,
                     perf["mean"] - perf["std"],
                     perf["mean"] + perf["std"],
                     color=color, alpha=0.25)
plt.xlabel("Generation")
plt.ylabel("Performance")
plt.title(f"Mean performance per generation ({TAG})")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / f"performance_mean_per_gen_{TAG}.pdf", format="pdf")
plt.close()

# --- Figure 1c: performance mean-only, no std band ---
plt.figure()
for perf, color, label in [(perf_a, COLOR_A, LABEL_A), (perf_b, COLOR_B, LABEL_B), (perf_c, COLOR_C, LABEL_C)]:
    if perf.empty:
        continue
    plt.plot(perf.index, perf["mean"], color=color, label=f"{label} mean")
plt.xlabel("Generation")
plt.ylabel("Performance")
plt.title(f"Mean performance per generation, no std ({TAG})")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / f"performance_mean_nostd_per_gen_{TAG}.pdf", format="pdf")
plt.close()

# --- Figure 2: learning delta ---
plt.figure()
for delta, color, label in [(delta_a, COLOR_A, LABEL_A), (delta_b, COLOR_B, LABEL_B), (delta_c, COLOR_C, LABEL_C)]:
    if delta.empty:
        continue
    plt.plot(delta.index, delta["mean"], color=color, label=f"{label} mean ΔL")
    plt.fill_between(delta.index,
                     delta["mean"] - delta["std"],
                     delta["mean"] + delta["std"],
                     color=color, alpha=0.25)
plt.xlabel("Generation")
plt.ylabel("Learning delta")
plt.title(f"Mean learning delta per generation ({TAG})")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / f"learning_delta_per_gen_{TAG}.pdf", format="pdf")
plt.close()

# ============================================================
# FIGURE 2d: Mean-fitness decomposition — morphology (pre) vs learning (ΔL)
# Top line  = mean final fitness per generation.
# Middle line = mean fitness_pre (the pre-learning / morphology baseline).
# The gap between them is the learning contribution. The middle line is
# annotated with how much pre-learning vs learning each contribute to the
# final mean fitness at the last generation.
# ============================================================

def mean_decomp(df):
    """Per-generation mean fitness and mean fitness_pre."""
    if df.empty:
        return pd.DataFrame()
    g = (df.groupby("generation")
           .agg(fit=("fitness", "mean"), pre=("fitness_pre", "mean"))
           .sort_index())
    g["learn"] = g["fit"] - g["pre"]
    return g

decomp_a = mean_decomp(df_a)
decomp_b = mean_decomp(df_b)
decomp_c = mean_decomp(df_c)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, (decomp, color, label) in zip(
        axes, [(decomp_a, COLOR_A, LABEL_A), (decomp_b, COLOR_B, LABEL_B),
               (decomp_c, COLOR_C, LABEL_C)]):
    if decomp.empty:
        ax.set_title(f"{label} — no data")
        continue
    gens = decomp.index.to_numpy()
    fit = decomp["fit"].to_numpy()
    pre = decomp["pre"].to_numpy()

    # morphology band (0 -> pre) and learning band (pre -> total fitness)
    ax.fill_between(gens, 0, pre, color=color, alpha=0.30,
                    label="Morphology (fitness_pre)")
    ax.fill_between(gens, pre, fit, color=color, alpha=0.12,
                    label="Learning contribution (ΔL)")
    ax.plot(gens, fit, color=color, linewidth=1.8, label="Mean fitness")
    # the "line in the middle": the pre-learning baseline
    ax.plot(gens, pre, color=color, linewidth=1.3, linestyle="--")

    # annotate the split at the final generation
    f_fin, p_fin = float(fit[-1]), float(pre[-1])
    l_fin = f_fin - p_fin
    if f_fin != 0:
        pre_pct = 100.0 * p_fin / f_fin
        learn_pct = 100.0 * l_fin / f_fin
        ax.annotate(
            f"pre-learning: {pre_pct:.0f}% of final\n"
            f"learning ΔL: {learn_pct:.0f}% of final",
            xy=(gens[-1], p_fin),
            xytext=(0.50, 0.50), textcoords="axes fraction",
            fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )
    ax.set_xlabel("Generation")
    ax.set_title(label)
    ax.legend(fontsize=9, loc="upper left")

axes[0].set_ylabel("Mean fitness")
fig.suptitle(f"Mean fitness decomposition: morphology vs learning ({TAG})",
             fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / f"mean_fitness_decomposition_{TAG}.pdf", format="pdf")
plt.close()

# ============================================================
# FIGURE 3: scatter fitness vs ΔL at the chosen generations
# ============================================================

def gen_points(df, generation, top_k=None):
    sel = df[df["generation"] == generation]
    if top_k is not None:
        # top_k by fitness within EACH run (seed), then pooled.
        sel = (sel.groupby("seed", group_keys=False)
                  .apply(lambda g: g.nlargest(top_k, "fitness")))
    fit = sel["fitness"].to_numpy()
    ld = (sel["fitness"] - sel["fitness_pre"]).to_numpy()
    return fit, ld


def make_scatter(top_k, out_name, title_suffix):
    fig, axes = plt.subplots(1, len(GENERATIONS),
                             figsize=(6 * len(GENERATIONS), 6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, gen in zip(axes, GENERATIONS):
        fit_a, ld_a = gen_points(df_a, gen, top_k=top_k)
        fit_b, ld_b = gen_points(df_b, gen, top_k=top_k)
        fit_c, ld_c = gen_points(df_c, gen, top_k=top_k)
        ax.scatter(ld_a, fit_a, color=COLOR_A, alpha=0.5, s=4,
                   label=f"{LABEL_A} (n={len(fit_a)})")
        ax.scatter(ld_b, fit_b, color=COLOR_B, alpha=0.6, s=4,
                   label=f"{LABEL_B} (n={len(fit_b)})")
        ax.scatter(ld_c, fit_c, color=COLOR_C, alpha=0.6, s=4,
                   label=f"{LABEL_C} (n={len(fit_c)})")
        for x, y, color in [(ld_a, fit_a, COLOR_A), (ld_b, fit_b, COLOR_B),
                            (ld_c, fit_c, COLOR_C)]:
            if len(x) > 1:
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 100)
                ax.plot(xs, m * xs + b, color=color, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Learning Delta", fontsize=11)
        ax.set_title(f"Generation {gen}", fontsize=12)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Fitness", fontsize=11)
    fig.suptitle(f"Fitness vs Learning Delta ({TAG}){title_suffix}", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=150)
    plt.close()


make_scatter(None, f"scatter_fitness_vs_ldelta_{TAG}.png", "")
make_scatter(20, f"scatter_fitness_vs_ldelta_top20_{TAG}.png",
             " — top 20 by fitness")

# ============================================================
# FIGURE 4: learning curves at the chosen generations
# ============================================================

def load_curves_by_condition_gen(generations, top_k=None):
    grouped = {}
    gen_set = set(generations)

    key_of = {"fitness": "learning", "ldelta": "ldelta", "nsga2": "nsga2"}
    sources = []
    for cond in ("fitness", "ldelta", "nsga2"):
        for seed in SEEDS:
            sources.append((key_of[cond], _path(seed, cond)))

    for cond, path in sources:
        if not path.exists():
            continue
        rows_by_gen = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    gen = int(r["generation"])
                except (TypeError, ValueError, KeyError):
                    continue
                if gen in gen_set:
                    rows_by_gen.setdefault(gen, []).append(r)

        for gen, rows in rows_by_gen.items():
            recs = []
            for r in rows:
                raw_curve = r.get("learning_curve")
                raw_pre = r.get("fitness_pre")
                raw_fit = r.get("fitness")
                if not raw_curve or raw_pre is None or raw_fit is None:
                    continue
                try:
                    curve = json.loads(raw_curve)
                    pre = float(raw_pre)
                    fit = float(raw_fit)
                except (ValueError, TypeError):
                    continue
                if not curve or not math.isfinite(pre) or not math.isfinite(fit):
                    continue
                recs.append((fit, pre, curve))

            if top_k is not None:
                recs.sort(key=lambda t: t[0], reverse=True)
                recs = recs[:top_k]

            for _fit, pre, curve in recs:
                arr = np.array(curve, dtype=float)
                arr = np.where((arr <= -1e29) | ~np.isfinite(arr), -np.inf, arr)
                arr = np.maximum.accumulate(arr)
                arr = np.where(np.isfinite(arr), arr, pre)
                grouped.setdefault((cond, gen), []).append(arr - pre)

    return grouped


def stack_to_array(curves):
    if not curves:
        return np.empty((0, 0))
    L = min(len(c) for c in curves)
    return np.array([c[:L] for c in curves])


grouped = load_curves_by_condition_gen(GENERATIONS, top_k=TOP_K_CURVES)

fig, axes = plt.subplots(1, len(GENERATIONS),
                         figsize=(6 * len(GENERATIONS), 5),
                         sharey=True)
axes = np.atleast_1d(axes)
CONDITIONS = [("learning", COLOR_A, LABEL_A), ("ldelta", COLOR_B, LABEL_B),
              ("nsga2", COLOR_C, LABEL_C)]
for ax, gen in zip(axes, GENERATIONS):
    for cond, color, label in CONDITIONS:
        arr = stack_to_array(grouped.get((cond, gen), []))
        if arr.size == 0:
            continue
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, color=color, linewidth=1.5,
                label=f"{label} (n={arr.shape[0]})")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel("CMA Iteration", fontsize=11)
    ax.set_title(f"Generation {gen}", fontsize=12)
    ax.legend(fontsize=9)
axes[0].set_ylabel("Cumulative-max gain (curve − fitness_pre)", fontsize=11)
scope = "all alive individuals" if TOP_K_CURVES is None else f"top-{TOP_K_CURVES} by fitness"
fig.suptitle(f"Learning curves per generation ({TAG}) — {scope}", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / f"learning_curves_{TAG}.png", dpi=150)
plt.close()

# ============================================================
# FIGURE 5/6/7: archive heatmaps (binned population view)
# ============================================================
ARCHIVE_DIMS = (20, 20)
MEASURE_RANGES = ((0.0, 1.0), (0.0, 4.0))  # (branching, limb_std_length)
MEASURE_LABELS = ("branching", "limb_std_length")


def _bin_idx(meas):
    idx = []
    for v, (lo, hi), n in zip(meas, MEASURE_RANGES, ARCHIVE_DIMS):
        if not math.isfinite(v):
            return None
        t = (v - lo) / (hi - lo)
        idx.append(int(np.clip(np.floor(t * n), 0, n - 1)))
    return idx


def build_archive(df, generation):
    sel = df[df["generation"] == generation]
    grid = np.full(ARCHIVE_DIMS, np.nan)
    for _, row in sel.iterrows():
        raw = row.get("measures")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            meas = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if len(meas) < 2:
            continue
        idx = _bin_idx(meas)
        if idx is None:
            continue
        fit = float(row["fitness"])
        if not math.isfinite(fit):
            continue
        i0, i1 = idx
        if np.isnan(grid[i0, i1]) or fit > grid[i0, i1]:
            grid[i0, i1] = fit
    return grid


all_grids = {}
for cond, df in [("learning", df_a), ("ldelta", df_b), ("nsga2", df_c)]:
    for gen in GENERATIONS:
        all_grids[(cond, gen)] = build_archive(df, gen)

finite_vals = np.concatenate([g[np.isfinite(g)].ravel() for g in all_grids.values()
                              if np.any(np.isfinite(g))]) if all_grids else np.array([])
vmin = float(finite_vals.min()) if finite_vals.size else 0.0
vmax = float(finite_vals.max()) if finite_vals.size else 1.0

fig, axes = plt.subplots(3, len(GENERATIONS),
                         figsize=(5 * len(GENERATIONS), 13),
                         squeeze=False)
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad(color="lightgray")

for row_i, (cond, label) in enumerate([("learning", LABEL_A), ("ldelta", LABEL_B),
                                       ("nsga2", LABEL_C)]):
    for col_i, gen in enumerate(GENERATIONS):
        ax = axes[row_i, col_i]
        grid = all_grids[(cond, gen)]
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked.T, origin="lower", aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[MEASURE_RANGES[0][0], MEASURE_RANGES[0][1],
                               MEASURE_RANGES[1][0], MEASURE_RANGES[1][1]])
        filled = int(np.sum(np.isfinite(grid)))
        total = ARCHIVE_DIMS[0] * ARCHIVE_DIMS[1]
        ax.set_title(f"{label} — gen {gen}  ({filled}/{total} cells)", fontsize=11)
        if row_i == 2:
            ax.set_xlabel(MEASURE_LABELS[0])
        if col_i == 0:
            ax.set_ylabel(MEASURE_LABELS[1])

fig.suptitle(f"Archive (max fitness per cell) — {TAG}", fontsize=13)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="fitness")
plt.savefig(OUT_DIR / f"archive_heatmap_{TAG}.png", dpi=150, bbox_inches="tight")
plt.close()


def build_dl_archive(df, generation):
    sel = df[df["generation"] == generation]
    grid_fit = np.full(ARCHIVE_DIMS, np.nan)
    grid_dl = np.full(ARCHIVE_DIMS, np.nan)
    for _, row in sel.iterrows():
        raw = row.get("measures")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            meas = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if len(meas) < 2:
            continue
        idx = _bin_idx(meas)
        if idx is None:
            continue
        fit = float(row["fitness"])
        pre = float(row["fitness_pre"])
        if not (math.isfinite(fit) and math.isfinite(pre)):
            continue
        i0, i1 = idx
        if np.isnan(grid_fit[i0, i1]) or fit > grid_fit[i0, i1]:
            grid_fit[i0, i1] = fit
            grid_dl[i0, i1] = fit - pre
    return grid_dl


dl_grids = {}
for cond, df in [("learning", df_a), ("ldelta", df_b), ("nsga2", df_c)]:
    for gen in GENERATIONS:
        dl_grids[(cond, gen)] = build_dl_archive(df, gen)

finite_dl = np.concatenate([g[np.isfinite(g)].ravel() for g in dl_grids.values()
                            if np.any(np.isfinite(g))]) if dl_grids else np.array([])
dl_abs = float(np.max(np.abs(finite_dl))) if finite_dl.size else 1.0
dl_vmin, dl_vmax = -dl_abs, dl_abs

fig, axes = plt.subplots(3, len(GENERATIONS),
                         figsize=(5 * len(GENERATIONS), 13),
                         squeeze=False)
cmap_dl = plt.get_cmap("RdBu").copy()
cmap_dl.set_bad(color="lightgray")

for row_i, (cond, label) in enumerate([("learning", LABEL_A), ("ldelta", LABEL_B),
                                       ("nsga2", LABEL_C)]):
    for col_i, gen in enumerate(GENERATIONS):
        ax = axes[row_i, col_i]
        grid = dl_grids[(cond, gen)]
        masked = np.ma.masked_invalid(grid)
        im_dl = ax.imshow(masked.T, origin="lower", aspect="auto",
                          cmap=cmap_dl, vmin=dl_vmin, vmax=dl_vmax,
                          extent=[MEASURE_RANGES[0][0], MEASURE_RANGES[0][1],
                                  MEASURE_RANGES[1][0], MEASURE_RANGES[1][1]])
        filled = int(np.sum(np.isfinite(grid)))
        total = ARCHIVE_DIMS[0] * ARCHIVE_DIMS[1]
        ax.set_title(f"{label} — gen {gen}  ({filled}/{total} cells)", fontsize=11)
        if row_i == 2:
            ax.set_xlabel(MEASURE_LABELS[0])
        if col_i == 0:
            ax.set_ylabel(MEASURE_LABELS[1])

fig.suptitle(f"Archive (ΔL of cell elite) — {TAG}", fontsize=13)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im_dl, cax=cbar_ax, label="ΔL  (fitness − fitness_pre)")
plt.savefig(OUT_DIR / f"archive_dl_heatmap_{TAG}.png", dpi=150, bbox_inches="tight")
plt.close()


def build_max_dl_archive(df):
    grid = np.full(ARCHIVE_DIMS, np.nan)
    for _, row in df.iterrows():
        raw = row.get("measures")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            meas = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if len(meas) < 2:
            continue
        idx = _bin_idx(meas)
        if idx is None:
            continue
        fit = float(row["fitness"])
        pre = float(row["fitness_pre"])
        if not (math.isfinite(fit) and math.isfinite(pre)):
            continue
        dl = fit - pre
        i0, i1 = idx
        if np.isnan(grid[i0, i1]) or dl > grid[i0, i1]:
            grid[i0, i1] = dl
    return grid


max_dl_grids = {
    "learning": build_max_dl_archive(df_a),
    "ldelta": build_max_dl_archive(df_b),
    "nsga2": build_max_dl_archive(df_c),
}

finite_max = np.concatenate([g[np.isfinite(g)].ravel() for g in max_dl_grids.values()
                             if np.any(np.isfinite(g))])
max_abs = float(np.max(np.abs(finite_max))) if finite_max.size else 1.0
max_vmin, max_vmax = -max_abs, max_abs

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), squeeze=False)
cmap_max = plt.get_cmap("RdBu").copy()
cmap_max.set_bad(color="lightgray")

for col_i, (cond, label) in enumerate([("learning", LABEL_A), ("ldelta", LABEL_B),
                                       ("nsga2", LABEL_C)]):
    ax = axes[0, col_i]
    grid = max_dl_grids[cond]
    masked = np.ma.masked_invalid(grid)
    im_max = ax.imshow(masked.T, origin="lower", aspect="auto",
                       cmap=cmap_max, vmin=max_vmin, vmax=max_vmax,
                       extent=[MEASURE_RANGES[0][0], MEASURE_RANGES[0][1],
                               MEASURE_RANGES[1][0], MEASURE_RANGES[1][1]])
    filled = int(np.sum(np.isfinite(grid)))
    total = ARCHIVE_DIMS[0] * ARCHIVE_DIMS[1]
    ax.set_title(f"{label} — max ΔL ever  ({filled}/{total} cells touched)",
                 fontsize=11)
    ax.set_xlabel(MEASURE_LABELS[0])
    if col_i == 0:
        ax.set_ylabel(MEASURE_LABELS[1])

fig.suptitle(f"Archive (best learner per cell, whole run) — {TAG}", fontsize=13)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im_max, cax=cbar_ax, label="max ΔL")
plt.savefig(OUT_DIR / f"archive_max_dl_heatmap_{TAG}.png", dpi=150,
            bbox_inches="tight")
plt.close()

print(f"\nWrote plots to {OUT_DIR}/")
for p in sorted(OUT_DIR.glob(f"*{TAG}*")):
    print(f"  {p.name}")
