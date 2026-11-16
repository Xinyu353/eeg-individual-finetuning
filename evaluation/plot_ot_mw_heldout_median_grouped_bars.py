#!/usr/bin/env python3
"""Merge OT and MW 5-seed boxplots onto one Fig.2-style participant axis.

Same geometry as the separate median-category boxplots: boxes, min-max
whiskers, seed points, 4-decimal median labels. OT is solid, MW is hatched.
Legend sits inside the axes (two columns) without overlapping boxes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RUNS = {
    "ot": PROJECT_ROOT / "results" / "ot_heldout_seed_sensitivity_all_final" / "seed_sensitivity_runs.csv",
    "mw": PROJECT_ROOT / "results" / "mw_heldout_seed_sensitivity_all_final" / "seed_sensitivity_runs.csv",
}
OUT_DIR = PROJECT_ROOT / "results" / "heldout_80_20_combined_figures"
PARTICIPANTS = list(range(1, 31))
OFFSET = {"ot": -0.18, "mw": 0.18}
HATCH = {"ot": "", "mw": "///"}
BOX_WIDTH = 0.30
CATEGORY_COLORS = {"Good": "#2e7d32", "Borderline": "#f57c00", "Poor": "#c62828"}
SEED_FACE = "#7e57c2"
SEED_EDGE = "#4a3284"


def categorize(value: float) -> str:
    if value >= 0.60:
        return "Good"
    if value >= 0.50:
        return "Borderline"
    return "Poor"


def load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["participant"] = df["participant"].astype(int)
    return df


def values_for(runs: pd.DataFrame, pid: int) -> np.ndarray:
    return runs.loc[runs["participant"].eq(pid), "heldout_PCA_overlap_RF"].dropna().to_numpy(float)


def count_text(runs: pd.DataFrame) -> str:
    meds = runs.groupby("participant")["heldout_PCA_overlap_RF"].median()
    cats = meds.map(categorize)
    counts = cats.value_counts()
    return (
        f"Good={int(counts.get('Good', 0))}, "
        f"Borderline={int(counts.get('Borderline', 0))}, "
        f"Poor={int(counts.get('Poor', 0))}"
    )


def draw_condition(ax, runs: pd.DataFrame, condition: str, rng: np.random.Generator) -> None:
    data, positions, colors, meds = [], [], [], []
    for i, pid in enumerate(PARTICIPANTS):
        vals = values_for(runs, pid)
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        data.append(vals)
        positions.append(i + OFFSET[condition])
        colors.append(CATEGORY_COLORS[categorize(med)])
        meds.append(med)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        medianprops={"color": "black", "linewidth": 1.25},
        whiskerprops={"color": "0.30", "linewidth": 0.9},
        capprops={"color": "0.30", "linewidth": 0.9},
        zorder=2,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
        patch.set_hatch(HATCH[condition])

    for pos, vals, med in zip(positions, data, meds):
        jitter = rng.uniform(-0.07, 0.07, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=16,
            color=SEED_FACE,
            alpha=1.0,
            edgecolors=SEED_EDGE,
            linewidths=0.28,
            zorder=3,
        )
        y_label = min(0.97, max(vals.max() + 0.018, med + 0.018))
        ax.text(
            pos,
            y_label,
            f"{med:.4f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            rotation=90,
            color="0.15",
            zorder=4,
        )


def main() -> None:
    ot = load_runs(RUNS["ot"])
    mw = load_runs(RUNS["mw"])
    rng = np.random.default_rng(42)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 12
    plt.rcParams["hatch.linewidth"] = 0.7
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10

    fig, ax = plt.subplots(figsize=(16.8, 5.2), facecolor="white")
    ax.set_facecolor("white")
    ax.axhline(0.60, color="#2e7d32", linestyle="--", linewidth=1.4, alpha=0.85, zorder=1)
    ax.axhline(0.50, color="#c62828", linestyle="--", linewidth=1.4, alpha=0.85, zorder=1)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.30, zorder=0)

    draw_condition(ax, ot, "ot", rng)
    draw_condition(ax, mw, "mw", rng)

    ax.set_xlim(-0.7, len(PARTICIPANTS) - 0.3)
    ax.set_ylim(0.25, 1.0)
    ax.set_yticks(np.arange(0.3, 1.01, 0.1))
    ax.set_xticks(np.arange(len(PARTICIPANTS)))
    ax.set_xticklabels([f"P{pid}" for pid in PARTICIPANTS], rotation=45, ha="center", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=8)
    ax.set_xlabel("Participant", fontweight="bold", fontsize=14)
    ax.set_ylabel("Held-out PCA overlap\n(real vs. fake)", fontweight="bold", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cat_handles = [
        Patch(facecolor=CATEGORY_COLORS["Good"], edgecolor="black", alpha=0.80, label="Good"),
        Patch(facecolor=CATEGORY_COLORS["Borderline"], edgecolor="black", alpha=0.80, label="Borderline"),
        Patch(facecolor=CATEGORY_COLORS["Poor"], edgecolor="black", alpha=0.80, label="Poor"),
    ]
    type_handles = [
        Patch(facecolor="0.75", edgecolor="black", alpha=0.80, hatch="", label="OT"),
        Patch(facecolor="0.75", edgecolor="black", alpha=0.80, hatch="///", label="MW"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SEED_FACE,
            markeredgecolor=SEED_EDGE,
            markersize=5,
            label="Fine-tuning runs",
        ),
    ]
    legend_kw = dict(
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.5,
        labelspacing=0.35,
        framealpha=0.0,
        fontsize=10,
    )
    cat_legend = ax.legend(
        handles=cat_handles,
        loc="lower left",
        bbox_to_anchor=(0.012, 0.08),
        **legend_kw,
    )
    ax.add_artist(cat_legend)
    ax.legend(
        handles=type_handles,
        loc="lower left",
        bbox_to_anchor=(0.13, 0.08),
        **legend_kw,
    )
    ax.text(
        0.99,
        0.08,
        f"OT: {count_text(ot)}\nMW: {count_text(mw)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        clip_on=True,
        linespacing=1.45,
        zorder=5,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUT_DIR / "ot_mw_current_heldout_seed_boxplot_median_category.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")
    print(f"OT {count_text(ot)}")
    print(f"MW {count_text(mw)}")


if __name__ == "__main__":
    main()
