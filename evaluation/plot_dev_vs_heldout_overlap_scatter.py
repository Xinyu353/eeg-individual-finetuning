#!/usr/bin/env python3
"""Thesis supplement: development vs held-out PCA-overlap scatter.

One point per participant. x/y are five-run medians of calibration and
held-out PCA-overlap. OT and MW use different markers. The y=x line is the
no-generalization-gap reference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASES = {
    "ot": PROJECT_ROOT / "results" / "ot_heldout_seed_sensitivity_all_final",
    "mw": PROJECT_ROOT / "results" / "mw_heldout_seed_sensitivity_all_final",
}
OUT_DIR = PROJECT_ROOT / "results" / "heldout_80_20_combined_figures" / "thesis_supplement"

STYLE = {
    "ot": {
        "label": "On-task",
        "marker": "o",
        "facecolor": "#7e57c2",
        "edgecolor": "#4a3284",
        "markersize": 58,
    },
    "mw": {
        "label": "Mind-wandering",
        "marker": "o",
        "facecolor": "#2e7d32",
        "edgecolor": "#1b5e20",
        "markersize": 58,
    },
}
LABEL_SIZE = 13
TICK_SIZE = 11
LEGEND_SIZE = 11


def load_participant_medians(condition: str) -> pd.DataFrame:
    base = BASES[condition]
    runs = pd.read_csv(base / "seed_sensitivity_runs.csv")
    official = pd.read_csv(
        base / "comparison" / "tables" / "paper_source" / "heldout_participant_median_iqr.csv"
    )
    rows = []
    for pid, sub in runs.groupby("participant", sort=True):
        if len(sub) != 5:
            raise ValueError(f"{condition} P{pid}: expected 5 seeds, got {len(sub)}")
        off = official[official["participant"] == pid].iloc[0]
        rows.append(
            {
                "condition": condition,
                "participant": int(pid),
                "development_PCA_overlap_median": float(sub["calibration_PCA_overlap_RF"].median()),
                "heldout_PCA_overlap_median": float(off["median"]),
                "delta_heldout_minus_development": float(off["median"])
                - float(sub["calibration_PCA_overlap_RF"].median()),
                "category": str(off["category"]),
            }
        )
    return pd.DataFrame(rows)


def apply_fonts() -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.labelsize"] = LABEL_SIZE
    plt.rcParams["xtick.labelsize"] = TICK_SIZE
    plt.rcParams["ytick.labelsize"] = TICK_SIZE
    plt.rcParams["legend.fontsize"] = LEGEND_SIZE
    plt.rcParams["axes.titlepad"] = 6


def main() -> None:
    frames = [load_participant_medians(cond) for cond in ("ot", "mw")]
    df = pd.concat(frames, ignore_index=True)
    apply_fonts()

    fig, ax = plt.subplots(figsize=(6.2, 6.0), facecolor="white")
    lo = float(min(df["development_PCA_overlap_median"].min(), df["heldout_PCA_overlap_median"].min()))
    hi = float(max(df["development_PCA_overlap_median"].max(), df["heldout_PCA_overlap_median"].max()))
    lo = np.floor(lo * 20.0) / 20.0
    hi = np.ceil(hi * 20.0) / 20.0
    lims = (lo, hi)

    ax.plot(lims, lims, color="0.35", linestyle="--", linewidth=1.15, zorder=1, label=r"$y=x$")
    ax.axhline(0.60, color="#2e7d32", linestyle="--", linewidth=1.4, alpha=0.85, zorder=1)
    ax.axhline(0.50, color="#c62828", linestyle="--", linewidth=1.4, alpha=0.85, zorder=1)
    for condition, spec in STYLE.items():
        sub = df[df["condition"] == condition]
        ax.scatter(
            sub["development_PCA_overlap_median"],
            sub["heldout_PCA_overlap_median"],
            s=spec["markersize"],
            marker=spec["marker"],
            facecolor=spec["facecolor"],
            edgecolor=spec["edgecolor"],
            linewidths=0.7,
            alpha=0.92,
            zorder=3,
            label=spec["label"],
        )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Development PCA-overlap$_{\mathrm{RF}}$")
    ax.set_ylabel(r"Held-out PCA-overlap$_{\mathrm{RF}}$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    handles = [
        Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.15, label=r"$y=x$"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=STYLE["ot"]["facecolor"],
            markeredgecolor=STYLE["ot"]["edgecolor"],
            markersize=8,
            label="On-task",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=STYLE["mw"]["facecolor"],
            markeredgecolor=STYLE["mw"]["edgecolor"],
            markersize=8,
            label="Mind-wandering",
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", handletextpad=0.45)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUT_DIR / "figS_dev_vs_heldout_pca_overlap_scatter.png"
    table_path = OUT_DIR / "figS_dev_vs_heldout_pca_overlap_scatter.csv"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(table_path, index=False)
    print(f"Saved {plot_path}")
    print(f"Saved {table_path}")
    print(f"OT n={int((df.condition=='ot').sum())}, MW n={int((df.condition=='mw').sum())}")


if __name__ == "__main__":
    main()
