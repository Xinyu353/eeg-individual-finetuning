#!/usr/bin/env python3
"""Paper quality bars: 5-seed median held-out PCA overlap, 4 decimal labels.

Does not overwrite the existing 5-seed mean / 3-decimal same-style bars.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASES = {
    "ot": PROJECT_ROOT / "results" / "ot_heldout_seed_sensitivity_all_final",
    "mw": PROJECT_ROOT / "results" / "mw_heldout_seed_sensitivity_all_final",
}
STEMS = {
    "ot": "current_heldout_pca_overlap_bar_median_4dp",
    "mw": "mw_current_heldout_pca_overlap_bar_median_4dp",
}
COLOR_MAP = {
    "Good": "#2e7d32",
    "Borderline": "#f57c00",
    "Poor": "#c62828",
}


def categorize_overlap(value: float) -> str:
    if value >= 0.60:
        return "Good"
    if value >= 0.50:
        return "Borderline"
    return "Poor"


def plot_median_bar(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    plot_df = df.sort_values("participant").copy()
    plot_df["category"] = plot_df["median"].map(categorize_overlap)
    plot_df["label"] = plot_df["median"].map(lambda v: f"{v:.4f}")

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 12

    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(max(14, 0.62 * n), 6), facecolor="white")
    ax.set_facecolor("white")
    ax.set_ylabel("PCA Overlap", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1.0)

    ax.axhline(y=0.60, color="#2e7d32", linestyle="--", linewidth=2.5, alpha=0.8, zorder=1)
    ax.axhline(y=0.50, color="#c62828", linestyle="--", linewidth=2.5, alpha=0.8, zorder=1)

    colors = [COLOR_MAP[cat] for cat in plot_df["category"]]
    bars = ax.bar(
        range(n),
        plot_df["median"],
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        zorder=2,
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"P{int(pid)}" for pid in plot_df["participant"]], rotation=45, ha="right")

    for bar, label in zip(bars, plot_df["label"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    legend_elements = [
        mpatches.Patch(
            facecolor=COLOR_MAP["Good"],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label="Good (≥0.60)",
        ),
        mpatches.Patch(
            facecolor=COLOR_MAP["Borderline"],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label="Borderline (0.50-0.60)",
        ),
        mpatches.Patch(
            facecolor=COLOR_MAP["Poor"],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label="Poor (<0.50)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=False)
    ax.set_title("PCA Overlap Across Participants After Fine-tuning", fontsize=18, fontweight="bold", pad=20)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def main() -> None:
    for key, base in BASES.items():
        src = base / "comparison" / "tables" / "paper_source" / "heldout_participant_median_iqr.csv"
        df = pd.read_csv(src)
        stem = STEMS[key]
        plot_path = base / "comparison" / "plots" / "overlap_bars" / f"{stem}.png"
        table_path = base / "comparison" / "tables" / "plot_data" / f"{stem}.csv"
        plot_df = plot_median_bar(df, plot_path)
        out = plot_df[["participant", "n_seeds", "Q1", "median", "Q3", "IQR", "category", "label"]].copy()
        out = out.rename(columns={"label": "median_4dp"})
        table_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(table_path, index=False)
        print(f"{key.upper()} saved {plot_path}")
        print(f"{key.upper()} saved {table_path}")
        print(plot_df["category"].value_counts().to_string())
        print()


if __name__ == "__main__":
    main()
