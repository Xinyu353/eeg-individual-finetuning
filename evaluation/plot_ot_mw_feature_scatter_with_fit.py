#!/usr/bin/env python3
"""
Plot one predictor-vs-outcome scatter with a linear fit line for OT and MW.

This is for visual inspection only. The main summary figures should not use fit
lines because their x-axis is already a correlation coefficient.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from round1_predictors_to_outcomes_analysis import pretty_name


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_OT_TABLE = PROJECT_ROOT / "results" / "round1_all_predictors" / "round1_all_predictors_wide.csv"
DEFAULT_MW_TABLE = PROJECT_ROOT / "mw_pipeline" / "results" / "round1_all_predictors" / "round1_all_predictors_wide.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "ot_mw_predictor_summary_compare" / "scatter_fit_examples"


def plot_panel(ax, df: pd.DataFrame, predictor: str, outcome: str, title: str, color: str) -> None:
    sub = df[[predictor, outcome]].dropna()
    x = sub[predictor].to_numpy(dtype=float)
    y = sub[outcome].to_numpy(dtype=float)
    ax.scatter(x, y, s=34, alpha=0.85, color=color, edgecolors="white", linewidths=0.6)

    if len(sub) >= 3 and np.nanstd(x) > 0:
        fit = linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, fit.intercept + fit.slope * xs, color="black", linewidth=1.2, alpha=0.85)
        rho, p = spearmanr(x, y)
        text = f"ρ={rho:.2f}, p={p:.3g}\nr={fit.rvalue:.2f}, p={fit.pvalue:.3g}"
    else:
        text = "Not enough data"

    ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(pretty_name(predictor))
    ax.set_ylabel(pretty_name(outcome))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OT/MW scatter with fit line for one predictor.")
    parser.add_argument("--predictor", default="Petrosian_Fractal_Dimension__abs_dev_z")
    parser.add_argument("--outcome", default="pca_overlap", choices=["pca_overlap", "SWD"])
    parser.add_argument("--ot-table", default=str(DEFAULT_OT_TABLE))
    parser.add_argument("--mw-table", default=str(DEFAULT_MW_TABLE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ot = pd.read_csv(args.ot_table)
    mw = pd.read_csv(args.mw_table)
    for name, df in (("OT", ot), ("MW", mw)):
        missing = [c for c in [args.predictor, args.outcome] if c not in df.columns]
        if missing:
            raise ValueError(f"{name} table missing columns: {missing}")

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), facecolor="white")
    plot_panel(axes[0], ot, args.predictor, args.outcome, "OT", "#4C566A")
    plot_panel(axes[1], mw, args.predictor, args.outcome, "MW", "#2F6DB3")
    fig.suptitle(f"{pretty_name(args.predictor)} vs {pretty_name(args.outcome)}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_pred = args.predictor.replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"scatter_fit_{args.outcome}_vs_{safe_pred}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
