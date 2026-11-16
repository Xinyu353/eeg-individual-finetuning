#!/usr/bin/env python3
"""
Generate predictor-vs-outcome scatter plots with linear fit lines.

This complements the Spearman ρ summary plots by showing the raw participant
points for each predictor/outcome pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from round1_predictors_to_outcomes_analysis import (
    get_outcome_columns,
    get_predictor_columns,
    pretty_name,
)
from round1_all_predictors_correlation import (
    GROUP_CANDIDATES,
    GROUP_COLOR,
    choose_text_and_legend_locations,
    normalize_group,
    resolve_column,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_OT_INPUT = PROJECT_ROOT / "results" / "round1_all_predictors" / "round1_all_predictors_wide.csv"
DEFAULT_OT_SUMMARY = PROJECT_ROOT / "results" / "round1_predictors_to_outcomes" / "predictor_to_outcome_correlation_summary.csv"
DEFAULT_OT_OUT_DIR = PROJECT_ROOT / "results" / "round1_predictors_to_outcomes" / "scatter_fit_all"

DEFAULT_MW_INPUT = PROJECT_ROOT / "mw_pipeline" / "results" / "round1_all_predictors" / "round1_all_predictors_wide.csv"
DEFAULT_MW_SUMMARY = PROJECT_ROOT / "mw_pipeline" / "results" / "round1_predictors_to_outcomes" / "predictor_to_outcome_correlation_summary.csv"
DEFAULT_MW_OUT_DIR = PROJECT_ROOT / "mw_pipeline" / "results" / "round1_predictors_to_outcomes" / "scatter_fit_all"
GROUP_ORDER = ("good", "borderline", "poor")


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")


def make_scatter_with_fit(
    df: pd.DataFrame,
    summary_row: pd.Series,
    predictor: str,
    outcome: str,
    group_col: str,
    output_path: Path,
) -> None:
    sub_fit = df[[predictor, outcome]].dropna()
    x = sub_fit[predictor].to_numpy(dtype=float)
    y = sub_fit[outcome].to_numpy(dtype=float)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 8

    fig, ax = plt.subplots(figsize=(4.8, 3.8), facecolor="white")
    ax.set_facecolor("white")

    group_values = list(df[group_col].dropna().unique())
    group_values = sorted(
        group_values,
        key=lambda v: GROUP_ORDER.index(normalize_group(v)) if normalize_group(v) in GROUP_ORDER else len(GROUP_ORDER),
    )
    for group_raw in group_values:
        group_df = df[df[group_col] == group_raw]
        group = normalize_group(group_raw)
        ax.scatter(
            group_df[predictor],
            group_df[outcome],
            s=30,
            alpha=0.85,
            color=GROUP_COLOR.get(group, "#455a64"),
            edgecolors="white",
            linewidths=0.5,
            label=str(group_raw),
        )

    if len(sub_fit) >= 3 and np.nanstd(x) > 0:
        fit = linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, fit.intercept + fit.slope * xs, color="black", linewidth=1.2, alpha=0.85)

    annotation = (
        f"ρ={summary_row['spearman_rho']:.2f}, p={summary_row['spearman_p_value']:.3g}\n"
        f"r={summary_row['pearson_r']:.3f}, p={summary_row['pearson_p_value']:.3g}"
    )
    (ha, va, tx, ty), legend_loc = choose_text_and_legend_locations(df, predictor, outcome)
    ax.text(tx, ty, annotation, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5)

    ax.set_xlabel(pretty_name(predictor))
    ax.set_ylabel(pretty_name(outcome))
    ax.set_title(f"{pretty_name(outcome)} vs {pretty_name(predictor)}", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc=legend_loc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def select_summary_rows(summary: pd.DataFrame, outcome: str, top_k: int | None) -> pd.DataFrame:
    rows = summary[summary["outcome"] == outcome].copy()
    rows = rows.sort_values("spearman_p_value", ascending=True)
    if top_k is not None and top_k > 0:
        rows = rows.head(top_k)
    return rows


def run_one(input_csv: Path, summary_csv: Path, out_dir: Path, outcome_cols: List[str] | None, top_k: int | None) -> None:
    df = pd.read_csv(input_csv)
    summary = pd.read_csv(summary_csv)
    group_col = resolve_column(df, GROUP_CANDIDATES, "group")
    predictors = set(get_predictor_columns(df))
    outcomes = get_outcome_columns(df, outcome_cols)

    made = 0
    for outcome in outcomes:
        rows = select_summary_rows(summary, outcome, top_k)
        for _, row in rows.iterrows():
            predictor = str(row["predictor"])
            if predictor not in predictors:
                continue
            output_path = out_dir / outcome / f"scatter_fit_{outcome}_vs_{safe_name(predictor)}.png"
            make_scatter_with_fit(df, row, predictor, outcome, group_col, output_path)
            made += 1
    print(f"Saved {made} scatter fit plots under: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch plot predictor scatters with linear fit lines.")
    parser.add_argument("--task", choices=["ot", "mw"], default="ot")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--outcome-cols", nargs="*", default=None, help="Default: pca_overlap SWD")
    parser.add_argument("--top-k", type=int, default=0, help="0 means all predictors; otherwise top-k by Spearman p-value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "mw":
        input_csv = Path(args.input_csv) if args.input_csv else DEFAULT_MW_INPUT
        summary_csv = Path(args.summary_csv) if args.summary_csv else DEFAULT_MW_SUMMARY
        out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_MW_OUT_DIR
    else:
        input_csv = Path(args.input_csv) if args.input_csv else DEFAULT_OT_INPUT
        summary_csv = Path(args.summary_csv) if args.summary_csv else DEFAULT_OT_SUMMARY
        out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OT_OUT_DIR

    top_k = None if args.top_k == 0 else args.top_k
    run_one(input_csv, summary_csv, out_dir, args.outcome_cols, top_k)


if __name__ == "__main__":
    main()
