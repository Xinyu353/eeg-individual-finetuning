#!/usr/bin/env python3
"""
Complete round-1 analysis:

participant-level real-data predictors -> participant-level fine-tuning outcomes

Predictors (A; real relative to group):
- 16 feature deviations (*__abs_dev_z)
- centroid distances (16D and PCA-2D)
- Mahalanobis distance
- local density
- ERP/PSD deviations

Outcomes (B; fine-tuning real vs fake):
- pca_overlap
- SWD

Outputs:
- predictor_to_outcome_correlation_summary.csv
- predictor_to_outcome_correlation_summary.txt
- one Spearman ρ summary plot per outcome
- optional scatter plots for top predictors per outcome
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from round1_all_predictors_correlation import (
    CATEGORY_COLOR,
    FEATURE_LABELS,
    GLOBAL_PREDICTORS,
    GROUP_COLOR,
    GROUP_CANDIDATES,
    PREDICTOR_CATEGORY,
    PREDICTOR_LABELS,
    choose_text_and_legend_locations,
    normalize_group,
    predictor_category,
    resolve_column,
    significance_stars,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = PROJECT_ROOT / "results" / "round1_all_predictors" / "round1_all_predictors_wide.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "round1_predictors_to_outcomes"

DEFAULT_OUTCOMES = ("pca_overlap", "SWD")
EXTRA_OUTCOMES: Tuple[str, ...] = ()
EXTRA_PREDICTORS = (
    "mahalanobis_distance",
    "centroid_local_density",
)
GROUP_ORDER = ("good", "borderline", "poor")


def pretty_name(col: str) -> str:
    labels = {
        **PREDICTOR_LABELS,
        "mahalanobis_distance": "Mahalanobis distance",
        "centroid_local_density": "Local density",
        "pca_overlap": "PCA overlap",
        "pca_overlap_rf": "PCA overlap (real vs. fake)",
        "SWD": "SWD",
        "SWD_rf": "SWD (real vs. fake)",
    }
    if col == "pca_overlap":
        return labels["pca_overlap_rf"]
    if col == "SWD":
        return labels["SWD_rf"]
    if col in labels:
        return labels[col]
    if col.endswith("__abs_dev_z"):
        feat = col.replace("__abs_dev_z", "")
        return FEATURE_LABELS.get(feat, feat.replace("_", " "))
    return col.replace("_", " ")


def get_predictor_columns(df: pd.DataFrame) -> List[str]:
    predictors = [c for c in GLOBAL_PREDICTORS if c in df.columns]
    predictors.extend(c for c in EXTRA_PREDICTORS if c in df.columns)
    predictors.extend(sorted(c for c in df.columns if c.endswith("__abs_dev_z")))
    if not predictors:
        raise ValueError("No predictor columns found.")
    return predictors


def get_outcome_columns(df: pd.DataFrame, requested: List[str] | None) -> List[str]:
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(f"Requested outcome columns missing: {missing}")
        return requested
    cols = [c for c in DEFAULT_OUTCOMES if c in df.columns]
    cols.extend(c for c in EXTRA_OUTCOMES if c in df.columns)
    if not cols:
        raise ValueError("No outcome columns found.")
    return cols


def predictor_category_extended(col: str) -> str:
    if col in ("mahalanobis_distance", "centroid_local_density"):
        return "Global distance"
    return predictor_category(col)


def corr_one(df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, float]:
    sub = df[[x_col, y_col]].dropna()
    n = len(sub)
    if n < 3:
        return {
            "n_samples": n,
            "spearman_rho": np.nan,
            "spearman_p_value": np.nan,
            "pearson_r": np.nan,
            "pearson_p_value": np.nan,
        }
    rho, ps = spearmanr(sub[x_col], sub[y_col])
    r, pp = pearsonr(sub[x_col], sub[y_col])
    return {
        "n_samples": n,
        "spearman_rho": float(rho),
        "spearman_p_value": float(ps),
        "pearson_r": float(r),
        "pearson_p_value": float(pp),
    }


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not np.any(mask):
        return q

    vals = p[mask]
    n = len(vals)
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_masked = np.empty_like(vals)
    q_masked[order] = adjusted
    q[mask] = q_masked
    return q


def add_fdr_by_outcome(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["spearman_p_value_fdr_bh"] = np.nan
    out["pearson_p_value_fdr_bh"] = np.nan
    for outcome, idx in out.groupby("outcome").groups.items():
        idx = list(idx)
        for p_col, q_col in (
            ("spearman_p_value", "spearman_p_value_fdr_bh"),
            ("pearson_p_value", "pearson_p_value_fdr_bh"),
        ):
            vals = out.loc[idx, p_col].to_numpy(dtype=float)
            out.loc[idx, q_col] = fdr_bh(vals)
    return out


def make_spearman_summary_plot(summary_one: pd.DataFrame, outcome: str, save_path: Path) -> None:
    plot_df = summary_one.copy()
    plot_df["abs_spearman_rho"] = plot_df["spearman_rho"].abs()
    plot_df = plot_df.sort_values("abs_spearman_rho", ascending=False).reset_index(drop=True)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 13

    height = max(5.0, 0.34 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.6, height), facecolor="white")
    ax.set_facecolor("white")

    y = np.arange(len(plot_df))
    rho = plot_df["spearman_rho"].to_numpy(dtype=float)
    q = plot_df["spearman_p_value_fdr_bh"].to_numpy(dtype=float)
    categories = [predictor_category_extended(p) for p in plot_df["predictor"]]
    colors = [CATEGORY_COLOR.get(c, "#455a64") for c in categories]

    for yi, ri, c in zip(y, rho, colors):
        ax.hlines(yi, 0, ri, color=c, linewidth=2.0, alpha=0.85)
    ax.scatter(rho, y, s=36, color=colors, edgecolors="white", linewidths=0.6, zorder=3)

    for yi, ri, qv in zip(y, rho, q):
        stars = significance_stars(qv)
        dx = 0.018 if ri >= 0 else -0.018
        ha = "left" if ri >= 0 else "right"
        ax.text(ri + dx, yi, f"{ri:.2f}{stars}", va="center", ha=ha, fontsize=11, color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["predictor_pretty"].tolist())
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Spearman ρ with {pretty_name(outcome)}")
    ax.set_xlim(-1.15, 1.15)
    ax.set_xticks(np.linspace(-1.0, 1.0, 5))
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_order = ["Global distance", "Time-domain", "Frequency-domain", "Complexity", "Entropy"]
    category_set = set(categories)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CATEGORY_COLOR[c],
               markeredgecolor="white", markersize=8, label=c)
        for c in legend_order
        if c in category_set
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    ax.text(
        0.26,
        0.02,
        "* FDR-p < 0.05\n** FDR-p < 0.01\n*** FDR-p < 0.001",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_scatter(df: pd.DataFrame, predictor: str, outcome: str, group_col: str, row: pd.Series, save_path: Path) -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9

    fig, ax = plt.subplots(figsize=(4.8, 3.8), facecolor="white")
    ax.set_facecolor("white")
    group_values = list(df[group_col].dropna().unique())
    group_values = sorted(
        group_values,
        key=lambda v: GROUP_ORDER.index(normalize_group(v)) if normalize_group(v) in GROUP_ORDER else len(GROUP_ORDER),
    )
    for g_raw in group_values:
        sub = df[df[group_col] == g_raw]
        g = normalize_group(g_raw)
        ax.scatter(
            sub[predictor],
            sub[outcome],
            s=30,
            alpha=0.85,
            color=GROUP_COLOR.get(g, "#757575"),
            edgecolors="white",
            linewidths=0.5,
            label=str(g_raw),
        )

    annotation = f"ρ={row['spearman_rho']:.2f}, p={row['spearman_p_value']:.3g}"
    (ha, va, tx, ty), legend_loc = choose_text_and_legend_locations(df, predictor, outcome)
    ax.text(tx, ty, annotation, transform=ax.transAxes, ha=ha, va=va, fontsize=9)
    ax.set_xlabel(pretty_name(predictor))
    ax.set_ylabel(pretty_name(outcome))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc=legend_loc)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Round1 predictors-to-outcomes analysis.")
    ap.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--outcome-cols", nargs="*", default=None, help="Default: pca_overlap SWD")
    ap.add_argument("--plot-top-k", type=int, default=6, help="Scatter plots per outcome for top-k predictors.")
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input table not found: {input_csv}")

    df = pd.read_csv(input_csv)
    group_col = resolve_column(df, GROUP_CANDIDATES, "group")
    predictors = get_predictor_columns(df)
    outcomes = get_outcome_columns(df, args.outcome_cols)

    rows: List[Dict[str, object]] = []
    for outcome in outcomes:
        for predictor in predictors:
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_pretty": pretty_name(outcome),
                    "predictor": predictor,
                    "predictor_pretty": pretty_name(predictor),
                    "predictor_category": predictor_category_extended(predictor),
                    **corr_one(df, predictor, outcome),
                }
            )

    summary = add_fdr_by_outcome(pd.DataFrame(rows))
    summary = summary.sort_values(["outcome", "spearman_p_value"], ascending=[True, True]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "predictor_to_outcome_correlation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    summary_txt = out_dir / "predictor_to_outcome_correlation_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Round1 real-data predictors -> fine-tuning outcomes\n")
        f.write("=" * 56 + "\n\n")
        for outcome in outcomes:
            f.write(f"[Outcome: {outcome}]\n")
            tmp = summary[summary["outcome"] == outcome].sort_values("spearman_p_value")
            for _, r in tmp.head(10).iterrows():
                f.write(
                    f"  {r['predictor']}: ρ={r['spearman_rho']:.4f}, "
                    f"p={r['spearman_p_value']:.4g}, q={r['spearman_p_value_fdr_bh']:.4g}; "
                    f"r={r['pearson_r']:.4f}, p={r['pearson_p_value']:.4g}, q={r['pearson_p_value_fdr_bh']:.4g}\n"
                )
            f.write("\n")

    for outcome in outcomes:
        tmp = summary[summary["outcome"] == outcome]
        make_spearman_summary_plot(
            tmp,
            outcome,
            out_dir / f"spearman_rho_summary_{outcome}.png",
        )
        top = tmp.sort_values("spearman_p_value").head(max(0, int(args.plot_top_k)))
        for _, row in top.iterrows():
            predictor = str(row["predictor"])
            safe_pred = predictor.replace("/", "_").replace(" ", "_")
            make_scatter(
                df,
                predictor,
                outcome,
                group_col,
                row,
                out_dir / "scatter_top" / outcome / f"scatter_{outcome}_vs_{safe_pred}.png",
            )

    print(f"Input: {input_csv}")
    print(f"Predictors: {len(predictors)}")
    print(f"Outcomes: {outcomes}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved text: {summary_txt}")
    print(f"Saved outcome summary plots and top scatter plots under: {out_dir}")


if __name__ == "__main__":
    main()

