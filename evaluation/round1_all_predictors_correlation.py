#!/usr/bin/env python3
"""
Correlation analysis and plots for all round-1 predictors in one table.

Input:
  results/round1_all_predictors/round1_all_predictors_wide.csv

Predictors:
  - 4 global predictors:
      distance_to_centroid
      pca2d_distance_to_centroid
      ERP_deviation
      PSD_deviation
  - all columns ending with "__abs_dev_z" (16 feature-level predictors)

Outputs:
  - all_predictors_correlation_summary.csv
  - all_predictors_correlation_summary.txt
  - plots/*.png
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = PROJECT_ROOT / "results" / "round1_all_predictors" / "round1_all_predictors_wide.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "round1_all_predictors_correlation"

OVERLAP_CANDIDATES = ("overlap", "pca_overlap", "PCA_overlap")
GROUP_CANDIDATES = ("group", "category", "Group", "Category", "group_scheme2")
GLOBAL_PREDICTORS = (
    "distance_to_centroid",
    "pca2d_distance_to_centroid",
    "ERP_deviation",
    "PSD_deviation",
)

GROUP_COLOR = {
    "good": "#2e7d32",
    "borderline": "#f57c00",
    "poor": "#c62828",
}

PREDICTOR_LABELS = {
    "distance_to_centroid": "16D centroid",
    "pca2d_distance_to_centroid": "PCA-2D centroid",
    "ERP_deviation": "ERP dev.",
    "PSD_deviation": "PSD dev.",
}

FEATURE_LABELS = {
    "P3_Amplitude": r"P3$_A$",
    "P3_Latency": r"P3$_L$",
    "Activity": "HA",
    "Mobility": "HM",
    "Complexity": "HC",
    "Petrosian_Fractal_Dimension": "PFD",
    "Delta_PSD": r"PSD$_\delta$",
    "Theta_PSD": r"PSD$_\theta$",
    "Alpha_PSD": r"PSD$_\alpha$",
    "Beta_PSD": r"PSD$_\beta$",
    "Gamma_PSD": r"PSD$_\gamma$",
    "Delta_Entropy": r"E$_\delta$",
    "Theta_Entropy": r"E$_\theta$",
    "Alpha_Entropy": r"E$_\alpha$",
    "Beta_Entropy": r"E$_\beta$",
    "Gamma_Entropy": r"E$_\gamma$",
}

PREDICTOR_CATEGORY = {
    "distance_to_centroid": "Global distance",
    "pca2d_distance_to_centroid": "Global distance",
    "ERP_deviation": "Time-domain",
    "PSD_deviation": "Frequency-domain",
    "P3_Amplitude": "Time-domain",
    "P3_Latency": "Time-domain",
    "Activity": "Complexity",
    "Mobility": "Complexity",
    "Complexity": "Complexity",
    "Petrosian_Fractal_Dimension": "Complexity",
    "Delta_PSD": "Frequency-domain",
    "Theta_PSD": "Frequency-domain",
    "Alpha_PSD": "Frequency-domain",
    "Beta_PSD": "Frequency-domain",
    "Gamma_PSD": "Frequency-domain",
    "Delta_Entropy": "Entropy",
    "Theta_Entropy": "Entropy",
    "Alpha_Entropy": "Entropy",
    "Beta_Entropy": "Entropy",
    "Gamma_Entropy": "Entropy",
}

CATEGORY_COLOR = {
    "Global distance": "#4C566A",    # refined blue grey
    "Time-domain": "#7E57C2",        # muted violet
    "Frequency-domain": "#2F6DB3",   # scholarly blue
    "Complexity": "#3B8C5A",         # muted green
    "Entropy": "#B66A2E",            # ochre / burnt orange
}


def resolve_column(df: pd.DataFrame, candidates: Tuple[str, ...], name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find {name} column from candidates: {candidates}")


def pretty_name(col: str) -> str:
    if col in PREDICTOR_LABELS:
        return PREDICTOR_LABELS[col]
    if col.endswith("__abs_dev_z"):
        feature = col.replace("__abs_dev_z", "")
        return FEATURE_LABELS.get(feature, feature.replace("_", " "))
    return col.replace("__abs_dev_z", " deviation").replace("_", " ")


def predictor_category(col: str) -> str:
    if col.endswith("__abs_dev_z"):
        col = col.replace("__abs_dev_z", "")
    return PREDICTOR_CATEGORY.get(col, "Other")


def normalize_group(v: str) -> str:
    x = str(v).strip().lower()
    return x if x in GROUP_COLOR else "other"


def get_predictor_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in GLOBAL_PREDICTORS if c in df.columns]
    feature_cols = sorted(c for c in df.columns if c.endswith("__abs_dev_z"))
    cols.extend(feature_cols)
    if not cols:
        raise ValueError("No predictor columns found.")
    return cols


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
    rho, p_s = spearmanr(sub[x_col], sub[y_col])
    r, p_p = pearsonr(sub[x_col], sub[y_col])
    return {
        "n_samples": n,
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_s),
        "pearson_r": float(r),
        "pearson_p_value": float(p_p),
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


def add_fdr(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for p_col in ("spearman_p_value", "pearson_p_value"):
        vals = out[p_col].to_numpy(dtype=float)
        out[f"{p_col}_fdr_bh"] = fdr_bh(vals)
    return out


def choose_text_and_legend_locations(df: pd.DataFrame, x_col: str, y_col: str):
    xy = df[[x_col, y_col]].dropna().to_numpy(dtype=float)
    if len(xy) == 0:
        return ("left", "top", 0.03, 0.97), "lower right"

    x_min, x_max = np.nanmin(xy[:, 0]), np.nanmax(xy[:, 0])
    y_min, y_max = np.nanmin(xy[:, 1]), np.nanmax(xy[:, 1])
    x_span = max(1e-12, x_max - x_min)
    y_span = max(1e-12, y_max - y_min)
    xn = (xy[:, 0] - x_min) / x_span
    yn = (xy[:, 1] - y_min) / y_span

    w, h = 0.34, 0.16
    candidates = [
        ("left", "top", 0.03, 0.97),
        ("right", "top", 0.97, 0.97),
        ("left", "bottom", 0.03, 0.03),
        ("right", "bottom", 0.97, 0.03),
    ]

    def count(ha: str, va: str, x0: float, y0: float) -> int:
        if ha == "left":
            xl, xr = x0, min(1.0, x0 + w)
        else:
            xl, xr = max(0.0, x0 - w), x0
        if va == "top":
            yb, yt = max(0.0, y0 - h), y0
        else:
            yb, yt = y0, min(1.0, y0 + h)
        return int(np.sum((xn >= xl) & (xn <= xr) & (yn >= yb) & (yn <= yt)))

    text_candidate = min([(count(*c), c) for c in candidates], key=lambda z: z[0])[1]
    text_key = (text_candidate[0], text_candidate[1])
    legend_candidates = [c for c in candidates if (c[0], c[1]) != text_key]
    legend_candidate = min([(count(*c), c) for c in legend_candidates], key=lambda z: z[0])[1]
    legend_loc_map = {
        ("left", "top"): "upper left",
        ("right", "top"): "upper right",
        ("left", "bottom"): "lower left",
        ("right", "bottom"): "lower right",
    }
    return text_candidate, legend_loc_map[(legend_candidate[0], legend_candidate[1])]


def make_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    annotation: str,
    save_path: Path,
) -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9

    fig, ax = plt.subplots(figsize=(4.8, 3.8), facecolor="white")
    ax.set_facecolor("white")

    for g_raw in sorted(df[group_col].dropna().unique(), key=lambda v: str(v)):
        sub = df[df[group_col] == g_raw]
        g = normalize_group(g_raw)
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=30,
            alpha=0.85,
            color=GROUP_COLOR.get(g, "#455a64"),
            edgecolors="white",
            linewidths=0.5,
            label=str(g_raw),
        )

    (ha, va, tx, ty), legend_loc = choose_text_and_legend_locations(df, x_col, y_col)
    ax.text(tx, ty, annotation, transform=ax.transAxes, ha=ha, va=va, fontsize=9, bbox=None)

    ax.set_xlabel(pretty_name(x_col))
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {pretty_name(x_col)}", loc="left")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc=legend_loc)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_annotation(row: pd.Series, method: str) -> str:
    if method == "spearman":
        return f"ρ={row['spearman_rho']:.2f}, p={row['spearman_p_value']:.3g}"
    if method == "pearson":
        return f"r={row['pearson_r']:.3f}, p={row['pearson_p_value']:.3g}"
    return (
        f"ρ={row['spearman_rho']:.2f}, p={row['spearman_p_value']:.3g}\n"
        f"r={row['pearson_r']:.3f}, p={row['pearson_p_value']:.3g}"
    )


def significance_stars(q_value: float) -> str:
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def make_spearman_summary_plot(
    summary: pd.DataFrame,
    save_path: Path,
    p_col: str = "spearman_p_value_fdr_bh",
    note: str = "* FDR-p<0.05, ** FDR-p<0.01, *** FDR-p<0.001",
) -> None:
    """Horizontal coefficient summary plot for Spearman ρ."""
    plot_df = summary.copy()
    plot_df["abs_spearman_rho"] = plot_df["spearman_rho"].abs()
    plot_df = plot_df.sort_values("abs_spearman_rho", ascending=False).reset_index(drop=True)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 13

    height = max(5.0, 0.34 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.4, height), facecolor="white")
    ax.set_facecolor("white")

    y = np.arange(len(plot_df))
    rho = plot_df["spearman_rho"].to_numpy(dtype=float)
    q = plot_df[p_col].to_numpy(dtype=float)

    if "predictor_category" in plot_df.columns:
        categories = plot_df["predictor_category"].tolist()
    else:
        categories = [predictor_category(p) for p in plot_df["predictor"]]
    colors = [CATEGORY_COLOR.get(c, "#455a64") for c in categories]

    for yi, ri, c in zip(y, rho, colors):
        ax.hlines(yi, 0, ri, color=c, linewidth=2.0, alpha=0.85)
    ax.scatter(rho, y, s=36, color=colors, edgecolors="white", linewidths=0.6, zorder=3)

    # Annotate rho values and significance stars from p_col.
    for yi, ri, qv in zip(y, rho, q):
        stars = significance_stars(qv)
        dx = 0.018 if ri >= 0 else -0.018
        ha = "left" if ri >= 0 else "right"
        ax.text(ri + dx, yi, f"{ri:.2f}{stars}", va="center", ha=ha, fontsize=11, color="black")

    labels = plot_df["predictor_pretty"].tolist()
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman ρ with overlap")
    ax.set_xlim(-1.15, 1.15)
    ax.set_xticks(np.linspace(-1.0, 1.0, 5))
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_order = [
        "Global distance",
        "Time-domain",
        "Frequency-domain",
        "Complexity",
        "Entropy",
    ]
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CATEGORY_COLOR[c],
               markeredgecolor="white", markersize=8, label=c)
        for c in legend_order
        if c in set(categories)
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)

    ax.text(0.0, -0.08, note, transform=ax.transAxes, ha="left", va="top", fontsize=11)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Correlation analysis for all round-1 predictors.")
    ap.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--corr-method", type=str, default="both", choices=["spearman", "pearson", "both"])
    ap.add_argument(
        "--plot-mode",
        type=str,
        default="all",
        choices=["all", "top"],
        help="Plot all predictors or only top-k ranked by selected p-value.",
    )
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input csv not found: {input_csv}")

    df = pd.read_csv(input_csv)
    overlap_col = resolve_column(df, OVERLAP_CANDIDATES, "overlap")
    group_col = resolve_column(df, GROUP_CANDIDATES, "group")
    predictor_cols = get_predictor_columns(df)

    rows: List[Dict[str, object]] = []
    for x_col in predictor_cols:
        rows.append(
            {
                "predictor": x_col,
                "predictor_pretty": pretty_name(x_col),
                "y_variable": overlap_col,
                **corr_one(df, x_col, overlap_col),
            }
        )

    summary = add_fdr(pd.DataFrame(rows))
    sort_col = "spearman_p_value" if args.corr_method in ("spearman", "both") else "pearson_p_value"
    summary = summary.sort_values(sort_col, ascending=True).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "all_predictors_correlation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    summary_txt = out_dir / "all_predictors_correlation_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("All round-1 predictors correlation summary\n")
        f.write("=" * 48 + "\n\n")
        for _, r in summary.iterrows():
            f.write(
                f"{r['predictor']}: "
                f"ρ={r['spearman_rho']:.4f}, p_s={r['spearman_p_value']:.4g}, q_s={r['spearman_p_value_fdr_bh']:.4g}; "
                f"r={r['pearson_r']:.4f}, p_p={r['pearson_p_value']:.4g}, q_p={r['pearson_p_value_fdr_bh']:.4g}; "
                f"n={int(r['n_samples'])}\n"
            )

    if args.plot_mode == "all":
        to_plot = summary["predictor"].tolist()
    else:
        to_plot = summary["predictor"].head(max(0, args.top_k)).tolist()

    plot_dir = out_dir / "plots"
    for predictor in to_plot:
        row = summary[summary["predictor"] == predictor].iloc[0]
        annotation = make_annotation(row, args.corr_method)
        safe_name = predictor.replace("/", "_").replace(" ", "_")
        fig_path = plot_dir / f"scatter_{args.corr_method}_{overlap_col}_vs_{safe_name}.png"
        make_scatter(df, predictor, overlap_col, group_col, annotation, fig_path)

    coef_plot = out_dir / "spearman_rho_summary_plot.png"
    make_spearman_summary_plot(summary, coef_plot)

    print(f"Input: {input_csv}")
    print(f"Predictors: {len(predictor_cols)}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved text: {summary_txt}")
    print(f"Saved plots ({len(to_plot)}): {plot_dir}")
    print(f"Saved Spearman ρ summary plot: {coef_plot}")


if __name__ == "__main__":
    main()

