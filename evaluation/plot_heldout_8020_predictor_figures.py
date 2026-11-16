#!/usr/bin/env python3
"""Redraw 80/20 OT/MW predictor figures with the original analysis.

Outcome is held-out PCA overlap, summarized as 5-seed mean or median.
Stars and scatter annotations both use uncorrected Spearman p-values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from round1_all_predictors_correlation import (  # noqa: E402
    GLOBAL_PREDICTORS,
    GROUP_COLOR,
    add_fdr,
    make_spearman_summary_plot,
)
from round1_all_predictors_correlation import corr_one as corr_one_df  # noqa: E402
from round1_predictors_to_outcomes_analysis import (  # noqa: E402
    EXTRA_PREDICTORS,
    get_predictor_columns,
    pretty_name,
    predictor_category_extended,
)

DEFAULT_BASES = {
    "ot": PROJECT_ROOT / "results" / "ot_heldout_seed_sensitivity_all_final",
    "mw": PROJECT_ROOT / "results" / "mw_heldout_seed_sensitivity_all_final",
}
PREDICTOR_CANDIDATES = (
    "calibration_predictors_global_completed.csv",
    "calibration_predictors.csv",
)
GROUP_ORDER = ("good", "borderline", "poor")
SCATTER_ALIASES = {
    "distance_to_centroid": "16d_centroid",
    "pca2d_distance_to_centroid": "pca2d_centroid",
    "ERP_deviation": "ERP_dev",
    "PSD_deviation": "PSD_dev",
    "mahalanobis_distance": "Mahalanobis",
    "centroid_local_density": "local_density",
    "P3_Amplitude__abs_dev_z": "P3A",
    "P3_Latency__abs_dev_z": "P3L",
    "Activity__abs_dev_z": "HA",
    "Mobility__abs_dev_z": "HM",
    "Complexity__abs_dev_z": "HC",
    "Petrosian_Fractal_Dimension__abs_dev_z": "PFD",
    "Delta_PSD__abs_dev_z": "PSD_delta",
    "Theta_PSD__abs_dev_z": "PSD_theta",
    "Alpha_PSD__abs_dev_z": "PSD_alpha",
    "Beta_PSD__abs_dev_z": "PSD_beta",
    "Gamma_PSD__abs_dev_z": "PSD_gamma",
    "Delta_Entropy__abs_dev_z": "E_delta",
    "Theta_Entropy__abs_dev_z": "E_theta",
    "Alpha_Entropy__abs_dev_z": "E_alpha",
    "Beta_Entropy__abs_dev_z": "E_beta",
    "Gamma_Entropy__abs_dev_z": "E_gamma",
}


def categorize_overlap(value: float) -> str:
    if value >= 0.60:
        return "Good"
    if value >= 0.50:
        return "Borderline"
    return "Poor"


def normalize_group(value: str) -> str:
    key = str(value).strip().lower()
    return key if key in GROUP_COLOR else "other"


def load_participant_table(base: Path) -> pd.DataFrame:
    pred_path = next((base / name for name in PREDICTOR_CANDIDATES if (base / name).exists()), None)
    if pred_path is None:
        raise FileNotFoundError(f"No calibration predictor table in {base}")
    predictors = pd.read_csv(pred_path)
    runs = pd.read_csv(base / "seed_sensitivity_runs.csv")
    overlap = (
        runs.groupby(["condition", "participant"], sort=True)["heldout_PCA_overlap_RF"]
        .agg(
            heldout_PCA_overlap_RF_mean="mean",
            heldout_PCA_overlap_RF_median="median",
            heldout_PCA_overlap_RF_sd="std",
        )
        .reset_index()
    )
    merged = predictors.merge(overlap, on=["condition", "participant"], how="inner")
    keep = [c for c in GLOBAL_PREDICTORS if c in merged.columns]
    keep.extend(c for c in EXTRA_PREDICTORS if c in merged.columns)
    keep.extend(sorted(c for c in merged.columns if c.endswith("__abs_dev_z")))
    extra = [
        "condition",
        "participant",
        "heldout_PCA_overlap_RF_mean",
        "heldout_PCA_overlap_RF_median",
        "heldout_PCA_overlap_RF_sd",
    ]
    return merged[extra + keep].copy()


def correlation_summary(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    work = df.copy()
    work["pca_overlap"] = work[outcome]
    predictors = get_predictor_columns(work)
    rows = []
    for predictor in predictors:
        rows.append(
            {
                "outcome": outcome,
                "predictor": predictor,
                "predictor_pretty": pretty_name(predictor),
                "predictor_category": predictor_category_extended(predictor),
                **corr_one_df(work, predictor, outcome),
            }
        )
    return add_fdr(pd.DataFrame(rows)).sort_values("spearman_p_value").reset_index(drop=True)


def _corner_occupancy(ax, x: np.ndarray, y: np.ndarray) -> list[tuple[float, tuple, str]]:
    """Score plot corners by how many data points fall in each axes-fraction box."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_span = max(1e-12, xlim[1] - xlim[0])
    y_span = max(1e-12, ylim[1] - ylim[0])
    xn = (x - xlim[0]) / x_span
    yn = (y - ylim[0]) / y_span

    # Wider boxes so nearby points count as overlap.
    w, h = 0.46, 0.28
    candidates = [
        (("left", "top", 0.03, 0.97), "upper left"),
        (("right", "top", 0.97, 0.97), "upper right"),
        (("left", "bottom", 0.03, 0.03), "lower left"),
        (("right", "bottom", 0.97, 0.03), "lower right"),
    ]

    scored = []
    for (ha, va, x0, y0), legend_loc in candidates:
        if ha == "left":
            xl, xr = x0, min(1.0, x0 + w)
        else:
            xl, xr = max(0.0, x0 - w), x0
        if va == "top":
            yb, yt = max(0.0, y0 - h), y0
        else:
            yb, yt = y0, min(1.0, y0 + h)
        n_pts = int(np.sum((xn >= xl) & (xn <= xr) & (yn >= yb) & (yn <= yt)))
        scored.append((n_pts, (ha, va, x0, y0), legend_loc))
    scored.sort(key=lambda z: z[0])
    return scored


def make_scatter(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    save_path: Path,
) -> dict:
    sub = df[[predictor, outcome, "group"]].dropna()
    x = sub[predictor].to_numpy(dtype=float)
    y = sub[outcome].to_numpy(dtype=float)
    rho, p_value = spearmanr(x, y)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 8

    fig, ax = plt.subplots(figsize=(4.8, 3.8), facecolor="white")
    ax.set_facecolor("white")

    groups = list(sub["group"].dropna().unique())
    groups = sorted(
        groups,
        key=lambda v: GROUP_ORDER.index(normalize_group(v)) if normalize_group(v) in GROUP_ORDER else len(GROUP_ORDER),
    )
    for group_raw in groups:
        group_df = sub[sub["group"] == group_raw]
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

    if len(sub) >= 3 and np.nanstd(x) > 0:
        fit = linregress(x, y)
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xs, fit.intercept + fit.slope * xs, color="black", linewidth=1.2, alpha=0.85)

    # Extra margin so corner labels sit in empty space.
    x_pad = 0.10 * (float(x.max()) - float(x.min()) + 1e-12)
    y_pad = 0.12 * (float(y.max()) - float(y.min()) + 1e-12)
    ax.set_xlim(float(x.min()) - x_pad, float(x.max()) + x_pad)
    ax.set_ylim(float(y.min()) - y_pad, float(y.max()) + y_pad)

    scored = _corner_occupancy(ax, x, y)
    (ha, va, tx, ty) = scored[0][1]
    legend_loc = scored[1][2]
    annotation = f"ρ={rho:.2f}, p={p_value:.3g}"
    ax.text(
        tx,
        ty,
        annotation,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.9},
    )

    ax.set_xlabel(pretty_name(predictor))
    ax.set_ylabel("PCA overlap")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc=legend_loc, borderaxespad=0.4)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return {
        "outcome": outcome,
        "predictor": predictor,
        "n": int(len(sub)),
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_value),
        "path": str(save_path),
    }


def scatter_alias(predictor: str) -> str:
    if predictor in SCATTER_ALIASES:
        return SCATTER_ALIASES[predictor]
    return (
        pretty_name(predictor)
        .replace("$_", "")
        .replace("$", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "")
    )


def cleanup_old_scatter_locations(plot_dir: Path) -> None:
    for path in plot_dir.glob("*_scatter_heldout_PCA_overlap_RF_*.png"):
        path.unlink()
    for name in ("scatter_mean", "scatter_median"):
        old = plot_dir / name
        if old.is_dir():
            for path in old.glob("*.png"):
                path.unlink()
            old.rmdir()


def plot_one_outcome(
    df: pd.DataFrame,
    outcome: str,
    label: str,
    out_dir: Path,
) -> pd.DataFrame:
    plot_dir = out_dir / "plots"
    scatter_dir = plot_dir / "scatters" / label
    table_dir = out_dir / "tables" / "correlations"
    plot_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    work["group"] = work[outcome].apply(categorize_overlap)
    summary = correlation_summary(work, outcome)
    summary.to_csv(
        table_dir / f"current_heldout_{label}_all_predictors_correlation_summary_same_style.csv",
        index=False,
    )
    make_spearman_summary_plot(
        summary,
        plot_dir / f"current_heldout_{label}_all_predictors_spearman_summary_same_style.png",
        p_col="spearman_p_value",
        note="* p<0.05, ** p<0.01, *** p<0.001",
    )
    make_spearman_summary_plot(
        summary,
        plot_dir / f"current_heldout_{label}_all_predictors_spearman_summary_fdr.png",
        p_col="spearman_p_value_fdr_bh",
        note="* FDR q<0.05, ** FDR q<0.01, *** FDR q<0.001",
    )

    for predictor in get_predictor_columns(work):
        alias = scatter_alias(predictor)
        make_scatter(work, predictor, outcome, scatter_dir / f"{alias}.png")
    return summary


def main() -> None:
    for condition, base in DEFAULT_BASES.items():
        out_dir = base / "comparison"
        df = load_participant_table(base)
        if df.empty:
            raise RuntimeError(f"No merged 80/20 predictor/overlap rows in {base}")

        cleanup_old_scatter_locations(out_dir / "plots")
        mean_summary = plot_one_outcome(df, "heldout_PCA_overlap_RF_mean", "mean", out_dir)
        median_summary = plot_one_outcome(df, "heldout_PCA_overlap_RF_median", "median", out_dir)

        def sig_rows(summary: pd.DataFrame, p_col: str) -> pd.DataFrame:
            cols = ["predictor_pretty", "spearman_rho", "spearman_p_value", "spearman_p_value_fdr_bh"]
            return summary.loc[summary[p_col] < 0.05, cols]

        print(f"[{condition.upper()}] n participants: {len(df)}; n predictors: {len(mean_summary)}")
        print("  Mean uncorrected p<0.05:")
        print("  " + (sig_rows(mean_summary, "spearman_p_value").to_string(index=False).replace("\n", "\n  ") if len(sig_rows(mean_summary, "spearman_p_value")) else "none"))
        print("  Mean FDR q<0.05:")
        print("  " + (sig_rows(mean_summary, "spearman_p_value_fdr_bh").to_string(index=False).replace("\n", "\n  ") if len(sig_rows(mean_summary, "spearman_p_value_fdr_bh")) else "none"))
        print("  Median uncorrected p<0.05:")
        print("  " + (sig_rows(median_summary, "spearman_p_value").to_string(index=False).replace("\n", "\n  ") if len(sig_rows(median_summary, "spearman_p_value")) else "none"))
        print("  Median FDR q<0.05:")
        print("  " + (sig_rows(median_summary, "spearman_p_value_fdr_bh").to_string(index=False).replace("\n", "\n  ") if len(sig_rows(median_summary, "spearman_p_value_fdr_bh")) else "none"))
        print(f"  Wrote summaries under: {out_dir / 'plots'}")
        print(f"  Wrote scatters under: {out_dir / 'plots' / 'scatters'}")


if __name__ == "__main__":
    main()
