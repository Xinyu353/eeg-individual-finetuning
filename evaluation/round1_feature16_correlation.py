#!/usr/bin/env python3
"""
Round-1 feature-level participant-vs-group analysis (16 real features).

Goal:
- For each participant and each feature, measure how far participant is from
  group-level center in REAL data.
- Correlate that deviation with overlap (fine-tuning success proxy).

Inputs:
- Overlap table (participant-level), e.g. pca_overlap_categorized_enriched.csv
- Real feature files per participant: feature_extraction/real_ot/p{pid}_real.csv

Outputs:
- participant_feature_deviation_long.csv
- participant_feature_deviation_wide.csv
- feature_correlation_summary.csv
- optional scatter plots for top features
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_OVERLAP_CSV = PROJECT_ROOT / "results" / "pca_overlap_summary" / "pca_overlap_categorized_enriched.csv"
DEFAULT_REAL_OT_DIR = PROJECT_ROOT / "feature_extraction" / "real_ot"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "round1_feature16"

PID_CANDIDATES = ("participant_id", "participant", "Participant", "pid")
OVERLAP_CANDIDATES = ("overlap", "pca_overlap", "PCA_overlap")
GROUP_CANDIDATES = ("group", "category", "Group", "Category", "group_scheme2")

GROUP_COLOR = {
    "good": "#2e7d32",
    "borderline": "#f57c00",
    "poor": "#c62828",
}


def resolve_column(df: pd.DataFrame, candidates: Tuple[str, ...], name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find {name} column from candidates: {candidates}")


def normalize_group(v: str) -> str:
    x = str(v).strip().lower()
    return x if x in GROUP_COLOR else "other"


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


def load_participant_feature_means(real_ot_dir: Path, participants: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for pid in sorted(set(participants)):
        fp = real_ot_dir / f"p{pid}_real.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")
        df = pd.read_csv(fp)
        if "SampleID" in df.columns:
            feat_df = df.drop(columns=["SampleID"])
        else:
            feat_df = df.copy()
        feat_df = feat_df.select_dtypes(include=[np.number])
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if feat_df.empty:
            raise ValueError(f"No valid feature rows in {fp}")
        means = feat_df.mean(axis=0)
        row = {"participant_id": pid, "n_trials_feature": int(len(feat_df))}
        row.update({k: float(v) for k, v in means.items()})
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    return out


def compute_deviation_tables(mean_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in mean_df.columns if c not in ("participant_id", "n_trials_feature")]
    global_mean = mean_df[feature_cols].mean(axis=0)
    global_std = mean_df[feature_cols].std(axis=0, ddof=0).replace(0, np.nan)

    long_rows = []
    for _, r in mean_df.iterrows():
        pid = int(r["participant_id"])
        for feat in feature_cols:
            part_val = float(r[feat])
            g = float(global_mean[feat])
            s = float(global_std[feat]) if pd.notna(global_std[feat]) else np.nan
            signed = part_val - g
            abs_dev = abs(signed)
            z_abs = abs_dev / s if pd.notna(s) and s > 0 else np.nan
            long_rows.append(
                {
                    "participant_id": pid,
                    "feature": feat,
                    "participant_mean": part_val,
                    "group_mean": g,
                    "signed_dev": signed,
                    "abs_dev": abs_dev,
                    "abs_dev_z": z_abs,
                }
            )
    long_df = pd.DataFrame(long_rows)

    # wide table: one row per participant, columns = feature__abs_dev_z
    wide = long_df.pivot(index="participant_id", columns="feature", values="abs_dev_z")
    wide.columns = [f"{c}__abs_dev_z" for c in wide.columns]
    wide = wide.reset_index()
    return long_df, wide


def corr_one(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    sub = pd.concat([x, y], axis=1).dropna()
    if len(sub) < 3:
        return {"n": len(sub), "spearman_rho": np.nan, "spearman_p": np.nan, "pearson_r": np.nan, "pearson_p": np.nan}
    rho, p_s = spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
    r, p_p = pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])
    return {"n": len(sub), "spearman_rho": float(rho), "spearman_p": float(p_s), "pearson_r": float(r), "pearson_p": float(p_p)}


def make_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    text: str,
    save_path: Path,
    feature_name: str | None = None,
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
    for g_raw in sorted(df[group_col].dropna().unique(), key=lambda z: str(z)):
        sub = df[df[group_col] == g_raw]
        g = normalize_group(g_raw)
        color = GROUP_COLOR.get(g, "#455a64")
        ax.scatter(sub[x_col], sub[y_col], s=30, alpha=0.85, color=color, edgecolors="white", linewidths=0.5, label=str(g_raw))

    if feature_name:
        ax.set_xlabel(f"{feature_name} deviation (abs_dev_z)")
        ax.set_title(f"{y_col} vs {feature_name} deviation", loc="left")
    else:
        ax.set_xlabel(x_col)
        ax.set_title(f"{y_col} vs {x_col}", loc="left")
    ax.set_ylabel(y_col)
    ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox=None)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature16 participant-vs-group correlation with overlap.")
    ap.add_argument("--overlap-csv", type=str, default=str(DEFAULT_OVERLAP_CSV))
    ap.add_argument("--real-ot-dir", type=str, default=str(DEFAULT_REAL_OT_DIR))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--corr-method", type=str, default="spearman", choices=["spearman", "pearson", "both"])
    ap.add_argument("--plot-top-k", type=int, default=6, help="Generate scatter plots for top-k features by selected p-value.")
    args = ap.parse_args()

    overlap_csv = Path(args.overlap_csv)
    real_ot_dir = Path(args.real_ot_dir)
    out_dir = Path(args.output_dir)

    if not overlap_csv.exists():
        raise FileNotFoundError(f"Missing overlap csv: {overlap_csv}")
    if not real_ot_dir.is_dir():
        raise FileNotFoundError(f"Missing real_ot dir: {real_ot_dir}")

    ov = pd.read_csv(overlap_csv)
    pid_col = resolve_column(ov, PID_CANDIDATES, "participant id")
    overlap_col = resolve_column(ov, OVERLAP_CANDIDATES, "overlap")
    group_col = resolve_column(ov, GROUP_CANDIDATES, "group")

    ov = ov.copy()
    ov["participant_id"] = ov[pid_col].astype(int)
    participants = sorted(ov["participant_id"].unique().tolist())

    mean_df = load_participant_feature_means(real_ot_dir, participants)
    long_dev, wide_dev = compute_deviation_tables(mean_df)

    # merge for per-feature correlations
    corr_base = ov[["participant_id", overlap_col, group_col]].drop_duplicates()
    corr_rows: List[Dict[str, object]] = []
    for feat, sub in long_dev.groupby("feature"):
        m = corr_base.merge(sub[["participant_id", "abs_dev_z"]], on="participant_id", how="left")
        stats = corr_one(m["abs_dev_z"], m[overlap_col])
        corr_rows.append(
            {
                "feature": feat,
                "x_variable": "abs_dev_z",
                "y_variable": overlap_col,
                **stats,
            }
        )
    corr_df = pd.DataFrame(corr_rows)

    # Multiple-comparison correction (FDR) for both p columns
    for pcol in ("spearman_p", "pearson_p"):
        pv = corr_df[pcol].to_numpy(dtype=float)
        corr_df[f"{pcol}_fdr_bh"] = fdr_bh(pv)

    # Sorting
    if args.corr_method == "spearman":
        corr_df = corr_df.sort_values("spearman_p", ascending=True)
        p_for_rank = "spearman_p"
    elif args.corr_method == "pearson":
        corr_df = corr_df.sort_values("pearson_p", ascending=True)
        p_for_rank = "pearson_p"
    else:
        corr_df = corr_df.sort_values("spearman_p", ascending=True)
        p_for_rank = "spearman_p"

    out_dir.mkdir(parents=True, exist_ok=True)
    long_csv = out_dir / "participant_feature_deviation_long.csv"
    wide_csv = out_dir / "participant_feature_deviation_wide.csv"
    corr_csv = out_dir / "feature_correlation_summary.csv"
    long_dev.to_csv(long_csv, index=False)
    wide_dev.to_csv(wide_csv, index=False)
    corr_df.to_csv(corr_csv, index=False)

    # scatter for top-k features
    k = max(0, int(args.plot_top_k))
    top_feats = corr_df["feature"].head(k).tolist()
    for feat in top_feats:
        d = long_dev[long_dev["feature"] == feat][["participant_id", "abs_dev_z"]]
        m = corr_base.merge(d, on="participant_id", how="left")
        row = corr_df[corr_df["feature"] == feat].iloc[0]
        if args.corr_method == "spearman":
            text = f"ρ={row['spearman_rho']:.2f}, p={row['spearman_p']:.3g}"
            fn = f"scatter_spearman_{overlap_col}_vs_{feat}__abs_dev_z.png"
        elif args.corr_method == "pearson":
            text = f"r={row['pearson_r']:.3f}, p={row['pearson_p']:.3g}"
            fn = f"scatter_pearson_{overlap_col}_vs_{feat}__abs_dev_z.png"
        else:
            text = (
                f"ρ={row['spearman_rho']:.2f}, p={row['spearman_p']:.3g}\n"
                f"r={row['pearson_r']:.3f}, p={row['pearson_p']:.3g}"
            )
            fn = f"scatter_both_{overlap_col}_vs_{feat}__abs_dev_z.png"
        make_scatter(
            m,
            "abs_dev_z",
            overlap_col,
            group_col,
            text,
            out_dir / "plots" / fn,
            feature_name=feat,
        )

    txt = out_dir / "README_round1_feature16.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Round1 feature16 participant-vs-group analysis\n")
        f.write("=" * 48 + "\n\n")
        f.write("Deviation metric: abs_dev_z = |participant_mean - group_mean| / group_std\n")
        f.write(f"Participants: {len(participants)}\n")
        f.write(f"Ranking p-value column: {p_for_rank}\n")
        f.write("Outputs:\n")
        f.write(f"- {long_csv}\n- {wide_csv}\n- {corr_csv}\n")

    print(f"Input overlap: {overlap_csv}")
    print(f"Real feature dir: {real_ot_dir}")
    print(f"Participants: {len(participants)}")
    print(f"Saved: {long_csv}")
    print(f"Saved: {wide_csv}")
    print(f"Saved: {corr_csv}")
    print(f"Saved: {txt}")
    if k > 0:
        print(f"Saved top-{k} scatter plots under: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()

