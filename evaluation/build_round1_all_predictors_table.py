#!/usr/bin/env python3
"""
Build one participant-level table containing all round-1 real-data predictors.

Output one wide CSV with:
- original participant-level overlap/group columns
- existing fine-tuning distance outcome if available:
  - SWD
- 4 global predictors:
  - distance_to_centroid
  - pca2d_distance_to_centroid
  - ERP_deviation
  - PSD_deviation
- additional real-data isolation/density predictors:
  - mahalanobis_distance
  - centroid_local_density
- 16 feature-level deviation predictors:
  - {feature}__abs_dev_z

This script does not use fake data and does not recompute overlap.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from enrich_participant_round1_real import (
    compute_distance_to_centroid,
    compute_erp_psd_deviations,
    compute_pca2d_distance_to_centroid,
    load_feature_matrix_for_participants,
    load_raw_ot_matrix,
    resolve_participant_column,
)
from round1_feature16_correlation import (
    compute_deviation_tables,
    load_participant_feature_means,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = PROJECT_ROOT / "results" / "pca_overlap_summary" / "pca_overlap_categorized.csv"
DEFAULT_REAL_OT_DIR = PROJECT_ROOT / "feature_extraction" / "real_ot"
DEFAULT_RAW_CSV = PROJECT_ROOT / "real_EEG_pretrained_model" / "ot_vs_c_Pz_4753.csv"
DEFAULT_DISTANCE_CSV = PROJECT_ROOT / "evaluation" / "ot_distance_metrics.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "round1_all_predictors" / "round1_all_predictors_wide.csv"


def compute_mahalanobis_and_density_predictors(
    X: np.ndarray,
    pid_row: np.ndarray,
    participant_ids: List[int],
    k_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Real-data participant isolation/density predictors in standardized 16-D feature space.

    mahalanobis_distance:
      Distance between participant centroid and global centroid using LedoitWolf
      covariance fitted on all real trials.

    centroid_local_density:
      1 / mean distance to k nearest other participant centroids.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    mu_global = Xs.mean(axis=0)

    lw = LedoitWolf().fit(Xs)
    precision = lw.precision_

    centroid_rows = []
    for pid in participant_ids:
        mask = pid_row == pid
        if not np.any(mask):
            continue
        mu_p = Xs[mask].mean(axis=0)
        diff = mu_p - mu_global
        maha = float(np.sqrt(max(diff @ precision @ diff, 0.0)))
        centroid_rows.append({"participant_id": pid, "mahalanobis_distance": maha, "centroid": mu_p})

    centroids = np.vstack([r["centroid"] for r in centroid_rows])
    n = len(centroid_rows)
    k = min(max(1, k_neighbors), max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(centroids)
    distances, _ = nn.kneighbors(centroids)
    # First neighbor is the point itself (distance 0), remove it.
    neighbor_distances = distances[:, 1:]

    eps = 1e-12
    rows = []
    for row, dists in zip(centroid_rows, neighbor_distances):
        mean_knn_dist = float(np.mean(dists)) if len(dists) else np.nan
        rows.append(
            {
                "participant_id": row["participant_id"],
                "mahalanobis_distance": row["mahalanobis_distance"],
                "centroid_local_density": float(1.0 / (mean_knn_dist + eps)),
            }
        )

    return pd.DataFrame(rows)


def merge_existing_distance_outcomes(out: pd.DataFrame, distance_csv: Path, pid_col: str) -> pd.DataFrame:
    if not distance_csv.exists():
        print(f"[warn] Distance outcome file not found, skip: {distance_csv}")
        return out
    d = pd.read_csv(distance_csv)
    required = {"participant_id", "epoch"}
    if not required.issubset(set(d.columns)):
        print(f"[warn] Distance outcome file lacks participant_id/epoch, skip: {distance_csv}")
        return out

    # Keep only SWD for the current round-1 outcome analysis.
    # MMD_RBF and WD are intentionally excluded from this unified table.
    keep_cols = [c for c in ["participant_id", "epoch", "SWD", "num_trials"] if c in d.columns]
    d = d[keep_cols].copy()
    merged = out.merge(
        d,
        left_on=[pid_col, "epoch"] if "epoch" in out.columns else [pid_col],
        right_on=["participant_id", "epoch"] if "epoch" in out.columns else ["participant_id"],
        how="left",
        suffixes=("", "_distance"),
    )
    if "participant_id_distance" in merged.columns:
        merged = merged.drop(columns=["participant_id_distance"])
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a single wide CSV with all round-1 predictors.")
    ap.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--real-ot-dir", "--real-feature-dir", dest="real_ot_dir", type=str, default=str(DEFAULT_REAL_OT_DIR))
    ap.add_argument("--raw-csv", type=str, default=str(DEFAULT_RAW_CSV))
    ap.add_argument("--distance-csv", type=str, default=str(DEFAULT_DISTANCE_CSV))
    ap.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    real_ot_dir = Path(args.real_ot_dir)
    raw_csv = Path(args.raw_csv)
    distance_csv = Path(args.distance_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input table not found: {input_csv}")
    if not real_ot_dir.is_dir():
        raise FileNotFoundError(f"real_ot directory not found: {real_ot_dir}")
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw OT csv not found: {raw_csv}")

    base = pd.read_csv(input_csv)
    pid_col = resolve_participant_column(base)
    participant_ids: List[int] = sorted(int(x) for x in base[pid_col].unique())

    print(f"Loaded base table: {input_csv}")
    print(f"Participant column: {pid_col}; n={len(participant_ids)}")

    # 4 global real-data predictors.
    X_feat, pid_row, feat_names = load_feature_matrix_for_participants(real_ot_dir, participant_ids)
    distance_map = compute_distance_to_centroid(X_feat, pid_row, participant_ids)
    pca2d_distance_map = compute_pca2d_distance_to_centroid(X_feat, pid_row, participant_ids)

    X_bc, pid_raw = load_raw_ot_matrix(raw_csv)
    erp_map, psd_map = compute_erp_psd_deviations(X_bc, pid_raw, participant_ids)

    out = base.copy()
    # Keep pca_overlap as the main PCA real-vs-fake outcome; drop JS for this analysis round.
    out = out.drop(columns=[c for c in ["pca_js", "MMD_RBF", "WD"] if c in out.columns])
    out = merge_existing_distance_outcomes(out, distance_csv, pid_col)

    out["distance_to_centroid"] = out[pid_col].map(distance_map)
    out["pca2d_distance_to_centroid"] = out[pid_col].map(pca2d_distance_map)
    out["ERP_deviation"] = out[pid_col].map(erp_map)
    out["PSD_deviation"] = out[pid_col].map(psd_map)

    density_df = compute_mahalanobis_and_density_predictors(X_feat, pid_row, participant_ids)
    out = out.merge(density_df, left_on=pid_col, right_on="participant_id", how="left", suffixes=("", "_density"))
    if "participant_id_density" in out.columns:
        out = out.drop(columns=["participant_id_density"])

    # 16 feature-level participant-vs-group deviation predictors.
    mean_df = load_participant_feature_means(real_ot_dir, participant_ids)
    _, feature_wide = compute_deviation_tables(mean_df)
    out = out.merge(feature_wide, left_on=pid_col, right_on="participant_id", how="left", suffixes=("", "_feature"))
    if "participant_id_feature" in out.columns:
        out = out.drop(columns=["participant_id_feature"])

    missing_cols = [
        "distance_to_centroid",
        "pca2d_distance_to_centroid",
        "ERP_deviation",
        "PSD_deviation",
        "mahalanobis_distance",
        "centroid_local_density",
    ] + [f"{f}__abs_dev_z" for f in feat_names]
    n_missing_rows = out[missing_cols].isna().any(axis=1).sum()
    if n_missing_rows:
        print(f"[warn] {n_missing_rows} rows contain NaN in predictor columns.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"Feature predictors added: {len(feat_names)}")
    print(f"Global/density predictors added: 6")
    print(f"Total rows: {len(out)}")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()

