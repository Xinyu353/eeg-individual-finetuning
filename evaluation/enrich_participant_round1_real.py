#!/usr/bin/env python3
"""
Round-1 enrichment: add participant-level explanatory variables from REAL data only.

Reads the existing summary table (e.g. pca_overlap_categorized.csv) and adds:
  - distance_to_centroid
  - pca2d_distance_to_centroid
  - ERP_deviation
  - PSD_deviation

Does NOT recompute overlap, does not use fake data.

Definitions (documented for downstream analysis):
  distance_to_centroid
    Pool all trials from feature_extraction/real_ot/p{pid}_real.csv (16 feature columns).
    Fit StandardScaler on the pooled matrix. Let mu_global be the mean vector of all
    pooled standardized trials. For each participant, mu_p = mean of their trials in
    the same standardized space. distance_to_centroid = ||mu_p - mu_global||_2.

  pca2d_distance_to_centroid
    Fit PCA(n_components=2) on the same pooled standardized 16-D real feature matrix.
    Compute the Euclidean distance between each participant centroid and the global
    centroid in this 2-D PCA space. This matches the PCA visualization space, but is
    a lossy summary compared with distance_to_centroid.

  ERP_deviation
    Load raw OT trials from ot_vs_c_Pz_4753.csv (same layout as finetune_ot_participant).
    Per-trial baseline correction: subtract mean over time points where time in [-398,0] ms
    (matching training code: baseline_start=-0.398s, baseline_end=0s, fs=256, 409 samples).
    erp_global = mean waveform over ALL valid trials (all participants).
    erp_p = mean waveform over participant p's valid trials.
    ERP_deviation = RMSE(erp_p - erp_global) = sqrt(mean_t (erp_p(t)-erp_global(t))^2).

  PSD_deviation
    Welch PSD on erp_p and erp_global (same fs, nperseg=min(256, n_time)).
    PSD_deviation = sqrt(mean_f (log10(P_p(f)+eps) - log10(P_g(f)+eps))^2)  (log-domain RMSE).

Usage:
  python3 evaluation/enrich_participant_round1_real.py \\
    --input-csv results/pca_overlap_summary/pca_overlap_categorized.csv \\
    --output-csv results/pca_overlap_summary/pca_overlap_categorized_enriched.csv
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = PROJECT_ROOT / "results" / "pca_overlap_summary" / "pca_overlap_categorized.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "pca_overlap_summary" / "pca_overlap_categorized_enriched.csv"
DEFAULT_REAL_OT_DIR = PROJECT_ROOT / "feature_extraction" / "real_ot"
DEFAULT_RAW_CSV = PROJECT_ROOT / "real_EEG_pretrained_model" / "ot_vs_c_Pz_4753.csv"

EEG_LENGTH = 409
FS = 256.0
TIME_START_MS = -398.0
TIME_END_MS = 1195.0
BASELINE_START_S = -0.398
BASELINE_END_S = 0.0


def resolve_participant_column(df: pd.DataFrame) -> str:
    for c in ("participant_id", "participant", "Participant", "pid"):
        if c in df.columns:
            return c
    raise ValueError(
        "Input table must contain a participant id column: "
        "participant_id / participant / Participant / pid"
    )


def baseline_correct_trials(X: np.ndarray) -> np.ndarray:
    """X: (n_trials, n_time), same baseline window as finetune_ot_participant.load_real_data."""
    time_ms = np.linspace(TIME_START_MS, TIME_END_MS, X.shape[1])
    time_s = time_ms / 1000.0
    baseline_window = (time_s >= BASELINE_START_S) & (time_s <= BASELINE_END_S)
    b = X[:, baseline_window].mean(axis=1, keepdims=True)
    return X - b


def load_raw_ot_matrix(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X_bc: (n_trials, 409) baseline-corrected
      pid: (n_trials,) int participant id from first column
    """
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {csv_path}")
    pid = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    if X.shape[1] != EEG_LENGTH:
        raise ValueError(f"Expected {EEG_LENGTH} time columns, got {X.shape[1]}")
    valid = np.isfinite(X).all(axis=1) & np.isfinite(pid)
    X = X[valid]
    pid = pid[valid].astype(int)
    X_bc = baseline_correct_trials(X)
    return X_bc, pid


def compute_erp_psd_deviations(
    X_bc: np.ndarray, pid: np.ndarray, participant_ids: List[int]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    erp_global = X_bc.mean(axis=0)
    freqs_g, psd_g = welch(erp_global, fs=FS, nperseg=min(256, erp_global.size))
    psd_g = np.maximum(psd_g, 1e-20)

    erp_dev: Dict[int, float] = {}
    psd_dev: Dict[int, float] = {}

    for p in participant_ids:
        mask = pid == p
        if not np.any(mask):
            erp_dev[p] = float("nan")
            psd_dev[p] = float("nan")
            continue
        erp_p = X_bc[mask].mean(axis=0)
        erp_dev[p] = float(np.sqrt(np.mean((erp_p - erp_global) ** 2)))

        _, psd_p = welch(erp_p, fs=FS, nperseg=min(256, erp_p.size))
        m = min(psd_p.size, psd_g.size)
        lp = np.log10(np.maximum(psd_p[:m], 1e-20))
        lg = np.log10(np.maximum(psd_g[:m], 1e-20))
        psd_dev[p] = float(np.sqrt(np.mean((lp - lg) ** 2)))

    return erp_dev, psd_dev


def load_feature_matrix_for_participants(
    real_ot_dir: Path, participant_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Stack per-trial features for listed participants.
    Returns X (n_rows, n_features), pid_per_row (n_rows,), feature_names
    """
    feature_names: List[str] | None = None
    rows: List[np.ndarray] = []
    pids: List[int] = []

    for p in sorted(set(participant_ids)):
        path = real_ot_dir / f"p{p}_real.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing real feature file: {path}")
        df = pd.read_csv(path)
        if "SampleID" in df.columns:
            feat_df = df.drop(columns=["SampleID"])
        else:
            feat_df = df
        if feature_names is None:
            feature_names = list(feat_df.columns)
        else:
            if list(feat_df.columns) != feature_names:
                raise ValueError(f"Feature column mismatch in {path}")
        Xp = feat_df.to_numpy(dtype=np.float64)
        if not np.isfinite(Xp).all():
            Xp = Xp[np.isfinite(Xp).all(axis=1)]
        rows.append(Xp)
        pids.extend([p] * Xp.shape[0])

    X = np.vstack(rows)
    pid_row = np.asarray(pids, dtype=int)
    assert feature_names is not None
    return X, pid_row, feature_names


def compute_distance_to_centroid(
    X: np.ndarray, pid_row: np.ndarray, participant_ids: List[int]
) -> Dict[int, float]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    mu_global = Xs.mean(axis=0)

    out: Dict[int, float] = {}
    for p in participant_ids:
        m = pid_row == p
        if not np.any(m):
            out[p] = float("nan")
            continue
        mu_p = Xs[m].mean(axis=0)
        out[p] = float(np.linalg.norm(mu_p - mu_global))
    return out


def compute_pca2d_distance_to_centroid(
    X: np.ndarray, pid_row: np.ndarray, participant_ids: List[int]
) -> Dict[int, float]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)
    mu_global = X2.mean(axis=0)

    out: Dict[int, float] = {}
    for p in participant_ids:
        m = pid_row == p
        if not np.any(m):
            out[p] = float("nan")
            continue
        mu_p = X2[m].mean(axis=0)
        out[p] = float(np.linalg.norm(mu_p - mu_global))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich participant summary with real-data explanatory variables.")
    ap.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--real-ot-dir", type=str, default=str(DEFAULT_REAL_OT_DIR))
    ap.add_argument("--raw-csv", type=str, default=str(DEFAULT_RAW_CSV))
    args = ap.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    real_ot_dir = Path(args.real_ot_dir)
    raw_csv = Path(args.raw_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw OT CSV not found: {raw_csv}")
    if not real_ot_dir.is_dir():
        raise FileNotFoundError(f"real_ot directory not found: {real_ot_dir}")

    df = pd.read_csv(input_path)
    pid_col = resolve_participant_column(df)
    participant_ids = sorted(int(x) for x in df[pid_col].unique())

    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Participant id column: {pid_col!r}, n_unique={len(participant_ids)}")

    # --- Feature-space distance ---
    X_feat, pid_row, feat_names = load_feature_matrix_for_participants(real_ot_dir, participant_ids)
    print(f"Pooled real_ot features: shape={X_feat.shape}, n_features={len(feat_names)}")
    dist_map = compute_distance_to_centroid(X_feat, pid_row, participant_ids)
    pca2d_dist_map = compute_pca2d_distance_to_centroid(X_feat, pid_row, participant_ids)

    # --- ERP / PSD deviations from raw trials ---
    X_bc, pid_raw = load_raw_ot_matrix(raw_csv)
    print(f"Raw OT trials (baseline-corrected): {X_bc.shape[0]} trials, fs={FS} Hz")
    erp_map, psd_map = compute_erp_psd_deviations(X_bc, pid_raw, participant_ids)

    df = df.copy()
    df["distance_to_centroid"] = df[pid_col].map(dist_map)
    df["pca2d_distance_to_centroid"] = df[pid_col].map(pca2d_dist_map)
    df["ERP_deviation"] = df[pid_col].map(erp_map)
    df["PSD_deviation"] = df[pid_col].map(psd_map)

    missing = df[
        ["distance_to_centroid", "pca2d_distance_to_centroid", "ERP_deviation", "PSD_deviation"]
    ].isna().any(axis=1).sum()
    if missing:
        print(f"[warn] {missing} rows have NaN in new columns (missing participant in source data?)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote enriched table: {output_path}")


if __name__ == "__main__":
    main()
