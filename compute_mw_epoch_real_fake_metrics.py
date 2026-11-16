#!/usr/bin/env python3
"""
Compute MW real-vs-fake metrics for every generated epoch.

This helps manually choose the best epoch per participant after MW fine-tuning.
For each participant_*_full_* folder, the script pairs:

  real: mw_pipeline/participant_data/participant_{id}_real_data.csv
  fake: generated_data/generated_trials_epoch_*.csv

and computes waveform-level metrics for each epoch.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FINETUNED_DIR = PROJECT_ROOT / "mw_pipeline" / "finetuned_mw_participant"
DEFAULT_PARTICIPANT_DATA_DIR = PROJECT_ROOT / "mw_pipeline" / "participant_data"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "mw_pipeline" / "evaluation" / "mw_epoch_real_fake_metrics.csv"

FS = 256
TIME_MS = np.linspace(-398, 1195, 409)
TIME_S = TIME_MS / 1000.0
BASELINE_MASK = (TIME_S >= -0.398) & (TIME_S <= 0.0)


def extract_participant_id(folder_name: str) -> int | None:
    match = re.search(r"participant_(\d+)_", folder_name)
    return int(match.group(1)) if match else None


def extract_epoch(filename: str) -> int | None:
    match = re.search(r"generated_trials_epoch_(\d+)\.csv$", filename)
    return int(match.group(1)) if match else None


def read_trials(path: Path) -> np.ndarray:
    x = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D trials x time matrix: {path}")
    if x.shape[1] != 409:
        raise ValueError(f"Expected 409 timepoints, got {x.shape[1]}: {path}")
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) == 0:
        raise ValueError(f"No finite trials in: {path}")
    return x


def baseline_correct(x: np.ndarray) -> np.ndarray:
    return x - x[:, BASELINE_MASK].mean(axis=1, keepdims=True)


def match_counts(x: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(x), size=n, replace=False)
    iy = rng.choice(len(y), size=n, replace=False)
    return x[ix], y[iy]


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_projections: int = 256, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    directions = rng.normal(size=(n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    vals = []
    for direction in directions:
        vals.append(wasserstein_distance(x @ direction, y @ direction))
    return float(np.mean(vals))


def mmd_rbf(x: np.ndarray, y: np.ndarray, max_samples: int = 500, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y), max_samples)
    x = x[rng.choice(len(x), size=n, replace=False)]
    y = y[rng.choice(len(y), size=n, replace=False)]
    z = np.vstack([x, y])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    tri = d2[np.triu_indices_from(d2, k=1)]
    sigma2 = float(np.median(tri[tri > 0])) if np.any(tri > 0) else 1.0
    sigma2 = max(sigma2, 1e-12)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dist2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-dist2 / (2.0 * sigma2))

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def wd_timepoint_mean(x: np.ndarray, y: np.ndarray, max_samples: int = 500, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y), max_samples)
    x = x[rng.choice(len(x), size=n, replace=False)]
    y = y[rng.choice(len(y), size=n, replace=False)]
    return float(np.mean([wasserstein_distance(x[:, i], y[:, i]) for i in range(x.shape[1])]))


def erp_metrics(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    real_bc = baseline_correct(real)
    fake_bc = baseline_correct(fake)
    erp_real = real_bc.mean(axis=0)
    erp_fake = fake_bc.mean(axis=0)
    diff = erp_fake - erp_real
    corr = np.corrcoef(erp_real, erp_fake)[0, 1]
    return {
        "ERP_RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "ERP_MAE": float(np.mean(np.abs(diff))),
        "ERP_corr": float(corr),
    }


def psd_metrics(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    real_bc = baseline_correct(real)
    fake_bc = baseline_correct(fake)
    _, psd_real = welch(real_bc, fs=FS, axis=1, nperseg=min(256, real_bc.shape[1]))
    _, psd_fake = welch(fake_bc, fs=FS, axis=1, nperseg=min(256, fake_bc.shape[1]))
    log_real = 10.0 * np.log10(np.maximum(psd_real, 1e-12))
    log_fake = 10.0 * np.log10(np.maximum(psd_fake, 1e-12))
    mean_real = log_real.mean(axis=0)
    mean_fake = log_fake.mean(axis=0)
    diff = mean_fake - mean_real
    corr = np.corrcoef(mean_real, mean_fake)[0, 1]
    return {
        "PSD_log_RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "PSD_log_MAE": float(np.mean(np.abs(diff))),
        "PSD_log_corr": float(corr),
    }


def compute_epoch_metrics(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    real_matched, fake_matched = match_counts(real, fake)
    out: Dict[str, float] = {
        "n_real_trials": int(len(real)),
        "n_fake_trials": int(len(fake)),
        "n_matched_trials": int(len(real_matched)),
    }
    out.update(erp_metrics(real_matched, fake_matched))
    out.update(psd_metrics(real_matched, fake_matched))

    # Distance metrics use baseline-corrected trials to match the ERP selection plots.
    real_bc = baseline_correct(real_matched)
    fake_bc = baseline_correct(fake_matched)
    out["SWD"] = sliced_wasserstein(real_bc, fake_bc)
    out["MMD_RBF"] = mmd_rbf(real_bc, fake_bc)
    out["WD_timepoint_mean"] = wd_timepoint_mean(real_bc, fake_bc)
    return out


def add_within_participant_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lower_better = ["ERP_RMSE", "ERP_MAE", "PSD_log_RMSE", "PSD_log_MAE", "SWD", "MMD_RBF", "WD_timepoint_mean"]
    higher_better = ["ERP_corr", "PSD_log_corr"]

    rank_cols = []
    for col in lower_better:
        rank_col = f"{col}_rank"
        out[rank_col] = out.groupby("participant_id")[col].rank(method="min", ascending=True)
        rank_cols.append(rank_col)
    for col in higher_better:
        rank_col = f"{col}_rank"
        out[rank_col] = out.groupby("participant_id")[col].rank(method="min", ascending=False)
        rank_cols.append(rank_col)

    out["mean_metric_rank"] = out[rank_cols].mean(axis=1)
    out["suggested_best_by_mean_rank"] = False
    best_idx = out.groupby("participant_id")["mean_metric_rank"].idxmin()
    out.loc[best_idx, "suggested_best_by_mean_rank"] = True
    return out.sort_values(["participant_id", "epoch"]).reset_index(drop=True)


def iter_model_dirs(args: argparse.Namespace) -> List[Path]:
    if args.model_dir:
        return [Path(args.model_dir).resolve()]
    finetuned_dir = Path(args.finetuned_dir).resolve()
    return sorted(p for p in finetuned_dir.glob("participant_*_full_*") if p.is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MW real-vs-fake metrics for each generated epoch.")
    parser.add_argument("--model-dir", type=str, default=None, help="Process one participant_*_full_* folder.")
    parser.add_argument("--finetuned-dir", type=str, default=str(DEFAULT_FINETUNED_DIR))
    parser.add_argument("--participant-data-dir", type=str, default=str(DEFAULT_PARTICIPANT_DATA_DIR))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    participant_data_dir = Path(args.participant_data_dir).resolve()
    output_csv = Path(args.output_csv).resolve()

    rows = []
    for model_dir in iter_model_dirs(args):
        pid = extract_participant_id(model_dir.name)
        if pid is None:
            print(f"[skip] Cannot extract participant id: {model_dir}")
            continue
        real_path = participant_data_dir / f"participant_{pid}_real_data.csv"
        generated_dir = model_dir / "generated_data"
        if not real_path.exists():
            print(f"[skip] Missing real data for P{pid}: {real_path}")
            continue
        if not generated_dir.is_dir():
            print(f"[skip] Missing generated_data for P{pid}: {generated_dir}")
            continue

        real = read_trials(real_path)
        fake_files = sorted(
            generated_dir.glob("generated_trials_epoch_*.csv"),
            key=lambda p: extract_epoch(p.name) or -1,
        )
        print(f"P{pid}: real={real.shape}, epochs={len(fake_files)}")
        for fake_path in fake_files:
            epoch = extract_epoch(fake_path.name)
            if epoch is None:
                continue
            fake = read_trials(fake_path)
            metrics = compute_epoch_metrics(real, fake)
            rows.append(
                {
                    "participant_id": pid,
                    "epoch": epoch,
                    "model_dir": str(model_dir),
                    "real_file": str(real_path),
                    "fake_file": str(fake_path),
                    **metrics,
                }
            )
            print(
                f"  epoch {epoch}: ERP_RMSE={metrics['ERP_RMSE']:.3f}, "
                f"PSD_log_RMSE={metrics['PSD_log_RMSE']:.3f}, SWD={metrics['SWD']:.3f}"
            )

    if not rows:
        raise RuntimeError("No epoch metrics computed.")

    out = add_within_participant_ranks(pd.DataFrame(rows))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    best = out[out["suggested_best_by_mean_rank"]].sort_values("participant_id")
    best_csv = output_csv.with_name(output_csv.stem + "_suggested_best.csv")
    best.to_csv(best_csv, index=False)

    print("\nSaved epoch metrics:")
    print(f"  {output_csv}")
    print("Saved suggested best epochs by mean metric rank:")
    print(f"  {best_csv}")


if __name__ == "__main__":
    main()
