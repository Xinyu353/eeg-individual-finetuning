#!/usr/bin/env python3
"""
Held-out 80/20 participant-wise EEG fine-tuning and evaluation pipeline.

This script is intentionally separate from the existing OT/MW workflows so that
current results are not overwritten. It uses calibration trials for participant
fine-tuning, checkpoint selection, and predictor calculation, then evaluates the
frozen selected checkpoint only on held-out trials.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl_tmp"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_DIR = PROJECT_ROOT / "feature_extraction"
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

EVAL_DIR = PROJECT_ROOT / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from finetune_ot_participant import (  # noqa: E402
    CONFIG as BASE_FINETUNE_CONFIG,
    FullFineTuner,
    compute_num_fake_samples,
    load_pretrained_models,
)
from lora import apply_lora_to_model
from feature_extraction import compute_Wst, extract_all_features, extract_erp_features  # noqa: E402
from distribution_2d_analysis import (  # noqa: E402
    kde_bhattacharyya_2d,
    kde_js_2d,
    kde_overlap_2d,
)


RANDOM_SEED = 42
TEST_SIZE = 0.20
MIN_TOTAL_TRIALS = 100
EEG_LENGTH = 409
FS = 256.0
TIME_START_MS = -398.0
TIME_END_MS = 1195.0
BASELINE_START_S = -0.398
BASELINE_END_S = 0.0

CONDITION_CSV = {
    "ot": PROJECT_ROOT / "real_EEG_pretrained_model" / "ot_vs_c_Pz_4753.csv",
    "mw": PROJECT_ROOT / "real_EEG_pretrained_model" / "mw_vs_Pz_1789.csv",
}
CONDITION_MODEL = {
    "ot": PROJECT_ROOT / "real_EEG_pretrained_model" / "base_OT.pt",
    "mw": PROJECT_ROOT / "real_EEG_pretrained_model" / "base_MW.pt",
}

FEATURE_COLUMNS = [
    "P3_Amplitude",
    "P3_Latency",
    "Activity",
    "Mobility",
    "Complexity",
    "Petrosian_Fractal_Dimension",
    "Delta_PSD",
    "Theta_PSD",
    "Alpha_PSD",
    "Beta_PSD",
    "Gamma_PSD",
    "Delta_Entropy",
    "Theta_Entropy",
    "Alpha_Entropy",
    "Beta_Entropy",
    "Gamma_Entropy",
]

GLOBAL_PREDICTORS = [
    "distance_to_centroid",
    "pca2d_distance_to_centroid",
    "ERP_deviation",
    "PSD_deviation",
    "mahalanobis_distance",
    "centroid_local_density",
]


@dataclass
class ParticipantSplit:
    condition: str
    participant: int
    total_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    calibration_trials_bc: np.ndarray
    test_trials_bc: np.ndarray
    real_min: Optional[float] = None
    real_max: Optional[float] = None


@dataclass
class ParticipantRunResult:
    condition: str
    participant: int
    n_total: int
    n_calibration: int
    n_test: int
    save_dir: Path
    selected_epoch: int
    selected_checkpoint: Path
    calibration_pca_overlap: float
    heldout_pca_overlap: float


def baseline_mask(n_time: int = EEG_LENGTH) -> np.ndarray:
    time_ms = np.linspace(TIME_START_MS, TIME_END_MS, n_time)
    time_s = time_ms / 1000.0
    return (time_s >= BASELINE_START_S) & (time_s <= BASELINE_END_S)


BASELINE_MASK = baseline_mask()


def baseline_correct_trials(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D trial matrix, got shape={x.shape}")
    if x.shape[1] != EEG_LENGTH:
        raise ValueError(f"Expected {EEG_LENGTH} time points, got {x.shape[1]}")
    return x - x[:, BASELINE_MASK].mean(axis=1, keepdims=True)


def json_indices(indices: Sequence[int]) -> str:
    return json.dumps([int(i) for i in indices], separators=(",", ":"))


def parse_participant_filters(items: Sequence[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Participant filter must look like ot:1,2 or mw:4, got: {item}")
        condition, raw = item.split(":", 1)
        condition = condition.lower().strip()
        if condition not in CONDITION_CSV:
            raise ValueError(f"Unsupported condition in participant filter: {condition}")
        ids = [int(x) for x in raw.split(",") if x.strip()]
        out[condition] = ids
    return out


def build_config(condition: str, output_dir: Path, args: argparse.Namespace) -> dict:
    cfg = BASE_FINETUNE_CONFIG.copy()
    cfg["condition"] = condition
    cfg["model_path"] = str(CONDITION_MODEL[condition])
    cfg["csv_path"] = str(CONDITION_CSV[condition])
    cfg["output_dir"] = str(output_dir / "finetuned" / condition)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.save_interval is not None:
        cfg["save_interval"] = int(args.save_interval)
    lr_g = getattr(args, "lr_g", None)
    lr_d = getattr(args, "lr_d", None)
    if lr_g is not None:
        cfg["lr_g"] = float(lr_g)
    if lr_d is not None:
        cfg["lr_d"] = float(lr_d)
    return cfg


def load_condition_raw_trials(condition: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return valid baseline-corrected trials by participant and original CSV row indices."""
    csv_path = CONDITION_CSV[condition]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing raw CSV for {condition.upper()}: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    rows: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for pid_raw, sub in df.groupby(df.columns[0], sort=True):
        pid = int(pid_raw)
        original_idx = sub.index.to_numpy(dtype=int)
        x = sub.iloc[:, 1:].to_numpy(dtype=np.float64)
        if x.shape[1] != EEG_LENGTH:
            raise ValueError(f"{condition.upper()} P{pid}: expected {EEG_LENGTH} time points, got {x.shape[1]}")
        valid = np.isfinite(x).all(axis=1)
        x = x[valid]
        idx = original_idx[valid]
        if len(x):
            rows[pid] = (baseline_correct_trials(x), idx)
    return rows


def make_participant_splits(
    raw_by_condition: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    conditions: Sequence[str],
    participant_filters: Dict[str, List[int]],
) -> List[ParticipantSplit]:
    splits: List[ParticipantSplit] = []
    for condition in conditions:
        available = raw_by_condition[condition]
        if participant_filters:
            candidate_ids = participant_filters.get(condition, [])
        else:
            candidate_ids = sorted(available)

        for pid in candidate_ids:
            if pid not in available:
                print(f"[skip] {condition.upper()} P{pid}: not present in raw data")
                continue
            trials_bc, original_idx = available[pid]
            n_total = len(trials_bc)
            if n_total < MIN_TOTAL_TRIALS:
                print(f"[skip] {condition.upper()} P{pid}: total valid trials={n_total} < {MIN_TOTAL_TRIALS}")
                continue

            calibration_pos, test_pos = train_test_split(
                np.arange(n_total),
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED,
                shuffle=True,
            )
            calibration_indices = original_idx[np.asarray(calibration_pos, dtype=int)]
            test_indices = original_idx[np.asarray(test_pos, dtype=int)]
            assert set(calibration_indices).isdisjoint(set(test_indices)), (
                f"Split leakage: {condition.upper()} P{pid} calibration/test overlap"
            )

            splits.append(
                ParticipantSplit(
                    condition=condition,
                    participant=pid,
                    total_indices=original_idx,
                    calibration_indices=calibration_indices,
                    test_indices=test_indices,
                    calibration_trials_bc=trials_bc[calibration_pos],
                    test_trials_bc=trials_bc[test_pos],
                )
            )
    return splits


def write_split_manifest(splits: Sequence[ParticipantSplit], output_dir: Path) -> None:
    rows = []
    for sp in splits:
        rows.append(
            {
                "condition": sp.condition,
                "participant": sp.participant,
                "n_total": int(len(sp.total_indices)),
                "n_calibration": int(len(sp.calibration_indices)),
                "n_test": int(len(sp.test_indices)),
                "calibration_indices": json_indices(sp.calibration_indices),
                "test_indices": json_indices(sp.test_indices),
                "random_seed": RANDOM_SEED,
            }
        )
    out = output_dir / "split_manifest.csv"
    pd.DataFrame(rows).sort_values(["condition", "participant"]).to_csv(out, index=False)
    print(f"Saved split manifest: {out}")


def write_trial_matrix(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(x).to_csv(path, index=False, header=False)


def normalize_for_training(x_bc: np.ndarray, real_min: float, real_max: float, device: str) -> torch.Tensor:
    if real_max <= real_min:
        raise ValueError(f"Invalid normalization range: [{real_min}, {real_max}]")
    real = torch.tensor(x_bc, dtype=torch.float32, device=device)
    real = (real - real_min) / (real_max - real_min) * 2.0 - 1.0
    return real


def prepare_models_for_participant(device: str, config: dict):
    g, d, rmin, rmax = load_pretrained_models(device, config["model_path"], config)
    mode = config.get("finetune_mode", "full").lower()
    if mode == "lora" and config.get("use_lora", False):
        g = apply_lora_to_model(
            g,
            r=config["lora_r"],
            alpha=config["lora_alpha"],
            dropout=config["lora_dropout"],
            which=config["lora_targets"],
        )
    return g.to(device), d.to(device), rmin, rmax


def fine_tune_on_calibration(
    sp: ParticipantSplit,
    device: str,
    config: dict,
    output_dir: Path,
) -> Path:
    participant_config = config.copy()
    participant_config["participant_id"] = sp.participant

    g, d, rmin, rmax = prepare_models_for_participant(device, config)
    if rmin is None or rmax is None:
        # Avoid held-out leakage: fall back to calibration-only range.
        rmin = float(sp.calibration_trials_bc.min())
        rmax = float(sp.calibration_trials_bc.max())
        print(
            f"[warn] {sp.condition.upper()} P{sp.participant}: checkpoint lacks data range; "
            f"using calibration-only range [{rmin:.3f}, {rmax:.3f}]"
        )
    sp.real_min = float(rmin)
    sp.real_max = float(rmax)

    real = normalize_for_training(sp.calibration_trials_bc, sp.real_min, sp.real_max, device)
    num_fake_samples = compute_num_fake_samples(len(real))
    assert num_fake_samples == len(sp.calibration_trials_bc)

    output_dir.joinpath("finetuned", sp.condition).mkdir(parents=True, exist_ok=True)
    tuner = FullFineTuner(g, d, device, participant_config)
    save_dir = Path(
        tuner.train(
            real_data=real,
            data_min=sp.real_min,
            data_max=sp.real_max,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            save_interval=config["save_interval"],
            num_fake_samples=num_fake_samples,
        )
    )
    return save_dir


def extract_epoch(path: Path) -> Optional[int]:
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else None


def read_trials(path: Path) -> np.ndarray:
    x = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != EEG_LENGTH:
        raise ValueError(f"Expected trials x {EEG_LENGTH}, got {x.shape}: {path}")
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) == 0:
        raise ValueError(f"No valid trials in {path}")
    return x


def match_counts(x: np.ndarray, y: np.ndarray, seed: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    rng = np.random.default_rng(seed)
    return (
        x[rng.choice(len(x), size=n, replace=False)],
        y[rng.choice(len(y), size=n, replace=False)],
    )


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_projections: int = 256, seed: int = RANDOM_SEED) -> float:
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    directions = rng.normal(size=(n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    vals = [wasserstein_distance(x @ direction, y @ direction) for direction in directions]
    return float(np.mean(vals))


def pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    return a2 + b2 - 2.0 * (a @ b.T)


def mmd_rbf_checkpoint(x: np.ndarray, y: np.ndarray, max_samples: int = 500, seed: int = RANDOM_SEED) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y), max_samples)
    x = x[rng.choice(len(x), size=n, replace=False)]
    y = y[rng.choice(len(y), size=n, replace=False)]
    z = np.vstack([x, y])
    d2 = pairwise_sq_dists(z, z)
    tri = d2[np.triu_indices_from(d2, k=1)]
    sigma2 = float(np.median(tri[tri > 0])) if np.any(tri > 0) else 1.0
    sigma2 = max(sigma2, 1e-12)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-pairwise_sq_dists(a, b) / (2.0 * sigma2))

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())


def mmd_rbf_final(x: np.ndarray, y: np.ndarray) -> float:
    n, m = x.shape[0], y.shape[0]
    z = np.vstack([x, y])
    if z.shape[0] > 5000:
        idx = np.random.default_rng(RANDOM_SEED).choice(z.shape[0], size=5000, replace=False)
        zs = z[idx]
    else:
        zs = z
    d2 = pairwise_sq_dists(zs, zs)
    iu = np.triu_indices_from(d2, k=1)
    med2 = np.median(d2[iu])
    gamma = 1.0 / (2.0 * max(float(med2), 1e-12))
    kxx = np.exp(-gamma * pairwise_sq_dists(x, x))
    kyy = np.exp(-gamma * pairwise_sq_dists(y, y))
    kxy = np.exp(-gamma * pairwise_sq_dists(x, y))
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    mmd2 = kxx.sum() / (n * (n - 1) + 1e-12) + kyy.sum() / (m * (m - 1) + 1e-12) - 2.0 * kxy.mean()
    return float(np.sqrt(max(mmd2, 0.0)))


def wd_timepoint_mean(x: np.ndarray, y: np.ndarray, max_samples: int = 500, seed: int = RANDOM_SEED) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y), max_samples)
    x = x[rng.choice(len(x), size=n, replace=False)]
    y = y[rng.choice(len(y), size=n, replace=False)]
    return float(np.mean([wasserstein_distance(x[:, i], y[:, i]) for i in range(x.shape[1])]))


def erp_metrics(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    real_bc = baseline_correct_trials(real)
    fake_bc = baseline_correct_trials(fake)
    erp_real = real_bc.mean(axis=0)
    erp_fake = fake_bc.mean(axis=0)
    diff = erp_fake - erp_real
    corr = np.corrcoef(erp_real, erp_fake)[0, 1]
    return {
        "ERP_RMSE": float(np.sqrt(np.mean(diff**2))),
        "ERP_MAE": float(np.mean(np.abs(diff))),
        "ERP_corr": float(corr),
        "ERP_ICC": icc_2_1(erp_real, erp_fake),
    }


def psd_metrics(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    real_bc = baseline_correct_trials(real)
    fake_bc = baseline_correct_trials(fake)
    _, psd_real = welch(real_bc, fs=FS, axis=1, nperseg=min(256, real_bc.shape[1]))
    _, psd_fake = welch(fake_bc, fs=FS, axis=1, nperseg=min(256, fake_bc.shape[1]))
    log_real = 10.0 * np.log10(np.maximum(psd_real, 1e-12))
    log_fake = 10.0 * np.log10(np.maximum(psd_fake, 1e-12))
    mean_real = log_real.mean(axis=0)
    mean_fake = log_fake.mean(axis=0)
    diff = mean_fake - mean_real
    corr = np.corrcoef(mean_real, mean_fake)[0, 1]
    return {
        "PSD_log_RMSE": float(np.sqrt(np.mean(diff**2))),
        "PSD_log_MAE": float(np.mean(np.abs(diff))),
        "PSD_log_corr": float(corr),
        "PSD_log_ICC": icc_2_1(mean_real, mean_fake),
    }


def icc_2_1(a: np.ndarray, b: np.ndarray) -> float:
    """Two-way random single-measure ICC for paired waveforms."""
    ratings = np.column_stack([a, b]).astype(np.float64)
    n, k = ratings.shape
    if n < 2 or k != 2:
        return float("nan")
    mean_targets = ratings.mean(axis=1, keepdims=True)
    mean_raters = ratings.mean(axis=0, keepdims=True)
    grand = ratings.mean()
    ss_target = k * np.sum((mean_targets - grand) ** 2)
    ss_rater = n * np.sum((mean_raters - grand) ** 2)
    ss_error = np.sum((ratings - mean_targets - mean_raters + grand) ** 2)
    ms_target = ss_target / max(n - 1, 1)
    ms_rater = ss_rater / max(k - 1, 1)
    ms_error = ss_error / max((n - 1) * (k - 1), 1)
    denom = ms_target + (k - 1) * ms_error + (k * (ms_rater - ms_error) / max(n, 1))
    if abs(denom) < 1e-12:
        return float("nan")
    return float((ms_target - ms_error) / denom)


def compute_checkpoint_metrics(real_calibration_bc: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    real_matched, fake_matched = match_counts(real_calibration_bc, fake)
    out: Dict[str, float] = {
        "n_real_trials": int(len(real_calibration_bc)),
        "n_fake_trials": int(len(fake)),
        "n_matched_trials": int(len(real_matched)),
    }
    out.update(erp_metrics(real_matched, fake_matched))
    out.update(psd_metrics(real_matched, fake_matched))
    real_bc = baseline_correct_trials(real_matched)
    fake_bc = baseline_correct_trials(fake_matched)
    out["SWD"] = sliced_wasserstein(real_bc, fake_bc)
    out["MMD_RBF"] = mmd_rbf_checkpoint(real_bc, fake_bc)
    out["WD_timepoint_mean"] = wd_timepoint_mean(real_bc, fake_bc)
    return out


def add_checkpoint_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lower_better = [
        "ERP_RMSE",
        "ERP_MAE",
        "PSD_log_RMSE",
        "PSD_log_MAE",
        "SWD",
        "MMD_RBF",
        "WD_timepoint_mean",
    ]
    higher_better = ["ERP_ICC", "PSD_log_ICC"]
    rank_cols: List[str] = []
    for col in lower_better:
        rank_col = f"{col}_rank"
        out[rank_col] = out.groupby(["condition", "participant"])[col].rank(method="min", ascending=True)
        rank_cols.append(rank_col)
    for col in higher_better:
        rank_col = f"{col}_rank"
        out[rank_col] = out.groupby(["condition", "participant"])[col].rank(method="min", ascending=False)
        rank_cols.append(rank_col)
    out["average_rank"] = out[rank_cols].mean(axis=1)
    out["selected_checkpoint"] = False
    best_idx = out.groupby(["condition", "participant"])["average_rank"].idxmin()
    out.loc[best_idx, "selected_checkpoint"] = True
    return out.sort_values(["condition", "participant", "epoch"]).reset_index(drop=True)


def evaluate_checkpoints(sp: ParticipantSplit, save_dir: Path) -> pd.DataFrame:
    generated_dir = save_dir / "generated_data"
    fake_files = sorted(generated_dir.glob("generated_trials_epoch_*.csv"), key=lambda p: extract_epoch(p) or -1)
    if not fake_files:
        raise FileNotFoundError(f"No generated checkpoint CSVs under {generated_dir}")

    rows = []
    for fake_path in fake_files:
        epoch = extract_epoch(fake_path)
        if epoch is None:
            continue
        fake = read_trials(fake_path)
        assert len(fake) == len(sp.calibration_trials_bc), (
            f"{sp.condition.upper()} P{sp.participant} epoch {epoch}: fake count does not match calibration count"
        )
        metrics = compute_checkpoint_metrics(sp.calibration_trials_bc, fake)
        rows.append(
            {
                "condition": sp.condition,
                "participant": sp.participant,
                "epoch": int(epoch),
                "model_dir": str(save_dir),
                "real_subset": "calibration",
                "fake_file": str(fake_path),
                **metrics,
            }
        )
    return add_checkpoint_ranks(pd.DataFrame(rows))


def load_tuner_from_checkpoint(
    checkpoint_path: Path,
    device: str,
    config: dict,
    real_min: float,
    real_max: float,
    participant: int,
) -> FullFineTuner:
    participant_config = config.copy()
    participant_config["participant_id"] = participant
    g, d, _, _ = prepare_models_for_participant(device, config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    g.load_state_dict(ckpt["generator_state_dict"])
    if "discriminator_state_dict" in ckpt:
        d.load_state_dict(ckpt["discriminator_state_dict"])
    tuner = FullFineTuner(g, d, device, participant_config)
    token = ckpt.get("subject_token")
    if token is not None and tuner.subject_token is not None:
        tuner.subject_token.data.copy_(token.to(device))
    tuner.generator.eval()
    tuner.discriminator.eval()
    # Store for debugging and to make leakage checks explicit at the call site.
    tuner.config["heldout_real_min"] = real_min
    tuner.config["heldout_real_max"] = real_max
    return tuner


def extract_features_from_trials(trials: np.ndarray, verbose_prefix: str = "") -> pd.DataFrame:
    times = np.linspace(-0.4, 1.2, trials.shape[1]) * 1000.0
    rows = []
    total = len(trials)
    for idx, signal in enumerate(trials, start=1):
        if verbose_prefix and (idx == total or idx % 50 == 0):
            print(f"{verbose_prefix} feature extraction {idx}/{total}")
        wst, tlag, scale = compute_Wst(signal, times, plot_on=False, save_path=None)
        features = extract_all_features(signal, times)
        _, _, p3_amp, _, _, p3_lat = extract_erp_features(wst, tlag, scale, times)
        rows.append(
            {
                "SampleID": idx,
                "P3_Amplitude": p3_amp,
                "P3_Latency": p3_lat,
                "Activity": features["Activity"],
                "Mobility": features["Mobility"],
                "Complexity": features["Complexity"],
                "Petrosian_Fractal_Dimension": features["Petrosian Fractal Dimension"],
                "Delta_PSD": features["Delta PSD"],
                "Theta_PSD": features["Theta PSD"],
                "Alpha_PSD": features["Alpha PSD"],
                "Beta_PSD": features["Beta PSD"],
                "Gamma_PSD": features["Gamma PSD"],
                "Delta_Entropy": features["Delta Entropy"],
                "Theta_Entropy": features["Theta Entropy"],
                "Alpha_Entropy": features["Alpha Entropy"],
                "Beta_Entropy": features["Beta Entropy"],
                "Gamma_Entropy": features["Gamma Entropy"],
            }
        )
    return pd.DataFrame(rows)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Feature table missing columns: {missing}")
    x = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return x.to_numpy(dtype=np.float64)


def compute_pca_feature_metrics(real_features: pd.DataFrame, fake_features: pd.DataFrame) -> Dict[str, float]:
    xr = feature_matrix(real_features)
    xf = feature_matrix(fake_features)
    labels = np.array(["real"] * len(xr) + ["fake"] * len(xf))
    combined = np.vstack([xr, xf])
    xs = StandardScaler().fit_transform(combined)
    x2 = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(xs)
    real_2d = x2[labels == "real"]
    fake_2d = x2[labels == "fake"]
    return {
        "PCA_overlap_RF": float(kde_overlap_2d(real_2d, fake_2d)),
        "PCA_BC_RF": float(kde_bhattacharyya_2d(real_2d, fake_2d)),
        "PCA_JS_RF": float(kde_js_2d(real_2d, fake_2d)),
    }


def compute_final_quality_metrics(real_heldout_bc: np.ndarray, fake_heldout: np.ndarray) -> Dict[str, float]:
    real_matched, fake_matched = match_counts(real_heldout_bc, fake_heldout)
    metrics: Dict[str, float] = {
        "n_real_trials": int(len(real_heldout_bc)),
        "n_fake_trials": int(len(fake_heldout)),
        "n_matched_trials": int(len(real_matched)),
    }
    metrics.update(erp_metrics(real_matched, fake_matched))
    metrics.update(psd_metrics(real_matched, fake_matched))
    real_bc = baseline_correct_trials(real_matched)
    fake_bc = baseline_correct_trials(fake_matched)
    metrics["SWD"] = sliced_wasserstein(real_bc, fake_bc)
    metrics["MMD_RBF"] = mmd_rbf_final(real_bc, fake_bc)
    metrics["WD"] = wd_timepoint_mean(real_bc, fake_bc, max_samples=1000)
    return metrics


def save_feature_tables(
    output_dir: Path,
    sp: ParticipantSplit,
    selected_epoch: int,
    calibration_fake: np.ndarray,
    heldout_fake: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_root = output_dir / "features" / sp.condition
    real_cal_path = feature_root / "real_calibration" / f"p{sp.participant}_real_calibration.csv"
    real_test_path = feature_root / "real_heldout" / f"p{sp.participant}_real_heldout.csv"
    fake_cal_path = feature_root / "fake_calibration_selected" / f"p{sp.participant}_epoch_{selected_epoch}.csv"
    fake_test_path = feature_root / "fake_heldout_selected" / f"p{sp.participant}_epoch_{selected_epoch}.csv"

    real_cal = extract_features_from_trials(sp.calibration_trials_bc, f"{sp.condition.upper()} P{sp.participant} calibration")
    real_test = extract_features_from_trials(sp.test_trials_bc, f"{sp.condition.upper()} P{sp.participant} held-out")
    fake_cal = extract_features_from_trials(calibration_fake, f"{sp.condition.upper()} P{sp.participant} fake calibration")
    fake_test = extract_features_from_trials(heldout_fake, f"{sp.condition.upper()} P{sp.participant} fake held-out")

    for path, df in (
        (real_cal_path, real_cal),
        (real_test_path, real_test),
        (fake_cal_path, fake_cal),
        (fake_test_path, fake_test),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return real_cal, real_test, fake_cal, fake_test


def categorize_overlap(value: float) -> str:
    if value >= 0.60:
        return "Good"
    if value >= 0.50:
        return "Borderline"
    return "Poor"


def compute_condition_predictors(
    condition: str,
    splits: Sequence[ParticipantSplit],
    real_feature_by_pid: Dict[int, pd.DataFrame],
    group_reference: str,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    pids = sorted(sp.participant for sp in splits if sp.condition == condition)
    output_cols = [
        "condition",
        "participant",
        "n_calibration",
        "group_reference",
        *GLOBAL_PREDICTORS,
        *[f"{feature}__abs_dev_z" for feature in FEATURE_COLUMNS],
    ]
    if group_reference == "leave-one-out" and len(pids) < 2:
        print(f"[warn] {condition.upper()}: skip predictors; leave-one-out needs at least 2 participants.")
        return pd.DataFrame(columns=output_cols)
    split_by_pid = {sp.participant: sp for sp in splits if sp.condition == condition}
    feature_by_pid = {pid: feature_matrix(real_feature_by_pid[pid]) for pid in pids}
    mean_by_pid = {pid: feature_by_pid[pid].mean(axis=0) for pid in pids}

    for pid in pids:
        if group_reference == "leave-one-out":
            ref_pids = [x for x in pids if x != pid]
        else:
            ref_pids = list(pids)
        if not ref_pids:
            print(f"[warn] {condition.upper()} P{pid}: skip predictors; no reference participants available.")
            continue

        part_feat = feature_by_pid[pid]
        group_feat = np.vstack([feature_by_pid[x] for x in ref_pids])
        scaler = StandardScaler().fit(group_feat)
        group_scaled = scaler.transform(group_feat)
        part_scaled = scaler.transform(part_feat)
        group_mu = group_scaled.mean(axis=0)
        part_mu = part_scaled.mean(axis=0)

        distance_to_centroid = float(np.linalg.norm(part_mu - group_mu))
        pca = PCA(n_components=2, random_state=RANDOM_SEED).fit(group_scaled)
        group_2d = pca.transform(group_scaled)
        part_2d = pca.transform(part_scaled)
        pca2d_distance = float(np.linalg.norm(part_2d.mean(axis=0) - group_2d.mean(axis=0)))

        lw = LedoitWolf().fit(group_scaled)
        precision = lw.precision_
        diff = part_mu - group_mu
        mahalanobis = float(np.sqrt(max(diff @ precision @ diff, 0.0)))

        ref_centroids = []
        for ref_pid in ref_pids:
            ref_centroids.append(scaler.transform(feature_by_pid[ref_pid]).mean(axis=0))
        ref_centroids_arr = np.vstack(ref_centroids)
        k = min(5, len(ref_centroids_arr))
        if k > 0:
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_centroids_arr)
            distances, _ = nn.kneighbors(part_mu.reshape(1, -1))
            centroid_density = float(1.0 / (np.mean(distances[0]) + 1e-12))
        else:
            centroid_density = float("nan")

        sp = split_by_pid[pid]
        group_raw = np.vstack([split_by_pid[x].calibration_trials_bc for x in ref_pids])
        erp_group = group_raw.mean(axis=0)
        erp_part = sp.calibration_trials_bc.mean(axis=0)
        erp_dev = float(np.sqrt(np.mean((erp_part - erp_group) ** 2)))
        _, psd_group = welch(erp_group, fs=FS, nperseg=min(256, erp_group.size))
        _, psd_part = welch(erp_part, fs=FS, nperseg=min(256, erp_part.size))
        m = min(psd_group.size, psd_part.size)
        psd_dev = float(
            np.sqrt(
                np.mean(
                    (
                        np.log10(np.maximum(psd_part[:m], 1e-20))
                        - np.log10(np.maximum(psd_group[:m], 1e-20))
                    )
                    ** 2
                )
            )
        )

        row: Dict[str, float | int | str] = {
            "condition": condition,
            "participant": pid,
            "n_calibration": int(len(sp.calibration_trials_bc)),
            "group_reference": group_reference,
            "distance_to_centroid": distance_to_centroid,
            "pca2d_distance_to_centroid": pca2d_distance,
            "ERP_deviation": erp_dev,
            "PSD_deviation": psd_dev,
            "mahalanobis_distance": mahalanobis,
            "centroid_local_density": centroid_density,
        }

        if group_reference == "leave-one-out":
            stat_pids = [x for x in pids if x != pid]
        else:
            stat_pids = list(pids)
        stat_matrix = np.vstack([mean_by_pid[x] for x in stat_pids])
        stat_mean = stat_matrix.mean(axis=0)
        stat_std = stat_matrix.std(axis=0, ddof=0)
        part_mean = mean_by_pid[pid]
        for idx, feature in enumerate(FEATURE_COLUMNS):
            denom = stat_std[idx]
            row[f"{feature}__abs_dev_z"] = float(abs(part_mean[idx] - stat_mean[idx]) / denom) if denom > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows, columns=output_cols)


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


def compute_predictor_correlations(predictors: pd.DataFrame, final_metrics: pd.DataFrame) -> pd.DataFrame:
    corr_cols = ["condition", "outcome", "predictor", "n", "rho", "raw_p", "fdr_p"]
    if predictors.empty:
        return pd.DataFrame(columns=corr_cols)
    df = predictors.merge(final_metrics, on=["condition", "participant"], how="inner", suffixes=("", "_outcome"))
    predictor_cols = [c for c in GLOBAL_PREDICTORS if c in df.columns]
    predictor_cols.extend(sorted(c for c in df.columns if c.endswith("__abs_dev_z")))
    rows = []
    for condition, cond_df in df.groupby("condition", sort=True):
        for predictor in predictor_cols:
            sub = cond_df[[predictor, "PCA_overlap_RF"]].dropna()
            if len(sub) < 3:
                rho = raw_p = np.nan
            else:
                rho, raw_p = spearmanr(sub[predictor], sub["PCA_overlap_RF"])
            rows.append(
                {
                    "condition": condition,
                    "outcome": "PCA_overlap_RF",
                    "predictor": predictor,
                    "n": int(len(sub)),
                    "rho": float(rho) if np.isfinite(rho) else np.nan,
                    "raw_p": float(raw_p) if np.isfinite(raw_p) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=corr_cols)
    out["fdr_p"] = np.nan
    for condition, idx in out.groupby("condition").groups.items():
        idx = list(idx)
        out.loc[idx, "fdr_p"] = fdr_bh(out.loc[idx, "raw_p"].to_numpy(dtype=float))
    return out.sort_values(["condition", "raw_p"], na_position="last").reset_index(drop=True)


def save_summary_outputs(output_dir: Path, final_metrics: pd.DataFrame, correlations: pd.DataFrame) -> None:
    summary_dir = output_dir / "summary"
    plot_dir = output_dir / "plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["PCA_overlap_RF", "PCA_BC_RF", "PCA_JS_RF", "SWD", "MMD_RBF", "WD"]
    summary_rows = []
    for condition, cond_df in final_metrics.groupby("condition", sort=True):
        for metric in metrics:
            if metric not in cond_df.columns:
                continue
            vals = cond_df[metric].dropna()
            summary_rows.append(
                {
                    "condition": condition,
                    "n_participants": int(len(vals)),
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                    "mean_std_3dp": f"{vals.mean():.3f} +/- {vals.std(ddof=1):.3f}" if len(vals) > 1 else f"{vals.mean():.3f} +/- nan",
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_dir / "ot_vs_mw_summary_metrics.csv", index=False)
    summary_df.to_csv(summary_dir / "table_i_style_summary.csv", index=False)

    counts = final_metrics.copy()
    counts["quality_category"] = counts["PCA_overlap_RF"].apply(categorize_overlap)
    count_df = (
        counts.groupby(["condition", "quality_category"], sort=True)
        .size()
        .reset_index(name="n")
        .sort_values(["condition", "quality_category"])
    )
    count_df.to_csv(summary_dir / "good_borderline_poor_counts.csv", index=False)

    plot_pca_overlap_bars(counts, plot_dir / "participant_level_PCA_overlap_RF.png")

    for condition in sorted(correlations["condition"].dropna().unique()):
        cond_corr = correlations[correlations["condition"] == condition]
        cond_corr.to_csv(summary_dir / f"predictor_correlations_{condition}.csv", index=False)


def save_current_vs_heldout_comparison(output_dir: Path, final_metrics: pd.DataFrame) -> None:
    current_path = (
        PROJECT_ROOT
        / "results"
        / "final_quality_metrics_baseline_corrected"
        / "ot_mw_participant_quality_table_no_icc_wide.csv"
    )
    if not current_path.exists():
        print(f"[warn] Current protocol quality table not found, skip comparison: {current_path}")
        return

    current = pd.read_csv(current_path)
    rows = []
    for _, row in current.iterrows():
        participant = int(row["participant_id"])
        for condition in ("ot", "mw"):
            prefix = condition.upper()
            overlap_col = f"{prefix}_pca_overlap"
            epoch_col = f"{prefix}_epoch"
            if overlap_col not in current.columns or pd.isna(row.get(overlap_col)):
                continue
            rows.append(
                {
                    "condition": condition,
                    "participant": participant,
                    "current_epoch": int(row[epoch_col]) if epoch_col in current.columns and pd.notna(row[epoch_col]) else np.nan,
                    "current_PCA_overlap_RF": float(row[overlap_col]),
                    "current_quality_category": categorize_overlap(float(row[overlap_col])),
                }
            )
    if not rows:
        print(f"[warn] No current protocol rows parsed from: {current_path}")
        return

    current_long = pd.DataFrame(rows)
    heldout = final_metrics[
        ["condition", "participant", "selected_epoch", "PCA_overlap_RF"]
    ].copy()
    heldout["heldout_quality_category"] = heldout["PCA_overlap_RF"].apply(categorize_overlap)
    comparison = current_long.merge(heldout, on=["condition", "participant"], how="inner")
    if comparison.empty:
        print("[warn] No overlapping participants between current protocol table and held-out run.")
        return
    comparison = comparison.rename(
        columns={
            "selected_epoch": "heldout_selected_epoch",
            "PCA_overlap_RF": "heldout_PCA_overlap_RF",
        }
    )
    comparison["epoch_delta_heldout_minus_current"] = (
        comparison["heldout_selected_epoch"] - comparison["current_epoch"]
    )
    comparison["PCA_overlap_delta_heldout_minus_current"] = (
        comparison["heldout_PCA_overlap_RF"] - comparison["current_PCA_overlap_RF"]
    )
    comparison = comparison.sort_values(["condition", "participant"])
    out = output_dir / "summary" / "current_vs_heldout_protocol_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out, index=False)
    print(f"Saved current-vs-heldout comparison: {out}")


def plot_pca_overlap_bars(df: pd.DataFrame, save_path: Path) -> None:
    color_map = {"Good": "#2e7d32", "Borderline": "#f57c00", "Poor": "#c62828"}
    plot_df = df.sort_values(["condition", "participant"]).copy()
    labels = [f"{r.condition.upper()} P{int(r.participant)}" for r in plot_df.itertuples()]
    colors = [color_map[categorize_overlap(v)] for v in plot_df["PCA_overlap_RF"]]
    fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 0.35), 5), facecolor="white")
    ax.bar(np.arange(len(plot_df)), plot_df["PCA_overlap_RF"], color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.60, color="#2e7d32", linestyle="--", linewidth=1.5)
    ax.axhline(0.50, color="#c62828", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Held-out PCA overlap (real vs fake)")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_scatter_outputs(
    output_dir: Path,
    predictors: pd.DataFrame,
    final_metrics: pd.DataFrame,
    correlations: pd.DataFrame,
    top_k: int,
) -> None:
    merged = predictors.merge(final_metrics, on=["condition", "participant"], how="inner")
    scatter_dir = output_dir / "plots" / "figure3_style_scatters"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    for condition in sorted(merged["condition"].unique()):
        top = correlations[correlations["condition"] == condition].dropna(subset=["raw_p"]).head(top_k)
        cond_df = merged[merged["condition"] == condition]
        for row in top.itertuples():
            predictor = row.predictor
            if predictor not in cond_df.columns:
                continue
            sub = cond_df[[predictor, "PCA_overlap_RF"]].dropna()
            if len(sub) < 3:
                continue
            fig, ax = plt.subplots(figsize=(4.5, 3.6), facecolor="white")
            ax.scatter(sub[predictor], sub["PCA_overlap_RF"], s=30, color="#455a64", edgecolors="white", linewidths=0.5)
            try:
                slope, intercept = np.polyfit(sub[predictor], sub["PCA_overlap_RF"], 1)
                xs = np.linspace(sub[predictor].min(), sub[predictor].max(), 100)
                ax.plot(xs, slope * xs + intercept, color="#c62828", linewidth=1.5)
            except Exception:
                pass
            rho = row.rho
            raw_p = row.raw_p
            ax.text(0.04, 0.96, f"rho={rho:.2f}, p={raw_p:.3g}", transform=ax.transAxes, ha="left", va="top")
            ax.set_xlabel(predictor)
            ax.set_ylabel("Held-out PCA overlap (RF)")
            ax.set_title(f"{condition.upper()}: {predictor}", loc="left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            safe = predictor.replace("/", "_").replace(" ", "_")
            fig.savefig(scatter_dir / f"{condition}_PCA_overlap_RF_vs_{safe}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)


def process_participant(
    sp: ParticipantSplit,
    device: str,
    config: dict,
    output_dir: Path,
) -> Tuple[ParticipantRunResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 80)
    print(f"{sp.condition.upper()} Participant P{sp.participant}")
    print("=" * 80)
    print(f"Total = {len(sp.total_indices)}")
    print(f"Calibration = {len(sp.calibration_indices)}")
    print(f"Held-out = {len(sp.test_indices)}")
    assert set(sp.calibration_indices).isdisjoint(set(sp.test_indices))

    split_trial_dir = output_dir / "trials" / sp.condition / f"p{sp.participant}"
    write_trial_matrix(split_trial_dir / "calibration_real_trials.csv", sp.calibration_trials_bc)
    write_trial_matrix(split_trial_dir / "heldout_real_trials.csv", sp.test_trials_bc)

    save_dir = fine_tune_on_calibration(sp, device, config, output_dir)
    checkpoint_df = evaluate_checkpoints(sp, save_dir)
    selected_row = checkpoint_df[checkpoint_df["selected_checkpoint"]].iloc[0]
    selected_epoch = int(selected_row["epoch"])
    selected_checkpoint = save_dir / "models" / f"checkpoint_epoch_{selected_epoch}.pth"
    if not selected_checkpoint.exists():
        raise FileNotFoundError(f"Selected checkpoint not found: {selected_checkpoint}")
    print(f"Selected epoch = {selected_epoch}")

    # Held-out trials are first used below, after selected_epoch is fixed.
    assert selected_epoch is not None
    calibration_fake_path = save_dir / "generated_data" / f"generated_trials_epoch_{selected_epoch}.csv"
    calibration_fake = read_trials(calibration_fake_path)
    tuner = load_tuner_from_checkpoint(
        selected_checkpoint,
        device,
        config,
        real_min=float(sp.real_min),
        real_max=float(sp.real_max),
        participant=sp.participant,
    )
    heldout_fake = tuner.generate_trials(len(sp.test_trials_bc), float(sp.real_min), float(sp.real_max))
    assert len(heldout_fake) == len(sp.test_trials_bc)
    heldout_fake_path = split_trial_dir / f"heldout_fake_selected_epoch_{selected_epoch}.csv"
    write_trial_matrix(heldout_fake_path, heldout_fake)

    real_cal_feat, real_test_feat, fake_cal_feat, fake_test_feat = save_feature_tables(
        output_dir, sp, selected_epoch, calibration_fake, heldout_fake
    )
    calibration_pca = compute_pca_feature_metrics(real_cal_feat, fake_cal_feat)
    heldout_pca = compute_pca_feature_metrics(real_test_feat, fake_test_feat)
    final_metrics = compute_final_quality_metrics(sp.test_trials_bc, heldout_fake)
    final_metrics.update(heldout_pca)

    final_row = pd.DataFrame(
        [
            {
                "condition": sp.condition,
                "participant": sp.participant,
                "selected_epoch": selected_epoch,
                "selected_checkpoint": str(selected_checkpoint),
                "real_subset": "heldout",
                **final_metrics,
            }
        ]
    )

    calibration_row = pd.DataFrame(
        [
            {
                "condition": sp.condition,
                "participant": sp.participant,
                "selected_epoch": selected_epoch,
                "real_subset": "calibration",
                **calibration_pca,
            }
        ]
    )

    print(f"Calibration PCA overlap = {calibration_pca['PCA_overlap_RF']:.4f}")
    print(f"Held-out PCA overlap = {heldout_pca['PCA_overlap_RF']:.4f}")

    result = ParticipantRunResult(
        condition=sp.condition,
        participant=sp.participant,
        n_total=len(sp.total_indices),
        n_calibration=len(sp.calibration_indices),
        n_test=len(sp.test_indices),
        save_dir=save_dir,
        selected_epoch=selected_epoch,
        selected_checkpoint=selected_checkpoint,
        calibration_pca_overlap=calibration_pca["PCA_overlap_RF"],
        heldout_pca_overlap=heldout_pca["PCA_overlap_RF"],
    )
    return result, checkpoint_df, final_row, calibration_row


def append_or_write_csv(path: Path, df: pd.DataFrame, key_cols: Sequence[str], overwrite_existing_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite_existing_keys:
        old = pd.read_csv(path)
        out = pd.concat([old, df], ignore_index=True)
    elif path.exists():
        old = pd.read_csv(path)
        keys = set(tuple(x) for x in df[list(key_cols)].to_numpy())
        keep = [tuple(x) not in keys for x in old[list(key_cols)].to_numpy()]
        out = pd.concat([old.loc[keep], df], ignore_index=True)
    else:
        out = df.copy()
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out 80/20 EEG fine-tuning and evaluation pipeline.")
    parser.add_argument("--conditions", nargs="+", choices=["ot", "mw"], default=["ot", "mw"])
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "results" / "heldout_80_20"))
    parser.add_argument("--participants", nargs="*", default=[], help="Optional filters like ot:1,2 or mw:1 (illustrative IDs).")
    parser.add_argument("--group-reference", choices=["leave-one-out", "pooled-all"], default="leave-one-out")
    parser.add_argument("--epochs", type=int, default=None, help="For debugging only; default keeps existing 500 epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="For debugging only; default keeps existing batch size.")
    parser.add_argument("--save-interval", type=int, default=None, help="For debugging only; default keeps existing interval.")
    parser.add_argument("--plot-top-k", type=int, default=6)
    parser.add_argument("--overwrite-existing-keys", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_filters = parse_participant_filters(args.participants)
    conditions = [c for c in args.conditions if not participant_filters or c in participant_filters]
    if not conditions:
        raise ValueError("No conditions selected.")

    print("=" * 80)
    print("Held-out 80/20 participant EEG fine-tuning pipeline")
    print("=" * 80)
    print(f"Output dir: {output_dir}")
    print(f"Conditions: {conditions}")
    print(f"Group reference: {args.group_reference}")
    print(
        "[leakage note] This repository loads external base_OT/base_MW checkpoints; "
        "it does not contain group-pretraining code, so calibration-only group pretraining is not verified here."
    )

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception as exc:
        print(f"[warn] deterministic settings not fully enabled: {exc}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    raw_by_condition = {condition: load_condition_raw_trials(condition) for condition in conditions}
    splits = make_participant_splits(raw_by_condition, conditions, participant_filters)
    if not splits:
        raise RuntimeError("No eligible participants after filtering.")
    write_split_manifest(splits, output_dir)

    checkpoint_frames: List[pd.DataFrame] = []
    final_frames: List[pd.DataFrame] = []
    calibration_frames: List[pd.DataFrame] = []
    real_cal_features: Dict[str, Dict[int, pd.DataFrame]] = {condition: {} for condition in conditions}
    run_results: List[ParticipantRunResult] = []

    start = time.time()
    for condition in conditions:
        cfg = build_config(condition, output_dir, args)
        for sp in [s for s in splits if s.condition == condition]:
            result, checkpoint_df, final_df, calibration_df = process_participant(sp, device, cfg, output_dir)
            run_results.append(result)
            checkpoint_frames.append(checkpoint_df)
            final_frames.append(final_df)
            calibration_frames.append(calibration_df)
            real_feature_path = (
                output_dir
                / "features"
                / condition
                / "real_calibration"
                / f"p{sp.participant}_real_calibration.csv"
            )
            real_cal_features[condition][sp.participant] = pd.read_csv(real_feature_path)

            append_or_write_csv(
                output_dir / "checkpoint_selection.csv",
                checkpoint_df,
                key_cols=["condition", "participant", "epoch"],
                overwrite_existing_keys=True,
            )
            append_or_write_csv(
                output_dir / "final_heldout_metrics.csv",
                final_df,
                key_cols=["condition", "participant"],
                overwrite_existing_keys=True,
            )

    all_checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    all_final = pd.concat(final_frames, ignore_index=True)
    all_calibration = pd.concat(calibration_frames, ignore_index=True)
    all_checkpoints.to_csv(output_dir / "checkpoint_selection.csv", index=False)
    all_final.sort_values(["condition", "participant"]).to_csv(output_dir / "final_heldout_metrics.csv", index=False)
    all_calibration.sort_values(["condition", "participant"]).to_csv(
        output_dir / "calibration_selected_metrics.csv", index=False
    )

    predictor_frames = []
    for condition in conditions:
        condition_splits = [s for s in splits if s.condition == condition]
        predictor_frames.append(
            compute_condition_predictors(
                condition=condition,
                splits=condition_splits,
                real_feature_by_pid=real_cal_features[condition],
                group_reference=args.group_reference,
            )
        )
    predictors = pd.concat(predictor_frames, ignore_index=True).sort_values(["condition", "participant"])
    predictors.to_csv(output_dir / "participant_predictors_calibration.csv", index=False)

    correlations = compute_predictor_correlations(predictors, all_final)
    correlations.to_csv(output_dir / "predictor_correlations_heldout.csv", index=False)

    save_summary_outputs(output_dir, all_final, correlations)
    save_current_vs_heldout_comparison(output_dir, all_final)
    save_scatter_outputs(output_dir, predictors, all_final, correlations, top_k=args.plot_top_k)

    print("\n" + "=" * 80)
    print("Held-out run summary")
    print("=" * 80)
    for result in run_results:
        print(
            f"{result.condition.upper()} P{result.participant}: "
            f"Total={result.n_total}, Calibration={result.n_calibration}, Held-out={result.n_test}, "
            f"Selected epoch={result.selected_epoch}, "
            f"Calibration PCA overlap={result.calibration_pca_overlap:.4f}, "
            f"Held-out PCA overlap={result.heldout_pca_overlap:.4f}"
        )
    print(f"Elapsed minutes: {(time.time() - start) / 60.0:.2f}")
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
