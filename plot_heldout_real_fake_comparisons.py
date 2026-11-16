#!/usr/bin/env python3
"""Plot held-out real-vs-fake diagnostics for the 80/20 pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl_tmp"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from heldout_80_20_pipeline import FEATURE_COLUMNS, FS, TIME_END_MS, TIME_START_MS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "heldout_80_20"
REAL_LINE_COLOR = "black"
FAKE_LINE_COLOR = "red"
REAL_POINT_COLOR = "black"
FAKE_POINT_COLOR = "red"


def set_waveform_style() -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.titlesize"] = 32
    plt.rcParams["axes.labelsize"] = 32
    plt.rcParams["legend.fontsize"] = 28
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20


def set_distribution_style() -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20


def read_trials(path: Path) -> np.ndarray:
    data = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D trial matrix: {path}")
    return data


def mean_std(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=0) if data.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def plot_erp(real: np.ndarray, fake: np.ndarray, save_path: Path, title: str) -> None:
    set_waveform_style()
    time_s = np.linspace(TIME_START_MS, TIME_END_MS, real.shape[1]) / 1000.0
    real_mean, real_std = mean_std(real)
    fake_mean, fake_std = mean_std(fake)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.plot(time_s, real_mean, color=REAL_LINE_COLOR, linewidth=1.5, label="real")
    ax.fill_between(time_s, real_mean - real_std, real_mean + real_std, color=REAL_LINE_COLOR, alpha=0.1, linewidth=0)
    ax.plot(time_s, fake_mean, color=FAKE_LINE_COLOR, linewidth=1.5, label="fake")
    ax.fill_between(time_s, fake_mean - fake_std, fake_mean + fake_std, color=FAKE_LINE_COLOR, alpha=0.1, linewidth=0)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xticks(np.arange(-0.4, 1.21, 0.2))
    ax.set_xticklabels([f"{i:.1f}" for i in np.arange(-0.4, 1.21, 0.2)])
    ax.set_title(title, loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage ($\\mu$V)")
    ax.legend(frameon=False)
    ax.grid(False)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def trial_log_psd(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    freqs = None
    for trial in data:
        freqs, psd = welch(trial, fs=FS, nperseg=min(256, trial.size))
        rows.append(10.0 * np.log10(psd + 1e-10))
    return np.asarray(freqs), np.vstack(rows)


def plot_log_psd(real: np.ndarray, fake: np.ndarray, save_path: Path, title: str) -> None:
    set_waveform_style()
    freqs, real_psd = trial_log_psd(real)
    _, fake_psd = trial_log_psd(fake)
    real_mean, real_std = mean_std(real_psd)
    fake_mean, fake_std = mean_std(fake_psd)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.plot(freqs, real_mean, color=REAL_LINE_COLOR, linewidth=1.5, label="real")
    ax.fill_between(freqs, real_mean - real_std, real_mean + real_std, color=REAL_LINE_COLOR, alpha=0.1, linewidth=0)
    ax.plot(freqs, fake_mean, color=FAKE_LINE_COLOR, linewidth=1.5, label="fake")
    ax.fill_between(freqs, fake_mean - fake_std, fake_mean + fake_std, color=FAKE_LINE_COLOR, alpha=0.1, linewidth=0)
    ax.set_xlim(0, 45)
    ax.set_ylim(-50, 20)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Log PSD ($\\mu V^2/Hz$)")
    ax.legend(frameon=False)
    ax.grid(False)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def feature_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {path}: {missing}")
    return df[cols]


def plot_feature_pca(real_path: Path, fake_path: Path, save_path: Path, overlap: float) -> None:
    set_distribution_style()
    real = feature_matrix(real_path)
    fake = feature_matrix(fake_path)
    combined = pd.concat([real, fake], ignore_index=True)
    scaled = StandardScaler().fit_transform(combined)
    pca = PCA(n_components=2, random_state=42).fit(scaled)
    coords = pca.transform(scaled)
    n_real = len(real)

    fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="white")
    ax.scatter(coords[:n_real, 0], coords[:n_real, 1], color=REAL_POINT_COLOR, alpha=0.5, marker=".", label="Real")
    ax.scatter(coords[n_real:, 0], coords[n_real:, 1], color=FAKE_POINT_COLOR, alpha=0.5, marker=".", label="Fake")
    ax.set_title(f"PCA ({overlap:.4f})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(frameon=False)
    ax.grid(False)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_feature_delta(real_path: Path, fake_path: Path, save_path: Path, title: str, top_n: int = 10) -> None:
    set_distribution_style()
    real = feature_matrix(real_path)
    fake = feature_matrix(fake_path)
    combined = pd.concat([real, fake], ignore_index=True)
    mu = combined.mean(axis=0)
    sd = combined.std(axis=0, ddof=0).replace(0, np.nan)
    delta = ((fake.mean(axis=0) - real.mean(axis=0)) / sd).dropna().sort_values(key=lambda s: s.abs(), ascending=False)
    delta = delta.head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    colors = [FAKE_POINT_COLOR if v > 0 else REAL_POINT_COLOR for v in delta]
    ax.barh(delta.index, delta.values, color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Fake - real mean difference (pooled SD units)")
    ax.grid(False)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_participant(results_dir: Path, row: pd.Series) -> list[Path]:
    condition = str(row["condition"])
    participant = int(row["participant"])
    epoch = int(row["selected_epoch"])
    label = f"{condition.upper()} P{participant}"

    trial_dir = results_dir / "trials" / condition / f"p{participant}"
    feature_dir = results_dir / "features" / condition
    out_dir = results_dir / "plots" / "real_fake_comparisons" / condition / f"p{participant}"

    real_trials = read_trials(trial_dir / "heldout_real_trials.csv")
    fake_trials = read_trials(trial_dir / f"heldout_fake_selected_epoch_{epoch}.csv")
    real_features = feature_dir / "real_heldout" / f"p{participant}_real_heldout.csv"
    fake_features = feature_dir / "fake_heldout_selected" / f"p{participant}_epoch_{epoch}.csv"

    outputs = [
        out_dir / "erp_overlay.png",
        out_dir / "log_psd_overlay.png",
        out_dir / "feature_pca_distribution.png",
        out_dir / "feature_mean_delta_top10.png",
    ]
    plot_erp(real_trials, fake_trials, outputs[0], f"ERP ({label})")
    plot_log_psd(real_trials, fake_trials, outputs[1], f"PSD ({label})")
    plot_feature_pca(real_features, fake_features, outputs[2], float(row["PCA_overlap_RF"]))
    plot_feature_delta(real_features, fake_features, outputs[3], f"Feature delta ({label})")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot held-out real-vs-fake ERP, PSD, and feature distributions.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--participants", nargs="*", default=[], help="Optional filters like ot:26 mw:4")
    return parser.parse_args()


def participant_filter(items: Sequence[str]) -> set[tuple[str, int]]:
    filters: set[tuple[str, int]] = set()
    for item in items:
        condition, pid = item.split(":", 1)
        filters.add((condition.lower(), int(pid)))
    return filters


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    metrics_path = results_dir / "final_heldout_metrics.csv"
    final_metrics = pd.read_csv(metrics_path).sort_values(["condition", "participant"])
    filters = participant_filter(args.participants)
    if filters:
        final_metrics = final_metrics[
            final_metrics.apply(lambda r: (str(r["condition"]), int(r["participant"])) in filters, axis=1)
        ]
    if final_metrics.empty:
        raise RuntimeError("No participants matched for plotting.")

    generated = []
    for _, row in final_metrics.iterrows():
        generated.extend(plot_participant(results_dir, row))

    print(f"Generated {len(generated)} plots under {results_dir / 'plots' / 'real_fake_comparisons'}")


if __name__ == "__main__":
    main()
