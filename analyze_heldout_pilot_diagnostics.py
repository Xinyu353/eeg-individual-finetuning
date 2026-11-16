#!/usr/bin/env python3
"""Pilot diagnostics for original vs held-out quality and repeat generation stability."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl_tmp"))

import numpy as np
import pandas as pd
import torch

import heldout_80_20_pipeline as h


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "heldout_80_20"


def parse_data_range(summary_path: Path) -> tuple[float, float] | None:
    if not summary_path.exists():
        return None
    text = summary_path.read_text()
    match = re.search(r"Data range\s*:\s*\[([-+0-9.eE]+),\s*([-+0-9.eE]+)\]", text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def write_comparison_tables(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(summary_dir / "current_vs_heldout_protocol_comparison.csv")
    calibration = pd.read_csv(results_dir / "calibration_selected_metrics.csv")
    final = pd.read_csv(results_dir / "final_heldout_metrics.csv")
    split_manifest = pd.read_csv(results_dir / "split_manifest.csv")

    table = comparison[
        [
            "condition",
            "participant",
            "current_epoch",
            "current_PCA_overlap_RF",
            "current_quality_category",
            "heldout_selected_epoch",
            "heldout_PCA_overlap_RF",
            "heldout_quality_category",
        ]
    ].rename(
        columns={
            "current_epoch": "original_selected_epoch",
            "current_PCA_overlap_RF": "original_PCA_overlap_RF",
            "current_quality_category": "original_quality_category",
            "heldout_selected_epoch": "new_80_20_selected_epoch",
            "heldout_PCA_overlap_RF": "heldout_PCA_overlap_RF",
            "heldout_quality_category": "heldout_quality_category",
        }
    )
    table = table.merge(
        calibration[["condition", "participant", "PCA_overlap_RF", "PCA_BC_RF", "PCA_JS_RF"]].rename(
            columns={
                "PCA_overlap_RF": "calibration_PCA_overlap_RF",
                "PCA_BC_RF": "calibration_PCA_BC_RF",
                "PCA_JS_RF": "calibration_PCA_JS_RF",
            }
        ),
        on=["condition", "participant"],
        how="left",
    )
    table = table.merge(
        final[["condition", "participant", "PCA_BC_RF", "PCA_JS_RF", "SWD", "MMD_RBF", "WD"]].rename(
            columns={
                "PCA_BC_RF": "heldout_PCA_BC_RF",
                "PCA_JS_RF": "heldout_PCA_JS_RF",
                "SWD": "heldout_SWD",
                "MMD_RBF": "heldout_MMD_RBF",
                "WD": "heldout_WD",
            }
        ),
        on=["condition", "participant"],
        how="left",
    )
    table = table.merge(
        split_manifest[["condition", "participant", "n_total", "n_calibration", "n_test"]],
        on=["condition", "participant"],
        how="left",
    )
    table["calibration_minus_original_PCA_overlap_RF"] = (
        table["calibration_PCA_overlap_RF"] - table["original_PCA_overlap_RF"]
    )
    table["heldout_minus_original_PCA_overlap_RF"] = (
        table["heldout_PCA_overlap_RF"] - table["original_PCA_overlap_RF"]
    )
    table["heldout_minus_calibration_PCA_overlap_RF"] = (
        table["heldout_PCA_overlap_RF"] - table["calibration_PCA_overlap_RF"]
    )
    table = table.sort_values(["condition", "participant"])
    table.to_csv(summary_dir / "original_vs_new_calibration_vs_heldout.csv", index=False)

    epoch_table = table[
        [
            "condition",
            "participant",
            "original_selected_epoch",
            "new_80_20_selected_epoch",
            "n_total",
            "n_calibration",
            "n_test",
            "original_PCA_overlap_RF",
            "calibration_PCA_overlap_RF",
            "heldout_PCA_overlap_RF",
        ]
    ].copy()
    epoch_table["epoch_delta_80_20_minus_original"] = (
        epoch_table["new_80_20_selected_epoch"] - epoch_table["original_selected_epoch"]
    )
    epoch_table.to_csv(summary_dir / "selected_epoch_original_vs_80_20.csv", index=False)
    return table, epoch_table


def original_feature_paths(condition: str, participant: int, epoch: int) -> tuple[Path, Path]:
    if condition == "mw":
        real_path = PROJECT_ROOT / "mw_pipeline" / "features" / "real_mw" / f"p{participant}_real.csv"
        fake_path = PROJECT_ROOT / "mw_pipeline" / "features" / "mw_feature" / f"p{participant}_epoch_{epoch}.csv"
    else:
        real_path = PROJECT_ROOT / "ot_pipeline" / "features" / "real_ot" / f"p{participant}_real.csv"
        fake_path = PROJECT_ROOT / "ot_pipeline" / "features" / "ot_feature" / f"p{participant}_epoch_{epoch}.csv"
        if not real_path.exists():
            real_path = PROJECT_ROOT / "feature_extraction" / "real_ot" / f"p{participant}_real.csv"
        if not fake_path.exists():
            fake_path = PROJECT_ROOT / "feature_extraction" / "ot_feature" / f"p{participant}_epoch_{epoch}.csv"
    if not real_path.exists():
        raise FileNotFoundError(f"Original real feature file not found: {real_path}")
    if not fake_path.exists():
        raise FileNotFoundError(f"Original fake feature file not found: {fake_path}")
    return real_path, fake_path


def generate_real_subset_stability(
    results_dir: Path,
    n_repeats: int,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(results_dir / "summary" / "original_vs_new_calibration_vs_heldout.csv")
    rows = []
    for row in table.itertuples(index=False):
        condition = str(row.condition)
        participant = int(row.participant)
        original_epoch = int(row.original_selected_epoch)
        n_test = int(row.n_test)
        real_path, fake_path = original_feature_paths(condition, participant, original_epoch)
        real_features_all = pd.read_csv(real_path)
        fake_features_all = pd.read_csv(fake_path)
        if len(real_features_all) < n_test:
            raise ValueError(f"{condition.upper()} P{participant}: real rows {len(real_features_all)} < n_test {n_test}")

        for repeat_idx in range(1, n_repeats + 1):
            seed = base_seed + repeat_idx + participant * 1000 + (0 if condition == "ot" else 100000)
            rng = np.random.default_rng(seed)
            real_idx = rng.choice(len(real_features_all), size=n_test, replace=False)
            fake_replace = len(fake_features_all) < n_test
            fake_idx = rng.choice(len(fake_features_all), size=n_test, replace=fake_replace)
            real_subset = real_features_all.iloc[real_idx].reset_index(drop=True)
            fake_subset = fake_features_all.iloc[fake_idx].reset_index(drop=True)
            pca_metrics = h.compute_pca_feature_metrics(real_subset, fake_subset)
            rows.append(
                {
                    "condition": condition,
                    "participant": participant,
                    "original_selected_epoch": original_epoch,
                    "repeat": repeat_idx,
                    "seed": seed,
                    "n_subset_real": n_test,
                    "n_subset_fake": n_test,
                    "original_full_PCA_overlap_RF": float(row.original_PCA_overlap_RF),
                    "new_calibration_PCA_overlap_RF": float(row.calibration_PCA_overlap_RF),
                    "new_heldout_PCA_overlap_RF": float(row.heldout_PCA_overlap_RF),
                    **pca_metrics,
                    "original_real_features_path": str(real_path),
                    "original_fake_features_path": str(fake_path),
                }
            )
        print(f"{condition.upper()} P{participant}: real subset stability {n_repeats} repeats done")

    repeat_df = pd.DataFrame(rows).sort_values(["condition", "participant", "repeat"])
    repeat_df.to_csv(results_dir / "summary" / "original_real_subset_stability_20pct.csv", index=False)
    summary = (
        repeat_df.groupby(["condition", "participant", "original_selected_epoch"], sort=True)
        .agg(
            n_repeats=("PCA_overlap_RF", "size"),
            n_subset_real=("n_subset_real", "first"),
            original_full_PCA_overlap_RF=("original_full_PCA_overlap_RF", "first"),
            new_calibration_PCA_overlap_RF=("new_calibration_PCA_overlap_RF", "first"),
            new_heldout_PCA_overlap_RF=("new_heldout_PCA_overlap_RF", "first"),
            subset_mean_PCA_overlap_RF=("PCA_overlap_RF", "mean"),
            subset_sd_PCA_overlap_RF=("PCA_overlap_RF", "std"),
            subset_min_PCA_overlap_RF=("PCA_overlap_RF", "min"),
            subset_max_PCA_overlap_RF=("PCA_overlap_RF", "max"),
            subset_q025_PCA_overlap_RF=("PCA_overlap_RF", lambda s: float(s.quantile(0.025))),
            subset_q975_PCA_overlap_RF=("PCA_overlap_RF", lambda s: float(s.quantile(0.975))),
        )
        .reset_index()
    )
    summary["subset_range_PCA_overlap_RF"] = (
        summary["subset_max_PCA_overlap_RF"] - summary["subset_min_PCA_overlap_RF"]
    )
    summary.to_csv(results_dir / "summary" / "original_real_subset_stability_20pct_summary.csv", index=False)
    return repeat_df, summary


def build_splits(results_dir: Path) -> dict[tuple[str, int], h.ParticipantSplit]:
    final = pd.read_csv(results_dir / "final_heldout_metrics.csv")
    filters: dict[str, list[int]] = {}
    for condition, cond_df in final.groupby("condition"):
        filters[str(condition)] = sorted(cond_df["participant"].astype(int).tolist())
    conditions = sorted(filters)
    raw_by_condition = {condition: h.load_condition_raw_trials(condition) for condition in conditions}
    splits = h.make_participant_splits(raw_by_condition, conditions, filters)
    return {(sp.condition, sp.participant): sp for sp in splits}


def generate_stability(results_dir: Path, n_repeats: int, base_seed: int, device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(results_dir / "final_heldout_metrics.csv").sort_values(["condition", "participant"])
    split_by_key = build_splits(results_dir)
    cfg_args = SimpleNamespace(epochs=None, batch_size=None, save_interval=None)
    repeat_root = results_dir / "stability_repeats"
    rows = []

    for row in final.itertuples(index=False):
        condition = str(row.condition)
        participant = int(row.participant)
        selected_epoch = int(row.selected_epoch)
        checkpoint_path = Path(str(row.selected_checkpoint))
        sp = split_by_key[(condition, participant)]
        model_dir = checkpoint_path.parent.parent

        data_range = parse_data_range(model_dir / "training_summary.txt")
        if data_range is None:
            data_range = (float(sp.calibration_trials_bc.min()), float(sp.calibration_trials_bc.max()))
            print(f"[warn] {condition.upper()} P{participant}: fallback to calibration data range {data_range}")
        real_min, real_max = data_range

        cfg = h.build_config(condition, results_dir, cfg_args)
        tuner = h.load_tuner_from_checkpoint(
            checkpoint_path,
            device,
            cfg,
            real_min=real_min,
            real_max=real_max,
            participant=participant,
        )
        real_feature_path = (
            results_dir / "features" / condition / "real_heldout" / f"p{participant}_real_heldout.csv"
        )
        real_features = pd.read_csv(real_feature_path)
        n_trials = len(sp.test_trials_bc)

        for repeat_idx in range(1, n_repeats + 1):
            seed = base_seed + repeat_idx + participant * 1000 + (0 if condition == "ot" else 100000)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            fake_trials = tuner.generate_trials(n_trials, real_min, real_max)
            out_dir = repeat_root / condition / f"p{participant}" / f"repeat_{repeat_idx:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            fake_trial_path = out_dir / "heldout_fake_trials.csv"
            fake_feature_path = out_dir / "heldout_fake_features.csv"
            pd.DataFrame(fake_trials).to_csv(fake_trial_path, index=False, header=False)

            fake_features = h.extract_features_from_trials(
                fake_trials,
                verbose_prefix=f"{condition.upper()} P{participant} repeat {repeat_idx:02d}",
            )
            fake_features.to_csv(fake_feature_path, index=False)
            pca_metrics = h.compute_pca_feature_metrics(real_features, fake_features)
            rows.append(
                {
                    "condition": condition,
                    "participant": participant,
                    "selected_epoch": selected_epoch,
                    "repeat": repeat_idx,
                    "seed": seed,
                    "n_heldout_trials": n_trials,
                    "original_single_PCA_overlap_RF": float(row.PCA_overlap_RF),
                    **pca_metrics,
                    "fake_trials_path": str(fake_trial_path),
                    "fake_features_path": str(fake_feature_path),
                }
            )
            print(
                f"{condition.upper()} P{participant} repeat {repeat_idx:02d}: "
                f"PCA_overlap_RF={pca_metrics['PCA_overlap_RF']:.4f}"
            )

    repeat_df = pd.DataFrame(rows).sort_values(["condition", "participant", "repeat"])
    repeat_df.to_csv(results_dir / "summary" / "heldout_repeat_generation_stability.csv", index=False)

    summary = (
        repeat_df.groupby(["condition", "participant", "selected_epoch"], sort=True)
        .agg(
            n_repeats=("PCA_overlap_RF", "size"),
            original_single_PCA_overlap_RF=("original_single_PCA_overlap_RF", "first"),
            repeat_mean_PCA_overlap_RF=("PCA_overlap_RF", "mean"),
            repeat_std_PCA_overlap_RF=("PCA_overlap_RF", "std"),
            repeat_min_PCA_overlap_RF=("PCA_overlap_RF", "min"),
            repeat_max_PCA_overlap_RF=("PCA_overlap_RF", "max"),
        )
        .reset_index()
    )
    summary["repeat_range_PCA_overlap_RF"] = (
        summary["repeat_max_PCA_overlap_RF"] - summary["repeat_min_PCA_overlap_RF"]
    )
    summary["original_minus_repeat_mean_PCA_overlap_RF"] = (
        summary["original_single_PCA_overlap_RF"] - summary["repeat_mean_PCA_overlap_RF"]
    )
    summary.to_csv(results_dir / "summary" / "heldout_repeat_generation_stability_summary.csv", index=False)
    return repeat_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze held-out pilot diagnostics and repeat generation stability.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-real-subset-repeats", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=4242)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--skip-real-subset-stability", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    table, epoch_table = write_comparison_tables(results_dir)
    print(f"Wrote comparison rows: {len(table)}")
    print(f"Wrote epoch comparison rows: {len(epoch_table)}")
    if not args.skip_real_subset_stability:
        real_subset_df, real_subset_summary = generate_real_subset_stability(
            results_dir, args.n_real_subset_repeats, args.base_seed
        )
        print(f"Wrote real subset stability rows: {len(real_subset_df)}")
        print(f"Wrote real subset stability summary rows: {len(real_subset_summary)}")
    if args.skip_stability:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repeat_df, summary = generate_stability(results_dir, args.n_repeats, args.base_seed, device)
    print(f"Wrote repeat rows: {len(repeat_df)}")
    print(f"Wrote stability summary rows: {len(summary)}")


if __name__ == "__main__":
    main()
