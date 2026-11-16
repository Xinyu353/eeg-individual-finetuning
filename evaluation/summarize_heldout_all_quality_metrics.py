#!/usr/bin/env python3
"""Assemble already-computed held-out quality metrics into paper tables.

Does not retrain or regenerate trials. Reads official seed_sensitivity_runs.csv
(OT 25×5, MW 12×5) and writes participant-median summaries plus OT vs MW
Welch t-tests with BH-FDR across the six paper metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from round1_all_predictors_correlation import fdr_bh  # noqa: E402

BASES = {
    "ot": PROJECT_ROOT / "results" / "ot_heldout_seed_sensitivity_all_final",
    "mw": PROJECT_ROOT / "results" / "mw_heldout_seed_sensitivity_all_final",
}
OUT_DIR = PROJECT_ROOT / "results" / "heldout_80_20_combined_figures" / "heldout_quality_metrics"

METRICS = [
    ("heldout_PCA_overlap_RF", "PCA-overlap_RF", "higher"),
    ("heldout_PCA_BC_RF", "PCA-BC_RF", "higher"),
    ("heldout_PCA_JS_RF", "PCA-JS_RF", "lower"),
    ("heldout_quality_SWD", "SWD", "lower"),
    ("heldout_quality_MMD_RBF", "MMD-RBF", "lower"),
    ("heldout_quality_WD", "WD", "lower"),
]


def load_runs() -> pd.DataFrame:
    frames = []
    for condition, base in BASES.items():
        df = pd.read_csv(base / "seed_sensitivity_runs.csv")
        df["condition"] = condition
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def participant_median_iqr(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, pid), sub in runs.groupby(["condition", "participant"], sort=True):
        row = {
            "condition": condition,
            "participant": int(pid),
            "n_seeds": int(len(sub)),
        }
        for col, pretty, _better in METRICS:
            vals = sub[col].dropna().to_numpy(dtype=float)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            row[f"{pretty}_Q1"] = float(q1)
            row[f"{pretty}_median"] = float(med)
            row[f"{pretty}_Q3"] = float(q3)
            row[f"{pretty}_IQR"] = float(q3 - q1)
        rows.append(row)
    return pd.DataFrame(rows)


def condition_summary(part: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, sub in part.groupby("condition", sort=True):
        for _col, pretty, better in METRICS:
            vals = sub[f"{pretty}_median"].dropna().to_numpy(dtype=float)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            rows.append(
                {
                    "condition": condition,
                    "n_participants": int(len(vals)),
                    "metric": pretty,
                    "better": better,
                    "mean_of_participant_medians": float(vals.mean()),
                    "sd_of_participant_medians": float(vals.std(ddof=1)),
                    "median_of_participant_medians": float(med),
                    "q1_of_participant_medians": float(q1),
                    "q3_of_participant_medians": float(q3),
                    "iqr_of_participant_medians": float(q3 - q1),
                }
            )
    return pd.DataFrame(rows)


def welch_ot_vs_mw(part: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p_raw = []
    for _col, pretty, better in METRICS:
        ot = part.loc[part["condition"].eq("ot"), f"{pretty}_median"].dropna().to_numpy(dtype=float)
        mw = part.loc[part["condition"].eq("mw"), f"{pretty}_median"].dropna().to_numpy(dtype=float)
        stat = ttest_ind(ot, mw, equal_var=False)
        rows.append(
            {
                "metric": pretty,
                "better": better,
                "n_ot": int(len(ot)),
                "n_mw": int(len(mw)),
                "ot_mean": float(ot.mean()),
                "ot_sd": float(ot.std(ddof=1)),
                "mw_mean": float(mw.mean()),
                "mw_sd": float(mw.std(ddof=1)),
                "mean_diff_ot_minus_mw": float(ot.mean() - mw.mean()),
                "t": float(stat.statistic),
                "df": float(stat.df),
                "p": float(stat.pvalue),
            }
        )
        p_raw.append(float(stat.pvalue))
    out = pd.DataFrame(rows)
    out["p_FDR"] = fdr_bh(np.asarray(p_raw, dtype=float))
    out["significant_FDR_0.05"] = out["p_FDR"] < 0.05
    return out


def seed_level(runs: pd.DataFrame) -> pd.DataFrame:
    keep = ["condition", "participant", "seed", "selected_epoch"] + [m[0] for m in METRICS]
    out = runs[keep].copy()
    rename = {col: pretty for col, pretty, _ in METRICS}
    return out.rename(columns=rename).sort_values(["condition", "participant", "seed"])


def write_condition_copies(part: pd.DataFrame, seeds: pd.DataFrame) -> None:
    for condition, base in BASES.items():
        dest = base / "comparison" / "tables" / "paper_source"
        dest.mkdir(parents=True, exist_ok=True)
        part.loc[part["condition"].eq(condition)].to_csv(
            dest / "heldout_all_metrics_participant_median_iqr.csv", index=False
        )
        seeds.loc[seeds["condition"].eq(condition)].to_csv(
            dest / "heldout_all_metrics_seed_level.csv", index=False
        )


def main() -> None:
    runs = load_runs()
    metric_cols = [m[0] for m in METRICS]
    missing = runs[metric_cols].isna().sum()
    if missing.any():
        raise ValueError(f"Missing held-out metrics:\n{missing}")

    seeds = seed_level(runs)
    part = participant_median_iqr(runs)
    cond = condition_summary(part)
    tests = welch_ot_vs_mw(part)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(OUT_DIR / "heldout_all_metrics_seed_level.csv", index=False)
    part.to_csv(OUT_DIR / "heldout_all_metrics_participant_median_iqr.csv", index=False)
    cond.to_csv(OUT_DIR / "heldout_all_metrics_condition_summary.csv", index=False)
    tests.to_csv(OUT_DIR / "ot_vs_mw_heldout_metrics_welch_fdr.csv", index=False)
    write_condition_copies(part, seeds)

    print(f"OT rows={int(runs.condition.eq('ot').sum())} participants={runs.loc[runs.condition.eq('ot'), 'participant'].nunique()}")
    print(f"MW rows={int(runs.condition.eq('mw').sum())} participants={runs.loc[runs.condition.eq('mw'), 'participant'].nunique()}")
    print()
    print(cond.to_string(index=False))
    print()
    print(tests.to_string(index=False))
    print()
    print(f"Saved {OUT_DIR}")


if __name__ == "__main__":
    main()
