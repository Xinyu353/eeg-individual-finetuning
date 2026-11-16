#!/usr/bin/env python3
"""Export all-participant OT/MW thesis tables of five-run median metrics.

Each row is one participant. Development and held-out PCA-overlap are five-run
medians; delta is held-out median minus development median. Remaining columns
are five-run median [Q1-Q3] held-out quality metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASES = {
    "ot": PROJECT_ROOT / "results" / "ot_heldout_seed_sensitivity_all_final",
    "mw": PROJECT_ROOT / "results" / "mw_heldout_seed_sensitivity_all_final",
}
OUT_DIR = PROJECT_ROOT / "results" / "heldout_80_20_combined_figures" / "thesis_supplement"

OVERLAP_DIGITS = 4
METRIC_DIGITS = 4
HELD_OUT_METRICS = [
    ("heldout_PCA_BC_RF", "PCA-BC", METRIC_DIGITS),
    ("heldout_PCA_JS_RF", "PCA-JS", METRIC_DIGITS),
    ("heldout_quality_SWD", "SWD", METRIC_DIGITS),
    ("heldout_quality_MMD_RBF", "MMD-RBF", METRIC_DIGITS),
    ("heldout_quality_WD", "WD", METRIC_DIGITS),
]


def categorize(value: float) -> str:
    if value >= 0.60:
        return "Good"
    if value >= 0.50:
        return "Borderline"
    return "Poor"


def fmt_interval(median: float, q1: float, q3: float, digits: int = OVERLAP_DIGITS) -> str:
    return f"{median:.{digits}f} [{q1:.{digits}f}–{q3:.{digits}f}]"


def fmt_delta(value: float, digits: int = OVERLAP_DIGITS) -> str:
    if abs(value) < 0.5 * (10 ** -digits):
        return f"{0:.{digits}f}"
    return f"{value:+.{digits}f}"


def quartile_stats(values: np.ndarray) -> tuple[float, float, float]:
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return float(q1), float(med), float(q3)


def build_condition(condition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = BASES[condition]
    runs = pd.read_csv(base / "seed_sensitivity_runs.csv")
    official = pd.read_csv(base / "comparison" / "tables" / "paper_source" / "heldout_participant_median_iqr.csv")
    numeric_rows = []
    display_rows = []
    for pid, sub in runs.groupby("participant", sort=True):
        if len(sub) != 5:
            raise ValueError(f"{condition} P{pid}: expected 5 seeds, got {len(sub)}")
        off = official[official["participant"] == pid].iloc[0]
        heldout_q1 = float(off["Q1"])
        heldout_med = float(off["median"])
        heldout_q3 = float(off["Q3"])
        category = str(off["category"])
        if category != categorize(heldout_med):
            raise ValueError(f"{condition} P{pid}: category mismatch")

        dev_q1, dev_med, dev_q3 = quartile_stats(sub["calibration_PCA_overlap_RF"].to_numpy(dtype=float))
        delta = heldout_med - dev_med

        num = {
            "participant": int(pid),
            "n_total": int(sub["n_total"].iloc[0]),
            "n_cal": int(sub["n_train"].iloc[0]),
            "n_test": int(sub["n_test"].iloc[0]),
            "n_seeds": 5,
            "development_PCA_overlap_median": dev_med,
            "development_PCA_overlap_Q1": dev_q1,
            "development_PCA_overlap_Q3": dev_q3,
            "heldout_PCA_overlap_median": heldout_med,
            "heldout_PCA_overlap_Q1": heldout_q1,
            "heldout_PCA_overlap_Q3": heldout_q3,
            "delta_heldout_minus_development": float(delta),
            "category": category,
        }
        disp = {
            "Participant": f"P{int(pid)}",
            "n_cal": int(sub["n_train"].iloc[0]),
            "n_test": int(sub["n_test"].iloc[0]),
            "Development PCA-overlap": fmt_interval(dev_med, dev_q1, dev_q3),
            "Held-out PCA-overlap": fmt_interval(heldout_med, heldout_q1, heldout_q3),
            "Δ Held-out − Development": fmt_delta(delta),
            "Category": category,
        }
        for col, pretty, digits in HELD_OUT_METRICS:
            q1, med, q3 = quartile_stats(sub[col].to_numpy(dtype=float))
            num[f"{pretty}_median"] = med
            num[f"{pretty}_Q1"] = q1
            num[f"{pretty}_Q3"] = q3
            num[f"{pretty}_IQR"] = q3 - q1
            disp[pretty] = fmt_interval(med, q1, q3, digits)
        numeric_rows.append(num)
        display_rows.append(disp)
    return pd.DataFrame(numeric_rows), pd.DataFrame(display_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for condition in ("ot", "mw"):
        numeric, display = build_condition(condition)
        num_path = OUT_DIR / f"tableS_{condition}_heldout_participant_median_iqr.csv"
        disp_path = OUT_DIR / f"tableS_{condition}_heldout_participant_median_iqr_display.csv"
        numeric.to_csv(num_path, index=False)
        display.to_csv(disp_path, index=False)
        print(f"Saved {num_path}  ({len(numeric)} participants)")
        print(f"Saved {disp_path}")
        print(display.to_string(index=False))
        ranked = numeric.sort_values("delta_heldout_minus_development")
        print(
            f"{condition.upper()} largest development-to-held-out decrease: "
            + ", ".join(
                f"P{int(r.participant)} ({r.delta_heldout_minus_development:+.3f})"
                for r in ranked.head(3).itertuples()
            )
        )
        print()


if __name__ == "__main__":
    main()
