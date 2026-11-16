"""
Select 16 predefined feature columns (plus sample id) from features_ot_4753.csv.

Usage:
    python select_16_features.py

Output:
    Saves features_ot_4753_selected16.csv in the same directory as the input.
"""
import os
import sys
import pandas as pd


FEATURES_16 = [
    # ERP features (2)
    "P3_Amplitude",
    "P3_Latency",
    # Hjorth parameters (3)
    "Activity",
    "Mobility",
    "Complexity",
    # Fractal dimension (1)
    "Petrosian_Fractal_Dimension",
    # Band power PSD (5)
    "Delta_PSD",
    "Theta_PSD",
    "Alpha_PSD",
    "Beta_PSD",
    "Gamma_PSD",
    # Band spectral entropy (5)
    "Delta_Entropy",
    "Theta_Entropy",
    "Alpha_Entropy",
    "Beta_Entropy",
    "Gamma_Entropy",
]


def get_16_features():
    """Return the list of 16 selected feature column names."""
    return FEATURES_16


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(project_root, "real_EEG_pretrained_model/features_mw_1789.csv")
    output_csv = os.path.join(project_root, "real_EEG_pretrained_model/features_mw_1789_selected16.csv")

    if not os.path.exists(input_csv):
        print(f"[error] Input file not found: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)

    # Determine sample id column: prefer 'SampleID'; fallback to first column.
    if "SampleID" in df.columns:
        sample_col = "SampleID"
    else:
        sample_col = df.columns[0]
        print(f"[warn] 'SampleID' not found; using first column as sample id: {sample_col}")

    needed_cols = [sample_col] + FEATURES_16
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        print(f"[error] Missing columns in input CSV: {missing}")
        sys.exit(1)

    df_out = df[needed_cols]
    df_out.to_csv(output_csv, index=False)
    print(f"[ok] Saved selected features to: {output_csv}")


if __name__ == "__main__":
    main()

