# Feature extraction

Reusable EEG feature calculations and batch extraction utilities. No feature tables or EEG samples are included.

- `feature_extraction.py`: ERP, Hjorth, spectral, entropy, and fractal feature functions.
- `batch_extract_real_features.py`: features from local participant trial files.
- `batch_extract_selection_features.py`: features from locally selected generated trials.
- `extract_*.py`: configurable extraction workflows for individual trials and averaged ERPs.
- `select_feature*.py`: feature-column selection utilities.

Run scripts from the repository root. Review input paths, sampling settings, and each script's configuration before use.
