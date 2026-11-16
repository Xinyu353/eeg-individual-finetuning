# Evaluation and analysis

Reusable distribution comparisons, distance metrics, feature-deviation predictors, correlations, and plotting utilities. No participant results, plots, or derived tables are included.

- `compute_swd_mmd.py`: SWD, RBF-MMD, and time-point Wasserstein distances.
- `distribution_*.py`: feature-distribution comparisons.
- `participant_fake_overlay.py`: configurable real/generated distribution overlays.
- `build_round1_all_predictors_table.py`, `enrich_participant_round1_real.py`, and `round1_*.py`: participant-versus-group predictors and outcome analyses.
- `plot_*.py`, `summarize_heldout_all_quality_metrics.py`, and `export_thesis_participant_metric_tables.py`: plots and summaries from user-supplied intermediate results.

Run scripts from the repository root. Some summary scripts require multi-seed tables produced separately; the multi-seed orchestration code was not present in the supplied source. Configure local input paths before running.
