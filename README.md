# Individual Fine-tuning of Group-pretrained EEG Generative Models

General-purpose research code accompanying **Variability in Individual Fine-tuning of Group-pretrained EEG Generative Models**.

The code supports participant-wise fine-tuning of group-pretrained EEG generators, feature extraction, and real-versus-generated signal evaluation for on-task (OT) and mind-wandering (MW) conditions.

## Scope

This is a **code-only release**. It contains 43 Python scripts. Participant EEG, generated trials, extracted feature tables, participant-level results, figures, training logs, and model checkpoints are not included. Obtain appropriate data and compatible pretrained models separately.

The original directory did not contain group-pretraining code. This repository therefore starts from externally supplied pretrained checkpoints. It does not by itself establish that group pretraining excluded held-out trials or reproduce all manuscript results.

## Main components

| Component | Purpose |
| --- | --- |
| `wgan_cnn.py`, `filter.py` | CNN generator/discriminator and frequency filtering |
| `loss.py`, `loss_plotting.py` | Training losses and loss plots |
| `lora.py` | Low-rank adapters for linear and convolutional layers |
| `finetune_ot_participant.py` | Participant-wise fine-tuning for OT or MW |
| `heldout_80_20_pipeline.py` | Calibration/held-out split, checkpoint selection, and evaluation |
| `feature_extraction/` | ERP, Hjorth, spectral, entropy, and fractal features |
| `evaluation/` | Distribution distances, PCA/t-SNE comparisons, and predictor analyses |
| Other root scripts | ERP comparisons, distribution plots, and diagnostics |

## Environment

Use Python 3.10 or newer. Install the declared dependencies in a separate environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency file is not a locked reproduction environment. Full training and numerical reproduction have not been validated for this code-only release; all included Python files passed syntax compilation.

## Local inputs

The main fine-tuning and held-out pipelines expect headerless CSV rows containing a numeric participant ID followed by **409 EEG time points**. Defaults assume a sampling frequency of **256 Hz**, a time window from **-398 to 1195 ms**, and baseline correction over **-398 to 0 ms**. Review these settings before using another dataset.

The held-out pipeline's default local input locations are:

```text
real_EEG_pretrained_model/
  ot_vs_c_Pz_4753.csv
  mw_vs_Pz_1789.csv
  base_OT.pt
  base_MW.pt
```

These are expected filenames, not bundled data. Set `CONDITION_CSV` and `CONDITION_MODEL` in `heldout_80_20_pipeline.py` if your files are elsewhere. Checkpoints must match the model architecture and loading format in the scripts.

## Usage

Run commands from the repository root. These examples require your own local inputs.

Fine-tune an OT model with explicit input paths:

```bash
python finetune_ot_participant.py \
  --condition ot \
  --csv-path /path/to/your/ot_trials.csv \
  --model-path /path/to/your/base_OT.pt \
  --output-dir results/ot_finetuning \
  --finetune-mode full
```

For MW, use `--condition mw` with the corresponding CSV and pretrained checkpoint.

Run the held-out workflow after configuring its input paths:

```bash
python heldout_80_20_pipeline.py \
  --conditions ot mw \
  --output-dir results/heldout_80_20
```

This workflow uses calibration trials for fine-tuning, checkpoint selection, and predictors, then evaluates the selected model on held-out trials. Use `--participants` to select IDs from your own data. The original fixed pilot participant list has been removed. The default minimum is 100 valid trials per participant; inspect the script's constants and `--help` before running.

Extract features from local participant trial files:

```bash
python feature_extraction/batch_extract_real_features.py \
  --condition ot \
  --participant-data-dir /path/to/your/participant_data \
  --output-dir results/real_features
```

Additional scripts may expect intermediate tables or the original local directory layout. Use their command-line options or edit their configuration constants; no result tables are supplied.

## Release preparation

- Original model architectures, loss formulas, feature definitions, and analysis calculations were retained.
- Two machine-specific absolute paths were replaced with paths relative to the repository.
- Fixed participant examples and selected case-study figures were omitted; ERP comparison now requires an explicit participant ID and model directory.
- The supplied `lora.py` implementation is included. For LoRA fine-tuning, use `--finetune-mode lora --use-lora` and review the adapter configuration.
- Two wrappers depending on absent local modules were omitted. The supplied files also did not include the multi-seed orchestration script used by some downstream summary plots.
- `scikit-learn` and `statsmodels` were added to the dependency list because the included scripts import them.
- `.gitignore` restricts tracked content to code, README files, and the dependency file, and excludes data/model/output locations.

The source data and original files remain separate from this release. Do not add participant data or checkpoints to this repository.
