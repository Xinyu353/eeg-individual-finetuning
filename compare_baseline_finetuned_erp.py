#!/usr/bin/env python3
"""
Compare Baseline (pretrained) vs Fine-tuned ERP using fixed latent Z.

This script:
- Loads pretrained model (base_OT.pt) and fine-tuned model (from checkpoint)
- Generates ERP from fixed latent Z for both models
- Compares with Real ERP
- Saves comparison plot and metrics

Usage:
    python compare_baseline_finetuned_erp.py
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import json
import argparse
import glob
import re
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Get project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
sys.path.append(PROJECT_ROOT)

from lora import apply_lora_to_model
from wgan_cnn import Generator, Discriminator

# ============================================================================
# User Configuration
# ============================================================================
USER_CONFIG = {
    # Fine-tuned model directory (relative to PROJECT_ROOT)
    "model_dir": None,
    
    # Pretrained model path
    "pretrained_model": os.path.join(PROJECT_ROOT, 'real_EEG_pretrained_model', 'base_OT_2.pt'),
    
    # Real data CSV
    "csv_path": os.path.join(PROJECT_ROOT, 'real_EEG_pretrained_model', 'ot_vs_c_Pz_4753.csv'),
    "participant_id": None,  # Set explicitly for your own data.
    
    # Model config (should match training config)
    "latent_dim": 300,
    "eeg_length": 409,
    "cutoff_frequency": 40,
    "fs": 256,
    "time_start": -398,
    "time_end": 1195,
    "baseline_start": -0.398,
    "baseline_end": 0.0,
    
    # Fixed Z config
    "z_seed": 42,
    "num_trials": 256,  # Use min(num_real_trials, 256)
    
    # LoRA config (if fine-tuned model used LoRA)
    "use_lora": False,
    "lora_r": 2,
    "lora_alpha": 4.0,
    "lora_dropout": 0.05,
    "lora_targets": ("conv_first2", "final_linear"),
    "finetune_mode": "full",  # full / partial / lora
}

# ============================================================================
# Utility Functions
# ============================================================================

def make_time_axis(config: dict) -> np.ndarray:
    """Create time axis in seconds (matching visualization_utils.py)."""
    time_ms = np.linspace(config['time_start'], config['time_end'], config['eeg_length'])
    return time_ms / 1000.0  # Convert to seconds

def apply_baseline_window(x: np.ndarray, config: dict) -> np.ndarray:
    """Apply per-trial baseline correction to trials array [N, T]."""
    time_ms = np.linspace(config['time_start'], config['time_end'], config['eeg_length'])
    time_s = time_ms / 1000.0
    baseline_window = (time_s >= config['baseline_start']) & (time_s <= config['baseline_end'])
    baseline_means = x[:, baseline_window].mean(axis=1, keepdims=True)
    return x - baseline_means

def erp_from_trials(trials: np.ndarray, config: dict, real_min: float, real_max: float) -> np.ndarray:
    """
    Convert normalized trials to ERP: denormalize, baseline correct, then average.
    
    Args:
        trials: [N, T] array (normalized in [-1, 1])
        config: config dict
        real_min, real_max: denormalization range
    
    Returns:
        ERP array [T] in original scale (µV)
    """
    # Denormalize
    trials_denorm = (trials + 1.0) / 2.0 * (real_max - real_min) + real_min
    # Baseline correct
    trials_bc = apply_baseline_window(trials_denorm, config)
    # Average to ERP
    return trials_bc.mean(axis=0)

def save_np(path: str, arr: np.ndarray) -> None:
    """Save numpy array."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def load_np(path: str) -> Optional[np.ndarray]:
    """Load numpy array if exists."""
    if os.path.exists(path):
        return np.load(path)
    return None

def compute_mae(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Mean Absolute Error, optionally with mask."""
    if mask is not None:
        return float(np.mean(np.abs(a[mask] - b[mask])))
    return float(np.mean(np.abs(a - b)))

def save_metrics_json(path: str, metrics: dict) -> None:
    """Save metrics dict to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

def plot_three_curves(
    time: np.ndarray,  # Time in seconds (matching visualization_utils.py)
    real_mean: np.ndarray,
    real_std: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    finetuned_mean: np.ndarray,
    finetuned_std: np.ndarray,
    save_path: str,
    dpi: int = 600
) -> None:
    """
    Plot Real (black), Baseline (blue), Fine-tuned (red) ERP curves with std bands.
    Uses same style as visualization_utils.py (ERP plot style).
    """
    # Match visualization_utils.py ERP plot style exactly
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 32
    plt.rcParams['axes.labelsize'] = 32
    plt.rcParams['legend.fontsize'] = 16  # Smaller legend
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    
    f, ax = plt.subplots(1, 1, figsize=(6, 5), facecolor='white')
    
    # Plot order: real, fake, baseline (to match legend order)
    # Real (black) with std band
    ax.plot(time, real_mean, label='real', color='black', linewidth=1.5)
    ax.fill_between(time, real_mean - real_std, real_mean + real_std, color='black', alpha=0.1)
    
    # Fine-tuned (red) with std band - labeled as 'fake'
    ax.plot(time, finetuned_mean, label='fake', color='red', linewidth=1.5)
    ax.fill_between(time, finetuned_mean - finetuned_std, finetuned_mean + finetuned_std, color='red', alpha=0.1)
    
    # Baseline (blue) with std band
    ax.plot(time, baseline_mean, label='baseline', color='blue', linewidth=1.5, linestyle='--')
    ax.fill_between(time, baseline_mean - baseline_std, baseline_mean + baseline_std, color='blue', alpha=0.1)
    
    # Match visualization_utils.py ERP plot style exactly
    ax.set_xticks(np.arange(-0.4, 1.21, 0.2))
    ax.set_xticklabels([f'{i:.1f}' for i in np.arange(-0.4, 1.21, 0.2)])
    ax.set_ylabel('Voltage ($\\mu$V)')
    ax.set_xlabel('Time (s)')
    ax.set_title('ERP', loc='left')  # Simple title
    ax.legend(frameon=False, handlelength=1.5, handletextpad=0.3, labelspacing=0.3)
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.set_facecolor('white')
    f.patch.set_facecolor('white')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, format='png')
    plt.close()

def get_or_create_fixed_z(
    num_trials: int,
    latent_dim: int,
    device: str,
    seed: int = 42
) -> torch.Tensor:
    """
    Get or create fixed latent Z for reproducible ERP comparison.
    
    Args:
        num_trials: Number of trials (N)
        latent_dim: Latent dimension
        device: torch device
        seed: Random seed
    
    Returns:
        Z_fixed tensor [N, latent_dim] on device
    """
    z_path = os.path.join(PROJECT_ROOT, 'Z_fixed.npy')
    z_fixed = load_np(z_path)
    
    if z_fixed is None:
        # Create new fixed Z
        rng = np.random.RandomState(seed)
        z_fixed = rng.randn(num_trials, latent_dim).astype(np.float32)
        save_np(z_path, z_fixed)
        print(f"Created and saved fixed Z: {z_fixed.shape} to {z_path}")
    else:
        # Check shape compatibility
        if z_fixed.shape[1] != latent_dim:
            raise ValueError(f"Loaded Z_fixed has latent_dim={z_fixed.shape[1]}, expected {latent_dim}")
        if z_fixed.shape[0] < num_trials:
            raise ValueError(f"Loaded Z_fixed has {z_fixed.shape[0]} trials, need at least {num_trials}")
        # Use first N trials
        z_fixed = z_fixed[:num_trials]
        print(f"Loaded fixed Z: {z_fixed.shape} from {z_path}")
    
    return torch.tensor(z_fixed, device=device)

def generate_erp_from_fixed_z(
    generator: Generator,
    z_fixed: torch.Tensor,
    subject_token: Optional[torch.nn.Parameter],
    config: dict,
    real_min: float,
    real_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ERP mean and std from fixed Z using generator.
    Matches visualization_utils.py: per-trial baseline correction, then mean/std.
    
    Args:
        generator: Generator model
        z_fixed: Fixed latent [N, latent_dim]
        subject_token: Optional subject token to add to Z
        config: config dict
        real_min, real_max: Denormalization range
    
    Returns:
        (erp_mean, erp_std) arrays [T] in original scale
    """
    generator.eval()
    with torch.no_grad():
        z = z_fixed.clone()
        if subject_token is not None:
            z = z + subject_token
        
        fake = generator(z)
        if fake.dim() == 3:
            fake = fake.squeeze(1)
        
        fake_np = fake.cpu().numpy()
        
        # Debug: check normalized range
        print(f"  Generator output (normalized) range: [{fake_np.min():.4f}, {fake_np.max():.4f}]")
        
        # Denormalize: same formula as denormalize_data() in training script
        # Map from [-1, 1] back to original range [real_min, real_max]
        fake_denorm = (fake_np + 1.0) / 2.0 * (real_max - real_min) + real_min
        
        # Debug: check denormalized range
        print(f"  Fake denormalized range: [{fake_denorm.min():.2f}, {fake_denorm.max():.2f}] µV")
        print(f"  Using denormalization range: [{real_min:.2f}, {real_max:.2f}]")
        
        # Per-trial baseline correction (matching visualization_utils.py)
        # This matches how CSV files are processed: CSV is already denormalized, then baseline corrected
        time_ms = np.linspace(config['time_start'], config['time_end'], config['eeg_length'])
        time_s = time_ms / 1000.0
        baseline_window = (time_s >= config['baseline_start']) & (time_s <= config['baseline_end'])
        fake_denorm = fake_denorm - fake_denorm[:, baseline_window].mean(axis=1, keepdims=True)
        
        # Debug: check after baseline correction
        print(f"  After baseline correction range: [{fake_denorm.min():.2f}, {fake_denorm.max():.2f}] µV")
        
        # Compute mean and std (matching visualization_utils.py)
        erp_mean = fake_denorm.mean(axis=0)
        erp_std = fake_denorm.std(axis=0)
        
        # Debug: check ERP statistics
        print(f"  ERP mean absolute value: {np.mean(np.abs(erp_mean)):.2f} µV")
        print(f"  ERP mean range: [{erp_mean.min():.2f}, {erp_mean.max():.2f}] µV")
        print(f"  ERP std mean: {erp_std.mean():.2f} µV")
    
    return erp_mean, erp_std

def compute_erp_from_csv(csv_path: str, config: dict, real_min: float, real_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ERP mean and std from generated data CSV file.
    CSV files from training are already denormalized (original scale).
    Matches visualization_utils.py: per-trial baseline correction, then mean/std.
    
    Args:
        csv_path: Path to generated data CSV file
        config: config dict
        real_min, real_max: Data range (for consistency check, CSV should already be denormalized)
    
    Returns:
        (erp_mean, erp_std) arrays [T] in original scale
    """
    # Load generated data (already denormalized from training script)
    df = pd.read_csv(csv_path, header=None)
    trials = df.values.astype(np.float32)
    
    # Remove NaN/Inf
    valid_mask = np.isfinite(trials).all(axis=1)
    trials = trials[valid_mask]
    
    if trials.shape[0] == 0:
        raise ValueError(f"No valid trials in {csv_path}")
    
    if trials.shape[1] != config['eeg_length']:
        raise ValueError(f"Expected {config['eeg_length']} columns, got {trials.shape[1]}")
    
    # CSV data is already in original scale (denormalized), so no need to denormalize
    # Just apply per-trial baseline correction (matching visualization_utils.py)
    time_ms = np.linspace(config['time_start'], config['time_end'], config['eeg_length'])
    time_s = time_ms / 1000.0
    baseline_window = (time_s >= config['baseline_start']) & (time_s <= config['baseline_end'])
    trials = trials - trials[:, baseline_window].mean(axis=1, keepdims=True)
    
    # Compute mean and std (matching visualization_utils.py)
    erp_mean = trials.mean(axis=0)
    erp_std = trials.std(axis=0)
    
    return erp_mean, erp_std

def load_pretrained_generator(model_path: str, config: dict, device: str) -> Tuple[Generator, Optional[float], Optional[float]]:
    """
    Load pretrained generator and extract data range if available.
    
    Returns:
        (G, real_min, real_max) - real_min/real_max may be None if not in checkpoint
    """
    print(f'Loading pretrained model from: {model_path}')
    
    G = Generator(
        latent_dim=config['latent_dim'],
        eeg_length=config['eeg_length'],
        cutoff_frequency=config['cutoff_frequency'],
        fs=config['fs']
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location='cpu')
    
    # Debug: print checkpoint keys
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    
    # Try different checkpoint formats
    loaded = False
    if 'generator_state_dict' in ckpt:
        G.load_state_dict(ckpt['generator_state_dict'])
        print("✓ Loaded generator from checkpoint (key: 'generator_state_dict')")
        loaded = True
    elif 'generator' in ckpt:
        G.load_state_dict(ckpt['generator'])
        print("✓ Loaded generator from checkpoint (key: 'generator')")
        loaded = True
    elif 'model' in ckpt:
        G.load_state_dict(ckpt['model'])
        print("✓ Loaded generator from checkpoint (key: 'model')")
        loaded = True
    else:
        # Check if it's a direct state dict (no wrapper)
        try:
            G.load_state_dict(ckpt)
            print("✓ Loaded generator directly from checkpoint (assumed to be state_dict)")
            loaded = True
        except Exception as e:
            print(f"✗ Failed to load generator: {e}")
            raise
    
    if not loaded:
        raise ValueError("Could not load generator from checkpoint")
    
    # Try to get data range from checkpoint
    real_min = None
    real_max = None
    if 'real_min' in ckpt:
        try:
            real_min = float(ckpt['real_min'])
            print(f"✓ Found real_min in checkpoint: {real_min:.2f}")
        except Exception:
            pass
    if 'real_max' in ckpt:
        try:
            real_max = float(ckpt['real_max'])
            print(f"✓ Found real_max in checkpoint: {real_max:.2f}")
        except Exception:
            pass
    
    if real_min is None or real_max is None:
        print("⚠ No data range (real_min/real_max) found in checkpoint, will use real data range")
    else:
        print(f"✓ Using data range from pretrained checkpoint: [{real_min:.2f}, {real_max:.2f}]")
    
    G = G.to(device)
    G.eval()
    return G, real_min, real_max

def extract_epoch_number(filename: str) -> Optional[int]:
    """Extract epoch number from filename."""
    match = re.search(r'epoch_(\d+)\.pth', filename)
    if match:
        return int(match.group(1))
    return None

def load_finetuned_generator(model_dir: str, config: dict, device: str, checkpoint_path: Optional[str] = None) -> Tuple[Generator, Optional[torch.Tensor], float, float]:
    """
    Load fine-tuned generator from checkpoint.
    
    Args:
        model_dir: Model directory
        config: Config dict
        device: torch device
        checkpoint_path: Optional specific checkpoint path (if None, uses latest)
    
    Returns:
        (generator, subject_token, real_min, real_max)
    """
    models_dir = os.path.join(model_dir, 'models')
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    if checkpoint_path is None:
        # Find latest checkpoint
        checkpoint_files = [f for f in os.listdir(models_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {models_dir}")
        
        # Get latest checkpoint
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(models_dir, checkpoint_files[-1])
    
    print(f"Loading fine-tuned model from: {checkpoint_path}")
    
    G = Generator(
        latent_dim=config['latent_dim'],
        eeg_length=config['eeg_length'],
        cutoff_frequency=config['cutoff_frequency'],
        fs=config['fs']
    )
    
    # Apply LoRA if needed
    if config.get('use_lora', False) and config.get('finetune_mode', 'full') == 'lora':
        G = apply_lora_to_model(
            G,
            r=config['lora_r'],
            alpha=config['lora_alpha'],
            dropout=config['lora_dropout'],
            which=config['lora_targets'],
        )
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    G.load_state_dict(ckpt['generator_state_dict'])
    G = G.to(device)
    G.eval()
    
    # Load subject_token if exists
    subject_token = None
    if 'subject_token' in ckpt and ckpt['subject_token'] is not None:
        subject_token = ckpt['subject_token'].to(device)
        print("Loaded subject_token from checkpoint")
    
    # Try to get data range (optional)
    real_min = None
    real_max = None
    
    # Try to get from checkpoint
    if 'real_min' in ckpt:
        try:
            real_min = float(ckpt['real_min'])
        except Exception:
            real_min = None
    if 'real_max' in ckpt:
        try:
            real_max = float(ckpt['real_max'])
        except Exception:
            real_max = None
    
    # Try training summary (common filename is training_summary.txt)
    if (real_min is None or real_max is None):
        summary_path = os.path.join(model_dir, 'training_summary.txt')
        if os.path.exists(summary_path):
            import re
            with open(summary_path, 'r') as f:
                content = f.read()
            m_min = re.search(r'Data range\s*:\s*\[([-\d.]+)', content)
            m_max = re.search(r'Data range\s*:\s*\[[^,]+,\s*([-\d.]+)\]', content)
            if m_min:
                real_min = float(m_min.group(1))
            if m_max:
                real_max = float(m_max.group(1))
    
    # Do NOT raise here; let main() fall back to real data range if needed
    if real_min is None or real_max is None:
        print("[warn] real_min/real_max not found in checkpoint or summary; will fall back to real-data range in main().")
    else:
        print(f"Data range from checkpoint/summary: [{real_min:.2f}, {real_max:.2f}]")
    
    return G, subject_token, real_min, real_max

def load_real_data(csv_path: str, participant_id: int, config: dict) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Load real data and compute ERP mean and std.
    Matches visualization_utils.py: per-trial baseline correction, then mean/std.
    
    Returns:
        (real_erp_mean, real_erp_std, real_min, real_max)
    """
    print(f'Loading real data from: {csv_path}')
    print(f'Filtering rows where first column == {participant_id}')
    
    df = pd.read_csv(csv_path, header=None)
    filtered_df = df[df.iloc[:, 0] == participant_id]
    print(f"Participant {participant_id}: Found {len(filtered_df)} rows")
    
    X = filtered_df.iloc[:, 1:].values
    
    if X.shape[1] != config['eeg_length']:
        raise ValueError(f"Expected {config['eeg_length']} columns, got {X.shape[1]}")
    
    # Remove NaN/Inf
    valid_mask = np.isfinite(X).all(axis=1)
    X = X[valid_mask]
    print(f"Participant {participant_id}: {len(X)} trials after removing NaN/Inf")
    
    # Per-trial baseline correction (matching visualization_utils.py)
    time_ms = np.linspace(config['time_start'], config['time_end'], config['eeg_length'])
    time_s = time_ms / 1000.0
    baseline_window = (time_s >= config['baseline_start']) & (time_s <= config['baseline_end'])
    X = X - X[:, baseline_window].mean(axis=1, keepdims=True)
    
    # Compute range
    real_min = float(X.min())
    real_max = float(X.max())
    
    # Compute ERP mean and std (matching visualization_utils.py)
    real_erp_mean = X.mean(axis=0)
    real_erp_std = X.std(axis=0)
    
    return real_erp_mean, real_erp_std, real_min, real_max

# ============================================================================
# Main
# ============================================================================

def process_epoch_erp_comparison(
    model_dir: str,
    epoch_num: Optional[int],
    csv_path: str,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    real_mean: np.ndarray,
    real_std: np.ndarray,
    config: dict
) -> None:
    """
    Process ERP comparison for a specific epoch using generated data CSV.
    
    Args:
        model_dir: Model directory
        epoch_num: Epoch number (None for final)
        csv_path: Path to generated data CSV file
        baseline_mean, baseline_std: Baseline ERP mean and std (from pretrained model)
        real_mean, real_std: Real ERP mean and std
        config: Config dict
    """
    if epoch_num is None:
        print(f"\n--- Processing Final Generated Trials ---")
        epoch_output_dir = os.path.join(model_dir, 'visual', 'final')
    else:
        print(f"\n--- Processing Epoch {epoch_num} ---")
        epoch_output_dir = os.path.join(model_dir, 'visual', f'epoch_{epoch_num}')
    
    try:
        # Compute Fine-tuned ERP mean and std from generated data CSV
        # Note: CSV files are already denormalized, so no need to denormalize again
        finetuned_mean, finetuned_std = compute_erp_from_csv(csv_path, config, real_min=0.0, real_max=0.0)
        
        # Compute metrics (using means)
        time_s = make_time_axis(config)  # Time in seconds
        time_ms = time_s * 1000.0  # Convert to ms for window mask
        erp_window_mask = (time_ms >= 80) & (time_ms <= 300)
        
        mae_full_baseline = compute_mae(real_mean, baseline_mean)
        mae_full_finetuned = compute_mae(real_mean, finetuned_mean)
        mae_80_300_baseline = compute_mae(real_mean, baseline_mean, erp_window_mask)
        mae_80_300_finetuned = compute_mae(real_mean, finetuned_mean, erp_window_mask)
        
        # Count trials in CSV
        df = pd.read_csv(csv_path, header=None)
        n_trials = len(df)
        
        metrics = {
            "mae_full": {
                "baseline": mae_full_baseline,
                "finetuned": mae_full_finetuned
            },
            "mae_80_300": {
                "baseline": mae_80_300_baseline,
                "finetuned": mae_80_300_finetuned
            },
            "epoch": epoch_num if epoch_num is not None else "final",
            "N": n_trials
        }
        
        # Save ERPs and metrics
        save_np(os.path.join(epoch_output_dir, 'baseline_erp_mean.npy'), baseline_mean)
        save_np(os.path.join(epoch_output_dir, 'baseline_erp_std.npy'), baseline_std)
        save_np(os.path.join(epoch_output_dir, 'real_erp_mean.npy'), real_mean)
        save_np(os.path.join(epoch_output_dir, 'real_erp_std.npy'), real_std)
        save_np(os.path.join(epoch_output_dir, 'finetuned_erp_mean.npy'), finetuned_mean)
        save_np(os.path.join(epoch_output_dir, 'finetuned_erp_std.npy'), finetuned_std)
        save_metrics_json(os.path.join(epoch_output_dir, 'erp_metrics.json'), metrics)
        
        # Plot comparison
        plot_three_curves(
            time=time_s,  # Time in seconds
            real_mean=real_mean,
            real_std=real_std,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            finetuned_mean=finetuned_mean,
            finetuned_std=finetuned_std,
            save_path=os.path.join(epoch_output_dir, 'erp_compare.png'),
            dpi=600
        )
        
        if epoch_num is None:
            print(f"✓ Final generated trials processed successfully")
        else:
            print(f"✓ Epoch {epoch_num} processed successfully")
            
    except Exception as e:
        if epoch_num is None:
            print(f"✗ Error processing final: {e}")
        else:
            print(f"✗ Error processing epoch {epoch_num}: {e}")

def extract_epoch_number_from_csv(filename: str) -> Optional[int]:
    """Extract epoch number from CSV filename."""
    match = re.search(r'epoch_(\d+)\.csv', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description='Compare Baseline vs Fine-tuned ERP for all epochs')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Fine-tuned model directory (relative to PROJECT_ROOT)')
    parser.add_argument('--participant_id', type=int, default=None,
                        help='Participant ID')
    args = parser.parse_args()
    
    # Override config with command line args
    if args.model_dir:
        USER_CONFIG['model_dir'] = args.model_dir
    if args.participant_id is not None:
        USER_CONFIG['participant_id'] = args.participant_id
    
    if USER_CONFIG['participant_id'] is None:
        raise ValueError('Specify --participant_id for your own data.')

    if USER_CONFIG['model_dir'] is None:
        raise ValueError("Please specify model_dir in USER_CONFIG or use --model_dir argument")
    
    model_dir = os.path.join(PROJECT_ROOT, USER_CONFIG['model_dir'])
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print('=' * 80)
    print('ERP Comparison: Baseline vs Fine-tuned (All Epochs)')
    print('=' * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load pretrained generator (for Baseline ERP - no fine-tuning)
    print("\n" + "="*60)
    print("Loading pretrained model (for Baseline ERP)...")
    print("="*60)
    G_pretrained, pretrained_real_min, pretrained_real_max = load_pretrained_generator(
        USER_CONFIG['pretrained_model'],
        USER_CONFIG,
        device
    )
    
    # Load real data
    print("\n" + "="*60)
    print("Loading real data...")
    print("="*60)
    real_mean, real_std, real_min_data, real_max_data = load_real_data(
        USER_CONFIG['csv_path'],
        USER_CONFIG['participant_id'],
        USER_CONFIG
    )
    
    # Determine denormalization range for baseline ERP
    # Priority: pretrained checkpoint range > real data range
    if pretrained_real_min is not None and pretrained_real_max is not None:
        baseline_real_min = pretrained_real_min
        baseline_real_max = pretrained_real_max
        print(f"\n✓ Using pretrained checkpoint range for baseline denormalization: [{baseline_real_min:.2f}, {baseline_real_max:.2f}]")
    else:
        baseline_real_min = real_min_data
        baseline_real_max = real_max_data
        print(f"\n⚠ Pretrained checkpoint has no data range, using real data range for baseline: [{baseline_real_min:.2f}, {baseline_real_max:.2f}]")
    
    # For fine-tuned CSV files, they should be in the same scale as real data
    finetuned_real_min = real_min_data
    finetuned_real_max = real_max_data
    print(f"Using real data range for fine-tuned CSV (already denormalized): [{finetuned_real_min:.2f}, {finetuned_real_max:.2f}]")
    
    # Determine N for Baseline ERP generation
    df_all = pd.read_csv(USER_CONFIG['csv_path'], header=None)
    num_real_trials = len(df_all[df_all.iloc[:, 0] == USER_CONFIG['participant_id']])
    N = min(num_real_trials, USER_CONFIG['num_trials'])
    print(f"\nUsing N={N} trials for Baseline ERP generation")
    
    # Get or create fixed Z (for Baseline ERP generation)
    print("\n" + "="*60)
    print("Setting up fixed latent Z (for Baseline ERP)...")
    print("="*60)
    z_fixed = get_or_create_fixed_z(N, USER_CONFIG['latent_dim'], device, seed=USER_CONFIG['z_seed'])
    
    # Generate Baseline ERP (same for all epochs - from pretrained model)
    print("\n" + "="*60)
    print("Generating Baseline ERP from pretrained model...")
    print("="*60)
    print(f"Using denormalization range: [{baseline_real_min:.2f}, {baseline_real_max:.2f}]")
    
    baseline_mean, baseline_std = generate_erp_from_fixed_z(
        generator=G_pretrained,
        z_fixed=z_fixed,
        subject_token=None,  # Baseline: no subject_token (pre-individualized)
        config=USER_CONFIG,
        real_min=baseline_real_min,
        real_max=baseline_real_max
    )
    
    # Find all generated data CSV files
    generated_data_dir = os.path.join(model_dir, 'generated_data')
    if not os.path.exists(generated_data_dir):
        raise FileNotFoundError(f"Generated data directory not found: {generated_data_dir}")
    
    # Find epoch CSV files
    csv_pattern = os.path.join(generated_data_dir, "generated_trials_epoch_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    # Sort by epoch number
    csv_files_with_epoch = []
    for csv_file in csv_files:
        epoch_num = extract_epoch_number_from_csv(os.path.basename(csv_file))
        if epoch_num is not None:
            csv_files_with_epoch.append((epoch_num, csv_file))
    
    csv_files_with_epoch.sort(key=lambda x: x[0])
    
    # Also check for final_generated_trials.csv
    final_file = os.path.join(generated_data_dir, "final_generated_trials.csv")
    if os.path.exists(final_file):
        csv_files_with_epoch.append((None, final_file))  # None means final
    
    if not csv_files_with_epoch:
        raise FileNotFoundError(f"No generated data CSV files found in {generated_data_dir}")
    
    print(f"\nFound {len(csv_files_with_epoch)} generated data files to process")
    
    # Process each epoch
    for epoch_num, csv_path in csv_files_with_epoch:
        process_epoch_erp_comparison(
            model_dir=model_dir,
            epoch_num=epoch_num,
            csv_path=csv_path,
            baseline_mean=baseline_mean,  # Same baseline for all epochs
            baseline_std=baseline_std,
            real_mean=real_mean,
            real_std=real_std,
            config=USER_CONFIG
        )
    
    print(f"\n{'='*80}")
    print("All epochs processed!")
    print(f"Results saved in: {os.path.join(model_dir, 'visual')}")
    print('=' * 80)

if __name__ == '__main__':
    main()

