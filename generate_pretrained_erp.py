#!/usr/bin/env python3
"""
Generate ERP from two pretrained models (base_OT.pt and base_MW.pt).
Each model generates 1789 trials, then compute and plot ERP.
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Get project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
sys.path.append(PROJECT_ROOT)

from wgan_cnn import Generator

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Pretrained model paths
    "model_ot": os.path.join(PROJECT_ROOT, 'real_EEG_pretrained_model', 'base_OT_2.pt'),
    "model_mw": os.path.join(PROJECT_ROOT, 'real_EEG_pretrained_model', 'base_MW.pt'),
    
    # Model config
    "latent_dim": 300,
    "eeg_length": 409,
    "cutoff_frequency": 40,
    "fs": 256,
    "time_start": -398,
    "time_end": 1195,
    "baseline_start": -0.398,
    "baseline_end": 0.0,
    
    # Generation config
    "num_trials": 1789,
    "z_seed": 42,
    "batch_size": 200,
    
    # Output
    "output_dir": os.path.join(PROJECT_ROOT, 'pretrained_erp_comparison'),
}

def load_pretrained_generator(model_path: str, config: dict, device: str):
    """Load pretrained generator from checkpoint."""
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
    
    # Try different checkpoint formats
    if 'generator_state_dict' in ckpt:
        G.load_state_dict(ckpt['generator_state_dict'])
        print("✓ Loaded generator from checkpoint (key: 'generator_state_dict')")
    elif 'generator' in ckpt:
        G.load_state_dict(ckpt['generator'])
        print("✓ Loaded generator from checkpoint (key: 'generator')")
    elif 'model' in ckpt:
        G.load_state_dict(ckpt['model'])
        print("✓ Loaded generator from checkpoint (key: 'model')")
    else:
        G.load_state_dict(ckpt)
        print("✓ Loaded generator directly from checkpoint")
    
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
    
    G = G.to(device)
    G.eval()
    return G, real_min, real_max

def generate_trials(generator, num_trials: int, latent_dim: int, device: str, 
                    seed: int = 42, batch_size: int = 200):
    """Generate trials from generator using fixed random seed."""
    generator.eval()
    rng = np.random.RandomState(seed)
    
    trials = []
    with torch.no_grad():
        for i in range(0, num_trials, batch_size):
            b = min(batch_size, num_trials - i)
            z = torch.tensor(
                rng.randn(b, latent_dim).astype(np.float32),
                device=device
            )
            fake = generator(z)
            if fake.dim() == 3:
                fake = fake.squeeze(1)
            trials.append(fake.cpu().numpy())
    
    return np.concatenate(trials, axis=0)

def denormalize_trials(trials: np.ndarray, real_min: float, real_max: float) -> np.ndarray:
    """Denormalize trials from [-1, 1] to original range."""
    return (trials + 1.0) / 2.0 * (real_max - real_min) + real_min

def apply_baseline_correction(trials: np.ndarray, config: dict) -> np.ndarray:
    """Apply per-trial baseline correction."""
    time_ms = np.linspace(config['time_start'], config['time_end'], config['eeg_length'])
    time_s = time_ms / 1000.0
    baseline_window = (time_s >= config['baseline_start']) & (time_s <= config['baseline_end'])
    
    # Per-trial baseline correction
    baseline_means = trials[:, baseline_window].mean(axis=1, keepdims=True)
    trials_bc = trials - baseline_means
    return trials_bc

def compute_erp(trials: np.ndarray) -> tuple:
    """Compute ERP mean and std from trials."""
    erp_mean = trials.mean(axis=0)
    erp_std = trials.std(axis=0)
    return erp_mean, erp_std

def plot_erp_comparison(time: np.ndarray, ot_mean: np.ndarray, ot_std: np.ndarray,
                       mw_mean: np.ndarray, mw_std: np.ndarray, save_path: str):
    """Plot ERP comparison between OT and MW."""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 32
    plt.rcParams['axes.labelsize'] = 32
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    
    f, ax = plt.subplots(1, 1, figsize=(6, 5), facecolor='white')
    
    # OT (blue)
    ax.plot(time, ot_mean, label='OT', color='blue', linewidth=1.5)
    ax.fill_between(time, ot_mean - ot_std, ot_mean + ot_std, color='blue', alpha=0.1)
    
    # MW (red)
    ax.plot(time, mw_mean, label='MW', color='red', linewidth=1.5)
    ax.fill_between(time, mw_mean - mw_std, mw_mean + mw_std, color='red', alpha=0.1)
    
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Axis settings
    ax.set_xticks(np.arange(-0.4, 1.21, 0.2))
    ax.set_xticklabels([f'{i:.1f}' for i in np.arange(-0.4, 1.21, 0.2)])
    ax.set_ylabel('Voltage ($\\mu$V)')
    ax.set_xlabel('Time (s)')
    ax.set_title('ERP', loc='left')
    ax.legend(frameon=False)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.set_facecolor('white')
    f.patch.set_facecolor('white')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, format='png')
    plt.close()
    print(f"✓ Saved ERP comparison plot: {save_path}")

def main():
    print('=' * 80)
    print('Generate ERP from Pretrained Models (OT vs MW)')
    print('=' * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Time axis
    time_ms = np.linspace(CONFIG['time_start'], CONFIG['time_end'], CONFIG['eeg_length'])
    time_s = time_ms / 1000.0
    
    # Process OT model
    print("\n" + "="*60)
    print("Processing OT Model")
    print("="*60)
    G_ot, ot_real_min, ot_real_max = load_pretrained_generator(
        CONFIG['model_ot'], CONFIG, device
    )
    
    if ot_real_min is None or ot_real_max is None:
        print("⚠ Warning: OT model has no real_min/real_max in checkpoint.")
        print("  Using default range [-50, 50] µV")
        ot_real_min, ot_real_max = -50.0, 50.0
    else:
        print(f"✓ Using data range: [{ot_real_min:.2f}, {ot_real_max:.2f}]")
    
    print(f"\nGenerating {CONFIG['num_trials']} trials from OT model...")
    ot_trials_norm = generate_trials(
        G_ot, CONFIG['num_trials'], CONFIG['latent_dim'], 
        device, seed=CONFIG['z_seed'], batch_size=CONFIG['batch_size']
    )
    print(f"  Generated trials shape: {ot_trials_norm.shape}")
    print(f"  Normalized range: [{ot_trials_norm.min():.4f}, {ot_trials_norm.max():.4f}]")
    
    # Denormalize OT trials
    ot_trials = denormalize_trials(ot_trials_norm, ot_real_min, ot_real_max)
    print(f"  Denormalized range: [{ot_trials.min():.2f}, {ot_trials.max():.2f}] µV")
    
    # Baseline correction
    ot_trials_bc = apply_baseline_correction(ot_trials, CONFIG)
    print(f"  After baseline correction range: [{ot_trials_bc.min():.2f}, {ot_trials_bc.max():.2f}] µV")
    
    # Compute ERP
    ot_erp_mean, ot_erp_std = compute_erp(ot_trials_bc)
    print(f"  ERP mean range: [{ot_erp_mean.min():.2f}, {ot_erp_mean.max():.2f}] µV")
    print(f"  ERP mean absolute value: {np.mean(np.abs(ot_erp_mean)):.2f} µV")
    
    # Process MW model
    print("\n" + "="*60)
    print("Processing MW Model")
    print("="*60)
    G_mw, mw_real_min, mw_real_max = load_pretrained_generator(
        CONFIG['model_mw'], CONFIG, device
    )
    
    if mw_real_min is None or mw_real_max is None:
        print("⚠ Warning: MW model has no real_min/real_max in checkpoint.")
        print("  Using default range [-50, 50] µV")
        mw_real_min, mw_real_max = -50.0, 50.0
    else:
        print(f"✓ Using data range: [{mw_real_min:.2f}, {mw_real_max:.2f}]")
    
    print(f"\nGenerating {CONFIG['num_trials']} trials from MW model...")
    mw_trials_norm = generate_trials(
        G_mw, CONFIG['num_trials'], CONFIG['latent_dim'], 
        device, seed=CONFIG['z_seed'], batch_size=CONFIG['batch_size']
    )
    print(f"  Generated trials shape: {mw_trials_norm.shape}")
    print(f"  Normalized range: [{mw_trials_norm.min():.4f}, {mw_trials_norm.max():.4f}]")
    
    # Denormalize MW trials
    mw_trials = denormalize_trials(mw_trials_norm, mw_real_min, mw_real_max)
    print(f"  Denormalized range: [{mw_trials.min():.2f}, {mw_trials.max():.2f}] µV")
    
    # Baseline correction
    mw_trials_bc = apply_baseline_correction(mw_trials, CONFIG)
    print(f"  After baseline correction range: [{mw_trials_bc.min():.2f}, {mw_trials_bc.max():.2f}] µV")
    
    # Compute ERP
    mw_erp_mean, mw_erp_std = compute_erp(mw_trials_bc)
    print(f"  ERP mean range: [{mw_erp_mean.min():.2f}, {mw_erp_mean.max():.2f}] µV")
    print(f"  ERP mean absolute value: {np.mean(np.abs(mw_erp_mean)):.2f} µV")
    
    # Save data
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    # Save ERP data
    erp_df = pd.DataFrame({
        'time_s': time_s,
        'ot_mean': ot_erp_mean,
        'ot_std': ot_erp_std,
        'mw_mean': mw_erp_mean,
        'mw_std': mw_erp_std
    })
    erp_csv_path = os.path.join(CONFIG['output_dir'], 'erp_comparison.csv')
    erp_df.to_csv(erp_csv_path, index=False)
    print(f"✓ Saved ERP data: {erp_csv_path}")
    
    # Save trials (optional, can be large)
    # ot_trials_df = pd.DataFrame(ot_trials_bc)
    # ot_trials_df.to_csv(os.path.join(CONFIG['output_dir'], 'ot_trials_baseline_corrected.csv'), index=False, header=False)
    # mw_trials_df = pd.DataFrame(mw_trials_bc)
    # mw_trials_df.to_csv(os.path.join(CONFIG['output_dir'], 'mw_trials_baseline_corrected.csv'), index=False, header=False)
    
    # Plot comparison
    plot_path = os.path.join(CONFIG['output_dir'], 'erp_comparison.png')
    plot_erp_comparison(time_s, ot_erp_mean, ot_erp_std, mw_erp_mean, mw_erp_std, plot_path)
    
    print("\n" + "="*80)
    print("✓ All done!")
    print(f"Results saved in: {CONFIG['output_dir']}")
    print("="*80)

if __name__ == "__main__":
    main()

