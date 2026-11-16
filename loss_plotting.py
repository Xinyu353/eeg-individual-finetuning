#!/usr/bin/env python3
"""
Utility helpers to plot and save training loss curves.

Designed to match the user's previous plotting style while integrating with the
current fine-tuning scripts. Supports raw D/G losses, smoothed versions, and
auxiliary curves (e.g., Wasserstein terms, ERP loss). All paths remain relative
to the project root for HPC compatibility.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

try:
    from scipy.interpolate import make_interp_spline
except Exception:  # pragma: no cover - SciPy may be unavailable in some envs
    make_interp_spline = None


def _select_font() -> None:
    """Pick a preferred sans-serif font if available."""
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Arial", "Nimbus Sans", "DejaVu Sans"):
        if candidate in available:
            mpl.rcParams["font.family"] = candidate
            break
    mpl.rcParams["font.size"] = 12


def _clean_loss_array(losses: Sequence[float]) -> np.ndarray:
    """Convert to numpy array and replace NaN/Inf with finite values."""
    arr = np.asarray(losses, dtype=np.float32)
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr)
    max_val = np.nanmax(np.where(finite_mask, arr, -np.inf))
    min_val = np.nanmin(np.where(finite_mask, arr, np.inf))
    return np.nan_to_num(arr, nan=0.0, posinf=max_val, neginf=min_val)


def _smooth_curve(values: Sequence[float], num_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Create a smoothed curve using B-spline when possible."""
    arr = _clean_loss_array(values)
    if arr.size == 0:
        return np.array([]), np.array([])

    epochs = np.arange(arr.size, dtype=np.float32)
    if arr.size == 1:
        return epochs, arr

    epochs_smooth = np.linspace(epochs.min(), epochs.max(), num_points)
    if make_interp_spline is not None and arr.size >= 4:
        spline = make_interp_spline(epochs, arr, k=3)
        values_smooth = spline(epochs_smooth)
    else:
        values_smooth = np.interp(epochs_smooth, epochs, arr)
    return epochs_smooth, values_smooth


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _plot_and_save(x, curves, labels, colors, outfile, title):
    """Generic helper for plotting multiple curves."""
    if not curves:
        return
    _select_font()
    plt.figure(figsize=(3, 3))
    for y, label, color in zip(curves, labels, colors):
        if y.size == 0:
            continue
        plt.plot(x if y.shape == x.shape else np.arange(y.size), y, label=label, color=color,
                 linewidth=1, linestyle='-', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(title)
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(outfile, format='png', dpi=600)
    plt.close()


def plot_losses(d_loss_epoch: Sequence[float], g_loss_epoch: Sequence[float], save_dir: str) -> None:
    """Plot raw discriminator/generator losses."""
    _ensure_dir(save_dir)
    d_arr = _clean_loss_array(d_loss_epoch)
    g_arr = _clean_loss_array(g_loss_epoch)
    if d_arr.size == 0 or g_arr.size == 0:
        return
    _select_font()
    plt.figure(figsize=(3, 3))
    epochs = np.arange(1, d_arr.size + 1)
    plt.plot(epochs, d_arr, label='D Loss', color='firebrick', linewidth=1, linestyle='-', alpha=0.7)
    plt.plot(epochs, g_arr, label='G Loss', color='navy', linewidth=1, linestyle='-', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss.png'), format='png', dpi=600)
    plt.close()


def plot_losses_smooth(d_loss_epoch: Sequence[float], g_loss_epoch: Sequence[float], save_dir: str) -> None:
    """Plot smoothed discriminator/generator losses."""
    _ensure_dir(save_dir)
    d_arr = _clean_loss_array(d_loss_epoch)
    g_arr = _clean_loss_array(g_loss_epoch)
    if d_arr.size == 0 or g_arr.size == 0:
        return
    d_x, d_smooth = _smooth_curve(d_arr)
    g_x, g_smooth = _smooth_curve(g_arr)
    if d_x.size == 0 or g_x.size == 0:
        return
    _select_font()
    plt.figure(figsize=(3, 3))
    plt.plot(d_x, d_smooth, label='D Loss', color='firebrick', linewidth=1, alpha=0.7)
    plt.plot(g_x, g_smooth, label='G Loss', color='navy', linewidth=1, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch (Smooth)')
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_smooth.png'), format='png', dpi=600)
    plt.close()


def plot_aux_losses_smooth(
    wd_d_loss: Sequence[float],
    wd_g_loss: Sequence[float],
    erp_loss: Sequence[float],
    save_dir: str,
) -> None:
    """Plot smoothed auxiliary losses (e.g., WD/ERP)."""
    _ensure_dir(save_dir)
    wd_d_arr = _clean_loss_array(wd_d_loss)
    wd_g_arr = _clean_loss_array(wd_g_loss)
    erp_arr = _clean_loss_array(erp_loss)
    if wd_d_arr.size == 0 and wd_g_arr.size == 0 and erp_arr.size == 0:
        return

    x_vals = None
    curves = []
    labels = []
    colors = []

    if wd_d_arr.size:
        x_vals, wd_d_smooth = _smooth_curve(wd_d_arr)
        curves.append(wd_d_smooth)
        labels.append('WD(D) Loss')
        colors.append('indianred')
    if wd_g_arr.size:
        if x_vals is None:
            x_vals, wd_g_smooth = _smooth_curve(wd_g_arr)
        else:
            _, wd_g_smooth = _smooth_curve(wd_g_arr)
        curves.append(wd_g_smooth)
        labels.append('WD(G) Loss')
        colors.append('teal')
    if erp_arr.size:
        if x_vals is None:
            x_vals, erp_smooth = _smooth_curve(erp_arr)
        else:
            _, erp_smooth = _smooth_curve(erp_arr)
        curves.append(erp_smooth)
        labels.append('ERP Loss')
        colors.append('purple')

    if not curves or x_vals is None or x_vals.size == 0:
        return

    _select_font()
    plt.figure(figsize=(3, 3))
    for y, label, color in zip(curves, labels, colors):
        plt.plot(x_vals, y, label=label, linewidth=1, alpha=0.7, color=color)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Auxiliary Loss per Epoch (Smooth)')
    plt.legend(loc='best', frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_aux_smooth.png'), format='png', dpi=600)
    plt.close()


def save_loss_plots(
    d_losses: Sequence[float],
    g_losses: Sequence[float],
    wd_d_losses: Sequence[float],
    wd_g_losses: Sequence[float],
    erp_losses: Sequence[float],
    output_dir: str,
    subdir: str = 'plots',
) -> None:
    """Public entry point to export all loss plots."""
    if not output_dir:
        return
    plot_dir = os.path.join(output_dir, subdir)
    _ensure_dir(plot_dir)
    if not g_losses or not d_losses:
        return

    plot_losses(d_losses, g_losses, plot_dir)
    plot_losses_smooth(d_losses, g_losses, plot_dir)
    plot_aux_losses_smooth(wd_d_losses, wd_g_losses, erp_losses, plot_dir)


__all__ = [
    'plot_losses',
    'plot_losses_smooth',
    'plot_aux_losses_smooth',
    'save_loss_plots',
]

