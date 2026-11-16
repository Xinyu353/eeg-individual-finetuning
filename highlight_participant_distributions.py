#!/usr/bin/env python3
"""
highlight_participant_distributions.py

遍历所有参与者，每次选中一个参与者用红色高亮，其余样本用黑色，
在同一套 PCA / t-SNE 投影上绘制分布，保证整体分布一致，仅高亮位置变化。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from matplotlib.lines import Line2D

# 配置
FEATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_EEG_pretrained_model/features_mw_1789.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mw_pipeline/results/real_participant_distributions/highlight_mw')
RANDOM_SEED = 42  # 固定随机种子，保证投影一致

# 仅使用的16个特征列名（表头中应存在）
FEATURES_16 = [
    "P3_Amplitude",
    "P3_Latency",
    "Activity",
    "Mobility",
    "Complexity",
    "Petrosian_Fractal_Dimension",
    "Delta_PSD",
    "Theta_PSD",
    "Alpha_PSD",
    "Beta_PSD",
    "Gamma_PSD",
    "Delta_Entropy",
    "Theta_Entropy",
    "Alpha_Entropy",
    "Beta_Entropy",
    "Gamma_Entropy",
]


def get_grid_limits(Xr, Xf, q=0.01):
    """获取网格范围"""
    x_all = np.r_[Xr[:, 0], Xf[:, 0]]
    y_all = np.r_[Xr[:, 1], Xf[:, 1]]
    x_min, x_max = np.quantile(x_all, [q, 1 - q])
    y_min, y_max = np.quantile(y_all, [q, 1 - q])
    return (x_min, x_max, y_min, y_max)


def kde_on_grid(Xr, Xf, nx=200, ny=200, q=0.01):
    """
    在网格上计算KDE概率密度
    返回：xs, ys, P, Q, cell （P/Q 为在网格上的 KDE 概率密度，已归一化为 pmf，积分≈1）
    """
    x_min, x_max, y_min, y_max = get_grid_limits(Xr, Xf, q=q)
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')

    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    pooled = np.vstack([Xr, Xf])
    spread = float(np.mean(np.std(pooled, axis=0, ddof=1)))
    bandwidth = max(spread * (len(pooled) ** (-1.0 / 6.0)), 1e-3)
    kde_r = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(Xr)
    kde_f = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(Xf)

    Pr = np.exp(kde_r.score_samples(grid_points)).reshape(ny, nx)
    Pf = np.exp(kde_f.score_samples(grid_points)).reshape(ny, nx)

    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    cell = dx * dy

    # 归一化成离散 pmf（避免积分常数问题），保持与 t-SNE 口径一致
    Pr = Pr / (Pr.sum() * cell + 1e-12)
    Pf = Pf / (Pf.sum() * cell + 1e-12)
    return xs, ys, Pr, Pf, cell


def kde_overlap_2d(Xr, Xf, nx=200, ny=200, q=0.01):
    """计算2D KDE重叠度（带cell一致口径）"""
    _, _, Pr, Pf, cell = kde_on_grid(Xr, Xf, nx, ny, q)
    return float(np.minimum(Pr, Pf).sum() * cell)


def kde_metrics_2d_tsne(real_2d, fake_2d, grid_n=120, padding=0.05, eps=1e-12):
    """t-SNE专用的2D KDE指标计算"""
    xmin = min(real_2d[:,0].min(), fake_2d[:,0].min())
    xmax = max(real_2d[:,0].max(), fake_2d[:,0].max())
    ymin = min(real_2d[:,1].min(), fake_2d[:,1].min())
    ymax = max(real_2d[:,1].max(), fake_2d[:,1].max())
    dxm = (xmax - xmin) or 1.0
    dym = (ymax - ymin) or 1.0
    xmin -= padding * dxm; xmax += padding * dxm
    ymin -= padding * dym; ymax += padding * dym

    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    pos = np.column_stack([xx.ravel(), yy.ravel()])

    pooled = np.vstack([real_2d, fake_2d])
    spread = float(np.mean(np.std(pooled, axis=0, ddof=1)))
    bandwidth = max(spread * (len(pooled) ** (-1.0 / 6.0)), 1e-3)
    kde_r = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(real_2d)
    kde_f = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(fake_2d)
    pr = np.exp(kde_r.score_samples(pos)).reshape(grid_n, grid_n)
    pq = np.exp(kde_f.score_samples(pos)).reshape(grid_n, grid_n)

    dx = (xmax - xmin) / (grid_n - 1)
    dy = (ymax - ymin) / (grid_n - 1)
    cell = dx * dy

    pr = pr / (pr.sum() * cell)
    pq = pq / (pq.sum() * cell)

    overlap = np.minimum(pr, pq).sum() * cell
    return overlap


def frechet_distance_gaussian_from_stats(Xp, mu_g, Sigma_g, eps=1e-6):
    """
    Compute Fréchet distance between participant and global Gaussian distributions.
    
    FD = ||mu_p - mu_g||^2 + Tr(Sigma_p + Sigma_g - 2 * sqrtm(Sigma_p @ Sigma_g))
    
    Uses LedoitWolf covariance shrinkage for numerical stability.
    
    Args:
        Xp: Participant samples (N_p x 16)
        mu_g: Global mean vector (16,)
        Sigma_g: Global covariance matrix (16 x 16)
        eps: Small value for regularization if sqrtm fails
    
    Returns:
        float: Fréchet distance, or np.nan if Xp has fewer than 2 samples
    """
    if len(Xp) < 2:
        return np.nan
    
    # Compute participant mean and covariance
    mu_p = Xp.mean(axis=0)
    Sigma_p = LedoitWolf().fit(Xp).covariance_
    
    # First term: ||mu_p - mu_g||^2
    mean_diff = mu_p - mu_g
    term1 = np.dot(mean_diff, mean_diff)
    
    # Second term: Tr(Sigma_p + Sigma_g - 2 * sqrtm(Sigma_p @ Sigma_g))
    # Compute sqrtm(Sigma_p @ Sigma_g)
    try:
        sqrt_product = linalg.sqrtm(Sigma_p @ Sigma_g)
        if np.iscomplexobj(sqrt_product):
            sqrt_product = sqrt_product.real
        if not np.isfinite(sqrt_product).all():
            raise ValueError("sqrtm produced non-finite values")
    except (ValueError, np.linalg.LinAlgError):
        # Add regularization and recompute
        I = np.eye(Sigma_p.shape[0])
        Sigma_p_reg = Sigma_p + eps * I
        Sigma_g_reg = Sigma_g + eps * I
        try:
            sqrt_product = linalg.sqrtm(Sigma_p_reg @ Sigma_g_reg)
            if np.iscomplexobj(sqrt_product):
                sqrt_product = sqrt_product.real
        except (ValueError, np.linalg.LinAlgError):
            return np.nan
    
    # Compute trace
    term2 = np.trace(Sigma_p + Sigma_g - 2 * sqrt_product)
    
    return float(term1 + term2)


def main():
    print("=" * 80)
    print("Highlight Participant Distributions (PCA & t-SNE)")
    print("=" * 80)

    # 读取数据（含表头，首列为样本编号）
    print(f"Reading data: {FEATURES_FILE}")
    df = pd.read_csv(FEATURES_FILE)
    participant_col = df.columns[0]
    # 仅保留首列 + 16个指定特征
    missing = [c for c in FEATURES_16 if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    df = df[[participant_col] + FEATURES_16]
    participant_ids = sorted(df[participant_col].unique())
    print(f"Total rows: {len(df)}, features used (16): {len(FEATURES_16)}")
    print(f"Unique participants: {len(participant_ids)}")

    # 准备特征（只用16个特征降维）
    X = df.drop(columns=[participant_col]).values

    # 标准化
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # PCA（固定投影，所有参与者共享）
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_2d = pca.fit_transform(X_std)

    # t-SNE（固定投影，所有参与者共享）
    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_SEED,
        perplexity=min(30, len(X_std) - 1),
    )
    tsne_2d = tsne.fit_transform(X_std)

    # Compute global statistics once (for Fréchet distance)
    mu_g = X_std.mean(axis=0)
    Sigma_g = LedoitWolf().fit(X_std).covariance_

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 颜色方案：高亮深蓝，其他浅橙；图例顺序先高亮再其他
    highlight_color = "#0d47a1"  # very deep blue
    others_color = "#ffcc80"     # brighter orange
    legend_handles = [
        Line2D([0], [0], marker=".", color=highlight_color, linestyle="None", markersize=8, label="P{pid}"),
        Line2D([0], [0], marker=".", color=others_color, linestyle="None", markersize=8, label="all others"),
    ]

    # 存储所有participant的overlap结果
    overlap_results = []

    for pid in participant_ids:
        mask = df[participant_col] == pid
        others_mask = ~mask

        # 提取当前participant和其他participant的2D数据
        pca_highlight = pca_2d[mask]
        pca_others = pca_2d[others_mask]
        tsne_highlight = tsne_2d[mask]
        tsne_others = tsne_2d[others_mask]

        # 计算KDE指标
        # overlap值表示：高亮participant的分布占所有样本（others）分布的比例/重叠度
        # 使用全部数据，不进行下采样，以准确估计整体分布
        overlap_pca = kde_overlap_2d(pca_highlight, pca_others)
        overlap_tsne = kde_metrics_2d_tsne(tsne_highlight, tsne_others)

        # Compute Fréchet distance in original 16-D feature space
        Xp = X_std[mask]  # Participant samples
        FD16 = frechet_distance_gaussian_from_stats(Xp, mu_g, Sigma_g)

        # ---------- PCA ----------
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["axes.titlesize"] = 24
        plt.rcParams["axes.labelsize"] = 20
        plt.rcParams["legend.fontsize"] = 14
        plt.rcParams["xtick.labelsize"] = 18
        plt.rcParams["ytick.labelsize"] = 18

        fig, ax = plt.subplots(figsize=(5, 4), facecolor="white")
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")

        ax.scatter(
            pca_2d[others_mask, 0],
            pca_2d[others_mask, 1],
            c=others_color,
            alpha=0.25,
            s=20,
            marker=".",
            label="Others",
        )
        ax.scatter(
            pca_2d[mask, 0],
            pca_2d[mask, 1],
            c=highlight_color,
            alpha=0.7,
            s=20,
            marker=".",
            label="Highlighted",
        )

        ax.set_title(f"PCA (P{pid}) ({overlap_pca:.4f})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        # 动态更新 legend 中的当前高亮标签
        legend_handles[0].set_label(f"P{pid}")
        ax.legend(handles=legend_handles, frameon=False, labelspacing=0.3, handlelength=1.5)
        ax.grid(False)
        plt.tight_layout()
        pca_path = os.path.join(OUTPUT_DIR, f"pca_highlight_P{pid}.png")
        plt.savefig(pca_path, dpi=600)
        plt.close()

        # ---------- t-SNE ----------
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="white")
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")

        ax.scatter(
            tsne_2d[others_mask, 0],
            tsne_2d[others_mask, 1],
            c=others_color,
            alpha=0.25,
            s=20,
            marker=".",
            label="Others",
        )
        ax.scatter(
            tsne_2d[mask, 0],
            tsne_2d[mask, 1],
            c=highlight_color,
            alpha=0.7,
            s=20,
            marker=".",
            label="Highlighted",
        )

        ax.set_title(f"t-SNE (P{pid}) ({overlap_tsne:.4f})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        # t-SNE 不显示图例，保持简洁
        ax.grid(False)
        plt.tight_layout()
        tsne_path = os.path.join(OUTPUT_DIR, f"tsne_highlight_P{pid}.png")
        plt.savefig(tsne_path, dpi=600)
        plt.close()

        fd_str = "NaN" if np.isnan(FD16) else f"{FD16:.4f}"
        print(f"P{pid}: PCA_overlap={overlap_pca:.4f}, tSNE_overlap={overlap_tsne:.4f}, FD16={fd_str}")
        
        # 保存当前participant的overlap结果
        overlap_results.append({
            'Participant': pid,
            'PCA_overlap': overlap_pca,
            'tSNE_overlap': overlap_tsne,
            'FD16': FD16,
        })

    # 保存所有overlap结果到CSV文件
    results_df = pd.DataFrame(overlap_results)
    # 保留小数点后4位
    results_df['PCA_overlap'] = results_df['PCA_overlap'].round(4)
    results_df['tSNE_overlap'] = results_df['tSNE_overlap'].round(4)
    results_df['FD16'] = results_df['FD16'].round(4)
    csv_path = os.path.join(OUTPUT_DIR, 'overlap_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nOverlap results saved to: {csv_path}")
    print(f"Total participants: {len(overlap_results)}")

    print("=" * 80)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
