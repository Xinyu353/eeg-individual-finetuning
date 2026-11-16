#!/usr/bin/env python3
"""
distribution_2d_analysis.py
计算和可视化真实数据与生成数据的分布差异

- 读取 feature_extraction/real_ot/ 下的真实特征文件（如 p1_real.csv）
- 读取 feature_extraction/ot_feature/ 下的生成特征文件（如 p1_epoch_250.csv）
- 按 participant ID 一一对应进行对比（p1 对应 p1，p2 对应 p2 等）
- 计算 PCA 和 t-SNE 分布指标并画图保存
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob
import argparse
import scipy.stats as stats
from scipy.stats import wasserstein_distance, entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

# 特征名称简写映射（用于图表显示，使用LaTeX格式显示下标）
FEATURE_ABBREVIATIONS = {
    'P3_Amplitude': 'P3_A',
    'P3_Latency': 'P3_L',
    'Activity': 'HA',  # 注意：用户指定为HC
    'Mobility': 'HM',
    'Complexity': 'HC',  # 注意：用户指定为HC，可能与Activity重复
    'Petrosian_Fractal_Dimension': 'PFD',
    'Delta_PSD': r'PSD$_{\delta}$',
    'Theta_PSD': r'PSD$_{\theta}$',
    'Alpha_PSD': r'PSD$_{\alpha}$',
    'Beta_PSD': r'PSD$_{\beta}$',
    'Gamma_PSD': r'PSD$_{\gamma}$',
    'Delta_Entropy': r'E$_{\delta}$',
    'Theta_Entropy': r'E$_{\theta}$',
    'Alpha_Entropy': r'E$_{\alpha}$',
    'Beta_Entropy': r'E$_{\beta}$',
    'Gamma_Entropy': r'E$_{\gamma}$',
}

def get_feature_abbreviation(feature_name):
    """
    获取特征名称的简写（用于图表显示）
    
    Args:
        feature_name: 完整特征名称
    
    Returns:
        str: 简写名称（LaTeX格式），如果未找到映射则返回原名称
    """
    return FEATURE_ABBREVIATIONS.get(feature_name, feature_name)

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

    # sklearn KernelDensity avoids occasional scipy gaussian_kde segfaults on
    # ill-conditioned 2D participant clouds while preserving the same KDE-based
    # overlap interpretation.
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

def kde_bhattacharyya_2d(Xr, Xf, nx=200, ny=200, q=0.01):
    """计算2D Bhattacharyya系数（带cell一致口径）"""
    _, _, Pr, Pf, cell = kde_on_grid(Xr, Xf, nx, ny, q)
    return float(np.sqrt(Pr * Pf).sum() * cell)

def kde_js_2d(Xr, Xf, nx=200, ny=200, q=0.01, eps=1e-12):
    """计算2D JS散度（带cell一致口径）"""
    _, _, Pr, Pf, cell = kde_on_grid(Xr, Xf, nx, ny, q)
    P = (Pr + eps)
    Q = (Pf + eps)
    P = P / (P.sum() * cell)
    Q = Q / (Q.sum() * cell)

    M = 0.5 * (P + Q)
    js_div = 0.5 * (P * np.log(P / M)).sum() * cell + 0.5 * (Q * np.log(Q / M)).sum() * cell
    return float(js_div)

def kde_iou_2d(Xr, Xf, nx=200, ny=200, q=0.01):
    """计算2D IoU（无需cell系数）"""
    _, _, Pr, Pf, _ = kde_on_grid(Xr, Xf, nx, ny, q)
    inter = np.minimum(Pr, Pf).sum()
    union = np.maximum(Pr, Pf).sum()
    return float(inter / (union + 1e-12))

def hist2d_metrics(Xr, Xf, bins=50, q=0.01, eps=1e-12):
    """计算2D直方图度量"""
    x_min, x_max, y_min, y_max = get_grid_limits(Xr, Xf, q=q)
    ranges = [[x_min, x_max], [y_min, y_max]]

    Hr, xedges, yedges = np.histogram2d(Xr[:,0], Xr[:,1], bins=bins, range=ranges, density=False)
    Hf, _, _ = np.histogram2d(Xf[:,0], Xf[:,1], bins=[xedges, yedges], density=False)

    # 转 pmf
    Pr = Hr.astype(float) / (Hr.sum() + eps)
    Pf = Hf.astype(float) / (Hf.sum() + eps)

    # Bhattacharyya
    bc = np.sqrt(Pr * Pf).sum()

    # IoU
    iou = np.minimum(Pr, Pf).sum() / (np.maximum(Pr, Pf).sum() + eps)

    # KL(P||Q) 和 KL(Q||P)
    P = (Pr + eps) / (Pr.sum() + eps)
    Q = (Pf + eps) / (Pf.sum() + eps)
    kl_pq = np.sum(P * np.log(P / Q))
    kl_qp = np.sum(Q * np.log(Q / P))

    return {
        'BC_hist': bc,
        'IoU_hist': iou,
        'KL_P||Q_hist': kl_pq,
        'KL_Q||P_hist': kl_qp
    }

def sliced_wasserstein_2d(Xr, Xf, n_projections=128, seed=42):
    """计算2D Sliced Wasserstein距离"""
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, 2*np.pi, size=n_projections)

    vals = []
    for th in thetas:
        v = np.array([np.cos(th), np.sin(th)])
        pr = Xr @ v
        pf = Xf @ v
        vals.append(wasserstein_distance(pr, pf))
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std())

def analyze_pca_components(pca, feature_names, top_n=5):
    """
    分析PCA主成分的特征贡献
    
    Args:
        pca: 已拟合的PCA对象
        feature_names: 特征名称列表
        top_n: 返回每个主成分的前N个最重要特征
    
    Returns:
        dict: 包含每个主成分的特征贡献信息
    """
    components = pca.components_  # shape: (n_components, n_features)
    explained_variance = pca.explained_variance_ratio_  # 每个主成分解释的方差比例
    
    results = {}
    
    for comp_idx in range(components.shape[0]):
        comp = components[comp_idx]
        var_ratio = explained_variance[comp_idx]
        
        # 计算每个特征的绝对贡献（绝对值）
        abs_contributions = np.abs(comp)
        
        # 获取top N特征的索引
        top_indices = np.argsort(abs_contributions)[::-1][:top_n]
        
        # 构建特征贡献列表
        feature_contributions = []
        for idx in top_indices:
            feature_contributions.append({
                'feature': feature_names[idx],
                'coefficient': comp[idx],
                'abs_contribution': abs_contributions[idx]
            })
        
        results[f'PC{comp_idx + 1}'] = {
            'explained_variance_ratio': var_ratio,
            'top_features': feature_contributions
        }
    
    return results


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
    pos = np.vstack([xx.ravel(), yy.ravel()])

    kde_r = stats.gaussian_kde(real_2d.T)
    kde_f = stats.gaussian_kde(fake_2d.T)
    pr = kde_r(pos).reshape(grid_n, grid_n)
    pq = kde_f(pos).reshape(grid_n, grid_n)

    dx = (xmax - xmin) / (grid_n - 1)
    dy = (ymax - ymin) / (grid_n - 1)
    cell = dx * dy

    pr = pr / (pr.sum() * cell)
    pq = pq / (pq.sum() * cell)

    overlap = np.minimum(pr, pq).sum() * cell
    bc = np.sqrt(pr * pq).sum() * cell
    kl_pq = (pr * np.log((pr + eps) / (pq + eps))).sum() * cell
    kl_qp = (pq * np.log((pq + eps) / (pr + eps))).sum() * cell
    return overlap, bc, kl_pq, kl_qp

def process_comparison(real_data, fake_data, save_dir, comparison_name, participant_id):
    """处理单个比较"""
    print(f"\n{'='*60}")
    print(f"Processing: {comparison_name}")
    print(f"{'='*60}")
    
    # 创建分类文件夹
    pca_dir = os.path.join(save_dir, 'pca_plots')
    pca_components_dir = os.path.join(pca_dir, 'PCA_components')  # PCA components图单独文件夹
    tsne_dir = os.path.join(save_dir, 'tsne_plots')
    results_dir = os.path.join(save_dir, 'results')
    
    os.makedirs(pca_dir, exist_ok=True)
    os.makedirs(pca_components_dir, exist_ok=True)
    os.makedirs(tsne_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成文件名（去掉重复的p{id}）
    # comparison_name 格式: p{id}_epoch_{epoch}
    # 提取 epoch 部分
    epoch_match = re.search(r'_epoch_(\d+)$', comparison_name)
    epoch_str = epoch_match.group(1) if epoch_match else "unknown"
    # 新格式: p{id}_epoch_{epoch} -> p{id}_epoch_{epoch}（用于PCA散点图）
    # PCA components图: p{id}_PCA_components_epoch_{epoch}.png
    base_name = comparison_name  # 已经是 p{id}_epoch_{epoch} 格式
    components_base_name = f'p{participant_id}_PCA_components_epoch_{epoch_str}'  # 简化格式
    
    # 准备数据
    common_columns = list(set(real_data.columns) & set(fake_data.columns))
    real_data_clean = real_data[common_columns].copy()
    fake_data_clean = fake_data[common_columns].copy()
    
    real_data_clean['Label'] = 'Real'
    fake_data_clean['Label'] = 'Fake'
    combined = pd.concat([real_data_clean, fake_data_clean], axis=0, ignore_index=True)
    
    # 显式排除非特征列
    drop_cols = [c for c in ['SampleID', 'Label'] if c in combined.columns]
    X = combined.drop(columns=drop_cols).values
    labels = combined['Label'].to_numpy()
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # PCA分析
    print("\n--- PCA Analysis ---")
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(X_std)
    
    X_real_2d = pca_2d[labels == 'Real']
    X_fake_2d = pca_2d[labels == 'Fake']
    
    # 获取特征名称（排除非特征列）
    feature_names = [col for col in combined.drop(columns=drop_cols).columns]
    
    # 分析PCA主成分的特征贡献
    print("\n--- PCA Component Analysis ---")
    pca_component_analysis = analyze_pca_components(pca, feature_names, top_n=5)
    
    for pc_name, pc_info in pca_component_analysis.items():
        var_ratio = pc_info['explained_variance_ratio']
        print(f"\n{pc_name} (解释方差: {var_ratio*100:.2f}%):")
        print("  Top contributing features:")
        for i, feat_info in enumerate(pc_info['top_features'], 1):
            coef = feat_info['coefficient']
            sign = "+" if coef >= 0 else "-"
            print(f"    {i}. {feat_info['feature']:30s} "
                  f"coefficient: {sign}{abs(coef):.4f} "
                  f"(abs: {feat_info['abs_contribution']:.4f})")
    
    # 计算PCA的KDE指标（只计算需要的）
    print("\n== PCA-2D Feature-level Metrics ==")
    overlap_pca = kde_overlap_2d(X_real_2d, X_fake_2d)
    bc_pca = kde_bhattacharyya_2d(X_real_2d, X_fake_2d)
    js_pca = kde_js_2d(X_real_2d, X_fake_2d)
    
    print(f"Overlap_kde   : {overlap_pca:.4f}")
    print(f"BC_kde        : {bc_pca:.4f}")
    print(f"JS_kde        : {js_pca:.4f}")
    
    # 保存PCA结果
    pca_results = {
        'Overlap_kde': overlap_pca,
        'BC_kde': bc_pca,
        'JS_kde': js_pca,
        'component_analysis': pca_component_analysis
    }
    
    # 画PCA图
    min_row_count = min(800, len(fake_data_clean), len(real_data_clean))
    fake_sample = fake_data_clean.sample(n=min_row_count, random_state=42)
    real_sample = real_data_clean.sample(n=min_row_count, random_state=42)
    
    fake_sample['Label'] = 'Fake'
    real_sample['Label'] = 'Real'
    combined_data = pd.concat([fake_sample, real_sample], axis=0)
    
    features = combined_data.drop(columns=drop_cols)
    labels_plot = combined_data['Label']

    # 复用已拟合的 scaler/pca，确保与数值一致
    features_standardized = scaler.transform(features)
    pca_result = pca.transform(features_standardized)
    
    plt.figure(figsize=(4, 3.5), facecolor='white')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    
    ax = plt.gca()
    ax.set_facecolor('white')
    
    plt.scatter(
        pca_result[labels_plot == 'Real', 0], 
        pca_result[labels_plot == 'Real', 1], 
        label='Real', 
        alpha=0.5, 
        c='black', 
        marker='.'
    )
    
    plt.scatter(
        pca_result[labels_plot == 'Fake', 0], 
        pca_result[labels_plot == 'Fake', 1], 
        label='Fake', 
        alpha=0.5, 
        c='red', 
        marker='.'
    )
    
    plt.title(f'PCA ({overlap_pca:.4f})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.grid(False)
    
    plt.tight_layout()
    save_path_pca = os.path.join(pca_dir, f'p{participant_id}_PCA_{base_name}.png')
    plt.savefig(save_path_pca, format='png', dpi=600)
    plt.close()
    
    # 绘制PCA主成分特征贡献图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    # 启用LaTeX渲染以显示希腊字母下标
    plt.rcParams['text.usetex'] = False  # 使用matplotlib的数学文本渲染而不是LaTeX
    plt.rcParams['mathtext.default'] = 'regular'  # 使用常规数学文本
    
    # 计算累积方差解释率
    explained_variances = pca.explained_variance_ratio_
    cumulative_variance = explained_variances.sum()
    
    # 设置总标题：分两行，分别控制位置和样式
    # 第1行：主标题（加粗，字号略大）
    fig.suptitle(f'PCA Component Loadings (Cumulative Variance Explained: {cumulative_variance*100:.2f}%)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 第2行：说明文字（字号略小，位置稍低）
    fig.text(0.5, 0.94, 'Features are sorted by absolute loading (descending). The sign of a loading indicates direction.',
             ha='center', va='top', fontsize=12, style='normal')
    
    for comp_idx, ax in enumerate(axes):
        pc_name = f'PC{comp_idx + 1}'
        pc_info = pca_results['component_analysis'][pc_name]
        var_ratio = pc_info['explained_variance_ratio']
        
        # ============================================================
        # 条形图排序逻辑：按 |loading| 降序排列
        # ============================================================
        # 1. 获取PCA主成分的loading（系数）
        components = pca.components_
        comp = components[comp_idx]  # 当前主成分的所有特征loading
        
        # 2. 计算每个特征的 |loading|（绝对值）
        abs_contributions = np.abs(comp)
        
        # 3. 按 |loading| 降序排列（从大到小）
        #    np.argsort() 返回从小到大的索引，[::-1] 反转得到从大到小
        sorted_indices = np.argsort(abs_contributions)[::-1]
        
        # 4. 取前10个特征（|loading| 最大的前10个）
        top_n_plot = min(10, len(feature_names))
        top_indices = sorted_indices[:top_n_plot]
        
        # 5. 准备绘图数据（已按 |loading| 降序排列）
        top_features = [feature_names[i] for i in top_indices]
        top_features_abbrev = [get_feature_abbreviation(f) for f in top_features]
        top_coefficients = comp[top_indices]  # 原始loading值（保留正负号）
        top_abs = abs_contributions[top_indices]  # 绝对值（用于排序）
        
        # 6. 绘制水平条形图
        #    颜色：红色表示负loading，蓝色表示正loading
        colors = ['red' if c < 0 else 'blue' for c in top_coefficients]
        bars = ax.barh(range(len(top_features)), top_coefficients, color=colors, alpha=0.7)
        
        # 7. 设置Y轴：反转Y轴，让 |loading| 最大的特征显示在最上面
        #    这样读者第一眼就能看到最重要的特征
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features_abbrev)
        ax.invert_yaxis()  # 反转Y轴：索引0（|loading|最大）→ 顶部
        ax.set_xlabel('Loading', fontsize=12)
        
        # 子图标题：仅保留各自 explained variance
        # 调整pad参数，增加与总标题的间距
        ax.set_title(f'{pc_name} (Variance Explained: {var_ratio*100:.2f}%)', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # 统一 x 轴范围以便比较
        ax.set_xlim(-0.5, 0.5)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 获取x轴范围，用于判断标签位置
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        margin = x_range * 0.02  # 2%的边距
        
        # 添加数值标签（智能位置调整）
        for i, (bar, coef) in enumerate(zip(bars, top_coefficients)):
            # 判断标签应该放在条形内部还是外部
            # 如果系数为正且接近右边界，或系数为负且接近左边界，放在内部
            if coef >= 0:
                # 正系数：如果接近右边界，标签放在条形内部（左侧）
                if coef > x_max - margin * 3:
                    label_x = coef - margin
                    ha = 'right'
                else:
                    label_x = coef + margin
                    ha = 'left'
            else:
                # 负系数：如果接近左边界，标签放在条形内部（右侧）
                if coef < x_min + margin * 3:
                    label_x = coef + margin
                    ha = 'left'
                else:
                    label_x = coef - margin
                    ha = 'right'
            
            # 确保标签在x轴范围内
            label_x = max(x_min + margin, min(x_max - margin, label_x))
            
            ax.text(label_x, i, f'{coef:.3f}',
                   va='center', ha=ha, fontsize=9)
    
    # 添加图例：蓝色=positive loading，红色=negative loading
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Positive loading'),
        Patch(facecolor='red', alpha=0.7, label='Negative loading')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=False)
    
    # 使用tight_layout的rect参数预留顶部空间，确保标题不重叠
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    # 进一步调整，确保说明文字和子图标题之间有足够间距
    plt.subplots_adjust(top=0.85, bottom=0.12)
    save_path_components = os.path.join(pca_components_dir, f'{components_base_name}.png')
    plt.savefig(save_path_components, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA components plot saved to: {save_path_components}")
    
    # t-SNE分析（仅用于可视化）
    print("\n--- t-SNE Visualization ---")
    min_row_count_tsne = min(800, len(fake_data_clean), len(real_data_clean))
    fake_sample_tsne = fake_data_clean.sample(n=min_row_count_tsne, random_state=42).copy()
    real_sample_tsne = real_data_clean.sample(n=min_row_count_tsne, random_state=42).copy()
    
    fake_sample_tsne["Label"] = "Fake"
    real_sample_tsne["Label"] = "Real"
    
    combined_tsne = pd.concat([fake_sample_tsne, real_sample_tsne], ignore_index=True)
    
    X_tsne = combined_tsne.drop(columns=drop_cols).to_numpy()
    y_tsne = combined_tsne["Label"].to_numpy()
    
    scaler_tsne = StandardScaler()
    X_std_tsne = scaler_tsne.fit_transform(X_tsne)
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(X_std_tsne)
    
    mask_real = (y_tsne == "Real")
    mask_fake = (y_tsne == "Fake")
    tsne_real_2d = tsne_result[mask_real, :2]
    tsne_fake_2d = tsne_result[mask_fake, :2]

    # 计算t-SNE的Overlap（仅用于可视化标注）
    overlap_tsne, _, _, _ = kde_metrics_2d_tsne(tsne_real_2d, tsne_fake_2d)
    print(f"Overlap_kde (for visualization only): {overlap_tsne:.4f}")
    
    # 不保存t-SNE数值指标（因为不具可比性）
    tsne_results = {}
    
    # 画t-SNE图
    plt.figure(figsize=(4, 3.5))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    
    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.set_facecolor('white')
    
    plt.scatter(
        tsne_result[y_tsne == 'Real', 0], 
        tsne_result[y_tsne == 'Real', 1], 
        label='Real', 
        alpha=0.5, 
        c='black', 
        marker='.'
    )
    
    plt.scatter(
        tsne_result[y_tsne == 'Fake', 0], 
        tsne_result[y_tsne == 'Fake', 1], 
        label='Fake', 
        alpha=0.5, 
        c='red', 
        marker='.'
    )
    
    plt.title(f't-SNE ({overlap_tsne:.4f})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.grid(False)
    
    plt.tight_layout()
    save_path_tsne = os.path.join(tsne_dir, f'p{participant_id}_tsne_{base_name}.png')
    plt.savefig(save_path_tsne, format='png', dpi=600)
    plt.close()
    
    # 保存PCA结果到文本文件（只保存特征层面的指标）
    results_file = os.path.join(results_dir, f'p{participant_id}_results_{base_name}.txt')
    with open(results_file, 'w') as f:
        f.write(f"Feature-level Distribution Analysis Results for {comparison_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write("PCA-2D Feature-level Metrics:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Overlap_kde':20s}: {pca_results['Overlap_kde']:.4f}\n")
        f.write(f"{'BC_kde':20s}: {pca_results['BC_kde']:.4f}\n")
        f.write(f"{'JS_kde':20s}: {pca_results['JS_kde']:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("PCA Component Analysis (Feature Contributions)\n")
        f.write("="*60 + "\n\n")
        
        for pc_name, pc_info in pca_results['component_analysis'].items():
            var_ratio = pc_info['explained_variance_ratio']
            f.write(f"{pc_name} - Explained Variance: {var_ratio*100:.2f}%\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Rank':<6} {'Feature':<30} {'Coefficient':<15} {'Abs Contribution':<15}\n")
            f.write("-"*60 + "\n")
            
            for i, feat_info in enumerate(pc_info['top_features'], 1):
                coef = feat_info['coefficient']
                abs_contrib = feat_info['abs_contribution']
                f.write(f"{i:<6} {feat_info['feature']:<30} {coef:>14.4f} {abs_contrib:>14.4f}\n")
            f.write("\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Note: t-SNE metrics are not included as t-SNE does not preserve\n")
        f.write("distance and density, making numerical comparisons unreliable.\n")
    
    print(f"\nResults saved to: {results_file}")
    print(f"PCA plot saved to: {save_path_pca}")
    print(f"t-SNE plot saved to: {save_path_tsne}")
    
    return overlap_pca, overlap_tsne, bc_pca, js_pca

def extract_participant_id_from_filename(filename):
    """
    从文件名提取 participant_id
    
    Args:
        filename: 文件名，如 "p1_real.csv" 或 "p1_epoch_250.csv" 或 "p11_real.csv"
    
    Returns:
        int: participant_id，如 1, 11，如果无法提取则返回 None
    """
    # 匹配 p数字_real.csv 或 p数字_epoch_数字.csv
    match = re.match(r'p(\d+)(?:_real|_epoch_\d+)\.csv', filename)
    if match:
        return int(match.group(1))
    return None


def build_argparser():
    parser = argparse.ArgumentParser(description="2D real-vs-fake feature distribution analysis for OT or MW.")
    parser.add_argument("--condition", choices=["ot", "mw"], default="ot")
    parser.add_argument("--real-feature-dir", default=None)
    parser.add_argument("--fake-feature-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary-filename", default=None)
    return parser


def resolve_paths(args, project_root):
    real_feature_dir = args.real_feature_dir
    fake_feature_dir = args.fake_feature_dir
    output_dir = args.output_dir
    summary_filename = args.summary_filename

    if args.condition == "mw":
        real_feature_dir = real_feature_dir or os.path.join(project_root, "mw_pipeline", "features", "real_mw")
        fake_feature_dir = fake_feature_dir or os.path.join(project_root, "mw_pipeline", "features", "mw_feature")
        output_dir = output_dir or os.path.join(project_root, "mw_pipeline", "results", "mw_distributions_2d")
        summary_filename = summary_filename or "mw_distributions_2d_results.csv"
    else:
        real_feature_dir = real_feature_dir or os.path.join(project_root, "feature_extraction", "real_ot")
        fake_feature_dir = fake_feature_dir or os.path.join(project_root, "feature_extraction", "ot_feature")
        output_dir = output_dir or os.path.join(project_root, "results", "ot_distributions_2d")
        summary_filename = summary_filename or "ot_distributions_2d_results.csv"

    return real_feature_dir, fake_feature_dir, output_dir, summary_filename


def main():
    """主函数"""
    args = build_argparser().parse_args()
    print("="*80)
    print(f"Distribution 2D Analysis Program - {args.condition.upper()} Participant Comparison")
    print("="*80)
    
    # Get project root (parent of evaluation directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    real_feature_dir, fake_feature_dir, output_base_dir, summary_filename = resolve_paths(args, project_root)
    
    print(f"Real features directory: {real_feature_dir}")
    print(f"Fake features directory: {fake_feature_dir}")
    print(f"Output directory: {output_base_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(real_feature_dir):
        print(f"✗ Real features directory not found: {real_feature_dir}")
        return
    
    if not os.path.exists(fake_feature_dir):
        print(f"✗ Fake features directory not found: {fake_feature_dir}")
        return
    
    # 获取所有真实特征文件
    real_files = glob.glob(os.path.join(real_feature_dir, "p*_real.csv"))
    real_files.sort()
    
    # 获取所有生成特征文件
    fake_files = glob.glob(os.path.join(fake_feature_dir, "p*_epoch_*.csv"))
    fake_files.sort()
    
    print(f"\nFound {len(real_files)} real feature files")
    print(f"Found {len(fake_files)} fake feature files")
    
    # 创建 participant_id 到文件的映射
    real_dict = {}
    for real_file in real_files:
        pid = extract_participant_id_from_filename(os.path.basename(real_file))
        if pid is not None:
            real_dict[pid] = real_file
    
    fake_dict = {}
    for fake_file in fake_files:
        pid = extract_participant_id_from_filename(os.path.basename(fake_file))
        if pid is not None:
            if pid not in fake_dict:
                fake_dict[pid] = []
            fake_dict[pid].append(fake_file)
    
    # 找到所有匹配的 participant
    matched_participants = sorted(set(real_dict.keys()) & set(fake_dict.keys()))
    
    if not matched_participants:
        print("\n✗ No matching participants found between real and fake features")
        return
    
    print(f"\nFound {len(matched_participants)} matching participants:")
    for pid in matched_participants:
        print(f"  P{pid}: {len(fake_dict[pid])} fake file(s)")
    
    # 处理每个匹配的 participant
    all_results = []
    
    for pid in matched_participants:
        real_file = real_dict[pid]
        fake_files_for_pid = fake_dict[pid]
        
        print(f"\n{'='*80}")
        print(f"Processing Participant {pid}")
        print(f"{'='*80}")
        print(f"Real file: {os.path.basename(real_file)}")
        
        # 读取真实数据
        try:
            real_data = pd.read_csv(real_file)
            print(f"  Real data shape: {real_data.shape}")
        except Exception as e:
            print(f"✗ Error reading real file {real_file}: {e}")
            continue
        
        # 处理该 participant 的每个生成特征文件
        for fake_file in fake_files_for_pid:
            fake_filename = os.path.basename(fake_file)
            print(f"\n  Fake file: {fake_filename}")
            
            # 从文件名提取 epoch（用于命名）
            epoch_match = re.search(r'_epoch_(\d+)\.csv', fake_filename)
            epoch_str = epoch_match.group(1) if epoch_match else "unknown"
            
            # 读取生成数据
            try:
                fake_data = pd.read_csv(fake_file)
                print(f"    Fake data shape: {fake_data.shape}")
            except Exception as e:
                print(f"✗ Error reading fake file {fake_file}: {e}")
                continue
            
            # 生成保存目录和比较名称
            comparison_name = f"p{pid}_epoch_{epoch_str}"
            # 直接保存到 output_base_dir，不创建子文件夹
            save_dir = output_base_dir
            
            # 处理比较
            try:
                overlap_pca, overlap_tsne, bc_pca, js_pca = process_comparison(
                    real_data, fake_data, save_dir, comparison_name, pid
                )
                print(f"  ✓ Successfully processed {comparison_name}")
                print(f"    PCA Overlap: {overlap_pca:.4f}")
                print(f"    t-SNE Overlap: {overlap_tsne:.4f}")
                
                all_results.append({
                    'participant_id': pid,
                    'epoch': epoch_str,
                    'comparison_name': comparison_name,
                    'pca_overlap': overlap_pca,
                    'pca_bc': bc_pca,
                    'pca_js': js_pca,
                    'tsne_overlap': overlap_tsne
                })
            except Exception as e:
                print(f"✗ Error processing {comparison_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存汇总结果到 CSV（放在 results 文件夹）
    if all_results:
        results_df = pd.DataFrame(all_results)
        # 按 participant_id 和 epoch 排序
        results_df = results_df.sort_values(['participant_id', 'epoch'])
        # 保留4位小数
        results_df['pca_overlap'] = results_df['pca_overlap'].round(4)
        results_df['pca_bc'] = results_df['pca_bc'].round(4)
        results_df['pca_js'] = results_df['pca_js'].round(4)
        results_df['tsne_overlap'] = results_df['tsne_overlap'].round(4)
        
        results_dir = os.path.join(output_base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, summary_filename)
        results_df.to_csv(csv_path, index=False)
        print(f"\n✓ Summary CSV saved to: {csv_path}")
    
    # 打印总结
    print(f"\n{'='*80}")
    print("Processing Summary")
    print("="*80)
    print(f"Total comparisons processed: {len(all_results)}")
    print(f"\nAll files saved to: {output_base_dir}")
    print(f"\nDirectory structure:")
    print(f"  {output_base_dir}/")
    print(f"    ├── pca_plots/")
    print(f"    │   └── p{{id}}_PCA_p{{id}}_epoch_{{epoch}}.png")
    print(f"    ├── tsne_plots/")
    print(f"    │   └── p{{id}}_tsne_p{{id}}_epoch_{{epoch}}.png")
    print(f"    └── results/")
    print(f"        ├── p{{id}}_results_p{{id}}_epoch_{{epoch}}.txt")
    print(f"        └── {summary_filename} (summary)")
    print(f"\nExample files:")
    if all_results:
        first_result = all_results[0]
        pid = first_result['participant_id']
        epoch = first_result['epoch']
        print(f"  pca_plots/p{pid}_PCA_p{pid}_epoch_{epoch}.png")
        print(f"  tsne_plots/p{pid}_tsne_p{pid}_epoch_{epoch}.png")
        print(f"  results/p{pid}_results_p{pid}_epoch_{epoch}.txt")
        print(f"  results/{summary_filename}")
    print("="*80)

if __name__ == "__main__":
    main()
