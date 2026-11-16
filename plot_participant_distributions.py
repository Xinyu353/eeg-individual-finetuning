#!/usr/bin/env python3
"""
plot_participant_distributions.py
随机选择5个参与者，为每个参与者画PCA和t-SNE分布图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 设置随机种子
# 使用None或注释掉下面这行，每次运行会随机选择不同的样本
# 使用固定数字（如42）可以确保每次选择相同的样本（用于可重复性）
RANDOM_SEED = None  # 改为None让每次运行选择不同的样本
if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

# 文件路径
FEATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_EEG_pretrained_model/features_ot_4753.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'participant_distributions')

def plot_all_participants_together(participant_data_dict, selected_ids, save_dir):
    """
    将所有选中的参与者画在同一个图上，用不同颜色区分
    
    Parameters:
    -----------
    participant_data_dict : dict
        字典，key为participant_id，value为该参与者的数据DataFrame
    selected_ids : list
        选中的参与者ID列表
    save_dir : str
        保存目录
    """
    print(f"\n{'='*60}")
    print(f"Plotting all {len(selected_ids)} participants together")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 合并所有参与者的数据
    all_data_list = []
    all_labels = []
    for pid in selected_ids:
        data = participant_data_dict[pid]
        all_data_list.append(data)
        all_labels.extend([f'P{pid}'] * len(data))
    
    combined_data = pd.concat(all_data_list, axis=0, ignore_index=True)
    
    # 准备特征数据（排除第一列，即参与者ID）
    feature_cols = [col for col in combined_data.columns if col != combined_data.columns[0]]
    X = combined_data[feature_cols].values
    
    # 标准化（使用所有数据）
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # 定义颜色（黑色、红色、蓝色、黄色、深绿色）
    colors = ['#000000', '#d62728', '#1f77b4', '#ffd700', '#006400']  # 黑、红、蓝、黄、深绿
    color_map = {f'P{pid}': colors[i] for i, pid in enumerate(selected_ids)}
    
    # PCA分析
    print("\n--- PCA Analysis (All Participants) ---")
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(X_std)
    
    explained_var = pca.explained_variance_ratio_
    print(f"  Explained variance ratio: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}")
    print(f"  Total explained variance: {explained_var.sum():.4f}")
    
    # 生成标题和文件名中的样本编号字符串
    sample_ids_str = ','.join(map(str, selected_ids))
    filename_ids_str = '_'.join(map(str, selected_ids))
    
    # 画PCA图（所有参与者）
    # 调整figsize来改变图的大小：(width, height) 单位是英寸
    # 当前: (6, 5)，可以改为 (5, 4) 更小，或 (8, 6) 更大
    plt.figure(figsize=(5, 4), facecolor='white')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # 为每个参与者画点
    for pid in selected_ids:
        mask = np.array(all_labels) == f'P{pid}'
        plt.scatter(
            pca_2d[mask, 0], 
            pca_2d[mask, 1], 
            label=f'P{pid}',
            alpha=0.5, 
            c=color_map[f'P{pid}'], 
            marker='.',
            s=25
        )
    
    plt.title(f'PCA ({sample_ids_str})')
    plt.xlabel(f'Component 1 ({explained_var[0]:.2%})')
    plt.ylabel(f'Component 2 ({explained_var[1]:.2%})')
    plt.legend(loc='best', frameon=False, markerscale=2.5, labelspacing=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.grid(False)
    
    plt.tight_layout()
    save_path_pca = os.path.join(save_dir, f'pca_{filename_ids_str}.png')
    plt.savefig(save_path_pca, format='png', dpi=600)
    plt.close()
    print(f"  PCA plot saved to: {save_path_pca}")
    
    # t-SNE分析（如果数据点太多，进行采样）
    print("\n--- t-SNE Visualization (All Participants) ---")
    max_samples_per_participant = 200  # 每个参与者最多采样200个点
    sampled_indices = []
    sampled_labels = []
    
    for pid in selected_ids:
        mask = np.array(all_labels) == f'P{pid}'
        indices = np.where(mask)[0]
        if len(indices) > max_samples_per_participant:
            sampled_idx = np.random.choice(indices, size=max_samples_per_participant, replace=False)
            print(f"  Sampling {max_samples_per_participant} points from P{pid} ({len(indices)} trials)")
        else:
            sampled_idx = indices
        sampled_indices.extend(sampled_idx)
        sampled_labels.extend([f'P{pid}'] * len(sampled_idx))
    
    sampled_indices = np.array(sampled_indices)
    X_tsne_input = X_std[sampled_indices]
    
    # 如果RANDOM_SEED为None，t-SNE会使用随机状态
    tsne_kwargs = {'n_components': 2, 'perplexity': min(30, len(X_tsne_input)-1)}
    if RANDOM_SEED is not None:
        tsne_kwargs['random_state'] = RANDOM_SEED
    tsne = TSNE(**tsne_kwargs)
    tsne_2d = tsne.fit_transform(X_tsne_input)
    
    # 画t-SNE图（所有参与者）
    # 调整figsize来改变图的大小：(width, height) 单位是英寸
    plt.figure(figsize=(5, 4), facecolor='white')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # 为每个参与者画点
    for pid in selected_ids:
        mask = np.array(sampled_labels) == f'P{pid}'
        plt.scatter(
            tsne_2d[mask, 0], 
            tsne_2d[mask, 1], 
            label=f'P{pid}',
            alpha=0.5, 
            c=color_map[f'P{pid}'], 
            marker='.',
            s=25
        )
    
    plt.title(f't-SNE ({sample_ids_str})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # 不显示图例
    plt.grid(False)
    
    plt.tight_layout()
    save_path_tsne = os.path.join(save_dir, f'tsne_{filename_ids_str}.png')
    plt.savefig(save_path_tsne, format='png', dpi=600)
    plt.close()
    print(f"  t-SNE plot saved to: {save_path_tsne}")


def plot_participant_distribution(participant_data, participant_id, save_dir):
    """
    为单个参与者画PCA和t-SNE分布图
    
    Parameters:
    -----------
    participant_data : pd.DataFrame
        该参与者的数据
    participant_id : int or str
        参与者ID
    save_dir : str
        保存目录
    """
    print(f"\n{'='*60}")
    print(f"Processing Participant {participant_id}")
    print(f"  Number of trials: {len(participant_data)}")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备特征数据（排除第一列，即参与者ID）
    feature_cols = [col for col in participant_data.columns if col != participant_data.columns[0]]
    X = participant_data[feature_cols].values
    
    # 标准化
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # PCA分析
    print("\n--- PCA Analysis ---")
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(X_std)
    
    explained_var = pca.explained_variance_ratio_
    print(f"  Explained variance ratio: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}")
    print(f"  Total explained variance: {explained_var.sum():.4f}")
    
    # 画PCA图
    plt.figure(figsize=(4, 3), facecolor='white')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    
    ax = plt.gca()
    ax.set_facecolor('white')
    
    plt.scatter(
        pca_2d[:, 0], 
        pca_2d[:, 1], 
        alpha=0.6, 
        c='black', 
        marker='.',
        s=20
    )
    
    plt.title(f'Participant {participant_id} - PCA')
    plt.xlabel(f'Component 1 ({explained_var[0]:.2%})')
    plt.ylabel(f'Component 2 ({explained_var[1]:.2%})')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.grid(False)
    
    plt.tight_layout()
    save_path_pca = os.path.join(save_dir, f'PCA_participant_{participant_id}.png')
    plt.savefig(save_path_pca, format='png', dpi=600)
    plt.close()
    print(f"  PCA plot saved to: {save_path_pca}")
    
    # t-SNE分析（如果数据点太多，进行采样）
    print("\n--- t-SNE Visualization ---")
    max_samples = 1000
    if len(X_std) > max_samples:
        print(f"  Sampling {max_samples} points from {len(X_std)} trials for t-SNE")
        indices = np.random.choice(len(X_std), size=max_samples, replace=False)
        X_tsne_input = X_std[indices]
    else:
        X_tsne_input = X_std
        indices = np.arange(len(X_std))
    
    # 如果RANDOM_SEED为None，t-SNE会使用随机状态
    tsne_kwargs = {'n_components': 2, 'perplexity': min(30, len(X_tsne_input)-1)}
    if RANDOM_SEED is not None:
        tsne_kwargs['random_state'] = RANDOM_SEED
    tsne = TSNE(**tsne_kwargs)
    tsne_2d = tsne.fit_transform(X_tsne_input)
    
    # 画t-SNE图
    plt.figure(figsize=(4, 3), facecolor='white')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    plt.scatter(
        tsne_2d[:, 0], 
        tsne_2d[:, 1], 
        alpha=0.6, 
        c='black', 
        marker='.',
        s=20
    )
    
    plt.title(f'Participant {participant_id} - t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(False)
    
    plt.tight_layout()
    save_path_tsne = os.path.join(save_dir, f'tsne_participant_{participant_id}.png')
    plt.savefig(save_path_tsne, format='png', dpi=600)
    plt.close()
    print(f"  t-SNE plot saved to: {save_path_tsne}")
    
    # 保存统计信息
    stats_file = os.path.join(save_dir, f'stats_participant_{participant_id}.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Distribution Analysis for Participant {participant_id}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of trials: {len(participant_data)}\n")
        f.write(f"Number of features: {len(feature_cols)}\n\n")
        f.write("PCA Results:\n")
        f.write("-"*40 + "\n")
        f.write(f"PC1 explained variance: {explained_var[0]:.4f} ({explained_var[0]:.2%})\n")
        f.write(f"PC2 explained variance: {explained_var[1]:.4f} ({explained_var[1]:.2%})\n")
        f.write(f"Total explained variance: {explained_var.sum():.4f} ({explained_var.sum():.2%})\n")
        f.write(f"\nt-SNE: {len(X_tsne_input)} points visualized\n")
    
    print(f"  Statistics saved to: {stats_file}")


def main():
    """主函数"""
    print("="*80)
    print("Participant Distribution Analysis")
    print("="*80)
    
    # 读取数据
    print(f"\nReading data from: {FEATURES_FILE}")
    df = pd.read_csv(FEATURES_FILE)
    print(f"  Total shape: {df.shape}")
    
    # 获取第一列（参与者ID）
    participant_col = df.columns[0]
    participant_ids = df[participant_col].unique()
    print(f"  Total number of unique participants: {len(participant_ids)}")
    print(f"  Participant ID column: {participant_col}")
    
    # 随机选择5个参与者
    n_select = 5
    if len(participant_ids) < n_select:
        print(f"⚠ Warning: Only {len(participant_ids)} participants available, selecting all")
        selected_ids = participant_ids
    else:
        selected_ids = np.random.choice(participant_ids, size=n_select, replace=False)
        selected_ids = sorted(selected_ids)  # 排序以便输出更清晰
    
    print(f"\n{'='*80}")
    print(f"Selected {len(selected_ids)} participants (remember these IDs!):")
    print(f"{'='*80}")
    for i, pid in enumerate(selected_ids, 1):
        n_trials = len(df[df[participant_col] == pid])
        print(f"  {i}. Participant {pid}: {n_trials} trials")
    print(f"{'='*80}\n")
    
    # 保存选中的参与者ID到文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    selected_ids_file = os.path.join(OUTPUT_DIR, 'selected_participant_ids.txt')
    with open(selected_ids_file, 'w') as f:
        f.write("Selected Participant IDs\n")
        f.write("="*60 + "\n\n")
        for i, pid in enumerate(selected_ids, 1):
            n_trials = len(df[df[participant_col] == pid])
            f.write(f"{i}. Participant {pid}: {n_trials} trials\n")
    print(f"Selected participant IDs saved to: {selected_ids_file}\n")
    
    # 准备所有参与者的数据字典
    participant_data_dict = {}
    for pid in selected_ids:
        participant_data_dict[pid] = df[df[participant_col] == pid].copy()
    
    # 画所有参与者在同一个图上
    plot_all_participants_together(participant_data_dict, selected_ids, OUTPUT_DIR)
    
    # 为每个选中的参与者单独画分布图（可选）
    # for pid in selected_ids:
    #     participant_data = participant_data_dict[pid]
    #     participant_save_dir = os.path.join(OUTPUT_DIR, f'participant_{pid}')
    #     plot_participant_distribution(participant_data, pid, participant_save_dir)
    
    print(f"\n{'='*80}")
    print("All participants processed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

