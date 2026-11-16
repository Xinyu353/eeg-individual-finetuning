#!/usr/bin/env python3
"""
plot_pca_overlap_summary.py
绘制所有参与者的PCA_overlap条形图，按Good/Borderline/Poor分类着色。

读取 participant_distributions/highlight_ot/overlap_results.csv 或
results/ot_distributions_2d/results/ot_distributions_2d_results.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 设置matplotlib参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 600

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 数据文件路径（优先使用real vs fake的PCA_overlap数据）
DATA_FILE_1 = os.path.join(PROJECT_ROOT, 'results', 'ot_distributions_2d', 'results', 'ot_distributions_2d_results.csv')
DATA_FILE_2 = os.path.join(PROJECT_ROOT, 'participant_distributions', 'highlight_ot', 'overlap_results.csv')

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'pca_overlap_summary')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(data_file_1=DATA_FILE_1, data_file_2=DATA_FILE_2):
    """加载数据，优先使用real vs fake的PCA_overlap数据"""
    if os.path.exists(data_file_1):
        df = pd.read_csv(data_file_1)
        print(f"Loaded data from: {data_file_1}")
        print(f"Columns: {df.columns.tolist()}")
        # 确保列名一致
        if 'pca_overlap' not in df.columns and 'PCA_overlap' in df.columns:
            df = df.rename(columns={'PCA_overlap': 'pca_overlap'})
        # 如果有多行对应同一个participant，取最新的epoch
        if 'epoch' in df.columns:
            df = df.sort_values('epoch').groupby('participant_id').last().reset_index()
        return df
    elif data_file_2 and os.path.exists(data_file_2):
        df = pd.read_csv(data_file_2)
        print(f"Loaded data from: {data_file_2}")
        print(f"Columns: {df.columns.tolist()}")
        # 确保列名一致
        if 'Participant' in df.columns:
            df = df.rename(columns={'Participant': 'participant_id'})
        if 'PCA_overlap' in df.columns:
            df = df.rename(columns={'PCA_overlap': 'pca_overlap'})
        return df
    else:
        raise FileNotFoundError(f"Neither {data_file_1} nor {data_file_2} exists.")


def categorize_overlap(value):
    """根据PCA_overlap值分类"""
    if value >= 0.60:
        return 'Good'
    elif value >= 0.50:
        return 'Borderline'
    else:
        return 'Poor'


def plot_pca_overlap_bar(df, output_dir=OUTPUT_DIR):
    """绘制PCA_overlap条形图"""
    # 按participant_id排序
    df = df.sort_values('participant_id').copy()
    
    # 分类
    df['category'] = df['pca_overlap'].apply(categorize_overlap)
    
    # 统计
    category_counts = df['category'].value_counts()
    print("\nCategory counts:")
    print(f"  Good (≥0.60): {category_counts.get('Good', 0)}")
    print(f"  Borderline (0.50-0.60): {category_counts.get('Borderline', 0)}")
    print(f"  Poor (<0.50): {category_counts.get('Poor', 0)}")
    
    # 颜色映射
    color_map = {
        'Good': '#2e7d32',      # 深绿色
        'Borderline': '#f57c00',  # 橙色
        'Poor': '#c62828'        # 深红色
    }
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
    ax.set_facecolor('white')
    
    # 设置y轴
    ax.set_ylabel('PCA Overlap', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    
    # 先添加明显的分类阈值横线（在条形图之前，这样横线在条形图下方）
    # Good阈值线 (y=0.60) - 绿色
    ax.axhline(y=0.60, color='#2e7d32', linestyle='--', linewidth=2.5, alpha=0.8, zorder=1)
    
    # Borderline阈值线 (y=0.50) - 红色
    ax.axhline(y=0.50, color='#c62828', linestyle='--', linewidth=2.5, alpha=0.8, zorder=1)
    
    # 准备数据
    participants = df['participant_id'].values
    overlaps = df['pca_overlap'].values
    colors = [color_map[cat] for cat in df['category'].values]
    
    # 绘制条形图（在横线之后，zorder更高）
    bars = ax.bar(range(len(participants)), overlaps, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=0.5, zorder=2)
    
    # 设置x轴标签
    ax.set_xticks(range(len(participants)))
    ax.set_xticklabels([f'P{pid}' for pid in participants], rotation=45, ha='right')
    
    # 添加数值标签
    for i, (bar, overlap) in enumerate(zip(bars, overlaps)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{overlap:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=color_map['Good'], alpha=0.8, edgecolor='black', linewidth=0.5, label='Good (≥0.60)'),
        mpatches.Patch(facecolor=color_map['Borderline'], alpha=0.8, edgecolor='black', linewidth=0.5, label='Borderline (0.50-0.60)'),
        mpatches.Patch(facecolor=color_map['Poor'], alpha=0.8, edgecolor='black', linewidth=0.5, label='Poor (<0.50)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False)
    
    # 标题
    ax.set_title('PCA Overlap Across Participants After Fine-tuning', fontsize=18, fontweight='bold', pad=20)
    
    # 网格
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 布局
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, 'pca_overlap_bar.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    plt.close()
    
    return df


def plot_pca_fd16_comparison(df, output_dir=OUTPUT_DIR):
    """绘制PCA_overlap和FD16的对比图（如果FD16存在）"""
    if 'FD16' not in df.columns:
        print("\nFD16 column not found, skipping comparison plot.")
        return
    
    # 按participant_id排序
    df = df.sort_values('participant_id').copy()
    
    # 创建双y轴图
    fig, ax1 = plt.subplots(figsize=(14, 6), facecolor='white')
    ax1.set_facecolor('white')
    
    participants = df['participant_id'].values
    x_pos = np.arange(len(participants))
    
    # 左y轴：PCA_overlap
    color1 = '#1976d2'
    ax1.set_xlabel('Participant', fontsize=16, fontweight='bold')
    ax1.set_ylabel('PCA Overlap', fontsize=16, fontweight='bold', color=color1)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'P{pid}' for pid in participants], rotation=45, ha='right')
    
    # 添加分类阈值横线（在条形图之前）
    # Good阈值线 (y=0.60) - 绿色
    ax1.axhline(y=0.60, color='#2e7d32', linestyle='--', linewidth=2.5, alpha=0.8, zorder=1)
    # Borderline阈值线 (y=0.50) - 红色
    ax1.axhline(y=0.50, color='#c62828', linestyle='--', linewidth=2.5, alpha=0.8, zorder=1)
    
    # 绘制PCA_overlap条形图（在横线之后，zorder更高）
    bars1 = ax1.bar(x_pos - 0.2, df['pca_overlap'].values, width=0.4, 
                    color=color1, alpha=0.7, label='PCA Overlap', edgecolor='black', linewidth=0.5, zorder=2)
    
    # 右y轴：FD16（注意FD16越小越好，所以可能需要反转或使用不同的表示）
    ax2 = ax1.twinx()
    color2 = '#d32f2f'
    bars2 = ax2.bar(x_pos + 0.2, df['FD16'].values, width=0.4,
                    color=color2, alpha=0.7, label='FD16', edgecolor='black', linewidth=0.5, zorder=2)
    ax2.set_ylabel('FD16', fontsize=16, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)
    
    # 标题
    ax1.set_title('PCA Overlap and FD16 Values Across Participants', fontsize=18, fontweight='bold', pad=20)
    
    # 网格
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    
    # 布局
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, 'pca_overlap_fd16_comparison.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def build_argparser():
    parser = argparse.ArgumentParser(description="Plot participant PCA overlap summary for OT or MW.")
    parser.add_argument("--condition", choices=["ot", "mw"], default="ot")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--fallback-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def resolve_paths(args):
    if args.condition == "mw":
        input_csv = args.input_csv or os.path.join(
            PROJECT_ROOT, "mw_pipeline", "results", "mw_distributions_2d", "results", "mw_distributions_2d_results.csv"
        )
        fallback_csv = args.fallback_csv
        output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "mw_pipeline", "results", "pca_overlap_summary")
    else:
        input_csv = args.input_csv or DATA_FILE_1
        fallback_csv = args.fallback_csv or DATA_FILE_2
        output_dir = args.output_dir or OUTPUT_DIR
    return input_csv, fallback_csv, output_dir


def main():
    """主函数"""
    args = build_argparser().parse_args()
    input_csv, fallback_csv, output_dir = resolve_paths(args)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"Plotting PCA Overlap Summary ({args.condition.upper()})")
    print("=" * 80)
    
    # 加载数据
    df = load_data(input_csv, fallback_csv)
    print(f"\nLoaded {len(df)} participants")
    print(f"PCA_overlap range: [{df['pca_overlap'].min():.4f}, {df['pca_overlap'].max():.4f}]")
    print(f"PCA_overlap mean: {df['pca_overlap'].mean():.4f}")
    print(f"PCA_overlap std: {df['pca_overlap'].std():.4f}")
    
    # 绘制条形图
    df_categorized = plot_pca_overlap_bar(df, output_dir)
    
    # 绘制对比图（如果FD16存在）
    if 'FD16' in df.columns:
        plot_pca_fd16_comparison(df_categorized, output_dir)
        print(f"\nFD16 range: [{df['FD16'].min():.4f}, {df['FD16'].max():.4f}]")
        print(f"FD16 mean: {df['FD16'].mean():.4f}")
        print(f"FD16 std: {df['FD16'].std():.4f}")
    
    # 保存分类结果
    output_csv = os.path.join(output_dir, 'pca_overlap_categorized.csv')
    df_categorized.to_csv(output_csv, index=False)
    print(f"\nSaved categorized data to: {output_csv}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
