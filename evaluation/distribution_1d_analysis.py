#!/usr/bin/env python3
"""
distribution_1d_analysis.py

绘制1D分布曲线对比图，计算overlap指标
- 读取 EEG_features/selected_features/ 下的特征文件
- 相同特征画在一个子图上（4×4，共16个子图）
- real用黑色，其他用不同颜色
- 计算real与其他数据的overlap
- 结果保存到 results/3.distribution_1D/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import glob
from sklearn.preprocessing import StandardScaler

plt.rcParams['mathtext.default'] = 'rm'     # 直立，不斜体
plt.rcParams['mathtext.fontset'] = 'dejavusans'   # 使用Arial类似的字体
plt.rcParams['font.family'] = 'Arial'       # 设置默认字体为Arial
plt.rcParams['font.sans-serif'] = ['Arial'] # 设置sans-serif字体列表
plt.rcParams['axes.titlesize'] = 16         # 设置标题字体大小
plt.rcParams['axes.labelsize'] = 14         # 设置轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12        # 设置x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12        # 设置y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12        # 设置图例字体大小


def load_feature_data(features_dir):
    """
    加载所有特征文件
    
    Args:
        features_dir: 特征文件目录
    
    Returns:
        dict: {filename: DataFrame}
    """
    csv_files = glob.glob(os.path.join(features_dir, "*.csv"))
    data_dict = {}
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        try:
            df = pd.read_csv(csv_file)
            # 移除非特征列
            if 'SampleID' in df.columns:
                df = df.drop('SampleID', axis=1)
            if 'Label' in df.columns:
                df = df.drop('Label', axis=1)
            data_dict[filename] = df
            print(f"Loaded {filename}: {df.shape}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return data_dict

def calculate_overlap_1d(real_data, fake_data):
    """
    计算两个1D分布的overlap
    
    Args:
        real_data: 真实数据 (1D array)
        fake_data: 生成数据 (1D array)
    
    Returns:
        float: overlap值 [0, 1]
    """
    # 创建KDE
    kde_real = gaussian_kde(real_data)
    kde_fake = gaussian_kde(fake_data)
    
    # 确定数据范围
    min_val = min(real_data.min(), fake_data.min())
    max_val = max(real_data.max(), fake_data.max())
    
    # 创建网格点
    x = np.linspace(min_val, max_val, 1000)
    
    # 计算密度
    density_real = kde_real(x)
    density_fake = kde_fake(x)
    
    # 归一化
    density_real = density_real / np.trapz(density_real, x)
    density_fake = density_fake / np.trapz(density_fake, x)
    
    # 计算overlap
    overlap = np.trapz(np.minimum(density_real, density_fake), x)
    
    return overlap

def plot_distribution_comparison(data_dict, output_dir):
    """
    绘制分布对比图
    
    Args:
        data_dict: 数据字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 找到real数据
    real_file = None
    for filename in data_dict.keys():
        if 'real' in filename.lower():
            real_file = filename
            break
    
    if real_file is None:
        print("Error: No real data file found!")
        return
    
    real_data = data_dict[real_file]
    
    # 按照预定义的顺序排列文件
    file_order = [
        '0.real_mw.csv',           # 0: real
        '1.scratch_ep_700.csv',    # 1: scratch
        '2.no_train.csv',          # 2: no training
        '3.full_tune_ep_270.csv',  # 3: full
        '4.Gall_Dlast_ep_90.csv',  # 4: partial 1
        '5.first_last_ep_120.csv', # 5: partial 2
        '6.probe_ep_270.csv'       # 6: partial 3
    ]
    
    # 按照预定义顺序过滤和排列文件
    other_files = []
    for filename in file_order:
        if filename in data_dict and filename != real_file:
            other_files.append(filename)
    
    print(f"Real data: {real_file}")
    print(f"Other files (in order): {other_files}")
    
    # 获取所有特征名
    feature_names = real_data.columns.tolist()
    print(f"Features: {feature_names}")
    
    # 设置颜色
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # 创建4x4子图
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # 存储overlap结果
    overlap_results = []
    
    for i, feature in enumerate(feature_names):
        if i >= 16:  # 最多16个子图
            break
            
        ax = axes[i]
        
        # 计算所有数据的合理x轴范围（基于百分位数，避免极端值）
        all_values = []
        for filename in data_dict.keys():
            if filename in data_dict and feature in data_dict[filename].columns:
                values = data_dict[filename][feature].dropna()
                all_values.extend(values.tolist())
        
        if len(all_values) > 0:
            # 使用1%和99%百分位数作为x轴范围，避免极端值影响图形可读性
            x_min = np.percentile(all_values, 1)
            x_max = np.percentile(all_values, 99)
            x_range = np.linspace(x_min, x_max, 200)
        else:
            x_range = np.linspace(0, 1, 200)
        
        # 绘制real数据（黑色）
        real_values = real_data[feature].dropna()
        if len(real_values) > 0:
            kde_real = gaussian_kde(real_values)
            density_real = kde_real(x_range)
            density_real = density_real / np.trapz(density_real, x_range)  # 归一化
            ax.plot(x_range, density_real, color='black', linewidth=2, label='real')
        
        # 绘制其他数据
        for j, filename in enumerate(other_files):
            if j >= len(colors):
                break
                
            fake_data = data_dict[filename]
            if feature in fake_data.columns:
                fake_values = fake_data[feature].dropna()
                if len(fake_values) > 0:
                    kde_fake = gaussian_kde(fake_values)
                    density_fake = kde_fake(x_range)  # 使用统一的x_range
                    density_fake = density_fake / np.trapz(density_fake, x_range)  # 归一化
                    
                    # 使用自定义标签映射（按文件顺序0-6）
                    label_mapping = {
                        'real': 'real',                    # 0
                        'scratch_ep_700': 'scratch',       # 1
                        'no_train': 'no training',         # 2
                        'full_tune_ep_270': 'full',        # 3
                        'gall_dlast_ep_90': 'partial 1',   # 4
                        'first_last_ep_120': 'partial 2', # 5
                        'probe_ep_270': 'partial 3'        # 6
                    }
                    
                    # 根据文件名确定标签
                    filename_lower = filename.lower()
                    label = filename_lower.replace('.csv', '')
                    for key, value in label_mapping.items():
                        if key in filename_lower:
                            label = value
                            break
                    
                    ax.plot(x_range, density_fake, color=colors[j], linewidth=1.5, 
                           alpha=0.8, label=label)
                    
                    # 计算overlap
                    overlap = calculate_overlap_1d(real_values, fake_values)
                    overlap_results.append({
                        'feature': feature,
                        'fake_file': filename,
                        'overlap': overlap
                    })
        
        # 特征名称映射
        feature_mapping = {
            'P3_Amplitude': r'$P3_{\mathrm{A}}$',
            'P3_Latency' : r'$P3_{\mathrm{L}}$',

            'Activity'  : 'HA',
            'Mobility'  : 'HM',
            'Complexity': 'HC',
            'Petrosian_Fractal_Dimension': 'PFD',

            'Delta_PSD':  r'$PSD_{\delta}$',
            'Theta_PSD':  r'$PSD_{\theta}$',
            'Alpha_PSD':  r'$PSD_{\alpha}$',
            'Beta_PSD':   r'$PSD_{\beta}$',
            'Gamma_PSD':  r'$PSD_{\gamma}$',

            'Delta_Entropy': r'$E_{\delta}$',
            'Theta_Entropy': r'$E_{\theta}$',
            'Alpha_Entropy': r'$E_{\alpha}$',
            'Beta_Entropy':  r'$E_{\beta}$',
            'Gamma_Entropy': r'$E_{\gamma}$',
        }

        
        # 使用映射后的标题，如果没有映射则使用原名称
        title = feature_mapping.get(feature, feature)
        ax.set_title(f'{title}', fontsize=16, fontfamily='Arial')
        
        # 只在每行的第一个子图显示y轴标签
        if i % 4 == 0:
            ax.set_ylabel('density', fontsize=22, fontfamily='Arial')
        else:
            ax.set_ylabel('')
        
        # 只在每行的第一个子图显示x轴标签
        if i >= 12:  # 最后一行
            ax.set_xlabel('value', fontsize=22, fontfamily='Arial')
        else:
            ax.set_xlabel('')
        
        ax.grid(False)  # 移除网格
        
        # 设置坐标轴刻度标签字体
        ax.tick_params(axis='both', which='major', labelsize=12)
        for label in ax.get_xticklabels():
            label.set_fontfamily('Arial')
        for label in ax.get_yticklabels():
            label.set_fontfamily('Arial')
        
        # 只在第三行第一个子图（索引8）显示图例
        if i == 8:
            legend = ax.legend(fontsize=12, loc='upper right', frameon=False)
            # 设置图例字体为Arial
            for text in legend.get_texts():
                text.set_fontfamily('Arial')
    
    # 隐藏多余的子图
    for i in range(len(feature_names), 16):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_dir, 'distribution_1d_comparison.png')
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plot saved to: {save_path}")
    
    # 重新组织overlap结果为矩阵格式
    overlap_matrix = []
    
    # 获取所有特征
    all_features = sorted(list(set([r['feature'] for r in overlap_results])))
    
    # 使用与绘图时相同的文件顺序
    ordered_fake_files = []
    for filename in file_order:
        if filename in data_dict and filename != real_file:
            ordered_fake_files.append(filename)
    
    # 创建标签映射
    label_mapping = {
        'real': 'real',                    # 0
        'scratch_ep_700': 'scratch',       # 1
        'no_train': 'no training',         # 2
        'full_tune_ep_270': 'full',        # 3
        'gall_dlast_ep_90': 'partial 1',   # 4
        'first_last_ep_120': 'partial 2', # 5
        'probe_ep_270': 'partial 3'        # 6
    }
    
    # 使用有序的文件列表创建列名
    columns = ['feature'] + [label_mapping.get(f.replace('.csv', '').lower(), f.replace('.csv', '')) for f in ordered_fake_files]
    
    # 填充数据
    for feature in all_features:
        row = {'feature': feature}
        
        for fake_file in ordered_fake_files:
            # 找到对应的overlap值
            overlap_value = None
            for result in overlap_results:
                if result['feature'] == feature and result['fake_file'] == fake_file:
                    overlap_value = result['overlap']
                    break
            
            # 使用标签作为列名
            label = label_mapping.get(fake_file.replace('.csv', '').lower(), fake_file.replace('.csv', ''))
            row[label] = overlap_value if overlap_value is not None else 0.0000
        
        overlap_matrix.append(row)
    
    # 转换为DataFrame
    overlap_df = pd.DataFrame(overlap_matrix)
    
    # 格式化数值，保留4位小数，包括0
    for col in overlap_df.columns:
        if col != 'feature':
            overlap_df[col] = overlap_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "0.0000")
    
    # 保存结果
    overlap_path = os.path.join(output_dir, 'overlap_results.csv')
    overlap_df.to_csv(overlap_path, index=False)
    
    print(f"Overlap results saved to: {overlap_path}")
    
    # 打印overlap统计
    print("\nOverlap Statistics:")
    print("=" * 50)
    for feature in all_features:
        feature_overlaps = []
        for result in overlap_results:
            if result['feature'] == feature:
                feature_overlaps.append(result['overlap'])
        
        if feature_overlaps:
            feature_overlaps = np.array(feature_overlaps)
            print(f"{feature}:")
            print(f"  Mean: {feature_overlaps.mean():.4f}")
            print(f"  Std:  {feature_overlaps.std():.4f}")
            print(f"  Min:  {feature_overlaps.min():.4f}")
            print(f"  Max:  {feature_overlaps.max():.4f}")
            print()

def main():
    """主函数"""
    print("=" * 80)
    print("1D Distribution Analysis")
    print("=" * 80)
    
    # Get project root (parent of evaluation directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 设置路径
    features_dir = os.path.join(project_root, "EEG_features", "selected_features")
    output_dir = os.path.join(project_root, "results", "3.distribution_1D")
    
    # 检查输入目录
    if not os.path.exists(features_dir):
        print(f"Error: Features directory not found: {features_dir}")
        return
    
    # 加载数据
    print("Loading feature data...")
    data_dict = load_feature_data(features_dir)
    
    if not data_dict:
        print("Error: No feature files found!")
        return
    
    # 绘制分布对比图
    print("Generating distribution plots...")
    plot_distribution_comparison(data_dict, output_dir)
    
    print("=" * 80)
    print("Analysis completed!")
    print(f"Results saved in: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
