#!/usr/bin/env python3
"""
p_value_analysis.py

计算real数据与其他fake数据的p值
- 读取 EEG_features/selected_features/ 下的特征文件
- real: 0.real_mw.csv
- fake: 1-6对应的csv文件
- 计算KS test和Mann-Whitney U test的p值
- 结果保存到 results/4.p_value_analysis/
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import glob

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

def bootstrap_ks_test_Z(data, feature, subsample_size1=100, subsample_size2=100, n_iterations=1000):
    """
    Bootstrap KS test with Z-score calculation
    
    Args:
        data: DataFrame with 'label' column (0=real, 1=fake)
        feature: feature name
        subsample_size1: subsample size for group 0
        subsample_size2: subsample size for group 1
        n_iterations: number of bootstrap iterations
    
    Returns:
        tuple: (mean_p_value, mean_ks_statistic)
    """
    group_0 = data[data['label'] == 0][feature].to_numpy()
    group_1 = data[data['label'] == 1][feature].to_numpy()

    p_values = []
    ks_statistics = []

    for _ in range(n_iterations):
        sample_0 = np.random.choice(group_0, size=min(subsample_size1, len(group_0)), replace=True)
        sample_1 = np.random.choice(group_1, size=min(subsample_size2, len(group_1)), replace=True)

        ks_stat, p_value = ks_2samp(sample_0, sample_1)
        p_values.append(p_value)
        ks_statistics.append(ks_stat)

    p_adjusted = multipletests(p_values, method='fdr_bh')[1]
    z_values = [norm.ppf(1 - p / 2) for p in p_adjusted]

    mean_z = np.mean(z_values)
    mean_p = 2 * (1 - norm.cdf(mean_z))
    mean_ks_stat = np.mean(ks_statistics)

    return mean_p, mean_ks_stat

def mann_whitney_u_test(data, feature, subsample_size=100, n_iterations=1000):
    """
    Bootstrap Mann-Whitney U test
    
    Args:
        data: DataFrame with 'label' column (0=real, 1=fake)
        feature: feature name
        subsample_size: subsample size for each group
        n_iterations: number of bootstrap iterations
    
    Returns:
        tuple: (mean_p_value, mean_u_statistic)
    """
    group_0 = data[data['label'] == 0][feature].to_numpy()
    group_1 = data[data['label'] == 1][feature].to_numpy()

    p_values = []
    u_statistics = []
    z_scores = []

    for _ in range(n_iterations):
        # Randomly sample from the two groups
        sample_0 = np.random.choice(group_0, size=min(subsample_size, len(group_0)), replace=True)
        sample_1 = np.random.choice(group_1, size=min(subsample_size, len(group_1)), replace=True)

        # Perform Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(sample_0, sample_1, alternative='two-sided')
        u_statistics.append(u_stat)

        mean_u = len(sample_0) * len(sample_1) / 2
        std_u = np.sqrt(len(sample_0) * len(sample_1) * (len(sample_0) + len(sample_1) + 1) / 12)
        z_score = (u_stat - mean_u) / std_u
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        z_scores.append(z_score)

        p_values.append(p_value)

    # Calculate mean U statistic and mean p-value
    mean_u = np.mean(u_statistics)
    mean_p = np.mean(p_values)
    return mean_p, mean_u

def calculate_p_values(data_dict, output_dir):
    """
    计算p值并保存结果
    
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
    other_files = [f for f in data_dict.keys() if f != real_file]
    
    print(f"Real data: {real_file}")
    print(f"Other files: {other_files}")
    
    # 获取所有特征名
    feature_names = real_data.columns.tolist()
    print(f"Features: {feature_names}")
    
    # 存储所有结果
    all_results = []
    
    # 标签映射
    label_mapping = {
        'real': 'real',                    # 0
        'scratch_ep_700': 'scratch',       # 1
        'no_train': 'no training',         # 2
        'full_tune_ep_270': 'full',        # 3
        'gall_dlast_ep_90': 'partial 1',   # 4
        'first_last_ep_120': 'partial 2', # 5
        'probe_ep_270': 'partial 3'        # 6
    }
    
    for fake_file in other_files:
        print(f"\nProcessing {fake_file}...")
        
        fake_data = data_dict[fake_file]
        
        # 确定标签
        filename_lower = fake_file.lower()
        label = filename_lower.replace('.csv', '')
        for key, value in label_mapping.items():
            if key in filename_lower:
                label = value
                break
        
        # 对每个特征计算p值
        for feature in feature_names:
            if feature in fake_data.columns:
                # 准备数据
                real_values = real_data[feature].dropna()
                fake_values = fake_data[feature].dropna()
                
                if len(real_values) > 0 and len(fake_values) > 0:
                    # 准备数据：real=0, fake=1
                    combined_data = pd.DataFrame({
                        'values': np.concatenate([real_values, fake_values]),
                        'label': np.concatenate([np.zeros(len(real_values)), np.ones(len(fake_values))])
                    })
                    
                    # Bootstrap KS test
                    ks_p, ks_stat = bootstrap_ks_test_Z(combined_data, 'values', n_iterations=100)
                    
                    # Bootstrap Mann-Whitney U test
                    mw_p, mw_stat = mann_whitney_u_test(combined_data, 'values', n_iterations=100)
                    
                    all_results.append({
                        'feature': feature,
                        'fake_file': fake_file,
                        'fake_label': label,
                        'ks_p_value': ks_p,
                        'ks_statistic': ks_stat,
                        'mann_whitney_p_value': mw_p,
                        'mann_whitney_statistic': mw_stat
                    })
                    
                    print(f"  {feature}: KS p={ks_p:.4f}, MW p={mw_p:.4f}")
    
    # 保存详细结果到CSV
    results_df = pd.DataFrame(all_results)
    results_df = results_df.round(4)  # 保留4位小数
    
    detailed_csv_path = os.path.join(output_dir, 'p_value_results_detailed.csv')
    results_df.to_csv(detailed_csv_path, index=False)
    
    print(f"\nDetailed results saved to: {detailed_csv_path}")
    
    # 创建矩阵格式的KS p-value表
    ks_matrix = []
    
    # 获取所有特征和所有fake文件
    all_features = sorted(list(set([r['feature'] for r in all_results])))
    all_fake_files = sorted(list(set([r['fake_file'] for r in all_results])))
    
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
    
    # 创建列名
    columns = ['feature'] + [label_mapping.get(f.replace('.csv', '').lower(), f.replace('.csv', '')) for f in all_fake_files]
    
    # 填充KS p-value数据
    for feature in all_features:
        row = {'feature': feature}
        
        for fake_file in all_fake_files:
            # 找到对应的KS p-value
            ks_p_value = None
            for result in all_results:
                if result['feature'] == feature and result['fake_file'] == fake_file:
                    ks_p_value = result['ks_p_value']
                    break
            
            # 使用标签作为列名
            label = label_mapping.get(fake_file.replace('.csv', '').lower(), fake_file.replace('.csv', ''))
            
            # 格式化数值，保留4位小数，p<=0.05加*
            if ks_p_value is not None:
                formatted_value = f"{ks_p_value:.4f}"
                if ks_p_value <= 0.05:
                    formatted_value += "*"
                row[label] = formatted_value
            else:
                row[label] = "0.0000"
        
        ks_matrix.append(row)
    
    # 转换为DataFrame并保存
    ks_df = pd.DataFrame(ks_matrix)
    ks_csv_path = os.path.join(output_dir, 'ks_p_value_matrix.csv')
    ks_df.to_csv(ks_csv_path, index=False)
    
    print(f"KS p-value matrix saved to: {ks_csv_path}")
    
    # 打印统计摘要
    print("\nP-value Statistics Summary:")
    print("=" * 60)
    
    for fake_label in results_df['fake_label'].unique():
        fake_results = results_df[results_df['fake_label'] == fake_label]
        print(f"\n{fake_label}:")
        print(f"  KS p-values - Mean: {fake_results['ks_p_value'].mean():.4f}, "
              f"Std: {fake_results['ks_p_value'].std():.4f}")
        print(f"  MW p-values - Mean: {fake_results['mann_whitney_p_value'].mean():.4f}, "
              f"Std: {fake_results['mann_whitney_p_value'].std():.4f}")
        
        # 统计显著差异 (p < 0.05)
        ks_significant = (fake_results['ks_p_value'] < 0.05).sum()
        mw_significant = (fake_results['mann_whitney_p_value'] < 0.05).sum()
        total_features = len(fake_results)
        
        print(f"  Significant differences (p < 0.05): KS={ks_significant}/{total_features}, "
              f"MW={mw_significant}/{total_features}")

def main():
    """主函数"""
    print("=" * 80)
    print("P-value Analysis")
    print("=" * 80)
    
    # Get project root (parent of evaluation directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 设置路径
    features_dir = os.path.join(project_root, "EEG_features", "selected_features")
    output_dir = os.path.join(project_root, "results", "4.p_value_analysis")
    
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
    
    # 计算p值
    print("Calculating p-values...")
    calculate_p_values(data_dict, output_dir)
    
    print("=" * 80)
    print("Analysis completed!")
    print(f"Results saved in: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
