import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import (
    process_averaged_trials, 
    process_single_trial,
    extract_all_features,
    hjorth_parameters,
    petrosian_fractal_dimension,
    psd_bandpower,
    calculate_band_entropy,
    compute_Wst,
    extract_erp_features
)
import os
from datetime import datetime
from EntropyHub import SpecEn, ApEn



def get_dataset_name(input_file):
    """
    从输入文件路径中提取数据集名称
    
    Parameters:
    -----------
    input_file : str
        输入文件路径
        
    Returns:
    --------
    str
        数据集名称（不包含扩展名）
    """
    return os.path.splitext(os.path.basename(input_file))[0]

def create_output_dirs(input_file, base_output_dir=None):
    """
    创建输出目录结构
    
    Parameters:
    -----------
    input_file : str
        输入文件路径
    base_output_dir : str
        基础输出目录，如果为None则使用相对路径
        
    Returns:
    --------
    tuple
        (results_dir, plots_dir, features_dir)
    """
    # 获取数据集名称
    dataset_name = get_dataset_name(input_file)
    
    # 设置基础输出目录
    if base_output_dir is None:
        # 使用相对路径，适合HPC环境
        base_output_dir = os.path.join('feature_extraction', 'results')
    
    # 创建主目录
    results_dir = os.path.join(base_output_dir, dataset_name)
    plots_dir = os.path.join(results_dir, 'plots')
    features_dir = os.path.join(results_dir, 'features')
    
    # 创建子目录
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(os.path.join(plots_dir, 'wavelet_plots_single'), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, 'wavelet_plots_averaged'), exist_ok=True)
    
    return results_dir, plots_dir, features_dir

def process_single_trials(input_file, output_file, plot_on=True, scale_factor=1.0):
    """
    处理单个trials并提取所有特征
    
    Parameters:
    -----------
    input_file : str
        输入CSV文件路径
    output_file : str
        输出特征CSV文件路径
    plot_on : bool
        是否绘制时频图
    scale_factor : float
        EEG数据放大倍数
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 检查第一列是否为SampleID
    first_col = df.columns[0]
    if first_col.lower() in ['sampleid', 'sample_id', 'id', 'participant', 'participant_id']:
        sample_ids = df.iloc[:, 0]  # 第一列是编号
        time_series_data = df.iloc[:, 1:]  # 其余列是EEG数据
        print(f"Using {first_col} as SampleID")
    else:
        # 如果没有SampleID列，使用行号作为ID
        sample_ids = pd.Series(range(1, len(df) + 1))  # 从1开始的编号
        time_series_data = df  # 所有列都是EEG数据
        print("No SampleID column found, using row numbers as IDs")
    
    # 放大EEG数据
    time_series_data = time_series_data * scale_factor
    
    # 创建时间点数组
    times = np.linspace(-0.4, 1.2, time_series_data.shape[1]) * 1000  # 转换为毫秒
    
    # 创建输出目录
    _, plots_dir, _ = create_output_dirs(input_file)
    wavelet_plot_dir = os.path.join(plots_dir, 'wavelet_plots_single')
    
    # 初始化特征列表
    features_list = []
    
    # 处理每个trial
    total_trials = len(df)
    for idx, (sample_id, row) in enumerate(zip(sample_ids, time_series_data.itertuples(index=False))):
        print(f'Processing trial {idx + 1}/{total_trials} (SampleID: {sample_id})')
        
        # 转换为numpy数组
        signal = np.array(row)
        
        # 设置保存路径
        save_path = os.path.join(wavelet_plot_dir, f'wavelet_plot_{idx + 1}.png')
        
        # 计算小波变换
        Wst, tlag, scale = compute_Wst(signal, times, plot_on=plot_on, save_path=save_path)
        
        # 提取所有特征
        features = extract_all_features(signal, times)
        
        # 提取ERP特征
        p1_amp, n1_amp, p3_amp, p1_lat, n1_lat, p3_lat = extract_erp_features(Wst, tlag, scale, times)
        
        # 创建新的特征字典，只包含16个指定特征
        ordered_features = {
            'SampleID': sample_id,
            # ERP特征 (2个)
            'P3_Amplitude': p3_amp,
            'P3_Latency': p3_lat,
            # Hjorth参数 (3个)
            'Activity': features['Activity'],
            'Mobility': features['Mobility'],
            'Complexity': features['Complexity'],
            # 分形维数 (1个)
            'Petrosian_Fractal_Dimension': features['Petrosian Fractal Dimension'],
            # 各频段功率谱密度 (5个)
            'Delta_PSD': features['Delta PSD'],
            'Theta_PSD': features['Theta PSD'],
            'Alpha_PSD': features['Alpha PSD'],
            'Beta_PSD': features['Beta PSD'],
            'Gamma_PSD': features['Gamma PSD'],
            # 各频段谱熵 (5个)
            'Delta_Entropy': features['Delta Entropy'],
            'Theta_Entropy': features['Theta Entropy'],
            'Alpha_Entropy': features['Alpha Entropy'],
            'Beta_Entropy': features['Beta Entropy'],
            'Gamma_Entropy': features['Gamma Entropy']
        }
        
        # 添加到特征列表
        features_list.append(ordered_features)
    
    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    
    # 保存特征到CSV
    features_df.to_csv(output_file, index=False)
    print(f'Single trial features saved to {output_file}')
    
    return features_df

def process_averaged_erp_features(input_file, output_file, n_trials=20, random_seed=42, plot=True):
    """
    随机抽取trials并计算平均ERP特征，直到处理完所有数据
    
    Parameters:
    -----------
    input_file : str
        输入CSV文件路径
    output_file : str
        输出CSV文件路径
    n_trials : int
        每组要抽取的trial数量
    random_seed : int
        随机种子
    plot : bool
        是否绘制时频图
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 创建输出目录
    _, plots_dir, _ = create_output_dirs(input_file)
    wavelet_plot_dir = os.path.join(plots_dir, 'wavelet_plots_averaged')
    
    # 初始化特征列表
    features_list = []
    
    # 获取所有可用的trial索引
    all_indices = np.arange(len(df))
    np.random.shuffle(all_indices)  # 随机打乱索引
    
    # 计算可以完整处理的组数
    n_complete_groups = len(all_indices) // n_trials
    
    # 处理每一组
    for group_idx in range(n_complete_groups):
        start_idx = group_idx * n_trials
        end_idx = start_idx + n_trials
        
        # 获取当前组的trial索引
        selected_indices = all_indices[start_idx:end_idx]
        selected_df = df.iloc[selected_indices]
        
        # 判断是否有participant编号
        if len(df.columns) == 410:  # 有编号的情况
            time_series_data = selected_df.iloc[:, 1:]  # 其余列是EEG数据
        elif len(df.columns) == 409:  # 没有编号的情况
            time_series_data = selected_df  # 所有列都是EEG数据
        else:
            raise ValueError(f"Unexpected number of columns: {len(df.columns)}")
        
        # 创建时间点数组
        times = np.linspace(-0.4, 1.2, time_series_data.shape[1]) * 1000  # 转换为毫秒
        
        # 计算平均值
        averaged_signal = np.mean(time_series_data.values, axis=0)
        
        # 设置保存路径
        save_path = os.path.join(wavelet_plot_dir, f'wavelet_plot_group_{group_idx + 1}.png')
        
        # 计算小波变换
        Wst, tlag, scale = compute_Wst(averaged_signal, times, plot_on=plot, save_path=save_path)
        
        # 提取ERP特征
        p1_amp, n1_amp, p3_amp, p1_lat, n1_lat, p3_lat = extract_erp_features(Wst, tlag, scale, times)
        
        # 创建特征字典
        features = {
            'GroupID': group_idx + 1,
            'SelectedTrials': ','.join(map(str, selected_indices)),
            'P1_Amplitude': p1_amp,
            'P1_Latency': p1_lat,
            'N1_Amplitude': n1_amp,
            'N1_Latency': n1_lat,
            'P3_Amplitude': p3_amp,
            'P3_Latency': p3_lat
        }
        
        # 添加到特征列表
        features_list.append(features)
        
        print(f"Processed group {group_idx + 1}/{n_complete_groups}")
        if plot:
            print(f"Saved plot to {save_path}")
    
    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    
    # 保存特征到CSV
    features_df.to_csv(output_file, index=False)
    print(f'Averaged ERP features saved to {output_file}')
    
    return features_df

def main():
    """
    主函数 - 特征提取程序入口
    修改这里的路径来适应不同的数据文件
    """
    print("="*80)
    print("Feature Extraction Program")
    print("="*80)
    
    # ========================================
    # 数据路径设置 - 在这里修改你的数据路径
    # 目标: EEG_data/Gall_Dlast_ep_90.csv
    # 从 feature_extraction 目录运行时，使用如下相对路径:
    # ../EEG_data/Gall_Dlast_ep_90.csv
    # ========================================
    input_file = '../EEG_data/Gall_Dlast_ep_90.csv'
    
    print(f"Input data file: {input_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please check the file path in the main() function")
        return
    
    # 创建输出目录结构
    results_dir, plots_dir, features_dir = create_output_dirs(input_file)
    print(f"Output directory: {results_dir}")
    
    # 设置输出文件路径
    selected_features_output_file = os.path.join(features_dir, 'selected_16_features.csv')
    averaged_erp_output_file = os.path.join(features_dir, 'averaged_erp_features.csv')
    
    # 处理选定的16个特征
    print("\nProcessing selected 16 features for single trials...")
    single_features_df = process_single_trials(input_file, selected_features_output_file, plot_on=True)
    
    # 处理随机抽取的trials的平均ERP特征
    print("\nProcessing averaged ERP features from random trials...")
    averaged_features_df = process_averaged_erp_features(input_file, averaged_erp_output_file, n_trials=20)
    
    print("\n" + "="*80)
    print("Processing completed!")
    print("="*80)
    print(f"Results saved in: {os.path.abspath(results_dir)}")
    print(f"Selected 16 features saved to: {os.path.abspath(selected_features_output_file)}")
    print(f"Averaged ERP features saved to: {os.path.abspath(averaged_erp_output_file)}")
    print("Plots saved in:")
    print(f"- Single trial plots: {os.path.abspath(os.path.join(plots_dir, 'wavelet_plots_single'))}")
    print(f"- Averaged trial plots: {os.path.abspath(os.path.join(plots_dir, 'wavelet_plots_averaged'))}")

if __name__ == "__main__":
    main() 