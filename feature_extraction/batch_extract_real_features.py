#!/usr/bin/env python3
"""
批量提取 participant_data 文件夹下真实数据的特征

遍历 participant_data/ 文件夹下的所有 CSV 文件（如 participant_1_real_data.csv），
提取特征并保存为 p1_real.csv 格式。
"""

import os
import sys
import re
import glob
import argparse
import pandas as pd
import numpy as np

# 添加当前目录到路径，以便导入 feature_extraction 模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from feature_extraction import (
    extract_all_features,
    extract_erp_features,
    compute_Wst
)

# 配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PARTICIPANT_DATA_DIR = os.path.join(PROJECT_ROOT, 'participant_data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'real_ot')  # 默认保存在 feature_extraction/real_ot/


def extract_participant_id(filename):
    """
    从文件名提取 participant_id
    
    Args:
        filename: 文件名，如 "participant_1_real_data.csv" 或 "participant_11_real_data.csv"
    
    Returns:
        int: participant_id，如 1, 11，如果无法提取则返回 None
    """
    match = re.search(r'participant_(\d+)_real_data', filename)
    if match:
        return int(match.group(1))
    return None


def process_real_data_file(csv_file, output_dir):
    """
    处理单个真实数据 CSV 文件
    
    Args:
        csv_file: CSV 文件完整路径
    
    Returns:
        bool: 是否成功处理
    """
    filename = os.path.basename(csv_file)
    participant_id = extract_participant_id(filename)
    
    if participant_id is None:
        print(f"✗ Cannot extract participant_id from filename: {filename}")
        return False
    
    print(f"\n{'='*80}")
    print(f"Processing: {filename} (P{participant_id})")
    print(f"{'='*80}")
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"✗ CSV file not found: {csv_file}")
        return False
    
    # 设置输出文件名：p1_real.csv
    output_filename = f"p{participant_id}_real.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # 检查输出文件是否已存在
    if os.path.exists(output_path):
        print(f"⚠ Output file already exists: {output_path}")
        print(f"  Skipping... (delete it first if you want to reprocess)")
        return True
    
    try:
        # 读取无表头的 CSV 文件
        print(f"Reading CSV file (no header)...")
        df = pd.read_csv(csv_file, header=None)
        
        if df.empty:
            print(f"✗ CSV file is empty: {csv_file}")
            return False
        
        print(f"  Loaded {len(df)} trials, {df.shape[1]} timepoints")
        
        # 使用行号作为 SampleID
        sample_ids = pd.Series(range(1, len(df) + 1))
        time_series_data = df
        
        # 创建时间点数组
        times = np.linspace(-0.4, 1.2, time_series_data.shape[1]) * 1000  # 转换为毫秒
        
        # 初始化特征列表
        features_list = []
        
        # 处理每个trial
        total_trials = len(df)
        print(f"Extracting features from {total_trials} trials...")
        
        for idx, (sample_id, row) in enumerate(zip(sample_ids, time_series_data.itertuples(index=False))):
            if (idx + 1) % 50 == 0 or (idx + 1) == total_trials:
                print(f'  Processing trial {idx + 1}/{total_trials} (SampleID: {sample_id})')
            
            # 转换为numpy数组
            signal = np.array(row)
            
            # 计算小波变换（不绘制图）
            Wst, tlag, scale = compute_Wst(signal, times, plot_on=False, save_path=None)
            
            # 提取所有特征
            features = extract_all_features(signal, times)
            
            # 提取ERP特征
            p1_amp, n1_amp, p3_amp, p1_lat, n1_lat, p3_lat = extract_erp_features(Wst, tlag, scale, times)
            
            # 创建特征字典，只包含16个指定特征
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
        features_df.to_csv(output_path, index=False)
        
        print(f"✓ Features extracted successfully!")
        print(f"  Input: {csv_file}")
        print(f"  Output: {output_path}")
        print(f"  Rows: {len(features_df)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_argparser():
    parser = argparse.ArgumentParser(description="Batch feature extraction from participant real-data CSV files.")
    parser.add_argument("--condition", choices=["ot", "mw"], default="ot")
    parser.add_argument("--participant-data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def resolve_paths(args):
    participant_data_dir = args.participant_data_dir
    output_dir = args.output_dir
    if participant_data_dir is None:
        participant_data_dir = PARTICIPANT_DATA_DIR
        if args.condition == "mw":
            participant_data_dir = os.path.join(PROJECT_ROOT, "mw_pipeline", "participant_data")
    if output_dir is None:
        output_dir = OUTPUT_DIR
        if args.condition == "mw":
            output_dir = os.path.join(PROJECT_ROOT, "mw_pipeline", "features", "real_mw")
    return participant_data_dir, output_dir


def main():
    """主函数"""
    args = build_argparser().parse_args()
    participant_data_dir, output_dir = resolve_paths(args)

    print("="*80)
    print(f"Batch Feature Extraction from Real Data ({args.condition.upper()})")
    print("="*80)
    
    # 检查 participant_data 目录是否存在
    if not os.path.exists(participant_data_dir):
        print(f"✗ Participant data directory not found: {participant_data_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Participant data directory: {participant_data_dir}")
    print(f"Output directory: {output_dir}")
    
    # 获取所有 CSV 文件
    pattern = os.path.join(participant_data_dir, 'participant_*_real_data.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"✗ No participant_*_real_data.csv files found in {participant_data_dir}")
        return
    
    # 按文件名排序
    csv_files.sort()
    
    print(f"\nFound {len(csv_files)} CSV files to process:")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    
    # 处理每个 CSV 文件
    results = []
    for idx, csv_file in enumerate(csv_files, 1):
        success = process_real_data_file(csv_file, output_dir)
        results.append({'file': os.path.basename(csv_file), 'success': success})
    
    # 打印总结
    print(f"\n{'='*80}")
    print("Processing Summary")
    print("="*80)
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Total files: {len(csv_files)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successfully processed:")
        for r in successful:
            print(f"  - {r['file']}")
    
    if failed:
        print(f"\n✗ Failed:")
        for r in failed:
            print(f"  - {r['file']}")
    
    print(f"\nAll feature files saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

