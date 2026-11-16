#!/usr/bin/env python3
"""
批量提取 !selection/ 文件夹下所有 CSV 文件的特征

遍历 finetuned_ot_participant/!selection/ 下的所有文件夹，
找到 generated_trials_epoch_数字.csv 文件，提取特征并保存。
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
SELECTION_DIR = os.path.join(PROJECT_ROOT, 'ot_pipeline', 'finetuned_ot_participant', '!selection')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'ot_pipeline', 'features', 'ot_feature')


def extract_participant_and_epoch(folder_name):
    """
    从文件夹名提取 participant_id 和 epoch
    
    Args:
        folder_name: 文件夹名，如 "p1_epoch_250"
    
    Returns:
        tuple: (participant_id, epoch) 或 (None, None) 如果无法提取
    """
    match = re.match(r'p(\d+)_epoch_(\d+)', folder_name)
    if match:
        participant_id = int(match.group(1))
        epoch = int(match.group(2))
        return participant_id, epoch
    return None, None


def find_epoch_csv(folder_path):
    """
    在文件夹中查找 generated_trials_epoch_数字.csv 文件
    
    Args:
        folder_path: 文件夹路径
    
    Returns:
        str: CSV 文件路径，如果没找到返回 None
    """
    pattern = os.path.join(folder_path, 'generated_trials_epoch_*.csv')
    csv_files = glob.glob(pattern)
    
    if len(csv_files) == 0:
        return None
    elif len(csv_files) == 1:
        return csv_files[0]
    else:
        # 如果有多个，选择第一个（或者可以按epoch数字排序选择最新的）
        print(f"⚠ Found multiple epoch CSV files in {folder_path}, using: {csv_files[0]}")
        return csv_files[0]


def process_selection_folder(folder_path, folder_name, output_dir):
    """
    处理单个 selection 文件夹
    
    Args:
        folder_path: 文件夹完整路径
        folder_name: 文件夹名（如 "p1_epoch_250"）
    
    Returns:
        bool: 是否成功处理
    """
    # 提取 participant_id 和 epoch
    participant_id, epoch = extract_participant_and_epoch(folder_name)
    
    if participant_id is None or epoch is None:
        print(f"✗ Cannot extract participant_id and epoch from folder name: {folder_name}")
        return False
    
    print(f"\n{'='*80}")
    print(f"Processing: {folder_name} (P{participant_id}, epoch {epoch})")
    print(f"{'='*80}")
    
    # 查找 CSV 文件
    csv_file = find_epoch_csv(folder_path)
    if csv_file is None:
        print(f"✗ No generated_trials_epoch_*.csv found in {folder_path}")
        return False
    
    print(f"Found CSV: {os.path.basename(csv_file)}")
    
    # 检查文件是否存在且有数据
    try:
        # 读取第一行检查文件格式（无表头）
        df_test = pd.read_csv(csv_file, header=None, nrows=1)
        if df_test.empty:
            print(f"✗ CSV file is empty: {csv_file}")
            return False
    except Exception as e:
        print(f"✗ Error reading CSV file {csv_file}: {e}")
        return False
    
    # 设置输出文件名：p1_epoch_250.csv
    output_filename = f"p{participant_id}_epoch_{epoch}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # 检查输出文件是否已存在
    if os.path.exists(output_path):
        print(f"⚠ Output file already exists: {output_path}")
        print(f"  Skipping... (delete it first if you want to reprocess)")
        return True
    
    try:
        # 提取特征（不绘制图，只提取特征）
        print(f"Extracting features from {os.path.basename(csv_file)}...")
        
        # 读取无表头的 CSV 文件
        df = pd.read_csv(csv_file, header=None)
        
        if df.empty:
            print(f"✗ CSV file is empty: {csv_file}")
            return False
        
        # 使用行号作为 SampleID
        sample_ids = pd.Series(range(1, len(df) + 1))
        time_series_data = df
        
        # 创建时间点数组
        times = np.linspace(-0.4, 1.2, time_series_data.shape[1]) * 1000  # 转换为毫秒
        
        # 初始化特征列表
        features_list = []
        
        # 处理每个trial
        total_trials = len(df)
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
        print(f"✗ Error processing {folder_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_argparser():
    parser = argparse.ArgumentParser(description="Batch feature extraction from generated selection folders.")
    parser.add_argument("--condition", choices=["ot", "mw"], default="mw")
    parser.add_argument("--selection-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def resolve_paths(args):
    selection_dir = args.selection_dir
    output_dir = args.output_dir
    if selection_dir is None:
        selection_dir = SELECTION_DIR
        if args.condition == "mw":
            selection_dir = os.path.join(PROJECT_ROOT, "mw_pipeline", "finetuned_mw_participant", "!selection")
    if output_dir is None:
        output_dir = OUTPUT_DIR
        if args.condition == "mw":
            output_dir = os.path.join(PROJECT_ROOT, "mw_pipeline", "features", "mw_feature2")
    return selection_dir, output_dir


def main():
    """主函数"""
    args = build_argparser().parse_args()
    selection_dir, output_dir = resolve_paths(args)

    print("="*80)
    print(f"Batch Feature Extraction from !selection/ folders ({args.condition.upper()})")
    print("="*80)
    
    # 检查 selection 目录是否存在
    if not os.path.exists(selection_dir):
        print(f"✗ Selection directory not found: {selection_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Selection directory: {selection_dir}")
    print(f"Output directory: {output_dir}")
    
    # 获取所有文件夹
    all_items = os.listdir(selection_dir)
    folders = [item for item in all_items 
               if os.path.isdir(os.path.join(selection_dir, item)) 
               and item.startswith('p')]  # 只处理 p 开头的文件夹
    
    if not folders:
        print(f"✗ No folders found in {SELECTION_DIR}")
        return
    
    # 按文件夹名排序
    folders.sort()
    
    print(f"\nFound {len(folders)} folders to process:")
    for folder in folders:
        print(f"  - {folder}")
    
    # 处理每个文件夹
    results = []
    for idx, folder_name in enumerate(folders, 1):
        folder_path = os.path.join(selection_dir, folder_name)
        success = process_selection_folder(folder_path, folder_name, output_dir)
        results.append({'folder': folder_name, 'success': success})
    
    # 打印总结
    print(f"\n{'='*80}")
    print("Processing Summary")
    print("="*80)
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Total folders: {len(folders)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successfully processed:")
        for r in successful:
            print(f"  - {r['folder']}")
    
    if failed:
        print(f"\n✗ Failed:")
        for r in failed:
            print(f"  - {r['folder']}")
    
    print(f"\nAll feature files saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

