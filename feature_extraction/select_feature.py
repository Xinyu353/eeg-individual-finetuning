#!/usr/bin/env python3
"""
select_feature.py
从EEG_features文件夹下的所有特征文件中提取16个选定特征
删除SampleID列，只保留16个特征并保存为selected_16_features.csv
"""

import os
import pandas as pd
import glob
from pathlib import Path

def get_16_features():
    """
    返回16个选定特征的列名
    """
    return [
        # ERP特征 (2个)
        'P3_Amplitude',
        'P3_Latency',
        # Hjorth参数 (3个)
        'Activity',
        'Mobility',
        'Complexity',
        # 分形维数 (1个)
        'Petrosian_Fractal_Dimension',
        # 各频段功率谱密度 (5个)
        'Delta_PSD',
        'Theta_PSD',
        'Alpha_PSD',
        'Beta_PSD',
        'Gamma_PSD',
        # 各频段谱熵 (5个)
        'Delta_Entropy',
        'Theta_Entropy',
        'Alpha_Entropy',
        'Beta_Entropy',
        'Gamma_Entropy'
    ]

def find_feature_files(base_dir):
    """
    在base_dir下查找所有特征文件
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        list: 找到的特征文件路径列表
    """
    feature_files = []
    
    # 查找all_features.csv文件
    all_features_pattern = os.path.join(base_dir, "**", "features", "all_features.csv")
    all_features_files = glob.glob(all_features_pattern, recursive=True)
    feature_files.extend(all_features_files)
    
    # 查找selected_16_features.csv文件
    selected_features_pattern = os.path.join(base_dir, "**", "features", "selected_16_features.csv")
    selected_features_files = glob.glob(selected_features_pattern, recursive=True)
    feature_files.extend(selected_features_files)
    
    return feature_files

def process_feature_file(file_path, output_dir, base_dir):
    """
    处理单个特征文件，提取16个特征
    
    Args:
        file_path: 输入特征文件路径
        output_dir: 输出目录
        base_dir: 基础目录路径
        
    Returns:
        bool: 处理是否成功
    """
    try:
        print(f"Processing: {file_path}")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 获取16个特征列名
        target_features = get_16_features()
        
        # 检查哪些特征列存在
        available_features = [col for col in target_features if col in df.columns]
        missing_features = [col for col in target_features if col not in df.columns]
        
        if missing_features:
            print(f"  Warning: Missing features: {missing_features}")
        
        if not available_features:
            print(f"  Error: No target features found in {file_path}")
            return False
        
        # 提取可用的特征列
        selected_df = df[available_features].copy()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名（基于文件夹名）
        # 获取文件夹名（features的上一级目录）
        folder_path = os.path.dirname(os.path.dirname(file_path))
        folder_name = os.path.basename(folder_path)
        output_file = os.path.join(output_dir, f"{folder_name}.csv")
        
        # 保存处理后的特征
        selected_df.to_csv(output_file, index=False)
        
        print(f"  ✓ Saved {len(available_features)} features to: {output_file}")
        print(f"  ✓ Shape: {selected_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False

def main():
    """
    主函数
    """
    print("="*80)
    print("Feature Selection Program")
    print("="*80)
    
    # Get project root (parent of feature_extraction directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 设置路径
    base_dir = os.path.join(project_root, "EEG_features")
    output_dir = os.path.join(project_root, "EEG_features", "selected_features")
    
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # 检查基础目录是否存在
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    # 查找所有特征文件
    print("\nSearching for feature files...")
    feature_files = find_feature_files(base_dir)
    
    if not feature_files:
        print("No feature files found!")
        return
    
    print(f"Found {len(feature_files)} feature files:")
    for file_path in feature_files:
        print(f"  - {file_path}")
    
    # 处理每个特征文件
    print(f"\nProcessing feature files...")
    success_count = 0
    
    for file_path in feature_files:
        if process_feature_file(file_path, output_dir, base_dir):
            success_count += 1
    
    print(f"\n" + "="*80)
    print(f"Processing completed!")
    print(f"Successfully processed: {success_count}/{len(feature_files)} files")
    print(f"Results saved in: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
