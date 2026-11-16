#!/usr/bin/env python3
"""
select_feature_simple.py
简化版特征选择脚本，可以指定输入和输出目录
"""

import os
import pandas as pd
import glob
import sys

def get_16_features():
    """返回16个选定特征的列名"""
    return [
        'P3_Amplitude', 'P3_Latency',  # ERP特征 (2个)
        'Activity', 'Mobility', 'Complexity',  # Hjorth参数 (3个)
        'Petrosian_Fractal_Dimension',  # 分形维数 (1个)
        'Delta_PSD', 'Theta_PSD', 'Alpha_PSD', 'Beta_PSD', 'Gamma_PSD',  # 功率谱密度 (5个)
        'Delta_Entropy', 'Theta_Entropy', 'Alpha_Entropy', 'Beta_Entropy', 'Gamma_Entropy'  # 谱熵 (5个)
    ]

def process_single_file(input_file, output_file):
    """处理单个特征文件"""
    try:
        print(f"Processing: {input_file}")
        
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 获取16个特征列名
        target_features = get_16_features()
        available_features = [col for col in target_features if col in df.columns]
        
        if not available_features:
            print(f"  Error: No target features found")
            return False
        
        # 提取特征列（删除SampleID）
        selected_df = df[available_features].copy()
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存文件
        selected_df.to_csv(output_file, index=False)
        
        print(f"  ✓ Saved {len(available_features)} features to: {output_file}")
        print(f"  ✓ Shape: {selected_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("Usage: python select_feature_simple.py <input_file> <output_file>")
        print("Example: python select_feature_simple.py input.csv output.csv")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    print("="*60)
    print("Simple Feature Selection")
    print("="*60)
    
    success = process_single_file(input_file, output_file)
    
    if success:
        print("="*60)
        print("Processing completed successfully!")
        print("="*60)
    else:
        print("="*60)
        print("Processing failed!")
        print("="*60)

if __name__ == "__main__":
    main()
