#!/usr/bin/env python3
"""
compute_swd_mmd.py

- 读取 finetuned_ot_participant/!selection/ 下的所有文件夹（如 p1_epoch_250/）
- 读取 participant_data/ 文件夹下的真实数据 CSV（如 participant_1_real_data.csv）
- 匹配 participant（p1_epoch_250 对应 participant_1_real_data.csv）
- 计算每个匹配对的：
  - Sliced Wasserstein Distance (SWD)：多方向投影1D的Wasserstein距离的平均
  - MMD (RBF核；核宽度使用median heuristic)
  - Wasserstein Distance (WD)：逐时间点计算1D WD后取平均
- 结果保存至 evaluation/ot_distance_metrics.csv
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# -----------------------------
# Utilities
# -----------------------------

def read_real_csv(path):
    """读取真实数据：无表头，无样本编号列。"""
    df = pd.read_csv(path, header=None)
    return df.to_numpy(dtype=float)


def read_fake_csv(path):
    """读取对比数据：无样本编号列，无表头。"""
    df = pd.read_csv(path, header=None)
    return df.to_numpy(dtype=float)


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


def check_shape(X):
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")
    if X.shape[1] != 409:
        raise ValueError(f"Expected 409 timepoints per trial, got {X.shape[1]}")


def sliced_wasserstein(X, Y, n_projections=256, seed=42):
    """Compute Sliced Wasserstein Distance between two 2D arrays (n_samples, n_features).
    Project onto random 1D directions and average 1D Wasserstein distance.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    # 随机方向（高维到1D）：正态随机向量归一化
    directions = rng.normal(size=(n_projections, d))
    directions /= (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)

    vals = []
    for v in directions:
        proj_X = X @ v
        proj_Y = Y @ v
        vals.append(wasserstein_distance(proj_X, proj_Y))
    return float(np.mean(vals))


def pairwise_sq_dists(A, B):
    """Compute squared Euclidean distance matrix between rows of A and B."""
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    a2 = np.sum(A*A, axis=1, keepdims=True)
    b2 = np.sum(B*B, axis=1, keepdims=True).T
    return a2 + b2 - 2 * (A @ B.T)


def mmd_rbf(X, Y, gamma=None):
    """Compute unbiased MMD^2 with RBF kernel; return sqrt(MMD^2) for comparability (distance-like).
    gamma = 1/(2*sigma^2); sigma using median heuristic over pooled data.
    """
    n, m = X.shape[0], Y.shape[0]
    Z = np.vstack([X, Y])
    # median heuristic over all pairwise distances (use a sample if huge)
    if Z.shape[0] > 5000:
        idx = np.random.default_rng(42).choice(Z.shape[0], size=5000, replace=False)
        Zs = Z[idx]
    else:
        Zs = Z
    D2 = pairwise_sq_dists(Zs, Zs)
    # 取上三角非零部分的中位数
    iu = np.triu_indices_from(D2, k=1)
    med2 = np.median(D2[iu])
    sigma2 = max(med2, 1e-12)
    gamma = 1.0 / (2.0 * sigma2)

    # Kernels
    Kxx = np.exp(-gamma * pairwise_sq_dists(X, X))
    Kyy = np.exp(-gamma * pairwise_sq_dists(Y, Y))
    Kxy = np.exp(-gamma * pairwise_sq_dists(X, Y))

    # Unbiased MMD^2 estimator
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = (Kxx.sum() / (n*(n-1) + 1e-12)
            + Kyy.sum() / (m*(m-1) + 1e-12)
            - 2.0 * Kxy.mean())
    mmd2 = max(mmd2, 0.0)
    return float(np.sqrt(mmd2))


def match_counts(X, Y, max_trials=2000, seed=42):
    """Downsample to equal number of trials (same rows)."""
    rng = np.random.default_rng(seed)
    n = min(len(X), len(Y), max_trials)
    idxX = rng.choice(len(X), size=n, replace=False)
    idxY = rng.choice(len(Y), size=n, replace=False)
    return X[idxX], Y[idxY]


def wasserstein_distance_2d(X, Y, max_samples=1000, seed=42):
    """Compute Wasserstein Distance between two 2D arrays using per-timepoint method.
    Calculate 1D WD for each timepoint, then average.
    """
    rng = np.random.default_rng(seed)
    n = min(len(X), len(Y), max_samples)
    idxX = rng.choice(len(X), size=n, replace=False)
    idxY = rng.choice(len(Y), size=n, replace=False)
    X_sub = X[idxX]
    Y_sub = Y[idxY]
    
    # Calculate WD for each timepoint
    wd_list = []
    for i in range(X_sub.shape[1]):
        wd = wasserstein_distance(X_sub[:, i], Y_sub[:, i])
        wd_list.append(wd)
    
    return float(np.mean(wd_list))


def build_argparser():
    parser = argparse.ArgumentParser(description="Compute SWD/MMD/WD between real and generated participant data.")
    parser.add_argument("--condition", choices=["ot", "mw"], default="ot")
    parser.add_argument("--selection-dir", default=None)
    parser.add_argument("--participant-data-dir", default=None)
    parser.add_argument("--output-file", default=None)
    return parser


def resolve_paths(args, project_root, script_dir):
    if args.condition == "mw":
        selection_dir = args.selection_dir or os.path.join(project_root, "mw_pipeline", "finetuned_mw_participant", "!selection")
        participant_data_dir = args.participant_data_dir or os.path.join(project_root, "mw_pipeline", "participant_data")
        output_file = args.output_file or os.path.join(project_root, "mw_pipeline", "evaluation", "mw_distance_metrics.csv")
    else:
        selection_dir = args.selection_dir or os.path.join(project_root, 'finetuned_ot_participant', '!selection')
        participant_data_dir = args.participant_data_dir or os.path.join(project_root, 'participant_data')
        output_file = args.output_file or os.path.join(script_dir, 'ot_distance_metrics.csv')
    return selection_dir, participant_data_dir, output_file


def main():
    args = build_argparser().parse_args()
    # Get project root (parent of evaluation directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    selection_dir, participant_data_dir, output_file = resolve_paths(args, project_root, script_dir)
    
    print("="*80)
    print(f"Compute SWD/MMD/WD for {args.condition.upper()} Participant Data")
    print("="*80)
    print(f"Selection directory: {selection_dir}")
    print(f"Participant data directory: {participant_data_dir}")
    print(f"Output file: {output_file}")
    
    # 检查目录是否存在
    if not os.path.exists(selection_dir):
        print(f"✗ Selection directory not found: {selection_dir}")
        return
    
    if not os.path.exists(participant_data_dir):
        print(f"✗ Participant data directory not found: {participant_data_dir}")
        return
    
    # 获取所有 selection 文件夹
    all_items = os.listdir(selection_dir)
    selection_folders = [item for item in all_items 
                        if os.path.isdir(os.path.join(selection_dir, item)) 
                        and item.startswith('p')]  # 只处理 p 开头的文件夹
    
    if not selection_folders:
        print(f"✗ No selection folders found in {selection_dir}")
        return
    
    # 按文件夹名排序
    selection_folders.sort()
    
    print(f"\nFound {len(selection_folders)} selection folders to process")
    
    rows = []
    
    # 处理每个 selection 文件夹
    for folder_name in selection_folders:
        # 提取 participant_id 和 epoch
        participant_id, epoch = extract_participant_and_epoch(folder_name)
        
        if participant_id is None or epoch is None:
            print(f"✗ Cannot extract participant_id and epoch from: {folder_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing: {folder_name} (P{participant_id}, epoch {epoch})")
        print(f"{'='*80}")
        
        # 查找对应的真实数据文件
        real_data_file = os.path.join(participant_data_dir, f'participant_{participant_id}_real_data.csv')
        if not os.path.exists(real_data_file):
            print(f"✗ Real data file not found: {real_data_file}")
            continue
        
        # 查找生成的 CSV 文件
        folder_path = os.path.join(selection_dir, folder_name)
        fake_csv_file = find_epoch_csv(folder_path)
        if fake_csv_file is None:
            print(f"✗ No generated_trials_epoch_*.csv found in {folder_path}")
            continue
        
        try:
            # 读取真实数据
            print(f"Reading real data: {os.path.basename(real_data_file)}")
            X_real = read_real_csv(real_data_file)
            X_real = X_real[np.isfinite(X_real).all(axis=1)]
            check_shape(X_real)
            print(f"  Real data: {X_real.shape[0]} trials, {X_real.shape[1]} timepoints")
            
            # 读取生成数据
            print(f"Reading fake data: {os.path.basename(fake_csv_file)}")
            X_fake = read_fake_csv(fake_csv_file)
            X_fake = X_fake[np.isfinite(X_fake).all(axis=1)]
            check_shape(X_fake)
            print(f"  Fake data: {X_fake.shape[0]} trials, {X_fake.shape[1]} timepoints")
            
            # 匹配数量（下采样到相同数量）
            Xa, Xb = match_counts(X_real, X_fake)
            print(f"  After matching: {len(Xa)} trials each")
            
            # 计算距离指标
            print("Computing SWD...")
            swd = sliced_wasserstein(Xa, Xb)
            print("Computing MMD...")
            mmd = mmd_rbf(Xa, Xb)
            print("Computing WD...")
            wd = wasserstein_distance_2d(Xa, Xb)
            
            print(f"  SWD={swd:.3f}, MMD_RBF={mmd:.3f}, WD={wd:.3f}")
            
            # 保存结果
            rows.append({
                'participant_id': participant_id,
                'epoch': epoch,
                'folder': folder_name,
                'real_file': f'participant_{participant_id}_real_data.csv',
                'fake_file': os.path.basename(fake_csv_file),
                'num_trials': len(Xa),
                'SWD': swd,
                'MMD_RBF': mmd,
                'WD': wd
            })
            
            print(f"✓ Successfully processed P{participant_id}, epoch {epoch}")
            
        except Exception as e:
            print(f"✗ Error processing {folder_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    if not rows:
        print("\n✗ No results to save")
        return
    
    df = pd.DataFrame(rows)
    # 确保数值列保留3位小数
    df['SWD'] = df['SWD'].round(3)
    df['MMD_RBF'] = df['MMD_RBF'].round(3)
    df['WD'] = df['WD'].round(3)
    
    # 按 participant_id 和 epoch 排序
    df = df.sort_values(['participant_id', 'epoch'])
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, float_format='%.3f')
    
    print(f"\n{'='*80}")
    print("Processing Summary")
    print("="*80)
    print(f"Total processed: {len(rows)} pairs")
    print(f"Results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
