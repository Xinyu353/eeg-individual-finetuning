#!/usr/bin/env python3
"""
participant_fake_overlay.py

在全体参与者的 PCA / t-SNE 投影上，叠加单个参与者的真实分布与该参与者的
fake 分布，帮助直观看到 fake 分布相对真实分布与整体分布的位置。

数据来源（与 distribution_2d_analysis.py 相同）：
- 真实特征： feature_extraction/real_ot/p{id}_real.csv
- 生成特征： feature_extraction/ot_feature/p{id}_epoch_*.csv

输出：
- results/participant_fake_overlay/pca_plots/p{id}_overlay_epoch_{epoch}.png
- results/participant_fake_overlay/tsne_plots/p{id}_overlay_epoch_{epoch}.png

绘图内容：
- 灰色：所有其他参与者的真实样本
- 蓝色：该参与者的真实样本
- 红色：该参与者对应 epoch 的 fake 样本
"""

import os
import glob
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 复用与 distribution_2d_analysis.py 相同的特征简写
FEATURE_ABBREVIATIONS = {
    'P3_Amplitude': 'P3_A',
    'P3_Latency': 'P3_L',
    'Activity': 'HA',
    'Mobility': 'HM',
    'Complexity': 'HC',
    'Petrosian_Fractal_Dimension': 'PFD',
    'Delta_PSD': r'PSD$_{\delta}$',
    'Theta_PSD': r'PSD$_{\theta}$',
    'Alpha_PSD': r'PSD$_{\alpha}$',
    'Beta_PSD': r'PSD$_{\beta}$',
    'Gamma_PSD': r'PSD$_{\gamma}$',
    'Delta_Entropy': r'E$_{\delta}$',
    'Theta_Entropy': r'E$_{\theta}$',
    'Alpha_Entropy': r'E$_{\alpha}$',
    'Beta_Entropy': r'E$_{\beta}$',
    'Gamma_Entropy': r'E$_{\gamma}$',
}


def get_feature_abbreviation(feature_name: str) -> str:
    return FEATURE_ABBREVIATIONS.get(feature_name, feature_name)


def extract_pid_from_real(fname: str):
    m = re.match(r'p(\d+)_real\.csv', fname)
    return int(m.group(1)) if m else None


def extract_pid_epoch_from_fake(fname: str):
    m = re.match(r'p(\d+)_epoch_(\d+)\.csv', fname)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def load_all_real(real_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(real_dir, "p*_real.csv")))
    rows = []
    for path in files:
        pid = extract_pid_from_real(os.path.basename(path))
        if pid is None:
            continue
        df = pd.read_csv(path)
        df["Participant"] = pid
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No real files found in {real_dir}")
    real_df = pd.concat(rows, ignore_index=True)
    return real_df


def load_fake_for_pid(fake_dir: str, pid: int):
    pattern = os.path.join(fake_dir, f"p{pid}_epoch_*.csv")
    files = sorted(glob.glob(pattern))
    results = []
    for path in files:
        pid_found, epoch = extract_pid_epoch_from_fake(os.path.basename(path))
        if pid_found is None:
            continue
        df = pd.read_csv(path)
        df["Participant"] = pid_found
        df["Epoch"] = epoch
        results.append((epoch, df, path))
    return results  # list of tuples(epoch, df, path)


def make_dirs(base_out: str):
    pca_dir = os.path.join(base_out, "pca_plots")
    tsne_dir = os.path.join(base_out, "tsne_plots")
    os.makedirs(pca_dir, exist_ok=True)
    os.makedirs(tsne_dir, exist_ok=True)
    return pca_dir, tsne_dir


def plot_overlay(pca_2d, tsne_2d, labels, pid, epoch, pca_dir, tsne_dir):
    # labels dict: keys 'real_target', 'real_others', 'fake'
    colors = {
        "real_target": "#0d47a1",   # deep blue / blackish
        "real_others": "#ffcc80",   # light orange
        "fake": "#d32f2f",          # red
    }
    markers = {
        "real_target": ".",
        "real_others": ".",
        "fake": ".",
    }
    alpha = {"real_target": 0.6, "real_others": 0.25, "fake": 0.6}

    def scatter_plot(coords, title, save_path, is_pca: bool):
        # 和 distribution_2d_analysis.py 保持一致的图大小和字体设置
        plt.figure(figsize=(4, 3.5), facecolor="white")
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["axes.titlesize"] = 24
        plt.rcParams["axes.labelsize"] = 20
        plt.rcParams["legend.fontsize"] = 14  # 缩小图例字号，避免溢出
        plt.rcParams["xtick.labelsize"] = 16
        plt.rcParams["ytick.labelsize"] = 16

        ax = plt.gca()
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")

        for key in ["real_others", "real_target", "fake"]:
            pts = coords[labels == key]
            if len(pts) == 0:
                continue
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=colors[key],
                alpha=alpha[key],
                s=20,
                marker=markers[key],
                label={
                    "real_others": "Real (group)",
                    "real_target": f"Real (P{pid})",
                    "fake": f"Fake (P{pid})",
                }[key],
            )

        ax.set_title(title)
        if is_pca:
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
        else:
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")

        ax.legend(frameon=False, loc="best")
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close()

    # 标题与原程序风格一致：例如 "PCA (P1)" / "t-SNE (P1)"
    scatter_plot(
        pca_2d,
        f"PCA (P{pid})",
        os.path.join(pca_dir, f"p{pid}_overlay_epoch_{epoch}.png"),
        is_pca=True,
    )

    scatter_plot(
        tsne_2d,
        f"t-SNE (P{pid})",
        os.path.join(tsne_dir, f"p{pid}_overlay_epoch_{epoch}.png"),
        is_pca=False,
    )


def build_projection(real_all: pd.DataFrame, fake_df: pd.DataFrame, pid: int):
    """
    Fit PCA/t-SNE on (all real + current fake) with consistent columns.
    """
    # 对齐特征列：取交集（排除标识列）
    drop_cols = [c for c in ["SampleID", "Label", "Participant", "Epoch"] if c in real_all.columns or c in fake_df.columns]
    feature_cols = list(set(real_all.columns) & set(fake_df.columns))
    feature_cols = [c for c in feature_cols if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No common feature columns between real and fake data.")

    # 标记标签
    real_all_tmp = real_all.copy()
    real_all_tmp["Label"] = np.where(real_all_tmp["Participant"] == pid, "real_target", "real_others")

    fake_tmp = fake_df.copy()
    fake_tmp["Label"] = "fake"

    combined = pd.concat([real_all_tmp, fake_tmp], ignore_index=True)
    X = combined[feature_cols].values
    labels = combined["Label"].to_numpy()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X_std)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_std) - 1))
    tsne_2d = tsne.fit_transform(X_std)

    return pca_2d, tsne_2d, labels, feature_cols


def build_argparser():
    parser = argparse.ArgumentParser(description="Overlay one participant's real/fake distribution on the group projection.")
    parser.add_argument("--condition", choices=["ot", "mw"], default="ot")
    parser.add_argument("--real-feature-dir", default=None)
    parser.add_argument("--fake-feature-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def resolve_paths(args, project_root):
    if args.condition == "mw":
        real_dir = args.real_feature_dir or os.path.join(project_root, "mw_pipeline", "features", "real_mw")
        fake_dir = args.fake_feature_dir or os.path.join(project_root, "mw_pipeline", "features", "mw_feature")
        output_base = args.output_dir or os.path.join(project_root, "mw_pipeline", "results", "participant_fake_overlay")
    else:
        real_dir = args.real_feature_dir or os.path.join(project_root, "feature_extraction", "real_ot")
        fake_dir = args.fake_feature_dir or os.path.join(project_root, "feature_extraction", "ot_feature")
        output_base = args.output_dir or os.path.join(project_root, "results", "participant_fake_overlay")
    return real_dir, fake_dir, output_base


def main():
    args = build_argparser().parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    real_dir, fake_dir, output_base = resolve_paths(args, project_root)

    print(f"Condition: {args.condition.upper()}")
    print(f"Real features dir: {real_dir}")
    print(f"Fake features dir: {fake_dir}")
    print(f"Output dir: {output_base}")

    if not os.path.exists(real_dir):
        raise FileNotFoundError(f"Real feature directory not found: {real_dir}")
    if not os.path.exists(fake_dir):
        raise FileNotFoundError(f"Fake feature directory not found: {fake_dir}")

    # 载入所有真实数据
    real_all = load_all_real(real_dir)
    print(f"Loaded real data: {real_all.shape}")

    # fake 可用的 participant 列表
    fake_files = glob.glob(os.path.join(fake_dir, "p*_epoch_*.csv"))
    if not fake_files:
        raise FileNotFoundError(f"No fake feature files found in {fake_dir}")

    # 建立 pid -> list of fake files
    fake_pid_map = {}
    for path in fake_files:
        pid, epoch = extract_pid_epoch_from_fake(os.path.basename(path))
        if pid is None:
            continue
        fake_pid_map.setdefault(pid, []).append(path)

    # 输出目录（仅 2D）
    pca_dir, tsne_dir = make_dirs(output_base)

    for pid in sorted(fake_pid_map.keys()):
        # 该参与者的 fake 列表
        pid_files = sorted(fake_pid_map[pid])
        # 如果该参与者在 real 中不存在，跳过
        if pid not in real_all["Participant"].unique():
            print(f"Skip P{pid}: no real data found.")
            continue

        # 提取该参与者的真实数据（仅用于日志）
        real_pid_rows = (real_all["Participant"] == pid).sum()
        print(f"\nProcessing P{pid}: {len(pid_files)} fake file(s), real rows={real_pid_rows}")

        for fake_path in pid_files:
            fake_df = pd.read_csv(fake_path)
            pid_found, epoch = extract_pid_epoch_from_fake(os.path.basename(fake_path))
            if pid_found is None:
                continue
            fake_df["Participant"] = pid_found
            fake_df["Epoch"] = epoch

            print(f"  Fake file: {os.path.basename(fake_path)} (rows={len(fake_df)})")

            pca_2d, tsne_2d, labels, feature_cols = build_projection(real_all, fake_df, pid_found)

            print(f"  Common feature columns ({len(feature_cols)}): {feature_cols}")

            plot_overlay(
                pca_2d, tsne_2d, labels, pid_found, epoch,
                pca_dir, tsne_dir
            )

    print("\nDone. Check plots under:")
    print(f"  {pca_dir}")
    print(f"  {tsne_dir}")


if __name__ == "__main__":
    main()
