#!/usr/bin/env python3
"""
Fine-tune OT model for all participants

- Load base_OT.pt pretrained model
- Iterate through all participants in ot_vs_c_Pz_4753.csv (first column = participant_id)
- Fine-tune using real data for each participant
- Generate fake OT trials (1:1 ratio with real trials for each participant)
- All layers trainable (full fine-tuning)
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import sys
import time
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Get project root (relative path, suitable for HPC)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
sys.path.append(PROJECT_ROOT)

from lora import apply_lora_to_model
from wgan_cnn import Generator, Discriminator
from loss import WGP_loss
from loss_plotting import save_loss_plots

# ============================================================================
# USER CONFIGURATION - 所有可调参数集中在这里
# ============================================================================
# 运行脚本前如需修改参数，只需调整下面 CONFIG 中的字段

CONFIG = {
    "condition": "ot",
    # 数据路径（相对路径，适合HPC）
    "model_path": os.path.join(PROJECT_ROOT, 'real_EEG_pretrained_model', 'base_OT.pt'),
    "csv_path": os.path.join(PROJECT_ROOT, 'real_EEG_pretrained_model', 'ot_vs_c_Pz_4753.csv'),
    "output_dir": os.path.join(PROJECT_ROOT, 'finetuned_ot_participant'),
    
    # 训练超参数
    "epochs": 500,
    "batch_size": 32,  # 小样本微调建议用32或64
    "save_interval": 50,  # 每N个epoch保存一次checkpoint和生成数据
    # "num_fake_samples": 200,  # 已移除：将根据real data数量动态计算
    
    # 优化器参数
    "lr_g": 1e-4,  # Generator学习率
    "lr_d": 1e-5,  # Discriminator学习率
    "beta1": 0.5,  # Adam beta1
    "beta2": 0.999,  # Adam beta2
    
    # WGAN-GP损失权重
    "lambda_gp": 10.0,  # Gradient penalty权重
    "lambda_wd": 1.0,  # Wasserstein distance权重
    "lambda_erp": 300.0,  # ERP损失权重（建议200-500，避免压扁对抗学习）
    
    # 模型架构参数
    "latent_dim": 300,  # 潜在空间维度
    "eeg_length": 409,  # EEG时间点数
    "cutoff_frequency": 40,  # 滤波器截止频率
    "fs": 256,  # 采样频率
    
    # 训练策略
    "n_critic": 2,  # 每个G步前训练D的次数（WGAN-GP常用5）
    
    # 基线校正参数
    "baseline_start": -0.398,  # 基线窗口起始时间（秒）
    "baseline_end": 0.0,  # 基线窗口结束时间（秒）
    "time_start": -398,  # 时间序列起始点（毫秒）
    "time_end": 1195,  # 时间序列结束点（毫秒）
    
    # 生成参数
    "gen_batch_size": 128,  # 生成fake data时的batch size
    
    # LoRA / 微调模式设置
    "finetune_mode": "full",  # full / partial / lora / frozen（lora模式只训练LoRA参数+subject_token）
    "use_lora": False,  # full模式不需要LoRA
    "lora_r": 2,  # 小样本更稳，更保守的rank
    "lora_alpha": 4.0,  # 与r同量级或略大
    "lora_dropout": 0.05,
    "lora_targets": ("conv_first2", "final_linear"),
    
    # ERP warm-up
    "erp_warmup_frac": 0.0,      # 设为0关闭

    # 是否启用 subject_token（对每个 participant 的可训练 z 偏置）
    "use_subject_token": True,
}

# ============================================================================
# Utils
# ============================================================================

def denormalize_data(data: torch.Tensor, real_min: float, real_max: float) -> torch.Tensor:
    """Map from [-1, 1] back to original range using min/max from checkpoint."""
    data = (data + 1.0) / 2.0
    data = data * (real_max - real_min) + real_min
    return data


def compute_num_fake_samples(num_real_trials: int) -> int:
    """
    Compute number of fake samples: 1:1 ratio with real data.
    """
    return num_real_trials

# ---------- Full Fine-tuner Class ----------

class FullFineTuner:
    """Full fine-tuner: all parameters of G/D are trainable."""
    
    def __init__(self, generator: Generator, discriminator: Discriminator, device: str, config: dict):
        self.device = device
        self.config = config
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.mode = config.get("finetune_mode", "full").lower()
        
        # Loss function
        self.loss_fn = WGP_loss(
            lambda_gp=config['lambda_gp'],
            lambda_wd=config['lambda_wd'],
            lambda_erp=config['lambda_erp']
        )
        
        self.trainable_desc = "full"
        self._configure_trainable_params()
        
        # subject_token: 对该 participant 的可学习 z 偏置
        use_subject_token = self.config.get("use_subject_token", True)
        if use_subject_token:
            self.subject_token = nn.Parameter(
                torch.zeros(self.config['latent_dim'], device=self.device)
            )
        else:
            self.subject_token = None

        self.g_trainable_params = [p for p in self.generator.parameters() if p.requires_grad]
        self.d_trainable_params = [p for p in self.discriminator.parameters() if p.requires_grad]
        
        if self.mode == "frozen":
            self.optimizer_g = None
            self.optimizer_d = None
        else:
            # 将 subject_token 也加入 G 的可训练参数中
            if use_subject_token and self.subject_token is not None:
                self.subject_token.requires_grad = True
                self.g_trainable_params.append(self.subject_token)

            if not self.g_trainable_params:
                raise ValueError(f"No trainable generator parameters for mode='{self.mode}'.")
            if not self.d_trainable_params:
                raise ValueError(f"No trainable discriminator parameters for mode='{self.mode}'.")
            
            self.optimizer_g = optim.Adam(
                self.g_trainable_params,
                lr=config['lr_g'],
                betas=(config['beta1'], config['beta2'])
            )
            self.optimizer_d = optim.Adam(
                self.d_trainable_params,
                lr=config['lr_d'],
                betas=(config['beta1'], config['beta2'])
            )
        
        # Baseline correction function (per-trial) & ERP 时间窗
        time_ms = torch.linspace(
            config['time_start'],
            config['time_end'],
            config['eeg_length'],
            device=self.device,
        )
        time_s = time_ms / 1000.0
        self.baseline_mask = (time_s >= config['baseline_start']) & (time_s <= config['baseline_end'])
        
        # Histories
        self.g_losses = []
        self.d_losses = []
        self.wd_d_losses = []
        self.wd_g_losses = []
        self.erp_losses = []
    
    def _set_all_params(self, module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = flag
    
    def _freeze_all_params(self) -> None:
        self._set_all_params(self.generator, False)
        self._set_all_params(self.discriminator, False)
    
    def _unfreeze_generator_last_layer(self) -> None:
        last_layer = None
        for layer in reversed(self.generator.model):
            params = list(layer.parameters())
            if params:
                last_layer = layer
                for p in params:
                    p.requires_grad = True
                break
        if last_layer is None:
            raise RuntimeError("Failed to locate trainable layer in generator.")
    
    def _unfreeze_discriminator_last_fc(self) -> None:
        last_linear = None
        for layer in reversed(self.discriminator.fc):
            if isinstance(layer, nn.Linear):
                last_linear = layer
                for p in layer.parameters():
                    p.requires_grad = True
                break
        if last_linear is None:
            raise RuntimeError("Failed to locate Linear layer in discriminator.fc.")
    
    def _configure_trainable_params(self) -> None:
        mode = self.mode
        if mode == "full":
            self._set_all_params(self.generator, True)
            self._set_all_params(self.discriminator, True)
            self.trainable_desc = "full"
        elif mode == "partial":
            self._freeze_all_params()
            self._unfreeze_generator_last_layer()
            self._unfreeze_discriminator_last_fc()
            self.trainable_desc = "last_layer"
        elif mode == "lora":
            # assume LoRA wrappers already applied; keep their requires_grad settings
            self.trainable_desc = "lora"
        elif mode == "frozen":
            self._freeze_all_params()
            self.trainable_desc = "frozen"
        else:
            raise ValueError(f"Unsupported finetune_mode: {mode}")
    
    def _set_discriminator_grad(self, flag: bool) -> None:
        for p in self.d_trainable_params:
            p.requires_grad = flag
        
    def baseline_per_trial(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-trial baseline correction."""
        if x.dim() == 3:
            x = x.squeeze(1)
        b = x[:, self.baseline_mask].mean(dim=1, keepdim=True)
        return x - b
    
    def train_step_d(self, real_data: torch.Tensor, fake: torch.Tensor) -> float:
        """One discriminator training step."""
        self._set_discriminator_grad(True)
        
        self.optimizer_d.zero_grad()
        d_loss, wd_d = self.loss_fn.d_loss(self.discriminator, real_data, fake.detach())
        d_loss.backward()
        self.optimizer_d.step()
        
        return float(d_loss.item()), float(wd_d.item())
    
    def train_step_g(self, fake: torch.Tensor, real_data: torch.Tensor) -> tuple:
        """One generator training step."""
        self._set_discriminator_grad(False)
        
        self.optimizer_g.zero_grad()
        g_total, g_wgan, g_erp = self.loss_fn.g_loss(
            self.discriminator, fake, real_data, fs=self.config['fs']
        )
        g_total.backward()
        self.optimizer_g.step()
        
        # Restore D grad state
        self._set_discriminator_grad(True)
        
        return float(g_total.item()), float(g_wgan.item()), float(g_erp.item())
    
    def train_step(self, real_data: torch.Tensor, batch_size: int, n_critic: int) -> dict:
        """One training step with n_critic D updates per G update."""
        # 已在 load_real_data() 中完成每trial基线校正，这里不要再次校正
        # real_data = self.baseline_per_trial(real_data)
        
        # Sample and generate on device (GPU) using model forward
        noise = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
        if self.subject_token is not None:
            noise = noise + self.subject_token
        fake = self.generator(noise)
        
        if fake.dim() == 3:
            fake = fake.squeeze(1)
        
        # Multiple D steps (WGAN-GP standard practice)
        d_losses = []
        wd_d_losses = []
        for _ in range(n_critic):
            d_loss, wd_d = self.train_step_d(real_data, fake)
            d_losses.append(d_loss)
            wd_d_losses.append(wd_d)
            # Regenerate fake for next D step
            noise = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
            if self.subject_token is not None:
                noise = noise + self.subject_token
            fake = self.generator(noise)
            if fake.dim() == 3:
                fake = fake.squeeze(1)
        
        # One G step
        g_total, g_wgan, g_erp = self.train_step_g(fake, real_data)
        
        return {
            'd_loss': sum(d_losses) / len(d_losses),
            'g_loss': g_total,
            'wd_d': sum(wd_d_losses) / len(wd_d_losses),
            'wd_g': g_wgan,
            'erp_loss': g_erp,
        }
    
    @torch.no_grad()
    def generate_trials(self, num_trials: int, data_min: Optional[float], data_max: Optional[float]) -> np.ndarray:
        was_training = self.generator.training
        self.generator.train(False)
        
        trials = []
        gen_batch_size = self.config['gen_batch_size']
        for i in range(0, num_trials, gen_batch_size):
            b = min(gen_batch_size, num_trials - i)
            z = torch.randn(b, self.config['latent_dim'], device=self.device)
            if self.subject_token is not None:
                z = z + self.subject_token
            fake = self.generator(z)
            if fake.dim() == 3:
                fake = fake.squeeze(1)
            if data_min is not None and data_max is not None:
                fake = denormalize_data(fake, data_min, data_max)
            trials.append(fake.detach().cpu().numpy())
        
        if was_training:
            self.generator.train(True)
        
        return np.concatenate(trials, axis=0)
    
    def train(
        self,
        real_data: torch.Tensor,
        data_min: float,
        data_max: float,
        epochs: int = 300,
        batch_size: int = 128,
        save_interval: int = 30,
        num_fake_samples: int = 200,
    ) -> str:
        print("=" * 60)
        print(f"Starting {self.mode.upper()} fine-tuning (trainable: {self.trainable_desc})")
        print("=" * 60)
        
        # Train mode
        self.generator.train(True)
        self.discriminator.train(True)
        
        # Save dir (relative path) - include participant_id if available
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        participant_id = self.config.get('participant_id', 'all')
        if participant_id != 'all':
            save_dir = os.path.join(self.config['output_dir'], f'participant_{participant_id}_full_{timestamp}')
        else:
            save_dir = os.path.join(self.config['output_dir'], f'all_layers_finetune_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'generated_data'), exist_ok=True)
        
        print(f"Save dir: {save_dir}")
        print(f"epochs={epochs}, batch_size={batch_size}, lr_g={self.config['lr_g']}, lr_d={self.config['lr_d']}")
        print(f"lambda_erp={self.config['lambda_erp']}, latent_dim={self.config['latent_dim']}, n_critic={self.config['n_critic']}")
        print(f"finetune_mode={self.mode} (trainable: {self.trainable_desc})")
        
        self.training_start_time = time.time()
        n_critic = self.config['n_critic']
        warmup_frac = self.config.get("erp_warmup_frac", 0.0)
        target_lambda_erp = self.config["lambda_erp"]

        # Compute num_real_trials from real_data
        num_real_trials = len(real_data)
        
        if self.mode == "frozen":
            print("[frozen] Skip training, only export final generations/summary.")
            self._save_final(save_dir, data_min, data_max, num_fake_samples)
            self._save_summary(save_dir, epochs=0, batch_size=batch_size, data_min=data_min, data_max=data_max,
                             num_real_trials=num_real_trials, num_fake_samples=num_fake_samples)
            print(f"\nFrozen mode complete. Saved to: {save_dir}")
            return save_dir
        
        for epoch in range(epochs):
            if warmup_frac > 0:
                ratio = max(
                    0.0,
                    min(1.0, (epoch + 1 - epochs * warmup_frac) / max(1, epochs * (1 - warmup_frac)))
                )
                self.loss_fn.lambda_erp = target_lambda_erp * ratio
            else:
                self.loss_fn.lambda_erp = target_lambda_erp
            
            epoch_d = epoch_g = epoch_wdd = epoch_wdg = epoch_erp = 0.0
            num_batches = 0
            
            # shuffle
            idx = torch.randperm(len(real_data), device=real_data.device)
            
            # Use step-based slicing to handle small datasets
            for s in range(0, len(real_data), batch_size):
                e = min(s + batch_size, len(real_data))
                batch = real_data[idx[s:e]]  # already on device
                
                # 维度防呆断言（避免将来数据形状意外出错）
                assert batch.dim() == 2 and batch.size(1) == self.config['eeg_length'], \
                    f"expect [B,{self.config['eeg_length']}], got {tuple(batch.shape)}"
                
                actual_batch_size = batch.size(0)
                
                losses = self.train_step(batch, batch_size=actual_batch_size, n_critic=n_critic)
                epoch_d += losses['d_loss']
                epoch_g += losses['g_loss']
                epoch_wdd += losses['wd_d']
                epoch_wdg += losses['wd_g']
                epoch_erp += losses['erp_loss']
                num_batches += 1
            
            # averages
            avg_d = epoch_d / max(1, num_batches)
            avg_g = epoch_g / max(1, num_batches)
            avg_wdd = epoch_wdd / max(1, num_batches)
            avg_wdg = epoch_wdg / max(1, num_batches)
            avg_erp = epoch_erp / max(1, num_batches)
            
            self.d_losses.append(avg_d)
            self.g_losses.append(avg_g)
            self.wd_d_losses.append(avg_wdd)
            self.wd_g_losses.append(avg_wdg)
            self.erp_losses.append(avg_erp)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:3d}/{epochs}: D={avg_d:.4f}  G={avg_g:.4f}  "
                    f"WD_D={avg_wdd:.4f}  WD_G={avg_wdg:.4f}  ERP={avg_erp:.4f}"
                )
            
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch + 1, save_dir)
                self._save_generated(epoch + 1, save_dir, data_min, data_max, num_fake_samples)
        
        self._save_final(save_dir, data_min, data_max, num_fake_samples)
        self._save_summary(save_dir, epochs, batch_size, data_min, data_max,
                         num_real_trials=num_real_trials, num_fake_samples=num_fake_samples)
        
        print(f"\nTraining complete. Saved to: {save_dir}")
        return save_dir
    
    def _save_checkpoint(self, epoch: int, save_dir: str) -> None:
        ckpt = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'wd_d_losses': self.wd_d_losses,
            'wd_g_losses': self.wd_g_losses,
            'erp_losses': self.erp_losses,
            'subject_token': (self.subject_token.detach().cpu() if self.subject_token is not None else None),
        }
        torch.save(ckpt, os.path.join(save_dir, 'models', f'checkpoint_epoch_{epoch}.pth'))
    
    def _save_generated(self, epoch: int, save_dir: str, data_min: float, data_max: float, num_samples: int) -> None:
        gen = self.generate_trials(num_samples, data_min, data_max)
        pd.DataFrame(gen).to_csv(
            os.path.join(save_dir, 'generated_data', f'generated_trials_epoch_{epoch}.csv'),
            index=False, header=False
        )
    
    def _save_final(self, save_dir: str, data_min: float, data_max: float, num_samples: int) -> None:
        pd.DataFrame({
            'epoch': range(1, len(self.g_losses) + 1),
            'g_loss': self.g_losses,
            'd_loss': self.d_losses,
            'wd_d': self.wd_d_losses,
            'wd_g': self.wd_g_losses,
            'erp_loss': self.erp_losses,
        }).to_csv(os.path.join(save_dir, 'training_losses.csv'), index=False)
        
        save_loss_plots(
            d_losses=self.d_losses,
            g_losses=self.g_losses,
            wd_d_losses=self.wd_d_losses,
            wd_g_losses=self.wd_g_losses,
            erp_losses=self.erp_losses,
            output_dir=save_dir,
            subdir='plots',
        )
        
        final_gen = self.generate_trials(num_samples, data_min, data_max)
        pd.DataFrame(final_gen).to_csv(
            os.path.join(save_dir, 'final_generated_trials.csv'), index=False, header=False
        )
    
    def _save_summary(
        self,
        save_dir: str,
        epochs: int,
        batch_size: int,
        data_min: float,
        data_max: float,
        num_real_trials: int,
        num_fake_samples: int,
    ) -> None:
        g_total = sum(p.numel() for p in self.generator.parameters())
        d_total = sum(p.numel() for p in self.discriminator.parameters())
        g_train = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        d_train = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        t = time.time() - getattr(self, 'training_start_time', time.time())
        
        info = [
            '=' * 60,
            f'{self.mode.capitalize()} Fine-tuning Summary',
            '=' * 60,
            '',
            f"Training Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f'Time (min)    : {t/60:.2f}',
            f'Epochs        : {epochs}',
            f'Batch Size    : {batch_size}',
            f'LR(G/D)       : {self.config["lr_g"]} / {self.config["lr_d"]}',
            f'Lambda ERP    : {self.config["lambda_erp"]}',
            f'Latent Dim    : {self.config["latent_dim"]}',
            f'Finetune Mode : {self.mode} ({self.trainable_desc})',
            '',
            f'G params (total/trainable): {g_total:,} / {g_train:,}',
            f'D params (total/trainable): {d_total:,} / {d_train:,}',
            f'Total params               : {g_total + d_total:,}',
            '',
            f'Data range   : [{data_min:.2f}, {data_max:.2f}] (span {data_max - data_min:.2f})',
            f'Real trials  : {num_real_trials}',
            f'Fake trials  : {num_fake_samples}',
        ]
        
        if self.g_losses:
            info += [
                '', 'Final Losses:',
                f'  G: {self.g_losses[-1]:.4f}',
                f'  D: {self.d_losses[-1]:.4f}',
                f'  WD_D: {self.wd_d_losses[-1]:.4f}',
                f'  WD_G: {self.wd_g_losses[-1]:.4f}',
                f'  ERP: {self.erp_losses[-1]:.4f}',
            ]
        
        with open(os.path.join(save_dir, 'training_summary.txt'), 'w') as f:
            f.write('\n'.join(info))

# ---------- Load pretrained and data ----------

def load_pretrained_models(device: str, model_path: str, config: dict):
    """Load pretrained base_OT.pt model"""
    print(f'Loading pretrained model from: {model_path}')
    
    G = Generator(
        latent_dim=config['latent_dim'],
        eeg_length=config['eeg_length'],
        cutoff_frequency=config['cutoff_frequency'],
        fs=config['fs']
    )
    D = Discriminator(input_dim=config['eeg_length'])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location='cpu')
    
    # Try different checkpoint formats
    if 'generator_state_dict' in ckpt:
        G.load_state_dict(ckpt['generator_state_dict'])
        print("Loaded generator from checkpoint (key: 'generator_state_dict')")
    elif 'generator' in ckpt:
        G.load_state_dict(ckpt['generator'])
        print("Loaded generator from checkpoint (key: 'generator')")
    elif 'model' in ckpt:
        G.load_state_dict(ckpt['model'])
        print("Loaded generator from checkpoint (key: 'model')")
    else:
        # Assume the checkpoint is the generator itself
        G.load_state_dict(ckpt)
        print("Loaded generator directly from checkpoint")
    
    if 'discriminator_state_dict' in ckpt:
        D.load_state_dict(ckpt['discriminator_state_dict'])
        print("Loaded discriminator from checkpoint (key: 'discriminator_state_dict')")
    elif 'discriminator' in ckpt:
        D.load_state_dict(ckpt['discriminator'])
        print("Loaded discriminator from checkpoint (key: 'discriminator')")
    
    # Get data range from checkpoint if available, otherwise use defaults
    if 'real_min' in ckpt and 'real_max' in ckpt:
        real_min = float(ckpt['real_min'])
        real_max = float(ckpt['real_max'])
        print(f'Loaded data range from checkpoint: real_min={real_min:.2f}, real_max={real_max:.2f}')
    else:
        # Use default range (will be computed from data if not in checkpoint)
        real_min = None
        real_max = None
        print('No data range in checkpoint, will compute from data')
    
    return G, D, real_min, real_max

def load_real_data(device: str, csv_path: str, participant_id: int, config: dict,
                   real_min: Optional[float] = None, real_max: Optional[float] = None) -> tuple:
    """
    Load real data from CSV file for single participant
    - First column is participant_id
    - Only keep rows where first column equals participant_id
    - Return only the data columns (skip first column)
    """
    print(f'Loading real {config.get("condition", "ot").upper()} data from: {csv_path}')
    print(f'Filtering rows where first column == {participant_id}')
    
    df = pd.read_csv(csv_path, header=None)
    
    # Filter rows where first column equals participant_id
    filtered_df = df[df.iloc[:, 0] == participant_id]
    print(f"Participant {participant_id}: Found {len(filtered_df)} rows with participant_id={participant_id}")
    
    # Extract only the data columns (skip first column)
    X = filtered_df.iloc[:, 1:].values
    
    # Check shape
    if X.shape[1] != config['eeg_length']:
        raise ValueError(f"Expected {config['eeg_length']} columns, got {X.shape[1]}")
    
    # Remove rows with NaN or Inf
    valid_mask = np.isfinite(X).all(axis=1)
    X = X[valid_mask]
    num_trials = len(X)
    print(f"Participant {participant_id}: {num_trials} trials after removing NaN/Inf")
    
    # Per-trial baseline correction
    time_ms = np.linspace(config['time_start'], config['time_end'], config['eeg_length'])
    time_s = time_ms / 1000.0  # seconds
    baseline_window = (time_s >= config['baseline_start']) & (time_s <= config['baseline_end'])
    
    # Subtract baseline mean for each trial
    baseline_means = X[:, baseline_window].mean(axis=1, keepdims=True)
    X = X - baseline_means
    print(f'Applied per-trial baseline correction (window: {config["baseline_start"]} to {config["baseline_end"]} s)')
    
    # Compute data range if not provided (after baseline correction)
    if real_min is None or real_max is None:
        real_min = float(X.min())
        real_max = float(X.max())
        print(f'Computed data range (after baseline correction): real_min={real_min:.2f}, real_max={real_max:.2f}')
    
    # Normalize to [-1, 1]
    real = torch.tensor(X, dtype=torch.float32, device=device)
    real = (real - real_min) / (real_max - real_min) * 2.0 - 1.0  # [-1,1]
    
    print(f'Real normalized: {tuple(real.shape)}, range=[{real.min():.2f},{real.max():.2f}]')
    
    return real, real_min, real_max

# ---------- Main ----------

def get_all_participant_ids(csv_path: str) -> list:
    """Get all unique participant IDs from CSV file (first column)."""
    print(f'Reading participant IDs from: {csv_path}')
    df = pd.read_csv(csv_path, header=None)
    participant_ids = sorted(df.iloc[:, 0].unique().tolist())
    print(f'Found {len(participant_ids)} unique participants: {participant_ids}')
    return participant_ids


def train_single_participant(participant_id: int, device: str, config: dict) -> str:
    """Train a single participant model."""
    print('\n' + '=' * 80)
    print(f'Training participant {participant_id}')
    print('=' * 80)
    
    # Create a copy of config with participant_id
    participant_config = config.copy()
    participant_config['participant_id'] = participant_id
    
    # Load pretrained models (fresh copy for each participant)
    G, D, rmin, rmax = load_pretrained_models(device, config['model_path'], config)
    
    mode = config.get("finetune_mode", "full").lower()
    if mode == "lora" and config.get("use_lora", False):
        G = apply_lora_to_model(
            G,
            r=config["lora_r"],
            alpha=config["lora_alpha"],
            dropout=config["lora_dropout"],
            which=config["lora_targets"],
        )
    
    G = G.to(device)
    D = D.to(device)
    
    # Load real data for this participant
    real, rmin, rmax = load_real_data(
        device, config['csv_path'], participant_id, config, rmin, rmax
    )
    
    # Compute number of fake samples based on real data size
    num_real_trials = len(real)
    num_fake_samples = compute_num_fake_samples(num_real_trials)
    print(f'Real trials: {num_real_trials}, will generate {num_fake_samples} fake trials')
    
    # Create fine-tuner
    tuner = FullFineTuner(G, D, device, participant_config)
    
    # Train
    save_dir = tuner.train(
        real_data=real,
        data_min=rmin,
        data_max=rmax,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        save_interval=config['save_interval'],
        num_fake_samples=num_fake_samples,
    )
    
    return save_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune participant models for OT or MW.")
    parser.add_argument("--condition", choices=["ot", "mw"], default=CONFIG["condition"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--finetune-mode", choices=["full", "partial", "lora", "frozen"], default=None)
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA when --finetune-mode lora is used.")
    parser.add_argument("--no-subject-token", action="store_true")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> dict:
    cfg = CONFIG.copy()
    cfg["condition"] = args.condition
    if args.condition == "mw":
        cfg["model_path"] = os.path.join(PROJECT_ROOT, "real_EEG_pretrained_model", "base_MW.pt")
        cfg["csv_path"] = os.path.join(PROJECT_ROOT, "real_EEG_pretrained_model", "mw_vs_Pz_1789.csv")
        cfg["output_dir"] = os.path.join(PROJECT_ROOT, "mw_pipeline", "finetuned_mw_participant")

    for key, value in (
        ("model_path", args.model_path),
        ("csv_path", args.csv_path),
        ("output_dir", args.output_dir),
        ("epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("save_interval", args.save_interval),
        ("finetune_mode", args.finetune_mode),
    ):
        if value is not None:
            cfg[key] = value

    if args.use_lora:
        cfg["use_lora"] = True
    if args.no_subject_token:
        cfg["use_subject_token"] = False
    return cfg


def main() -> None:
    config = build_config_from_args(parse_args())
    print('=' * 80)
    print(f'Fine-tune {config["condition"].upper()} model for ALL participants')
    print('=' * 80)
    
    # Determinism for HPC
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    except Exception as e:
        print(f'[warn] deterministic off: {e}')
    
    # Auto-detect GPU (适合HPC环境)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Get all participant IDs
    participant_ids = get_all_participant_ids(config['csv_path'])
    
    if not participant_ids:
        print('No participants found. Exiting.')
        return
    
    print(f'\nWill train {len(participant_ids)} participants: {participant_ids}')
    print(f'Finetune mode: {config.get("finetune_mode", "full")}')
    print(f'Epochs: {config["epochs"]}, Batch size: {config["batch_size"]}')
    print(f'Save interval: {config["save_interval"]} epochs')
    print(f'Output dir: {config["output_dir"]}\n')
    
    # Train each participant
    results = []
    for idx, pid in enumerate(participant_ids, 1):
        print(f'\n[{idx}/{len(participant_ids)}] Processing participant {pid}...')
        try:
            save_dir = train_single_participant(pid, device, config)
            results.append({'participant_id': pid, 'status': 'success', 'save_dir': save_dir})
            print(f'✓ Participant {pid} completed. Saved to: {save_dir}')
        except Exception as e:
            print(f'✗ Participant {pid} failed: {e}')
            import traceback
            traceback.print_exc()
            results.append({'participant_id': pid, 'status': 'failed', 'error': str(e)})
    
    # Print summary
    print('\n' + '=' * 80)
    print('Training Summary')
    print('=' * 80)
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    print(f'Total participants: {len(participant_ids)}')
    print(f'Successful: {len(successful)}')
    print(f'Failed: {len(failed)}')
    
    if failed:
        print('\nFailed participants:')
        for r in failed:
            print(f"  Participant {r['participant_id']}: {r.get('error', 'Unknown error')}")
    
    # Save summary to file
    summary_dir = config['output_dir']
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(summary_dir, 'training_summary_all_participants.txt')
    with open(summary_file, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write(f'Fine-tuning Summary ({config["condition"].upper()}, All Participants)\n')
        f.write('=' * 80 + '\n\n')
        f.write(f'Training Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Total participants: {len(participant_ids)}\n')
        f.write(f'Successful: {len(successful)}\n')
        f.write(f'Failed: {len(failed)}\n\n')
        f.write('Successful participants:\n')
        for r in successful:
            f.write(f"  Participant {r['participant_id']}: {r['save_dir']}\n")
        if failed:
            f.write('\nFailed participants:\n')
            for r in failed:
                f.write(f"  Participant {r['participant_id']}: {r.get('error', 'Unknown error')}\n")
    print(f'\nSummary saved to: {summary_file}')

if __name__ == '__main__':
    main()
