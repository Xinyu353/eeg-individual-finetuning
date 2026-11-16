import numpy as np
import torch
from scipy.signal import welch, butter, filtfilt

import torch.autograd as autograd




def compute_psd(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    return freqs, psd


def normalize_signal(signal, range_min=-10, range_max=10):
    signal_min = signal.min()
    signal_max = signal.max()
    return (range_max - range_min) * (signal - signal_min) / (signal_max - signal_min) + range_min



def psd_loss(real_signal, fake_signal, fs, freq_range=(0, 30), weight_range1=(4, 12), weight_factor1=1.5,
             weight_range2=(12, 30), weight_factor2=2, norm_method='minmax'):
    loss = 0.0

    for i in range(real_signal.shape[0]):
        real_freqs, real_psd = compute_psd(real_signal[i].cpu().detach().numpy(), fs)
        fake_freqs, fake_psd = compute_psd(fake_signal[i].cpu().detach().numpy(), fs)

        # Select frequencies within the specified range
        freq_idx = np.where((real_freqs >= freq_range[0]) & (real_freqs <= freq_range[1]))[0]
        real_psd = real_psd[freq_idx]
        fake_psd = fake_psd[freq_idx]
        selected_freqs = real_freqs[freq_idx]
        # Normalize PSD
        if norm_method == 'standard':
            real_psd = (real_psd - np.mean(real_psd)) / np.std(real_psd)
            fake_psd = (fake_psd - np.mean(fake_psd)) / np.std(fake_psd)
        elif norm_method == 'minmax':
            real_psd = normalize_signal(real_psd)
            fake_psd = normalize_signal(fake_psd)
        else:
            raise ValueError("Normalization method must be 'standard' or 'minmax'")

        # Apply weighting
        # weights = torch.tensor(np.ones_like(real_psd), dtype=torch.float32)
        # weight_idx = np.where((real_freqs[freq_idx] >= weight_range[0]) & (real_freqs[freq_idx] <= weight_range[1]))[0]
        # weights[weight_idx] = weight_factor
        weights = np.ones_like(real_psd, dtype=np.float32)

        # Apply weight factor 1 in the first frequency range
        weight_idx1 = np.where((selected_freqs >= weight_range1[0]) & (selected_freqs <= weight_range1[1]))[0]
        weights[weight_idx1] = weight_factor1

        # Apply weight factor 2 in the second frequency range
        weight_idx2 = np.where((selected_freqs >= weight_range2[0]) & (selected_freqs <= weight_range2[1]))[0]
        weights[weight_idx2] = weight_factor2

        # Apply the weighted loss (e.g., MSE)
        weighted_loss = np.mean(weights * (real_psd - fake_psd) ** 2)
        # weighted_loss = np.mean(weights * np.abs((real_psd - fake_psd) / real_psd))
        loss += weighted_loss

        # loss += np.mean(weights*(real_psd - fake_psd) ** 2)

    return torch.tensor(loss / real_signal.shape[0], dtype=torch.float32).to(real_signal.device)
    # return torch.tensor(loss, dtype=torch.float32).to(real_signal.device)



def erp_loss(fake_samples, Fs=256):
    """
    ERP loss: penalize high-frequency components in the average ERP.
    Filters out 0.5-5 Hz and computes MSE between original and filtered.
    """
    # Compute the average ERP for fake samples
    avg_fake = np.mean(fake_samples.cpu().detach().numpy(), axis=0)

    # Design a 4th-order Butterworth bandpass filter (0.5-5 Hz)
    lowcut = 0.5
    highcut = 5
    order = 4
    nyq = 0.5 * Fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to the average ERP
    filtered_fake = filtfilt(b, a, avg_fake)

    # Compute the difference between original and filtered ERP
    difference = avg_fake - filtered_fake

    # Calculate the mean squared error as the loss
    erp_loss_value = np.mean(difference ** 2)

    return torch.tensor(erp_loss_value, dtype=torch.float32).to(fake_samples.device)



class WGP_loss:
    def __init__(self, lambda_gp=10.0, lambda_wd=10.0, lambda_erp=0.0):
        self.lambda_gp = lambda_gp
        self.lambda_wd = lambda_wd
        self.lambda_erp = lambda_erp

    def d_loss(self, discriminator, real_samples, fake_samples):
        # Move tensors to the correct device
        device = real_samples.device
        real_samples = real_samples.to(device)
        fake_samples = fake_samples.to(device)

        # Calculate standard WGAN discriminator loss
        real_scores = discriminator(real_samples)
        fake_scores = discriminator(fake_samples)
        d_loss = -torch.mean(real_scores) + torch.mean(fake_scores)

        # Calculate gradient penalty
        gradient_penalty = self._gp(discriminator, real_samples, fake_samples)

        # Total discriminator loss
        return d_loss + gradient_penalty, d_loss

    def wd_loss(self, discriminator, fake_samples):
        # Move tensors to the correct device
        device = fake_samples.device
        fake_samples = fake_samples.to(device)

        # Generator loss for WGAN is negative mean of discriminator scores for fake samples
        fake_scores = discriminator(fake_samples)
        return -torch.mean(fake_scores)

    def g_loss(self, discriminator, fake_samples, real_samples, fs):
        """
        Generator loss: WGAN loss + ERP loss
        """
        g_loss_wgan_value = self.lambda_wd * self.wd_loss(discriminator, fake_samples)
        g_loss_erp = self.lambda_erp * erp_loss(fake_samples, Fs=fs)

        g_loss_total = g_loss_wgan_value + g_loss_erp

        return g_loss_total, g_loss_wgan_value, g_loss_erp

    def _gp(self, discriminator, real_samples, fake_samples):
        """Calculates the gradient penalty for WGAN-GP"""
        batch_size = real_samples.size(0)
        device = real_samples.device

        # Generate random epsilon

        epsilon = torch.rand(batch_size, 1, device=device, requires_grad=True) #上次是1，1
        # epsilon = epsilon.expand_as(real_samples)


        # Interpolate between real and fake samples
        interpolated_samples = epsilon * real_samples + (1 - epsilon) * fake_samples
        interpolated_samples = interpolated_samples.to(device)
        interpolated_samples.detach().requires_grad_(True)

        # Calculate critic scores for interpolated samples
        interpolated_scores = discriminator(interpolated_samples)


        # Compute gradients of critic scores with respect to interpolated samples
        gradients = autograd.grad(outputs=interpolated_scores,
                                  inputs=interpolated_samples,
                                  grad_outputs=torch.ones_like(interpolated_scores).to(device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]

        # Flatten the gradients to compute the 2-norm across each sample
        # gradients = gradients.view(batch_size, -1)

        gradients = gradients.view(gradients.size()[0],-1)
        # gradients = gradients.reshape(gradients.size()[0], -1)


        # Calculate gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp


        return gradient_penalty
