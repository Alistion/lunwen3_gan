from __future__ import annotations

import torch
import torch.nn.functional as F


def amplitude_features(x: torch.Tensor, fs: int = 2048, max_freq: float = 100.0) -> torch.Tensor:
    if x.dim() == 3:
        x = x.squeeze(1)
    rms = torch.sqrt(torch.mean(x.pow(2), dim=1) + 1e-8)
    ptp = torch.max(x, dim=1).values - torch.min(x, dim=1).values
    spec = torch.abs(torch.fft.rfft(x, dim=-1))
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / float(fs)).to(x.device)
    fft_peak = spec[:, freqs <= float(max_freq)].max(dim=1).values
    return torch.stack([rms, ptp, fft_peak], dim=1)


def build_amplitude_templates(
    train_x,
    train_y,
    fs: int = 2048,
    max_freq: float = 100.0,
    num_classes: int = 5,
    device: torch.device | None = None,
) -> torch.Tensor:
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    feats = amplitude_features(x, fs=fs, max_freq=max_freq)
    global_template = feats.mean(dim=0)
    templates = []
    for cls in range(num_classes):
        idx = y == cls
        templates.append(feats[idx].mean(dim=0) if bool(idx.any()) else global_template)
    return torch.stack(templates, dim=0)


def amplitude_consistency_loss(
    x_fake: torch.Tensor,
    y: torch.Tensor,
    amplitude_templates: torch.Tensor,
    fs: int = 2048,
    max_freq: float = 100.0,
) -> torch.Tensor:
    fake_amp_feat = torch.log1p(amplitude_features(x_fake, fs=fs, max_freq=max_freq))
    real_amp_template = torch.log1p(amplitude_templates.to(device=x_fake.device, dtype=x_fake.dtype)[y])
    return F.l1_loss(fake_amp_feat, real_amp_template)
