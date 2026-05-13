from __future__ import annotations

import torch
import torch.nn.functional as F


def envelope_rms_sequence(x: torch.Tensor, segments: int = 16) -> torch.Tensor:
    if x.dim() == 3:
        x = x.squeeze(1)
    bsz, n = x.shape
    seg_len = n // int(segments)
    x = x[:, : seg_len * int(segments)].reshape(bsz, int(segments), seg_len)
    return torch.sqrt(torch.mean(x.pow(2), dim=-1) + 1e-8)


def build_envelope_templates(
    train_x,
    train_y,
    segments: int = 16,
    num_classes: int = 5,
    device: torch.device | None = None,
) -> torch.Tensor:
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    env = envelope_rms_sequence(x, segments=segments)
    global_template = env.mean(dim=0)
    templates = []
    for cls in range(num_classes):
        idx = y == cls
        templates.append(env[idx].mean(dim=0) if bool(idx.any()) else global_template)
    return torch.stack(templates, dim=0)


def envelope_consistency_loss(
    x_fake: torch.Tensor,
    y: torch.Tensor,
    envelope_templates: torch.Tensor,
    segments: int = 16,
) -> torch.Tensor:
    fake_env = torch.log1p(envelope_rms_sequence(x_fake, segments=segments))
    target_env = torch.log1p(envelope_templates.to(device=x_fake.device, dtype=x_fake.dtype)[y])
    return F.l1_loss(fake_env, target_env)


def autocorrelation_sequence(x: torch.Tensor, max_lag: int = 512) -> torch.Tensor:
    if x.dim() == 3:
        x = x.squeeze(1)
    x = x - x.mean(dim=-1, keepdim=True)
    n = x.shape[-1]
    fft_len = 2 * n
    spec = torch.fft.rfft(x, n=fft_len, dim=-1)
    acf = torch.fft.irfft(spec * torch.conj(spec), n=fft_len, dim=-1)[..., : n]
    acf = acf / (acf[:, :1] + 1e-8)
    return acf[:, 1 : int(max_lag) + 1]


def build_acf_templates(
    train_x,
    train_y,
    max_lag: int = 512,
    num_classes: int = 5,
    device: torch.device | None = None,
) -> torch.Tensor:
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    acf = autocorrelation_sequence(x, max_lag=max_lag)
    global_template = acf.mean(dim=0)
    templates = []
    for cls in range(num_classes):
        idx = y == cls
        templates.append(acf[idx].mean(dim=0) if bool(idx.any()) else global_template)
    return torch.stack(templates, dim=0)


def acf_consistency_loss(
    x_fake: torch.Tensor,
    y: torch.Tensor,
    acf_templates: torch.Tensor,
    max_lag: int = 512,
) -> torch.Tensor:
    fake_acf = autocorrelation_sequence(x_fake, max_lag=max_lag)
    target_acf = acf_templates.to(device=x_fake.device, dtype=x_fake.dtype)[y]
    return F.l1_loss(fake_acf, target_acf)


def temporal_smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(1)
    return torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))
