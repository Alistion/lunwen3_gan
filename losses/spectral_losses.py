from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_log_spectrum(x: torch.Tensor, fs: int = 2048, max_freq: float = 100.0, normalize: bool = True) -> torch.Tensor:
    if x.dim() == 3:
        x = x.squeeze(1)
    n = x.shape[-1]
    spec = torch.fft.rfft(x, dim=-1)
    mag = torch.log1p(torch.abs(spec))
    freqs = torch.fft.rfftfreq(n, d=1.0 / float(fs)).to(x.device)
    mask = freqs <= float(max_freq)
    mag = mag[:, mask]
    if normalize:
        mag = mag / (mag.sum(dim=-1, keepdim=True) + 1e-8)
    return mag


def spectrum_shape_loss(
    x_fake: torch.Tensor,
    x_real: torch.Tensor,
    y: torch.Tensor,
    fs: int = 2048,
    max_freq: float = 100.0,
    num_classes: int = 5,
    loss_type: str = "l1",
) -> tuple[torch.Tensor, int]:
    losses = []
    valid_classes = 0
    for cls in range(num_classes):
        idx = y == cls
        if int(idx.sum().item()) < 1:
            continue
        real_spec = compute_log_spectrum(x_real[idx], fs=fs, max_freq=max_freq, normalize=True).mean(dim=0)
        fake_spec = compute_log_spectrum(x_fake[idx], fs=fs, max_freq=max_freq, normalize=True).mean(dim=0)
        if loss_type == "mse":
            losses.append(F.mse_loss(fake_spec, real_spec))
        elif loss_type == "l1":
            losses.append(F.l1_loss(fake_spec, real_spec))
        else:
            raise ValueError(f"Unsupported spectrum loss type: {loss_type}")
        valid_classes += 1
    if not losses:
        return x_fake.sum() * 0.0, 0
    return torch.stack(losses).mean(), valid_classes


def build_real_spectrum_templates(
    train_x,
    train_y,
    fs: int = 2048,
    max_freq: float = 100.0,
    num_classes: int = 5,
    device: torch.device | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    specs = compute_log_spectrum(x, fs=fs, max_freq=max_freq, normalize=normalize)
    templates = []
    global_template = specs.mean(dim=0)
    for cls in range(num_classes):
        idx = y == cls
        templates.append(specs[idx].mean(dim=0) if bool(idx.any()) else global_template)
    return torch.stack(templates, dim=0)


def build_real_spectrum_template_pair(
    train_x,
    train_y,
    fs: int = 2048,
    max_freq: float = 100.0,
    num_classes: int = 5,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape_templates = build_real_spectrum_templates(
        train_x,
        train_y,
        fs=fs,
        max_freq=max_freq,
        num_classes=num_classes,
        device=device,
        normalize=True,
    )
    amp_templates = build_real_spectrum_templates(
        train_x,
        train_y,
        fs=fs,
        max_freq=max_freq,
        num_classes=num_classes,
        device=device,
        normalize=False,
    )
    return shape_templates, amp_templates


def template_spectrum_shape_loss(
    x_fake: torch.Tensor,
    y: torch.Tensor,
    real_spec_templates: torch.Tensor,
    fs: int = 2048,
    max_freq: float = 100.0,
    loss_type: str = "l1",
) -> tuple[torch.Tensor, int]:
    fake_spec = compute_log_spectrum(x_fake, fs=fs, max_freq=max_freq, normalize=True)
    target_spec = real_spec_templates.to(device=x_fake.device, dtype=x_fake.dtype)[y]
    if loss_type == "mse":
        loss = F.mse_loss(fake_spec, target_spec)
    elif loss_type == "l1":
        loss = F.l1_loss(fake_spec, target_spec)
    else:
        raise ValueError(f"Unsupported spectrum loss type: {loss_type}")
    return loss, int(torch.unique(y).numel())


def template_spectrum_shape_amp_loss(
    x_fake: torch.Tensor,
    y: torch.Tensor,
    real_shape_templates: torch.Tensor,
    real_amp_templates: torch.Tensor,
    fs: int = 2048,
    max_freq: float = 100.0,
    loss_type: str = "l1",
    lambda_spec_amp: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    fake_shape_spec = compute_log_spectrum(x_fake, fs=fs, max_freq=max_freq, normalize=True)
    fake_amp_spec = compute_log_spectrum(x_fake, fs=fs, max_freq=max_freq, normalize=False)
    target_shape = real_shape_templates.to(device=x_fake.device, dtype=x_fake.dtype)[y]
    target_amp = real_amp_templates.to(device=x_fake.device, dtype=x_fake.dtype)[y]
    if loss_type == "mse":
        shape_loss = F.mse_loss(fake_shape_spec, target_shape)
        amp_loss = F.mse_loss(fake_amp_spec, target_amp)
    elif loss_type == "l1":
        shape_loss = F.l1_loss(fake_shape_spec, target_shape)
        amp_loss = F.l1_loss(fake_amp_spec, target_amp)
    else:
        raise ValueError(f"Unsupported spectrum loss type: {loss_type}")
    total_loss = shape_loss + float(lambda_spec_amp) * amp_loss
    return total_loss, shape_loss, amp_loss, int(torch.unique(y).numel())


def spectrum_debug_stats(
    x_fake: torch.Tensor,
    x_real: torch.Tensor,
    fs: int = 2048,
    max_freq: float = 100.0,
) -> dict[str, float | tuple[int, ...]]:
    real_spec = compute_log_spectrum(x_real, fs=fs, max_freq=max_freq, normalize=True)
    fake_spec = compute_log_spectrum(x_fake, fs=fs, max_freq=max_freq, normalize=True)
    return {
        "real_spec_mean": float(real_spec.mean().detach().cpu().item()),
        "real_spec_std": float(real_spec.std(unbiased=False).detach().cpu().item()),
        "fake_spec_mean": float(fake_spec.mean().detach().cpu().item()),
        "fake_spec_std": float(fake_spec.std(unbiased=False).detach().cpu().item()),
        "real_spec_shape": tuple(real_spec.shape),
        "fake_spec_shape": tuple(fake_spec.shape),
    }


def template_spectrum_debug_stats(
    x_fake: torch.Tensor,
    y: torch.Tensor,
    real_spec_templates: torch.Tensor,
    fs: int = 2048,
    max_freq: float = 100.0,
) -> dict[str, float | tuple[int, ...]]:
    real_spec = real_spec_templates.to(device=x_fake.device, dtype=x_fake.dtype)[y]
    fake_spec = compute_log_spectrum(x_fake, fs=fs, max_freq=max_freq, normalize=True)
    return {
        "real_spec_mean": float(real_spec.mean().detach().cpu().item()),
        "real_spec_std": float(real_spec.std(unbiased=False).detach().cpu().item()),
        "fake_spec_mean": float(fake_spec.mean().detach().cpu().item()),
        "fake_spec_std": float(fake_spec.std(unbiased=False).detach().cpu().item()),
        "real_spec_shape": tuple(real_spec.shape),
        "fake_spec_shape": tuple(fake_spec.shape),
    }
