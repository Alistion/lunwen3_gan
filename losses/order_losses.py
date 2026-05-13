from __future__ import annotations

import torch
import torch.nn.functional as F


ORDER_MULTIPLES = (0.5, 1.0, 1.5, 2.0, 3.0)


def compute_order_energy(x: torch.Tensor, fs: float = 2048.0, rpm: torch.Tensor | float = 740.0, band_width: float = 2.0) -> torch.Tensor:
    """Return [0.5X, 1X, 1.5X, 2X, 3X, broadband] spectral energies.

    Order energy aligns generated samples with rotating-machine fault
    mechanisms instead of only matching generic waveform appearance.
    """
    if x.dim() == 3:
        x = x.squeeze(1)
    bsz, n = x.shape
    spec = torch.fft.rfft(x, dim=-1)
    power = torch.abs(spec).pow(2)
    freqs = torch.fft.rfftfreq(n, d=1.0 / float(fs)).to(x.device)
    if not torch.is_tensor(rpm):
        rpm = torch.full((bsz,), float(rpm), device=x.device, dtype=x.dtype)
    rpm = rpm.to(device=x.device, dtype=x.dtype).view(-1)
    fr = rpm / 60.0
    energies = []
    for multiple in ORDER_MULTIPLES:
        center = fr * multiple
        mask = (freqs.unsqueeze(0) >= center.unsqueeze(1) - band_width) & (
            freqs.unsqueeze(0) <= center.unsqueeze(1) + band_width
        )
        energies.append((power * mask.float()).sum(dim=1))
    broadband = power[:, (freqs >= 80.0) & (freqs <= 500.0)].sum(dim=1)
    energies.append(broadband)
    return torch.stack(energies, dim=1)


def normalize_energy(energy: torch.Tensor) -> torch.Tensor:
    return energy / (energy.sum(dim=1, keepdim=True) + 1e-8)


def order_consistency_loss(
    real: torch.Tensor,
    fake: torch.Tensor,
    y: torch.Tensor,
    fs: float,
    rpm: torch.Tensor,
    num_classes: int = 5,
    band_width: float = 2.0,
) -> torch.Tensor:
    real_e = normalize_energy(compute_order_energy(real, fs=fs, rpm=rpm, band_width=band_width))
    fake_e = normalize_energy(compute_order_energy(fake, fs=fs, rpm=rpm, band_width=band_width))
    losses = []
    for cls in range(num_classes):
        mask = y == cls
        if bool(mask.any()):
            losses.append(F.mse_loss(fake_e[mask].mean(dim=0), real_e[mask].mean(dim=0)))
    if not losses:
        return fake.sum() * 0.0
    return torch.stack(losses).mean()


def order_consistency_components(
    real: torch.Tensor,
    fake: torch.Tensor,
    y: torch.Tensor,
    target_masks: torch.Tensor,
    fs: float,
    rpm: torch.Tensor,
    num_classes: int = 5,
    band_width: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    real_e = normalize_energy(compute_order_energy(real, fs=fs, rpm=rpm, band_width=band_width))
    fake_e = normalize_energy(compute_order_energy(fake, fs=fs, rpm=rpm, band_width=band_width))
    align_losses = []
    for cls in range(num_classes):
        mask = y == cls
        if bool(mask.any()):
            align_losses.append(F.mse_loss(fake_e[mask].mean(dim=0), real_e[mask].mean(dim=0)))
    if align_losses:
        align_loss = torch.stack(align_losses).mean()
    else:
        align_loss = fake.sum() * 0.0

    sample_masks = target_masks.to(device=fake.device, dtype=fake.dtype)[y]
    suppression_loss = (fake_e * (1.0 - sample_masks)).mean()

    unbalance_mask = y == 1
    if bool(unbalance_mask.any()):
        unbalance_suppression_loss = fake_e[unbalance_mask][:, [3, 4, 5]].mean()
    else:
        unbalance_suppression_loss = fake.sum() * 0.0
    return align_loss, suppression_loss, unbalance_suppression_loss


def statistical_features(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.squeeze(1)
    eps = 1e-8
    rms = torch.sqrt(torch.mean(x.pow(2), dim=1) + eps)
    ptp = torch.max(x, dim=1).values - torch.min(x, dim=1).values
    mean = torch.mean(x, dim=1, keepdim=True)
    var = torch.mean((x - mean).pow(2), dim=1)
    kurtosis = torch.mean((x - mean).pow(4), dim=1) / (var.pow(2) + eps)
    crest = torch.max(torch.abs(x), dim=1).values / (rms + eps)
    return torch.log1p(torch.stack([rms, ptp, kurtosis, crest], dim=1))


def statistical_consistency_loss(real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    real_s = statistical_features(real)
    fake_s = statistical_features(fake)
    losses = []
    for cls in range(num_classes):
        mask = y == cls
        if bool(mask.any()):
            losses.append(F.mse_loss(fake_s[mask].mean(dim=0), real_s[mask].mean(dim=0)))
    if not losses:
        return fake.sum() * 0.0
    return torch.stack(losses).mean()
