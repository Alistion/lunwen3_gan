from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class OMFM(nn.Module):
    """Order Mechanism Feature Modulation.

    The order template tells the generator which mechanism-related spectral
    bands should be emphasized, then gamma/beta modulate intermediate feature
    maps in the same spirit as conditional normalization.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class UpsampleResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch"):
        super().__init__()
        norm_layer = nn.BatchNorm1d if norm == "batch" else nn.InstanceNorm1d
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_channels, out_channels, 5, padding=2),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            norm_layer(out_channels),
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_channels, out_channels, 1),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class OrderGuidedGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 5,
        embed_dim: int = 32,
        base_channels: int = 128,
        order_dim: int = 6,
        use_tanh_output: bool = False,
        out_tanh: bool | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.use_tanh_output = use_tanh_output if out_tanh is None else out_tanh
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.rpm_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(embed_dim, embed_dim))
        self.order_mlp = nn.Sequential(
            nn.Linear(order_dim, embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        cond_dim = latent_dim + embed_dim * 3
        self.fc = nn.Sequential(nn.Linear(cond_dim, base_channels * 128), nn.LeakyReLU(0.2, inplace=True))
        self.blocks = nn.ModuleList(
            [
                UpsampleResBlock(base_channels, base_channels // 2),
                UpsampleResBlock(base_channels // 2, base_channels // 4),
                UpsampleResBlock(base_channels // 4, base_channels // 8),
                UpsampleResBlock(base_channels // 8, base_channels // 16),
            ]
        )
        self.omfm1 = OMFM(cond_dim, base_channels // 2)
        self.omfm2 = OMFM(cond_dim, base_channels // 8)
        self.out = nn.Conv1d(base_channels // 16, 1, kernel_size=7, padding=3)

    def condition_vector(self, z: torch.Tensor, y: torch.Tensor, rpm: torch.Tensor, order_template: torch.Tensor) -> torch.Tensor:
        rpm = rpm.float().view(-1, 1) / 1000.0
        return torch.cat([z, self.label_embed(y), self.rpm_mlp(rpm), self.order_mlp(order_template.float())], dim=1)

    def forward(self, z: torch.Tensor, y: torch.Tensor, rpm: torch.Tensor, order_template: torch.Tensor) -> torch.Tensor:
        cond = self.condition_vector(z, y, rpm, order_template)
        x = self.fc(cond).view(z.size(0), -1, 128)
        x = self.blocks[0](x)
        x = self.omfm1(x, cond)
        x = self.blocks[1](x)
        x = self.blocks[2](x)
        x = self.omfm2(x, cond)
        x = self.blocks[3](x)
        x = self.out(x)
        if self.use_tanh_output:
            x = torch.tanh(x)
        return x


def conv_block(in_channels: int, out_channels: int, use_sn: bool = True) -> nn.Sequential:
    conv = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
    if use_sn:
        conv = spectral_norm(conv)
    return nn.Sequential(conv, nn.LeakyReLU(0.2, inplace=True))


class TimeFrequencyDiscriminator(nn.Module):
    """Conditional critic with time and frequency branches.

    The time branch captures impacts and waveform morphology, while the
    frequency branch sees log rFFT magnitudes so the critic can judge spectral
    signatures that are crucial in rotating machinery faults.
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 32,
        base_channels: int = 32,
        order_dim: int = 6,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.rpm_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(embed_dim, embed_dim))
        self.order_mlp = nn.Sequential(nn.Linear(order_dim, embed_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(embed_dim, embed_dim))
        self.time_branch = nn.Sequential(
            conv_block(1, base_channels, use_spectral_norm),
            conv_block(base_channels, base_channels * 2, use_spectral_norm),
            conv_block(base_channels * 2, base_channels * 4, use_spectral_norm),
            conv_block(base_channels * 4, base_channels * 4, use_spectral_norm),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.freq_branch = nn.Sequential(
            conv_block(1, base_channels, use_spectral_norm),
            conv_block(base_channels, base_channels * 2, use_spectral_norm),
            conv_block(base_channels * 2, base_channels * 4, use_spectral_norm),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        feat_dim = base_channels * 8 + embed_dim * 3
        self.critic = nn.Sequential(nn.Linear(feat_dim, base_channels * 4), nn.LeakyReLU(0.2, inplace=True), nn.Linear(base_channels * 4, 1))
        self.aux = nn.Sequential(nn.Linear(feat_dim, base_channels * 4), nn.LeakyReLU(0.2, inplace=True), nn.Linear(base_channels * 4, num_classes))

    def cond(self, y: torch.Tensor, rpm: torch.Tensor, order_template: torch.Tensor) -> torch.Tensor:
        rpm = rpm.float().view(-1, 1) / 1000.0
        return torch.cat([self.label_embed(y), self.rpm_mlp(rpm), self.order_mlp(order_template.float())], dim=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, rpm: torch.Tensor, order_template: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        spec = torch.fft.rfft(x.squeeze(1), dim=-1)
        mag = torch.log1p(torch.abs(spec)).unsqueeze(1)
        feat = torch.cat([self.time_branch(x), self.freq_branch(mag), self.cond(y, rpm, order_template)], dim=1)
        return self.critic(feat), self.aux(feat)


def save_generator_checkpoint(path: Path, generator: OrderGuidedGenerator, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generator_state_dict": generator.state_dict(), **meta}, path)
