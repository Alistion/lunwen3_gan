from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mhta_ddpm_1d import ConditionalResBlock1D, Downsample1D, Upsample1D, timestep_embedding


class ConditionalUNet1D(nn.Module):
    """Plain conditional 1D U-Net baseline for DDPM noise prediction.

    This intentionally mirrors the main MHTA-DDPM backbone while removing all
    attention modules, so the comparison isolates the contribution of MHTA.
    """

    def __init__(
        self,
        signal_length: int = 2048,
        num_classes: int = 5,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 48,
        channel_mults: Iterable[int] = (1, 2, 4, 8),
        time_embed_dim: int = 192,
        label_embed_dim: int = 192,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.signal_length = int(signal_length)
        self.num_classes = int(num_classes)
        self.time_embed_dim = int(time_embed_dim)
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.GELU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        self.label_mlp = nn.Sequential(
            nn.Linear(label_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        channels = [base_channels * int(mult) for mult in channel_mults]
        self.input_conv = nn.Conv1d(in_channels, channels[0], kernel_size=31, padding=15)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = channels[0]
        for level, out_ch in enumerate(channels):
            self.down_blocks.append(ConditionalResBlock1D(in_ch, out_ch, time_embed_dim, dropout))
            in_ch = out_ch
            if level != len(channels) - 1:
                self.downsamples.append(Downsample1D(out_ch))

        self.mid_res = ConditionalResBlock1D(channels[-1], channels[-1], time_embed_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_channels = list(reversed(channels))
        in_ch = rev_channels[0]
        for level, out_ch in enumerate(rev_channels):
            self.up_blocks.append(ConditionalResBlock1D(in_ch + out_ch, out_ch, time_embed_dim, dropout))
            in_ch = out_ch
            if level != len(rev_channels) - 1:
                self.upsamples.append(Upsample1D(out_ch))

        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=channels[0])
        self.out_conv = nn.Conv1d(channels[0], out_channels, kernel_size=3, padding=1)

    def condition(self, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        time_emb = timestep_embedding(timesteps, self.time_embed_dim)
        return self.time_mlp(time_emb) + self.label_mlp(self.label_embed(labels))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        cond = self.condition(timesteps, labels)
        h = self.input_conv(x)
        skips = []
        for level, block in enumerate(self.down_blocks):
            h = block(h, cond)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)
        h = self.mid_res(h, cond)
        for level, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), cond)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)
        if h.shape[-1] != self.signal_length:
            h = F.interpolate(h, size=self.signal_length, mode="linear", align_corners=False)
        return self.out_conv(F.silu(self.out_norm(h)))
