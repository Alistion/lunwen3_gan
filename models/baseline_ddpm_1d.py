from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class DDPMDownsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DDPMUpsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class DDPMResBlock1D(nn.Module):
    """1D adaptation of the residual block used in the original DDPM U-Net."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = zero_module(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1))
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond_proj(F.silu(cond)).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class DDPMAttentionBlock1D(nn.Module):
    """1D self-attention block matching the original DDPM attention pattern."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, length = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        scale = channels ** -0.25
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight, dim=-1)
        h = torch.einsum("bts,bcs->bct", weight, v)
        return x + self.proj_out(h)


class ConditionalUNet1D(nn.Module):
    """Conditional 1D adaptation of the original DDPM U-Net configuration."""

    def __init__(
        self,
        signal_length: int = 2048,
        num_classes: int = 5,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 128,
        channel_mults: Iterable[int] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_levels: Iterable[int] = (1,),
        time_embed_dim: int | None = None,
        label_embed_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.signal_length = int(signal_length)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.time_embed_dim = int(time_embed_dim or base_channels * 4)
        self.label_embed_dim = int(label_embed_dim or self.time_embed_dim)
        self.num_res_blocks = int(num_res_blocks)
        self.attention_levels = set(int(level) for level in attention_levels)

        self.label_embed = nn.Embedding(num_classes, self.label_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.label_mlp = nn.Sequential(
            nn.Linear(self.label_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        channels = [base_channels * int(mult) for mult in channel_mults]
        self.input_conv = nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels: list[int] = [channels[0]]
        in_ch = channels[0]
        for level, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(DDPMResBlock1D(in_ch, out_ch, self.time_embed_dim, dropout))
                attns.append(DDPMAttentionBlock1D(out_ch) if level in self.attention_levels else nn.Identity())
                in_ch = out_ch
                skip_channels.append(in_ch)
            self.down_blocks.append(nn.ModuleDict({"res": blocks, "attn": attns}))
            if level != len(channels) - 1:
                self.downsamples.append(DDPMDownsample1D(in_ch))
                skip_channels.append(in_ch)

        self.mid_block1 = DDPMResBlock1D(in_ch, in_ch, self.time_embed_dim, dropout)
        self.mid_attn = DDPMAttentionBlock1D(in_ch)
        self.mid_block2 = DDPMResBlock1D(in_ch, in_ch, self.time_embed_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels))):
            out_ch = channels[level]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(self.num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                blocks.append(DDPMResBlock1D(in_ch + skip_ch, out_ch, self.time_embed_dim, dropout))
                attns.append(DDPMAttentionBlock1D(out_ch) if level in self.attention_levels else nn.Identity())
                in_ch = out_ch
            self.up_blocks.append(nn.ModuleDict({"res": blocks, "attn": attns}))
            if level != 0:
                self.upsamples.append(DDPMUpsample1D(in_ch))

        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=in_ch)
        self.out_conv = zero_module(nn.Conv1d(in_ch, out_channels, kernel_size=3, padding=1))

    def condition(self, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        time_emb = timestep_embedding(timesteps, self.base_channels)
        return self.time_mlp(time_emb) + self.label_mlp(self.label_embed(labels))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        cond = self.condition(timesteps, labels)
        h = self.input_conv(x)
        skips = [h]

        for level, block in enumerate(self.down_blocks):
            for res, attn in zip(block["res"], block["attn"]):
                h = attn(res(h, cond))
                skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)
                skips.append(h)

        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        for level, block in enumerate(self.up_blocks):
            for res, attn in zip(block["res"], block["attn"]):
                skip = skips.pop()
                if h.shape[-1] != skip.shape[-1]:
                    h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
                h = attn(res(torch.cat([h, skip], dim=1), cond))
            if level < len(self.upsamples):
                h = self.upsamples[level](h)

        if h.shape[-1] != self.signal_length:
            h = F.interpolate(h, size=self.signal_length, mode="linear", align_corners=False)
        return self.out_conv(F.silu(self.out_norm(h)))


class DDPMScheduler1D:
    """Baseline-local linear DDPM scheduler, isolated from the MHTA implementation."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        clip_range: float = 5.0,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_schedule = str(beta_schedule).lower()
        self.clip_sample = bool(clip_sample)
        self.clip_range = float(clip_range)
        if self.beta_schedule != "linear":
            raise ValueError(f"Baseline DDPMScheduler1D only supports beta_schedule='linear', got {beta_schedule!r}.")
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device: torch.device | str) -> "DDPMScheduler1D":
        device = torch.device(device)
        for name in ("betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev", "posterior_variance"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, timesteps.long())
        return out.view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = torch.sqrt(self._extract(self.alphas_cumprod, timesteps, x_start.shape))
        sqrt_one_minus_alpha = torch.sqrt(1.0 - self._extract(self.alphas_cumprod, timesteps, x_start.shape))
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha = self._extract(self.alphas_cumprod, timesteps, x_t.shape)
        x0 = (x_t - torch.sqrt(1.0 - alpha) * eps) / torch.sqrt(alpha)
        return torch.clamp(x0, -self.clip_range, self.clip_range) if self.clip_sample else x0

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        eps = model(x, timesteps, labels)
        beta_t = self._extract(self.betas, timesteps, x.shape)
        alpha_t = self._extract(self.alphas, timesteps, x.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, timesteps, x.shape)
        mean = (x - beta_t * eps / torch.sqrt(1.0 - alpha_cumprod_t)) / torch.sqrt(alpha_t)
        variance = self._extract(self.posterior_variance, timesteps, x.shape)
        noise = torch.randn_like(x)
        nonzero = (timesteps != 0).float().view(-1, *([1] * (x.dim() - 1)))
        sample = mean + nonzero * torch.sqrt(torch.clamp(variance, min=1e-20)) * noise
        if self.clip_sample:
            sample = torch.clamp(sample, -self.clip_range, self.clip_range)
        return sample

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, int, int], labels: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
        if device is None:
            device = labels.device
        device = torch.device(device)
        self.to(device)
        x = torch.randn(shape, device=device)
        for t_value in reversed(range(self.num_train_timesteps)):
            timesteps = torch.full((shape[0],), t_value, dtype=torch.long, device=device)
            x = self.p_sample(model, x, timesteps, labels)
        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int],
        labels: torch.Tensor,
        num_inference_steps: int = 100,
        eta: float = 0.0,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = labels.device
        device = torch.device(device)
        self.to(device)
        x = torch.randn(shape, device=device)
        steps = torch.linspace(self.num_train_timesteps - 1, 0, int(num_inference_steps), device=device).round().long()
        for i, timestep in enumerate(steps):
            t = torch.full((shape[0],), int(timestep.item()), dtype=torch.long, device=device)
            eps = model(x, t, labels)
            alpha_t = self._extract(self.alphas_cumprod, t, x.shape)
            x0 = self.predict_x0_from_eps(x, t, eps)
            if i == len(steps) - 1:
                x = torch.clamp(x0, -self.clip_range, self.clip_range) if self.clip_sample else x0
                continue
            t_next = torch.full((shape[0],), int(steps[i + 1].item()), dtype=torch.long, device=device)
            alpha_next = self._extract(self.alphas_cumprod, t_next, x.shape)
            sigma = (
                float(eta)
                * torch.sqrt((1.0 - alpha_next) / (1.0 - alpha_t))
                * torch.sqrt(torch.clamp(1.0 - alpha_t / alpha_next, min=0.0))
            )
            direction = torch.sqrt(torch.clamp(1.0 - alpha_next - sigma.square(), min=0.0)) * eps
            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(alpha_next) * x0 + direction + noise
            if self.clip_sample:
                x = torch.clamp(x, -self.clip_range, self.clip_range)
        return x
