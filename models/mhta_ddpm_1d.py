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


class RMSNorm1D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x.square(), dim=1, keepdim=True) + self.eps) * self.weight


class MultiDconvHeadTransposedAttention1D(nn.Module):
    """1D MHTA adapted from multi-dconv head transposed attention.

    Attention is computed across channel subspaces rather than across all time
    positions, while point-wise and depth-wise convolutions preserve local
    vibration details before the channel-wise attention step.
    """

    def __init__(self, channels: int, num_heads: int = 4, bias: bool = False):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv1d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.project_out = nn.Conv1d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, length = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = channels // self.num_heads
        q = q.view(bsz, self.num_heads, head_dim, length)
        k = k.view(bsz, self.num_heads, head_dim, length)
        v = v.view(bsz, self.num_heads, head_dim, length)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.reshape(bsz, channels, length)
        return self.project_out(out)


class MHTABlock1D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, ):
        super().__init__()
        self.norm1 = RMSNorm1D(channels)
        self.attn = MultiDconvHeadTransposedAttention1D(channels, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm1(x))


class ConditionalResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = RMSNorm1D(out_channels)
        self.cond = nn.Linear(cond_dim, out_channels * 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = RMSNorm1D(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.cond(cond).chunk(2, dim=1)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return self.skip(x) + h


class Downsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ConditionalMHTAUNet1D(nn.Module):
    """Conditional 1D U-Net for MHTA-DDPM vibration signal generation."""

    def __init__(
        self,
        signal_length: int = 2048,
        num_classes: int = 5,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 48,
        channel_mults: Iterable[int] = (1, 2, 4, 4),
        num_heads: int = 4,
        time_embed_dim: int = 192,
        label_embed_dim: int = 192,
        dropout: float = 0.05,
        attention_levels: Iterable[int] = (1, 2, 3),
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
        self.attention_levels = set(int(level) for level in attention_levels)
        self.input_conv = nn.Conv1d(in_channels, channels[0], kernel_size=7, padding=3)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = channels[0]
        for level, out_ch in enumerate(channels):
            self.down_blocks.append(
                nn.ModuleDict(
                    {
                        "res": ConditionalResBlock1D(in_ch, out_ch, time_embed_dim, dropout),
                        "attn": MHTABlock1D(out_ch, num_heads=num_heads) if level in self.attention_levels else nn.Identity(),
                    }
                )
            )
            in_ch = out_ch
            if level != len(channels) - 1:
                self.downsamples.append(Downsample1D(out_ch))

        self.mid_res = ConditionalResBlock1D(channels[-1], channels[-1], time_embed_dim, dropout)
        self.mid_attn = MHTABlock1D(channels[-1], num_heads=num_heads)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_channels = list(reversed(channels))
        in_ch = rev_channels[0]
        for level, out_ch in enumerate(rev_channels):
            original_level = len(channels) - 1 - level
            self.up_blocks.append(
                nn.ModuleDict(
                    {
                        "res": ConditionalResBlock1D(in_ch + out_ch, out_ch, time_embed_dim, dropout),
                        "attn": MHTABlock1D(out_ch, num_heads=num_heads)
                        if original_level in self.attention_levels
                        else nn.Identity(),
                    }
                )
            )
            in_ch = out_ch
            if level != len(rev_channels) - 1:
                self.upsamples.append(Upsample1D(out_ch))

        self.out_norm = RMSNorm1D(channels[0])
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
            h = block["res"](h, cond)
            h = block["attn"](h)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        h = self.mid_res(h, cond)
        h = self.mid_attn(h)

        for level, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block["res"](h, cond)
            h = block["attn"](h)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)

        if h.shape[-1] != self.signal_length:
            h = F.interpolate(h, size=self.signal_length, mode="linear", align_corners=False)
        return self.out_conv(F.silu(self.out_norm(h)))


class DDPMScheduler1D:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        clip_sample: bool = True,
        clip_range: float = 5.0,
        cosine_s: float = 0.008,
        max_beta: float = 0.999,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.clip_sample = bool(clip_sample)
        self.clip_range = float(clip_range)
        self.cosine_s = float(cosine_s)
        self.max_beta = float(max_beta)
        self.betas = self._cosine_beta_schedule(self.num_train_timesteps, self.cosine_s, self.max_beta)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(num_train_timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
        steps = torch.arange(num_train_timesteps + 1, dtype=torch.float32)
        x = steps / float(num_train_timesteps)
        alphas_cumprod = torch.cos(((x + float(s)) / (1.0 + float(s))) * math.pi * 0.5).square()
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, min=1e-8, max=float(max_beta))

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
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int],
        labels: torch.Tensor,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
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
