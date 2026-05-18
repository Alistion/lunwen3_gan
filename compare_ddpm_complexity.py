from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from models.baseline_ddpm_1d import ConditionalUNet1D, DDPMAttentionBlock1D
from models.dcgan_1d import ConditionalDiscriminator1D, ConditionalGenerator1D
from models.mhta_ddpm_1d import (
    ConditionalMHTAUNet1D,
    MultiDconvHeadTransposedAttention1D,
    TemporalAttention1D,
)
from models.wgan_1d import ConditionalCritic1D


BASELINE_CONFIG = {
    "signal_length": 2048,
    "num_classes": 5,
    "base_channels": 128,
    "channel_mults": (1, 2, 2, 2),
    "num_res_blocks": 2,
    "attention_levels": (1,),
    "time_embed_dim": 512,
    "label_embed_dim": 512,
    "dropout": 0.1,
}

MHTA_CONFIG = {
    "signal_length": 2048,
    "num_classes": 5,
    "base_channels": 48,
    "channel_mults": (1, 2, 4, 8),
    "num_heads": 4,
    "time_embed_dim": 192,
    "label_embed_dim": 192,
    "dropout": 0.05,
    "attention_levels": (1, 2, 3),
    "strict_paper_mode": False,
}

GAN_CONFIG = {
    "latent_dim": 128,
    "num_classes": 5,
    "signal_length": 2048,
    "base_channels": 64,
}


@dataclass
class Complexity:
    name: str
    params: int
    trainable_params: int
    conv_linear_macs: int
    attention_macs: int

    @property
    def total_macs(self) -> int:
        return self.conv_linear_macs + self.attention_macs

    @property
    def total_flops(self) -> int:
        # Common convention: one multiply-add = 2 FLOPs.
        return 2 * self.total_macs


class MacCounter:
    def __init__(self) -> None:
        self.conv_linear_macs = 0
        self.attention_macs = 0
        self.handles: list[Any] = []

    def attach(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                self.handles.append(module.register_forward_hook(self._conv1d_hook))
            elif isinstance(module, nn.ConvTranspose1d):
                self.handles.append(module.register_forward_hook(self._conv_transpose1d_hook))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_hook))
            elif isinstance(module, DDPMAttentionBlock1D):
                self.handles.append(module.register_forward_hook(self._baseline_attention_hook))
            elif isinstance(module, MultiDconvHeadTransposedAttention1D):
                self.handles.append(module.register_forward_hook(self._mhta_attention_hook))
            elif isinstance(module, TemporalAttention1D):
                self.handles.append(module.register_forward_hook(self._temporal_attention_hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _conv1d_hook(self, module: nn.Conv1d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        batch, out_channels, out_length = output.shape
        kernel_ops = (module.in_channels // module.groups) * module.kernel_size[0]
        self.conv_linear_macs += int(batch * out_channels * out_length * kernel_ops)

    def _linear_hook(self, module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output_elements = output.numel()
        self.conv_linear_macs += int(output_elements * module.in_features)

    def _conv_transpose1d_hook(self, module: nn.ConvTranspose1d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        batch, in_channels, in_length = inputs[0].shape
        kernel_ops = (module.out_channels // module.groups) * module.kernel_size[0]
        self.conv_linear_macs += int(batch * in_channels * in_length * kernel_ops)

    def _baseline_attention_hook(self, module: DDPMAttentionBlock1D, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        x = inputs[0]
        batch, channels, length = x.shape
        # q @ k^T and attn @ v, each costs B * L * L * C MACs.
        self.attention_macs += int(2 * batch * length * length * channels)

    def _mhta_attention_hook(self, module: MultiDconvHeadTransposedAttention1D, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        x = inputs[0]
        batch, channels, length = x.shape
        head_dim = channels // module.num_heads
        # q @ k^T and attn @ v across channel subspaces.
        # Each costs B * H * D * D * L MACs.
        self.attention_macs += int(2 * batch * module.num_heads * head_dim * head_dim * length)

    def _temporal_attention_hook(self, module: TemporalAttention1D, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        x = inputs[0]
        batch, channels, length = x.shape
        # q @ k^T and attn @ v across time positions.
        self.attention_macs += int(2 * batch * length * length * channels)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(params), int(trainable)


def profile_model(name: str, model: nn.Module, signal_length: int, device: torch.device) -> Complexity:
    model = model.to(device).eval()
    params, trainable = count_parameters(model)
    counter = MacCounter()
    counter.attach(model)
    try:
        with torch.no_grad():
            x = torch.randn(1, 1, signal_length, device=device)
            t = torch.tensor([500], dtype=torch.long, device=device)
            labels = torch.tensor([1], dtype=torch.long, device=device)
            _ = model(x, t, labels)
    finally:
        counter.remove()
    return Complexity(
        name=name,
        params=params,
        trainable_params=trainable,
        conv_linear_macs=counter.conv_linear_macs,
        attention_macs=counter.attention_macs,
    )


def profile_generator(name: str, model: nn.Module, latent_dim: int, device: torch.device) -> Complexity:
    model = model.to(device).eval()
    params, trainable = count_parameters(model)
    counter = MacCounter()
    counter.attach(model)
    try:
        with torch.no_grad():
            z = torch.randn(1, latent_dim, device=device)
            labels = torch.tensor([1], dtype=torch.long, device=device)
            _ = model(z, labels)
    finally:
        counter.remove()
    return Complexity(
        name=name,
        params=params,
        trainable_params=trainable,
        conv_linear_macs=counter.conv_linear_macs,
        attention_macs=counter.attention_macs,
    )


def human(n: int) -> str:
    for unit in ("", "K", "M", "G", "T"):
        if abs(n) < 1000:
            return f"{n:.3f}{unit}" if unit else str(int(n))
        n /= 1000.0
    return f"{n:.3f}P"


def print_table(rows: list[Complexity], sampling_steps: int) -> None:
    headers = (
        "Model",
        "Params",
        "Conv+Linear MACs/fwd",
        "Attention MACs/fwd",
        "Total MACs/fwd",
        "FLOPs/fwd",
        f"Approx total FLOPs @ {sampling_steps} steps",
    )
    values = []
    for row in rows:
        values.append((
            row.name,
            human(row.params),
            human(row.conv_linear_macs),
            human(row.attention_macs),
            human(row.total_macs),
            human(row.total_flops),
            human(row.total_flops * sampling_steps),
        ))
    widths = [max(len(str(item[i])) for item in [headers, *values]) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for value in values:
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(value)))


def print_gan_training_params(
    dcgan_generator: nn.Module,
    dcgan_discriminator: nn.Module,
    wgan_generator: nn.Module,
    wgan_critic: nn.Module,
) -> None:
    rows = []
    for name, generator, judge in (
        ("DCGAN", dcgan_generator, dcgan_discriminator),
        ("WGAN", wgan_generator, wgan_critic),
    ):
        g_params, _ = count_parameters(generator)
        judge_params, _ = count_parameters(judge)
        rows.append((name, human(g_params), human(judge_params), human(g_params + judge_params)))
    headers = ("GAN", "Generator Params", "Discriminator/Critic Params", "Train-time Total Params")
    widths = [max(len(str(item[i])) for item in [headers, *rows]) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampling_steps = 1000
    baseline = ConditionalUNet1D(**BASELINE_CONFIG)
    mhta = ConditionalMHTAUNet1D(**MHTA_CONFIG)
    dcgan_generator = ConditionalGenerator1D(**GAN_CONFIG)
    dcgan_discriminator = ConditionalDiscriminator1D(
        num_classes=GAN_CONFIG["num_classes"],
        signal_length=GAN_CONFIG["signal_length"],
        base_channels=GAN_CONFIG["base_channels"],
    )
    wgan_generator = ConditionalGenerator1D(**GAN_CONFIG)
    wgan_critic = ConditionalCritic1D(
        num_classes=GAN_CONFIG["num_classes"],
        signal_length=GAN_CONFIG["signal_length"],
        base_channels=GAN_CONFIG["base_channels"],
    )
    rows = [
        profile_model("Vanilla DDPM-1D", baseline, BASELINE_CONFIG["signal_length"], device),
        profile_model("MHTA-DDPM", mhta, MHTA_CONFIG["signal_length"], device),
        profile_generator("DCGAN Generator", dcgan_generator, GAN_CONFIG["latent_dim"], device),
        profile_generator("WGAN Generator", wgan_generator, GAN_CONFIG["latent_dim"], device),
    ]
    print(f"Device used for dummy forward: {device}")
    print_table(rows, sampling_steps=sampling_steps)
    print()
    print("GAN train-time parameter footprint:")
    print_gan_training_params(dcgan_generator, dcgan_discriminator, wgan_generator, wgan_critic)
    print()
    print("Notes:")
    print("- MACs include Conv1d and Linear layers plus explicit attention matrix multiplications.")
    print("- FLOPs use the convention 1 MAC = 2 FLOPs.")
    print("- DDPM sampling FLOPs assume 1000 denoising steps; GAN rows generate a sample in one forward pass.")
    print("- Normalization, activations, softmax, interpolation, and elementwise ops are omitted from the estimate.")


if __name__ == "__main__":
    main()
