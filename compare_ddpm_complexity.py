from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
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

BENCHMARK_CONFIG = {
    "warmup": 20,
    "repeats": 100,
}

OUTPUT_CSV = Path("runs/complexity/comparison_complexity_table.csv")


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


@dataclass
class ComparisonRow:
    name: str
    inference_module: str
    inference: Complexity
    sampling_steps: int
    auxiliary_module: str = "-"
    auxiliary: Complexity | None = None
    infer_latency_ms: float | None = None

    @property
    def inference_params(self) -> int:
        return self.inference.params

    @property
    def auxiliary_params(self) -> int:
        return 0 if self.auxiliary is None else self.auxiliary.params

    @property
    def train_time_params(self) -> int:
        return self.inference_params + self.auxiliary_params

    @property
    def auxiliary_flops(self) -> int:
        return 0 if self.auxiliary is None else self.auxiliary.total_flops

    @property
    def sample_flops(self) -> int:
        return self.inference.total_flops * int(self.sampling_steps)

    @property
    def sample_latency_ms(self) -> float | None:
        if self.infer_latency_ms is None:
            return None
        return self.infer_latency_ms * int(self.sampling_steps)


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


def profile_judge(name: str, model: nn.Module, signal_length: int, device: torch.device) -> Complexity:
    model = model.to(device).eval()
    params, trainable = count_parameters(model)
    counter = MacCounter()
    counter.attach(model)
    try:
        with torch.no_grad():
            x = torch.randn(1, 1, signal_length, device=device)
            labels = torch.tensor([1], dtype=torch.long, device=device)
            _ = model(x, labels)
    finally:
        counter.remove()
    return Complexity(
        name=name,
        params=params,
        trainable_params=trainable,
        conv_linear_macs=counter.conv_linear_macs,
        attention_macs=counter.attention_macs,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_call(fn, device: torch.device, warmup: int, repeats: int) -> float:
    with torch.no_grad():
        for _ in range(int(warmup)):
            fn()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(int(repeats)):
            fn()
        synchronize(device)
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / max(int(repeats), 1)


def benchmark_denoiser_forward(model: nn.Module, signal_length: int, device: torch.device, config: dict) -> float:
    model = model.to(device).eval()
    x = torch.randn(1, 1, int(signal_length), device=device)
    t = torch.tensor([500], dtype=torch.long, device=device)
    labels = torch.tensor([1], dtype=torch.long, device=device)
    return benchmark_call(lambda: model(x, t, labels), device, int(config["warmup"]), int(config["repeats"]))


def benchmark_generator_forward(model: nn.Module, latent_dim: int, device: torch.device, config: dict) -> float:
    model = model.to(device).eval()
    z = torch.randn(1, int(latent_dim), device=device)
    labels = torch.tensor([1], dtype=torch.long, device=device)
    return benchmark_call(lambda: model(z, labels), device, int(config["warmup"]), int(config["repeats"]))


def human(n: int) -> str:
    for unit in ("", "K", "M", "G", "T"):
        if abs(n) < 1000:
            return f"{n:.3f}{unit}" if unit else str(int(n))
        n /= 1000.0
    return f"{n:.3f}P"


def human_ms(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1000.0:
        return f"{value:.3f}ms"
    seconds = value / 1000.0
    if seconds < 60.0:
        return f"{seconds:.3f}s"
    minutes = seconds / 60.0
    return f"{minutes:.3f}min"


def print_table(rows: list[ComparisonRow]) -> None:
    headers = (
        "Model",
        "Inference Module",
        "Inference Params",
        "Infer FLOPs/fwd",
        "Infer time/fwd",
        "Sampling steps",
        "Approx FLOPs/sample",
        "Approx time/sample",
    )
    values = []
    for row in rows:
        values.append((
            row.name,
            row.inference_module,
            human(row.inference_params),
            human(row.inference.total_flops),
            human_ms(row.infer_latency_ms),
            str(int(row.sampling_steps)),
            human(row.sample_flops),
            human_ms(row.sample_latency_ms),
        ))
    widths = [max(len(str(item[i])) for item in [headers, *values]) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for value in values:
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(value)))


def write_table_csv(path: Path, rows: list[ComparisonRow], device: torch.device) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "inference_module",
        "inference_params",
        "inference_params_human",
        "infer_flops_per_forward",
        "infer_flops_per_forward_human",
        "infer_time_per_forward_ms",
        "infer_time_per_forward_human",
        "sampling_steps",
        "approx_flops_per_sample",
        "approx_flops_per_sample_human",
        "approx_time_per_sample_ms",
        "approx_time_per_sample_human",
        "benchmark_device",
        "benchmark_warmup",
        "benchmark_repeats",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.name,
                    "inference_module": row.inference_module,
                    "inference_params": row.inference_params,
                    "inference_params_human": human(row.inference_params),
                    "infer_flops_per_forward": row.inference.total_flops,
                    "infer_flops_per_forward_human": human(row.inference.total_flops),
                    "infer_time_per_forward_ms": row.infer_latency_ms if row.infer_latency_ms is not None else "",
                    "infer_time_per_forward_human": human_ms(row.infer_latency_ms),
                    "sampling_steps": int(row.sampling_steps),
                    "approx_flops_per_sample": row.sample_flops,
                    "approx_flops_per_sample_human": human(row.sample_flops),
                    "approx_time_per_sample_ms": row.sample_latency_ms if row.sample_latency_ms is not None else "",
                    "approx_time_per_sample_human": human_ms(row.sample_latency_ms),
                    "benchmark_device": str(device),
                    "benchmark_warmup": int(BENCHMARK_CONFIG["warmup"]),
                    "benchmark_repeats": int(BENCHMARK_CONFIG["repeats"]),
                }
            )


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


def print_gan_component_table(rows: list[Complexity]) -> None:
    headers = ("Component", "Params", "Conv+Linear MACs/fwd", "FLOPs/fwd")
    values = [
        (
            row.name,
            human(row.params),
            human(row.conv_linear_macs),
            human(row.total_flops),
        )
        for row in rows
    ]
    widths = [max(len(str(item[i])) for item in [headers, *values]) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for value in values:
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(value)))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampling_steps = 1000
    baseline = ConditionalUNet1D(**BASELINE_CONFIG)
    mhta = ConditionalMHTAUNet1D(**MHTA_CONFIG)
    dcgan_generator = ConditionalGenerator1D(**GAN_CONFIG)
    dcgan_generator_complexity = profile_generator(
        "DCGAN/WGAN Generator",
        dcgan_generator,
        GAN_CONFIG["latent_dim"],
        device,
    )
    baseline_latency_ms = benchmark_denoiser_forward(
        baseline,
        BASELINE_CONFIG["signal_length"],
        device,
        BENCHMARK_CONFIG,
    )
    mhta_latency_ms = benchmark_denoiser_forward(
        mhta,
        MHTA_CONFIG["signal_length"],
        device,
        BENCHMARK_CONFIG,
    )
    dcgan_latency_ms = benchmark_generator_forward(
        dcgan_generator,
        GAN_CONFIG["latent_dim"],
        device,
        BENCHMARK_CONFIG,
    )
    rows = [
        ComparisonRow(
            name="Vanilla DDPM-1D",
            inference_module="Denoiser UNet",
            inference=profile_model("Vanilla DDPM-1D", baseline, BASELINE_CONFIG["signal_length"], device),
            sampling_steps=sampling_steps,
            infer_latency_ms=baseline_latency_ms,
        ),
        ComparisonRow(
            name="MHTA-DDPM",
            inference_module="MHTA Denoiser",
            inference=profile_model("MHTA-DDPM", mhta, MHTA_CONFIG["signal_length"], device),
            sampling_steps=sampling_steps,
            infer_latency_ms=mhta_latency_ms,
        ),
        ComparisonRow(
            name="DCGAN/WGAN Generator",
            inference_module="Conditional Generator",
            inference=dcgan_generator_complexity,
            sampling_steps=1,
            infer_latency_ms=dcgan_latency_ms,
        ),
    ]
    print(f"Device used for dummy forward: {device}")
    print_table(rows)
    write_table_csv(OUTPUT_CSV, rows, device)
    print()
    print(f"Saved CSV table to: {OUTPUT_CSV}")
    print()
    print("Notes:")
    print("- Infer FLOPs/fwd counts the module used at generation time: DDPM denoiser or GAN generator.")
    print(
        f"- Infer time/fwd is measured on the current {device} device with "
        f"{BENCHMARK_CONFIG['warmup']} warmup and {BENCHMARK_CONFIG['repeats']} timed forwards."
    )
    print("- FLOPs use the convention 1 MAC = 2 FLOPs.")
    print("- DDPM sample FLOPs use the configured denoising steps; GAN generation uses one generator forward pass.")
    print("- DCGAN and WGAN use the same generator architecture in this codebase; they differ in adversarial training objective and discriminator/critic design.")
    print("- Normalization, activations, softmax, interpolation, and elementwise ops are omitted from the estimate.")


if __name__ == "__main__":
    main()
