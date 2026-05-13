from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import NPZSignalDataset
from losses.amplitude_losses import amplitude_consistency_loss, build_amplitude_templates
from losses.gan_losses import gradient_penalty
from losses.order_losses import order_consistency_components, statistical_consistency_loss
from losses.spectral_losses import (
    build_real_spectrum_template_pair,
    spectrum_shape_loss,
    template_spectrum_debug_stats,
    template_spectrum_shape_amp_loss,
)
from losses.temporal_losses import (
    acf_consistency_loss,
    build_acf_templates,
    build_envelope_templates,
    envelope_consistency_loss,
    envelope_rms_sequence,
    temporal_smoothness_loss,
)
from models.omc_tf_gan import OrderGuidedGenerator, TimeFrequencyDiscriminator, save_generator_checkpoint
from utils.seed import set_seed
from utils.signal_utils import CLASS_NAMES, order_template_for_labels

CONFIG = {
    "data_dir": Path("processed/omc_tf_gan_dataset"),
    "out_dir": Path("runs/omc_tf_gan_v2"),
    "epochs": 500,
    "batch_size": 32,
    "latent_dim": 128,
    "lr_g": 1e-4,
    "lr_d": 5e-5,
    "n_critic": 3,
    "lambda_gp": 15.0,
    "lambda_cls": 1.0,
    "lambda_ord": 5,     # 从 5.0 降到 0.5 或更低
    "lambda_stat": 1,    # 从 1.0 降到 0.1
    "lambda_spec": 2,    # 从 2.0 降到 0.5
    "lambda_spec_amp": 0.1,
    "lambda_amp": 0.5,     # 从 0.2 稍微提点到 0.5
    "lambda_env": 0.2,     # 保持 0.2
    "lambda_acf": 0.1,     # 保持 0.1
    "lambda_tv": 0.01,
    "lambda_sup": 0.5,
    "lambda_unbalance_sup": 0.5,
    "spec_use_template": True,
    "spec_loss_type": "l1",
    "spec_max_freq": 100,
    "env_segments": 16,
    "acf_max_lag": 512,
    "fs": 2048,
    "rpm": 740,
    "seed": 42,
    "device": "cuda_if_available",
    "instance_noise_std": 0.01,
    "instance_noise_decay": 0.995,
    "min_instance_noise_std": 0.001,
    "use_real_augment": True,
    "real_aug_scale_min": 0.95,
    "real_aug_scale_max": 1.05,
    "real_aug_shift": 20,
    "real_aug_noise_std": 0.005,
    "save_every": 20,
    "preview_every": 20,
    "use_tanh_output": False,
    "early_stop_bad_gan": False,
}

ORDER_TEMPLATES = {
    "normal": [0.05, 0.60, 0.03, 0.08, 0.03, 0.10],
    "unbalance": [0.03, 1.00, 0.03, 0.08, 0.03, 0.08],
    "misalignment": [0.03, 0.45, 0.03, 1.00, 0.18, 0.10],
    "crack": [0.03, 0.75, 0.05, 0.55, 0.30, 0.25],
    "looseness": [0.35, 0.55, 0.30, 0.45, 0.35, 0.75],
}
TARGET_MASKS = {
    "normal": [0, 1, 0, 0, 0, 0],
    "unbalance": [0, 1, 0, 0, 0, 0],
    "misalignment": [0, 1, 0, 1, 0, 0],
    "crack": [0, 1, 0, 1, 1, 0],
    "looseness": [1, 1, 1, 1, 1, 1],
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("train_gan")


def get_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def augment_real_signal(x: torch.Tensor, config: dict) -> torch.Tensor:
    if not config["use_real_augment"]:
        return x
    bsz = x.size(0)
    scale = torch.empty(bsz, 1, 1, device=x.device).uniform_(config["real_aug_scale_min"], config["real_aug_scale_max"])
    x_aug = x * scale
    max_shift = int(config["real_aug_shift"])
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (bsz,), device=x.device)
        x_aug = torch.stack([torch.roll(item, shifts=int(shift.item()), dims=-1) for item, shift in zip(x_aug, shifts)], dim=0)
    return x_aug + float(config["real_aug_noise_std"]) * torch.randn_like(x_aug)


def sample_fault_labels(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(1, len(CLASS_NAMES), (batch_size,), dtype=torch.long, device=device)


def sample_real_by_labels(class_real: dict[int, torch.Tensor], labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    samples = []
    for label in labels.detach().cpu().tolist():
        pool = class_real[int(label)]
        idx = torch.randint(0, pool.size(0), (1,), device=device)
        samples.append(pool[idx].squeeze(0))
    return torch.stack(samples, dim=0)


def labels_to_order(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    templates = np.asarray([ORDER_TEMPLATES[name] for name in CLASS_NAMES], dtype=np.float32)
    return torch.from_numpy(templates[labels.detach().cpu().numpy()]).float().to(device)


def target_masks_tensor(device: torch.device) -> torch.Tensor:
    masks = np.asarray([TARGET_MASKS[name] for name in CLASS_NAMES], dtype=np.float32)
    return torch.from_numpy(masks).float().to(device)


def compute_saturation_ratio(x: torch.Tensor | np.ndarray, use_tanh_output: bool = False) -> float:
    if torch.is_tensor(x):
        arr = x.detach().cpu().float()
        threshold = 0.98 if use_tanh_output else 3.0
        return float((arr.abs() > threshold).float().mean().item())
    arr = np.asarray(x, dtype=np.float32)
    threshold = 0.98 if use_tanh_output else 3.0
    return float(np.mean(np.abs(arr) > threshold))


def save_discriminator_checkpoint(path: Path, discriminator: TimeFrequencyDiscriminator, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"discriminator_state_dict": discriminator.state_dict(), **meta}, path)


def segmented_rms_np(x: np.ndarray, segments: int) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, segments), dtype=np.float32)
    seg_len = x.shape[1] // int(segments)
    x_seg = x[:, : seg_len * int(segments)].reshape(len(x), int(segments), seg_len)
    return np.sqrt(np.mean(np.square(x_seg), axis=-1) + 1e-8).astype(np.float32)


def save_envelope_plot(class_name: str, real_c: np.ndarray, fake: np.ndarray | None, out_path: Path, segments: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(1, int(segments) + 1)
    if len(real_c):
        real_env = segmented_rms_np(real_c, segments)
        for row in real_env:
            ax.plot(xs, row, color="tab:blue", linewidth=0.8, alpha=0.35)
        ax.plot(xs, real_env.mean(axis=0), color="tab:blue", linewidth=2.0, label="real mean")
    if fake is not None and len(fake):
        fake_env = segmented_rms_np(fake, segments)
        for row in fake_env:
            ax.plot(xs, row, color="tab:orange", linewidth=0.8, alpha=0.35)
        ax.plot(xs, fake_env.mean(axis=0), color="tab:orange", linewidth=2.0, label="fake mean")
    else:
        ax.text(0.5, 0.5, "normal is not generated", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{class_name} envelope RMS")
    ax.set_xlabel("Segment")
    ax.set_ylabel("RMS")
    ax.set_xticks(xs)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


@torch.no_grad()
def save_preview(
    generator: OrderGuidedGenerator,
    train_npz: Path,
    out_dir: Path,
    epoch: int,
    config: dict,
    device: torch.device,
) -> float:
    generator.eval()
    data = np.load(train_npz, allow_pickle=True)
    x_real = np.asarray(data["X"], dtype=np.float32)
    y_real = np.asarray(data["y"], dtype=np.int64)
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    fs = int(config["fs"])
    fr = float(config["rpm"]) / 60.0
    all_fake = []

    for label, class_name in enumerate(CLASS_NAMES):
        real_c = x_real[y_real == label][:3]
        fake = None
        if label != 0:
            y = torch.full((3,), label, dtype=torch.long, device=device)
            z = torch.randn(3, int(config["latent_dim"]), device=device)
            rpm = torch.full((3,), float(config["rpm"]), device=device)
            order = torch.from_numpy(order_template_for_labels(np.full(3, label))).to(device)
            fake = generator(z, y, rpm, order).squeeze(1).cpu().numpy().astype(np.float32)
            all_fake.append(fake)

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        sample_len = real_c.shape[1] if len(real_c) else (fake.shape[1] if fake is not None else int(config["fs"]))
        t = np.arange(sample_len) / float(fs)
        for arr in real_c:
            axes[0, 0].plot(t, arr, linewidth=0.8, alpha=0.85)
        if fake is not None:
            for arr in fake:
                axes[0, 1].plot(t, arr, linewidth=0.8, alpha=0.85)
        else:
            axes[0, 1].text(0.5, 0.5, "normal is not generated", ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 0].set_title(f"{class_name} real time")
        axes[0, 1].set_title(f"{class_name} fake time")

        for ax, arrs, title in [(axes[1, 0], real_c, "real spectrum"), (axes[1, 1], fake, "fake spectrum")]:
            if arrs is not None and len(arrs):
                spec = np.abs(np.fft.rfft(arrs, axis=1))
                freqs = np.fft.rfftfreq(arrs.shape[1], d=1.0 / fs)
                mask = freqs <= 200
                for spec_i in spec:
                    ax.plot(freqs[mask], spec_i[mask], linewidth=0.8, alpha=0.85)
            elif title == "fake spectrum":
                ax.text(0.5, 0.5, "normal is not generated", ha="center", va="center", transform=ax.transAxes)
            for mul, name in [(1.0, "1X"), (2.0, "2X"), (3.0, "3X")]:
                ax.axvline(fr * mul, color="tab:red", linestyle="--", linewidth=0.8)
                ax.text(fr * mul, ax.get_ylim()[1] * 0.9, name, rotation=90, va="top", ha="right", fontsize=8)
            ax.set_xlim(0, 200)
            ax.set_title(title)
        for ax in axes.flat:
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(epoch_dir / f"{class_name}_preview.png", dpi=220)
        plt.close(fig)
        save_envelope_plot(class_name, real_c, fake, epoch_dir / f"{class_name}_env.png", int(config["env_segments"]))

    generator.train()
    if not all_fake:
        return 0.0
    return compute_saturation_ratio(np.concatenate(all_fake, axis=0), config["use_tanh_output"])


def warn_if_needed(logger: logging.Logger, row: dict, severe_epochs: int, config: dict) -> int:
    severe = False
    if abs(row["D_loss"]) > 25:
        logger.warning("Critic may be too strong.")
        severe = True
    if abs(row["G_loss"]) > 80:
        logger.warning("Generator loss magnitude is too large.")
        severe = True
    if row["L_stat"] > 20:
        logger.warning("Statistical consistency loss is unstable.")
        severe = True
    if row["saturation_fake"] > 0.2:
        logger.warning("Generated signal may be saturated.")
        logger.warning("Generated signal may be saturated or exploded.")
        severe = True
    severe_epochs = severe_epochs + 1 if severe else 0
    if config["early_stop_bad_gan"] and severe_epochs >= 30:
        raise RuntimeError("Early stopped because severe GAN warnings persisted for 30 epochs.")
    return severe_epochs


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    set_seed(int(config["seed"]))
    device = get_device(config["device"])
    out_dir = config["out_dir"]
    ckpt_dir = out_dir / "checkpoints"
    preview_dir = out_dir / "generated_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using device: %s", device)

    train_npz = config["data_dir"] / "train.npz"
    dataset = NPZSignalDataset(train_npz, include_conditions=True)
    train_npz_data = np.load(train_npz, allow_pickle=True)
    train_x = np.asarray(train_npz_data["X"], dtype=np.float32)
    train_y = np.asarray(train_npz_data["y"], dtype=np.int64)
    class_real = {
        label: torch.from_numpy(train_x[train_y == label]).float().unsqueeze(1).to(device)
        for label in range(len(CLASS_NAMES))
    }
    real_spec_shape_templates, real_spec_amp_templates = build_real_spectrum_template_pair(
        train_x,
        train_y,
        fs=config["fs"],
        max_freq=config["spec_max_freq"],
        num_classes=len(CLASS_NAMES),
        device=device,
    )
    amplitude_templates = build_amplitude_templates(
        train_x,
        train_y,
        fs=config["fs"],
        max_freq=config["spec_max_freq"],
        num_classes=len(CLASS_NAMES),
        device=device,
    )
    envelope_templates = build_envelope_templates(
        train_x,
        train_y,
        segments=config["env_segments"],
        num_classes=len(CLASS_NAMES),
        device=device,
    )
    acf_templates = build_acf_templates(
        train_x,
        train_y,
        max_lag=config["acf_max_lag"],
        num_classes=len(CLASS_NAMES),
        device=device,
    )
    target_masks = target_masks_tensor(device)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=0, pin_memory=device.type == "cuda")
    generator = OrderGuidedGenerator(latent_dim=config["latent_dim"], base_channels=128, use_tanh_output=config["use_tanh_output"]).to(device)
    discriminator = TimeFrequencyDiscriminator(base_channels=32, use_spectral_norm=True).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=config["lr_g"], betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=(0.0, 0.9))

    fieldnames = [
        "epoch",
        "D_loss",
        "G_loss",
        "OrdAlign",
        "Sup",
        "UnbSup",
        "Broad",
        "Proto",
        "L_env",
        "L_acf",
        "L_tv",
        "L_amp",
        "L_ord",
        "L_stat",
        "L_spec",
        "SpecShape",
        "SpecAmp",
        "SpecCls",
        "L_cls",
        "GP",
        "noise_std",
        "saturation_fake",
    ]
    best_g = float("inf")
    best_metrics = {}
    global_step = 0
    severe_epochs = 0
    log_path = out_dir / "train_log.csv"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, int(config["epochs"]) + 1):
            noise_std = max(config["min_instance_noise_std"], config["instance_noise_std"] * (config["instance_noise_decay"] ** epoch))
            meters = defaultdict(float)
            steps = 0
            g_steps = 0
            spec_valid_total = 0
            last_fake = None

            for real, y, rpm, order in tqdm(loader, desc=f"GAN epoch {epoch}/{config['epochs']}", leave=False):
                real = augment_real_signal(real.to(device), config)
                y = y.to(device)
                rpm = rpm.to(device)
                order = labels_to_order(y, device)
                bsz = real.size(0)

                y_fake_d = sample_fault_labels(bsz, device)
                rpm_fake_d = torch.full((bsz,), float(config["rpm"]), device=device)
                order_fake_d = labels_to_order(y_fake_d, device)
                z = torch.randn(bsz, int(config["latent_dim"]), device=device)
                with torch.no_grad():
                    fake = generator(z, y_fake_d, rpm_fake_d, order_fake_d)
                x_real_d = real + float(noise_std) * torch.randn_like(real)
                x_fake_d = fake.detach() + float(noise_std) * torch.randn_like(fake)
                real_gp = augment_real_signal(sample_real_by_labels(class_real, y_fake_d, device), config)
                real_gp = real_gp + float(noise_std) * torch.randn_like(real_gp)
                real_score, real_cls = discriminator(x_real_d, y, rpm, order)
                fake_score, _ = discriminator(x_fake_d, y_fake_d, rpm_fake_d, order_fake_d)
                gp = gradient_penalty(discriminator, real_gp, x_fake_d, y_fake_d, rpm_fake_d, order_fake_d)
                cls_real = F.cross_entropy(real_cls, y)
                d_loss = fake_score.mean() - real_score.mean() + config["lambda_gp"] * gp + config["lambda_cls"] * cls_real
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

                meters["D_loss"] += float(d_loss.item())
                meters["GP"] += float(gp.item())
                global_step += 1
                steps += 1

                if global_step % int(config["n_critic"]) == 0:
                    y_fake = sample_fault_labels(bsz, device)
                    rpm_fake = torch.full((bsz,), float(config["rpm"]), device=device)
                    order_fake = labels_to_order(y_fake, device)
                    real_ref = sample_real_by_labels(class_real, y_fake, device)
                    real_ref = augment_real_signal(real_ref, config)
                    z = torch.randn(bsz, int(config["latent_dim"]), device=device)
                    fake = generator(z, y_fake, rpm_fake, order_fake)
                    fake_score, fake_cls = discriminator(fake, y_fake, rpm_fake, order_fake)
                    cls_fake = F.cross_entropy(fake_cls, y_fake)
                    l_ord_align, l_sup, l_unb_sup = order_consistency_components(
                        real_ref,
                        fake,
                        y_fake,
                        target_masks,
                        fs=config["fs"],
                        rpm=rpm_fake,
                    )
                    l_ord = l_ord_align + config["lambda_sup"] * l_sup + config["lambda_unbalance_sup"] * l_unb_sup
                    l_stat = statistical_consistency_loss(real_ref, fake, y_fake)
                    l_amp = amplitude_consistency_loss(
                        fake,
                        y_fake,
                        amplitude_templates,
                        fs=config["fs"],
                        max_freq=config["spec_max_freq"],
                    )
                    l_env = envelope_consistency_loss(
                        fake,
                        y_fake,
                        envelope_templates,
                        segments=config["env_segments"],
                    )
                    l_acf = acf_consistency_loss(
                        fake,
                        y_fake,
                        acf_templates,
                        max_lag=config["acf_max_lag"],
                    )
                    l_tv = temporal_smoothness_loss(fake)
                    if config["spec_use_template"]:
                        l_spec, l_spec_shape, l_spec_amp, spec_valid_classes = template_spectrum_shape_amp_loss(
                            fake,
                            y_fake,
                            real_spec_shape_templates,
                            real_spec_amp_templates,
                            fs=config["fs"],
                            max_freq=config["spec_max_freq"],
                            loss_type=config["spec_loss_type"],
                            lambda_spec_amp=config["lambda_spec_amp"],
                        )
                    else:
                        l_spec, spec_valid_classes = spectrum_shape_loss(
                            fake,
                            real_ref,
                            y_fake,
                            fs=config["fs"],
                            max_freq=config["spec_max_freq"],
                            loss_type=config["spec_loss_type"],
                        )
                        l_spec_shape = l_spec
                        l_spec_amp = fake.sum() * 0.0
                    g_adv = -fake_score.mean()
                    l_broad = fake.sum() * 0.0
                    l_proto = fake.sum() * 0.0
                    g_loss = (
                        g_adv
                        + config["lambda_cls"] * cls_fake
                        + config["lambda_ord"] * l_ord
                        + config["lambda_env"] * l_env
                        + config["lambda_acf"] * l_acf
                        + config["lambda_tv"] * l_tv
                        + config["lambda_amp"] * l_amp
                        + config["lambda_stat"] * l_stat
                        + config["lambda_spec"] * l_spec
                    )
                    if epoch == 1 and g_steps == 0:
                        stats = template_spectrum_debug_stats(
                            fake,
                            y_fake,
                            real_spec_shape_templates,
                            fs=config["fs"],
                            max_freq=config["spec_max_freq"],
                        )
                        logger.info(
                            "L_spec debug: raw=%.8e valid_classes=%d real_mean=%.8e real_std=%.8e fake_mean=%.8e fake_std=%.8e real_shape=%s fake_shape=%s",
                            float(l_spec.detach().cpu().item()),
                            spec_valid_classes,
                            stats["real_spec_mean"],
                            stats["real_spec_std"],
                            stats["fake_spec_mean"],
                            stats["fake_spec_std"],
                            stats["real_spec_shape"],
                            stats["fake_spec_shape"],
                        )
                    opt_g.zero_grad(set_to_none=True)
                    g_loss.backward()
                    opt_g.step()

                    meters["G_loss"] += float(g_loss.item())
                    meters["OrdAlign"] += float(l_ord_align.item())
                    meters["Sup"] += float(l_sup.item())
                    meters["UnbSup"] += float(l_unb_sup.item())
                    meters["Broad"] += float(l_broad.item())
                    meters["Proto"] += float(l_proto.item())
                    meters["L_env"] += float(l_env.item())
                    meters["L_acf"] += float(l_acf.item())
                    meters["L_tv"] += float(l_tv.item())
                    meters["L_amp"] += float(l_amp.item())
                    meters["L_ord"] += float(l_ord.item())
                    meters["L_stat"] += float(l_stat.item())
                    meters["L_spec"] += float(l_spec.item())
                    meters["SpecShape"] += float(l_spec_shape.item())
                    meters["SpecAmp"] += float(l_spec_amp.item())
                    meters["L_cls"] += float(cls_fake.item())
                    spec_valid_total += spec_valid_classes
                    g_steps += 1
                    last_fake = fake.detach()

            saturation_fake = compute_saturation_ratio(last_fake, config["use_tanh_output"]) if last_fake is not None else 0.0
            row = {"epoch": epoch, "noise_std": noise_std, "saturation_fake": saturation_fake}
            for key in ["D_loss", "GP"]:
                row[key] = meters[key] / max(1, steps)
            for key in ["G_loss", "OrdAlign", "Sup", "UnbSup", "Broad", "Proto", "L_env", "L_acf", "L_tv", "L_amp", "L_ord", "L_stat", "L_spec", "SpecShape", "SpecAmp", "L_cls"]:
                row[key] = meters[key] / max(1, g_steps)
            row["SpecCls"] = spec_valid_total / max(1, g_steps)
            writer.writerow(row)
            f.flush()
            logger.info(
                f"Epoch {epoch:03d} "
                f"D={row['D_loss']:.4f} G={row['G_loss']:.4f} "
                f"OrdAlign={row['OrdAlign']:.6f} "
                f"Sup={row['Sup']:.6f} "
                f"UnbSup={row['UnbSup']:.6f} "
                f"Broad={row['Broad']:.6f} "
                f"Proto={row['Proto']:.6f} "
                f"Env={row['L_env']:.6f} "
                f"ACF={row['L_acf']:.6f} "
                f"TV={row['L_tv']:.6f} "
                f"Amp={row['L_amp']:.6f} "
                f"Stat={row['L_stat']:.6f} "
                f"SpecShape={row['SpecShape']:.8e} "
                f"SpecAmp={row['SpecAmp']:.8e} "
                f"Spec={row['L_spec']:.8e} "
                f"SpecCls={row['SpecCls']:.2f} "
                f"sat={row['saturation_fake']:.4f}"
            )
            severe_epochs = warn_if_needed(logger, row, severe_epochs, config)

            save_generator_checkpoint(ckpt_dir / "latest_generator.pt", generator, latent_dim=config["latent_dim"], base_channels_g=128, class_names=CLASS_NAMES, use_tanh_output=config["use_tanh_output"])
            save_discriminator_checkpoint(ckpt_dir / "latest_discriminator.pt", discriminator)

            if epoch % int(config["preview_every"]) == 0 or epoch == int(config["epochs"]):
                preview_sat = save_preview(generator, train_npz, preview_dir, epoch, config, device)
                if preview_sat > 0.2:
                    logger.warning("Generated signal may be saturated.")

            if row["G_loss"] and row["G_loss"] < best_g:
                best_g = row["G_loss"]
                best_metrics = row.copy()
                save_generator_checkpoint(ckpt_dir / "best_generator.pt", generator, latent_dim=config["latent_dim"], base_channels_g=128, class_names=CLASS_NAMES, use_tanh_output=config["use_tanh_output"])
                save_discriminator_checkpoint(ckpt_dir / "best_discriminator.pt", discriminator)
                (ckpt_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

            if epoch % int(config["save_every"]) == 0:
                save_generator_checkpoint(ckpt_dir / f"generator_epoch_{epoch:03d}.pt", generator, latent_dim=config["latent_dim"], base_channels_g=128, class_names=CLASS_NAMES, use_tanh_output=config["use_tanh_output"])


if __name__ == "__main__":
    main()
