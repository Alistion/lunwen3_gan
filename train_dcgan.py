from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dcgan_1d import ConditionalDiscriminator1D, ConditionalGenerator1D
from utils.baseline_utils import SignalDataset, get_device, open_csv_logger, setup_logger, write_json
from utils.run_paths import create_run_dir
from utils.seed import set_seed
from utils.signal_utils import CLASS_NAMES

CONFIG = {
    "data_dir": "processed/mhta_base_dataset",
    "run_root": "runs/dcgan_v1",
    "run_id": None,
    "epochs": 5000,
    "batch_size": 32,
    "g_lr": 2e-4,
    "d_lr": 5e-5,
    "beta1": 0.5,
    "beta2": 0.999,
    "real_label": 0.9,
    "instance_noise_std": 0.05,
    "instance_noise_decay_epochs": 1000,
    "latent_dim": 128,
    "signal_length": 2048,
    "num_classes": 5,
    "base_channels": 64,
    "num_workers": 2,
    "save_every": 200,
    "seed": 42,
    "device": "cuda_if_available",
}


def model_kwargs(config: dict) -> dict:
    return {"latent_dim": int(config["latent_dim"]), "num_classes": int(config["num_classes"]), "signal_length": int(config["signal_length"]), "base_channels": int(config["base_channels"])}


def save_checkpoint(path: Path, generator, discriminator, g_opt, d_opt, epoch: int, g_loss: float, d_loss: float, config: dict) -> None:
    torch.save({
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "g_optimizer_state_dict": g_opt.state_dict(),
        "d_optimizer_state_dict": d_opt.state_dict(),
        "epoch": int(epoch),
        "g_loss": float(g_loss),
        "d_loss": float(d_loss),
        "config": config,
        "class_names": CLASS_NAMES,
        "model_kwargs": model_kwargs(config),
    }, path)


def main(config: dict = CONFIG) -> None:
    logger = setup_logger("train_dcgan")
    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    out_dir = create_run_dir(Path(config["run_root"]), config.get("run_id"))
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", config)
    dataset = SignalDataset(Path(config["data_dir"]) / "train.npz")
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda", drop_last=True)
    generator = ConditionalGenerator1D(**model_kwargs(config)).to(device)
    discriminator = ConditionalDiscriminator1D(num_classes=int(config["num_classes"]), signal_length=int(config["signal_length"]), base_channels=int(config["base_channels"])).to(device)
    g_opt = torch.optim.Adam(generator.parameters(), lr=float(config["g_lr"]), betas=(float(config["beta1"]), float(config["beta2"])))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=float(config["d_lr"]), betas=(float(config["beta1"]), float(config["beta2"])))
    best_g = float("inf")
    logger.info("Training conditional DCGAN on %d samples with device=%s", len(dataset), device)
    f, writer = open_csv_logger(out_dir / "train_log.csv", ["epoch", "g_loss", "d_loss", "best_g_loss"])
    with f:
        for epoch in range(1, int(config["epochs"]) + 1):
            g_losses, d_losses = [], []
            noise_decay = max(0.0, 1.0 - (epoch - 1) / max(1, int(config["instance_noise_decay_epochs"])))
            noise_std = float(config["instance_noise_std"]) * noise_decay
            for real, labels in tqdm(loader, desc=f"dcgan epoch {epoch:03d}", leave=False):
                real, labels = real.to(device), labels.to(device)
                bsz = real.size(0)
                z = torch.randn(bsz, int(config["latent_dim"]), device=device)
                fake = generator(z, labels)
                real_for_d = real + noise_std * torch.randn_like(real) if noise_std > 0 else real
                fake_for_d = fake.detach() + noise_std * torch.randn_like(fake) if noise_std > 0 else fake.detach()
                d_real = discriminator(real_for_d, labels)
                d_fake = discriminator(fake_for_d, labels)
                real_targets = torch.full_like(d_real, float(config["real_label"]))
                d_loss = F.binary_cross_entropy_with_logits(d_real, real_targets) + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                d_opt.zero_grad(set_to_none=True); d_loss.backward(); d_opt.step()
                z = torch.randn(bsz, int(config["latent_dim"]), device=device)
                fake = generator(z, labels)
                g_loss = F.binary_cross_entropy_with_logits(discriminator(fake, labels), torch.ones(bsz, device=device))
                g_opt.zero_grad(set_to_none=True); g_loss.backward(); g_opt.step()
                g_losses.append(float(g_loss.item())); d_losses.append(float(d_loss.item()))
            g_epoch, d_epoch = float(np.mean(g_losses)), float(np.mean(d_losses))
            if g_epoch < best_g:
                best_g = g_epoch
                save_checkpoint(ckpt_dir / "best_model.pt", generator, discriminator, g_opt, d_opt, epoch, g_epoch, d_epoch, config)
            save_checkpoint(ckpt_dir / "latest_model.pt", generator, discriminator, g_opt, d_opt, epoch, g_epoch, d_epoch, config)
            if epoch % int(config["save_every"]) == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", generator, discriminator, g_opt, d_opt, epoch, g_epoch, d_epoch, config)
            writer.writerow({"epoch": epoch, "g_loss": g_epoch, "d_loss": d_epoch, "best_g_loss": best_g}); f.flush()
            logger.info("epoch=%03d g_loss=%.6f d_loss=%.6f", epoch, g_epoch, d_epoch)


if __name__ == "__main__":
    main()
