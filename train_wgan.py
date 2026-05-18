from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.wgan_1d import ConditionalCritic1D, ConditionalGenerator1D
from utils.baseline_utils import SignalDataset, get_device, open_csv_logger, setup_logger, write_json
from utils.run_paths import create_run_dir
from utils.seed import set_seed
from utils.signal_utils import CLASS_NAMES

CONFIG = {
    "data_dir": "processed/mhta_base_dataset",
    "run_root": "runs/wgan_v1",
    "run_id": None,
    "epochs": 5000,
    "batch_size": 32,
    "lr": 5e-5,
    "latent_dim": 128,
    "signal_length": 2048,
    "num_classes": 5,
    "base_channels": 64,
    "critic_steps": 5,
    "weight_clip": 0.01,
    "num_workers": 2,
    "save_every": 200,
    "seed": 42,
    "device": "cuda_if_available",
}


def model_kwargs(config: dict) -> dict:
    return {"latent_dim": int(config["latent_dim"]), "num_classes": int(config["num_classes"]), "signal_length": int(config["signal_length"]), "base_channels": int(config["base_channels"])}


def save_checkpoint(path: Path, generator, critic, g_opt, c_opt, epoch: int, g_loss: float, c_loss: float, config: dict) -> None:
    torch.save({
        "generator_state_dict": generator.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "g_optimizer_state_dict": g_opt.state_dict(),
        "critic_optimizer_state_dict": c_opt.state_dict(),
        "epoch": int(epoch),
        "g_loss": float(g_loss),
        "critic_loss": float(c_loss),
        "config": config,
        "class_names": CLASS_NAMES,
        "model_kwargs": model_kwargs(config),
    }, path)


def main(config: dict = CONFIG) -> None:
    logger = setup_logger("train_wgan")
    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    out_dir = create_run_dir(Path(config["run_root"]), config.get("run_id"))
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", config)
    dataset = SignalDataset(Path(config["data_dir"]) / "train.npz")
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda", drop_last=True)
    generator = ConditionalGenerator1D(**model_kwargs(config)).to(device)
    critic = ConditionalCritic1D(num_classes=int(config["num_classes"]), signal_length=int(config["signal_length"]), base_channels=int(config["base_channels"])).to(device)
    g_opt = torch.optim.RMSprop(generator.parameters(), lr=float(config["lr"]))
    c_opt = torch.optim.RMSprop(critic.parameters(), lr=float(config["lr"]))
    best_g = float("inf")
    logger.info("Training conditional WGAN on %d samples with device=%s", len(dataset), device)
    f, writer = open_csv_logger(out_dir / "train_log.csv", ["epoch", "g_loss", "critic_loss", "best_g_loss"])
    with f:
        for epoch in range(1, int(config["epochs"]) + 1):
            g_losses, c_losses = [], []
            for real, labels in tqdm(loader, desc=f"wgan epoch {epoch:03d}", leave=False):
                real, labels = real.to(device), labels.to(device)
                bsz = real.size(0)
                for _ in range(int(config["critic_steps"])):
                    z = torch.randn(bsz, int(config["latent_dim"]), device=device)
                    fake = generator(z, labels).detach()
                    c_loss = -(critic(real, labels).mean() - critic(fake, labels).mean())
                    c_opt.zero_grad(set_to_none=True); c_loss.backward(); c_opt.step()
                    for p in critic.parameters():
                        p.data.clamp_(-float(config["weight_clip"]), float(config["weight_clip"]))
                z = torch.randn(bsz, int(config["latent_dim"]), device=device)
                g_loss = -critic(generator(z, labels), labels).mean()
                g_opt.zero_grad(set_to_none=True); g_loss.backward(); g_opt.step()
                g_losses.append(float(g_loss.item())); c_losses.append(float(c_loss.item()))
            g_epoch, c_epoch = float(np.mean(g_losses)), float(np.mean(c_losses))
            if g_epoch < best_g:
                best_g = g_epoch
                save_checkpoint(ckpt_dir / "best_model.pt", generator, critic, g_opt, c_opt, epoch, g_epoch, c_epoch, config)
            save_checkpoint(ckpt_dir / "latest_model.pt", generator, critic, g_opt, c_opt, epoch, g_epoch, c_epoch, config)
            if epoch % int(config["save_every"]) == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", generator, critic, g_opt, c_opt, epoch, g_epoch, c_epoch, config)
            writer.writerow({"epoch": epoch, "g_loss": g_epoch, "critic_loss": c_epoch, "best_g_loss": best_g}); f.flush()
            logger.info("epoch=%03d g_loss=%.6f critic_loss=%.6f", epoch, g_epoch, c_epoch)


if __name__ == "__main__":
    main()
