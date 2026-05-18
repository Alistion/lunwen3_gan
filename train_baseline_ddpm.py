from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline_ddpm_1d import ConditionalUNet1D
from models.mhta_ddpm_1d import DDPMScheduler1D
from utils.baseline_utils import SignalDataset, get_device, open_csv_logger, setup_logger, write_json
from utils.run_paths import create_run_dir
from utils.seed import set_seed
from utils.signal_utils import CLASS_NAMES

CONFIG = {
    "data_dir": "processed/mhta_base_dataset",
    "run_root": "runs/baseline_ddpm_v1",
    "run_id": None,
    "epochs": 5000,
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_train_timesteps": 1000,
    "clip_sample": True,
    "clip_range": 5.0,
    "signal_length": 2048,
    "num_classes": 5,
    "base_channels": 48,
    "channel_mults": [1, 2, 4, 8],
    "time_embed_dim": 192,
    "label_embed_dim": 192,
    "dropout": 0.05,
    "grad_clip": 1.0,
    "num_workers": 2,
    "save_every": 500,
    "seed": 42,
    "device": "cuda_if_available",
}


def model_kwargs(config: dict) -> dict:
    return {
        "signal_length": int(config["signal_length"]),
        "num_classes": int(config["num_classes"]),
        "base_channels": int(config["base_channels"]),
        "channel_mults": tuple(int(v) for v in config["channel_mults"]),
        "time_embed_dim": int(config["time_embed_dim"]),
        "label_embed_dim": int(config["label_embed_dim"]),
        "dropout": float(config["dropout"]),
    }


def scheduler_kwargs(config: dict) -> dict:
    return {
        "num_train_timesteps": int(config["num_train_timesteps"]),
        "clip_sample": bool(config["clip_sample"]),
        "clip_range": float(config["clip_range"]),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, loss: float, config: dict) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "loss": float(loss),
        "config": config,
        "class_names": CLASS_NAMES,
        "model_kwargs": model_kwargs(config),
        "scheduler_kwargs": scheduler_kwargs(config),
    }, path)


def main(config: dict = CONFIG) -> None:
    logger = setup_logger("train_baseline_ddpm")
    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    out_dir = create_run_dir(Path(config["run_root"]), config.get("run_id"))
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", config)

    dataset = SignalDataset(Path(config["data_dir"]) / "train.npz")
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda", drop_last=True)
    model = ConditionalUNet1D(**model_kwargs(config)).to(device)
    scheduler = DDPMScheduler1D(**scheduler_kwargs(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    best_loss = float("inf")
    logger.info("Training baseline DDPM on %d samples with device=%s", len(dataset), device)
    f, writer = open_csv_logger(out_dir / "train_log.csv", ["epoch", "loss", "best_loss"])
    with f:
        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            losses = []
            for x0, labels in tqdm(loader, desc=f"baseline-ddpm epoch {epoch:03d}", leave=False):
                x0, labels = x0.to(device), labels.to(device)
                noise = torch.randn_like(x0)
                t = torch.randint(0, int(config["num_train_timesteps"]), (x0.size(0),), device=device)
                pred = model(scheduler.q_sample(x0, t, noise), t, labels)
                loss = F.mse_loss(pred, noise)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
                optimizer.step()
                losses.append(float(loss.item()))
            epoch_loss = float(np.mean(losses))
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(ckpt_dir / "best_model.pt", model, optimizer, epoch, epoch_loss, config)
            save_checkpoint(ckpt_dir / "latest_model.pt", model, optimizer, epoch, epoch_loss, config)
            if epoch % int(config["save_every"]) == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, epoch_loss, config)
            writer.writerow({"epoch": epoch, "loss": epoch_loss, "best_loss": best_loss})
            f.flush()
            logger.info("epoch=%03d loss=%.6f best=%.6f", epoch, epoch_loss, best_loss)


if __name__ == "__main__":
    main()
