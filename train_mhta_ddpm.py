from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.mhta_ddpm_1d import ConditionalMHTAUNet1D, DDPMScheduler1D
from utils.seed import set_seed
from utils.signal_utils import CLASS_NAMES


CONFIG = {
    "data_dir": Path("processed/mhta_base_dataset"),
    "out_dir": Path("runs/omc_tf_mhta_ddpm_v1"),
    "epochs": 3000,
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_train_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "clip_sample": True,
    "clip_range": 5.0,
    "signal_length": 2048,
    "num_classes": 5,
    "base_channels": 48,
    "channel_mults": [1, 2, 4, 8],
    "num_heads": 4,
    "time_embed_dim": 192,
    "label_embed_dim": 192,
    "dropout": 0.05,
    "attention_levels": [1, 2, 3],
    "lambda_fft": 0.01,
    "grad_clip": 1.0,
    "num_workers": 2,
    "save_every": 500,
    "seed": 42,
    "device": "cuda_if_available",
}


class SignalDataset(Dataset):
    def __init__(self, npz_path: Path):
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing train set: {npz_path}. Please run python data_preprocess/preprocess.py first.")
        data = np.load(npz_path, allow_pickle=True)
        self.x = np.asarray(data["X"], dtype=np.float32)
        self.y = np.asarray(data["y"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]).float().unsqueeze(0), torch.tensor(int(self.y[index]), dtype=torch.long)


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("train_mhta_ddpm")


def get_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def serializable_config(config: dict) -> dict:
    return {key: (value.as_posix() if isinstance(value, Path) else value) for key, value in config.items()}


def model_kwargs(config: dict) -> dict:
    return {
        "signal_length": int(config["signal_length"]),
        "num_classes": int(config["num_classes"]),
        "base_channels": int(config["base_channels"]),
        "channel_mults": tuple(int(v) for v in config["channel_mults"]),
        "num_heads": int(config["num_heads"]),
        "time_embed_dim": int(config["time_embed_dim"]),
        "label_embed_dim": int(config["label_embed_dim"]),
        "dropout": float(config["dropout"]),
        "attention_levels": tuple(int(v) for v in config["attention_levels"]),
    }


def scheduler_kwargs(config: dict) -> dict:
    return {
        "num_train_timesteps": int(config["num_train_timesteps"]),
        "beta_start": float(config["beta_start"]),
        "beta_end": float(config["beta_end"]),
        "clip_sample": bool(config.get("clip_sample", True)),
        "clip_range": float(config.get("clip_range", 5.0)),
    }


def save_checkpoint(
    path: Path,
    model: ConditionalMHTAUNet1D,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "loss": float(loss),
            "config": serializable_config(config),
            "class_names": CLASS_NAMES,
            "model_kwargs": model_kwargs(config),
            "scheduler_kwargs": scheduler_kwargs(config),
        },
        path,
    )


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    out_dir = Path(config["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(serializable_config(config), f, indent=2)

    dataset = SignalDataset(Path(config["data_dir"]) / "train.npz")
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = ConditionalMHTAUNet1D(**model_kwargs(config)).to(device)
    scheduler = DDPMScheduler1D(**scheduler_kwargs(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    log_path = out_dir / "train_log.csv"
    best_loss = float("inf")

    logger.info("Training MHTA-DDPM on %d samples with device=%s", len(dataset), device)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "best_loss"])
        writer.writeheader()
        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            losses = []
            pbar = tqdm(dataloader, desc=f"mhta-ddpm epoch {epoch:03d}", leave=False)
            for x0, labels in pbar:
                x0 = x0.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                noise = torch.randn_like(x0)
                timesteps = torch.randint(
                    0,
                    int(config["num_train_timesteps"]),
                    (x0.size(0),),
                    dtype=torch.long,
                    device=device,
                )
                x_t = scheduler.q_sample(x0, timesteps, noise)
                pred_noise = model(x_t, timesteps, labels)

                # 1) 标准 DDPM 噪声预测损失（时域）
                noise_loss = F.mse_loss(pred_noise, noise)

                # 2) 根据预测噪声反推预测的干净信号 x0，并对标准化数据做宽松限幅
                alpha_bar_t = scheduler._extract(scheduler.alphas_cumprod, timesteps, x_t.shape)
                pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
                pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)

                # 3) 使用正交归一化 FFT，避免长序列带来的幅值膨胀
                fft_real = torch.fft.rfft(x0, dim=-1, norm="ortho")
                fft_pred = torch.fft.rfft(pred_x0, dim=-1, norm="ortho")

                # 4) 比较对数幅值谱，使大峰值与小峰值的优化更平滑
                mag_real = torch.log(torch.abs(fft_real) + 1e-7)
                mag_pred = torch.log(torch.abs(fft_pred) + 1e-7)
                fft_loss = F.l1_loss(mag_pred, mag_real)

                # 5) 联合优化时域噪声拟合与稳健的频域重建
                loss = noise_loss + float(config.get("lambda_fft", 0.01)) * fft_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
                optimizer.step()

                loss_value = float(loss.item())
                losses.append(loss_value)
                pbar.set_postfix(loss=f"{loss_value:.4f}")

            epoch_loss = float(np.mean(losses)) if losses else float("nan")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(ckpt_dir / "best_model.pt", model, optimizer, epoch, epoch_loss, config)
            save_checkpoint(ckpt_dir / "latest_model.pt", model, optimizer, epoch, epoch_loss, config)
            if epoch % int(config["save_every"]) == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, epoch_loss, config)

            writer.writerow({"epoch": epoch, "loss": epoch_loss, "best_loss": best_loss})
            f.flush()
            logger.info("epoch=%03d loss=%.6f best=%.6f", epoch, epoch_loss, best_loss)

    logger.info("Saved MHTA-DDPM checkpoints to %s", ckpt_dir)


if __name__ == "__main__":
    main()
