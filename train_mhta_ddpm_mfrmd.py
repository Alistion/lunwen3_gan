from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.mhta_ddpm_1d import ConditionalMHTAUNet1D, DDPMScheduler1D
from utils.run_paths import create_run_dir
from utils.seed import set_seed


CONFIG = {
    "data_dir": "processed/mfrmd_base_data",
    "run_root": "runs/omc_tf_mhta_ddpm_v1/mfrmd_data",
    "run_id": None,
    # Set per-class training sample counts here. Use None to load all samples.
    # Example: {"normal": 50, "unbalance": 50, "misalignment": 50, "looseness": 50}
    "train_counts": {
        "normal": 500,
        "unbalance": 500,
        "misalignment": 500,
        "looseness": 500,
    },
    "epochs": 5000,
    "batch_size": 32,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "clip_sample": True,
    "clip_range": 15.0,
    "signal_length": 2048,
    "base_channels": 48,
    "channel_mults": [1, 2, 4, 8],
    "num_heads": 4,
    "time_embed_dim": 192,
    "label_embed_dim": 192,
    "dropout": 0.05,
    "attention_levels": [1, 2, 3],
    "disable_temporal_attention": False,
    "disable_mhta": False,
    "use_qkv_dwconv": True,
    "input_kernel_size": 31,
    "lambda_fft": 0,
    "grad_clip": 1.0,
    "num_workers": 0,
    "save_every": 500,
    "seed": 42,
    "device": "cuda_if_available",
    "generate_after_train": True,
    "evaluate_after_generate": True,
    "plot_after_generate": True,
    "generated_subdir": "generated_dataset",
    "samples_per_class": 100,
    "generate_batch_size": 32,
    "sampler": "ddpm",
    "num_inference_steps": 200,
    "eta": 0.0,
    "fs": 2048,
    "max_freq": 300.0,
    "fd_downsample": 256,
    "max_pairs_per_class": 100,
    "plot_samples_per_class": 3,
    "eps": 1e-8,
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("train_mhta_ddpm_mfrmd")


def get_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def serializable_config(config: dict) -> dict:
    out = {}
    for key, value in config.items():
        if isinstance(value, Path):
            out[key] = value.as_posix()
        else:
            out[key] = value
    return out


def model_kwargs(config: dict, num_classes: int) -> dict:
    return {
        "signal_length": int(config["signal_length"]),
        "num_classes": int(num_classes),
        "base_channels": int(config["base_channels"]),
        "channel_mults": tuple(int(v) for v in config["channel_mults"]),
        "num_heads": int(config["num_heads"]),
        "time_embed_dim": int(config["time_embed_dim"]),
        "label_embed_dim": int(config["label_embed_dim"]),
        "dropout": float(config["dropout"]),
        "attention_levels": tuple(int(v) for v in config["attention_levels"]),
        "strict_paper_mode": False,
        "disable_temporal_attention": bool(config.get("disable_temporal_attention", False)),
        "disable_mhta": bool(config.get("disable_mhta", False)),
        "use_qkv_dwconv": bool(config.get("use_qkv_dwconv", True)),
        "input_kernel_size": int(config["input_kernel_size"]) if config.get("input_kernel_size") is not None else None,
    }


def scheduler_kwargs(config: dict) -> dict:
    return {
        "num_train_timesteps": int(config["num_train_timesteps"]),
        "beta_start": float(config["beta_start"]),
        "beta_end": float(config["beta_end"]),
        "clip_sample": bool(config.get("clip_sample", True)),
        "clip_range": float(config.get("clip_range", 5.0)),
    }


class MFRMDDataset(Dataset):
    def __init__(self, npz_path: Path, train_counts: dict[str, int] | None, seed: int):
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing train set: {npz_path}. Please run data_preprocess/preprocess_mfrmd.py first.")
        data = np.load(npz_path, allow_pickle=True)
        x = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        class_names = [str(v) for v in data["class_names"].tolist()]

        selected_idx = select_indices_by_count(y, class_names, train_counts, seed)
        self.x = x[selected_idx]
        self.y = y[selected_idx]
        self.indices = selected_idx
        self.class_names = class_names
        self.label_map = {name: i for i, name in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]).float().unsqueeze(0), torch.tensor(int(self.y[index]), dtype=torch.long)


def select_indices_by_count(
    labels: np.ndarray,
    class_names: list[str],
    train_counts: dict[str, int] | None,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    chosen = []
    for label, class_name in enumerate(class_names):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        if train_counts is None:
            need = len(idx)
        else:
            need = int(train_counts.get(class_name, len(idx)))
        if need > len(idx):
            raise ValueError(f"{class_name} has only {len(idx)} samples, but requested {need}.")
        chosen.extend(idx[:need].tolist())
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def compute_loss(
    pred_noise: torch.Tensor,
    noise: torch.Tensor,
    x0: torch.Tensor,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler1D,
    config: dict,
) -> torch.Tensor:
    loss_noise = F.mse_loss(pred_noise, noise)
    lambda_fft = float(config.get("lambda_fft", 0.05))
    if lambda_fft <= 0.0:
        return loss_noise

    alpha_bar_t = scheduler._extract(scheduler.alphas_cumprod, timesteps, x_t.shape)
    pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
    pred_x0 = torch.clamp(pred_x0, -25.0, 25.0)
    mag_real = torch.abs(torch.fft.rfft(x0, dim=-1, norm="ortho"))
    mag_pred = torch.abs(torch.fft.rfft(pred_x0, dim=-1, norm="ortho"))
    loss_fft = F.mse_loss(mag_pred[:, :, :300], mag_real[:, :, :300])
    return loss_noise + lambda_fft * loss_fft


def save_checkpoint(
    path: Path,
    model: ConditionalMHTAUNet1D,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: dict,
    class_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mk = model_kwargs(config, len(class_names))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "loss": float(loss),
            "config": serializable_config(config),
            "class_names": class_names,
            "model_kwargs": mk,
            "scheduler_kwargs": scheduler_kwargs(config),
        },
        path,
    )


def write_dataset_selection(out_dir: Path, dataset: MFRMDDataset) -> None:
    rows = []
    for local_index, source_index in enumerate(dataset.indices.tolist()):
        label = int(dataset.y[local_index])
        rows.append(
            {
                "local_index": local_index,
                "source_train_index": int(source_index),
                "label": label,
                "class_name": dataset.class_names[label],
            }
        )
    with (out_dir / "train_subset_indices.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(config: dict, out_dir: Path, logger: logging.Logger) -> tuple[Path, list[str]]:
    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    dataset = MFRMDDataset(Path(config["data_dir"]) / "train.npz", config.get("train_counts"), int(config["seed"]))
    class_names = dataset.class_names
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    write_dataset_selection(out_dir, dataset)
    model = ConditionalMHTAUNet1D(**model_kwargs(config, len(class_names))).to(device)
    scheduler = DDPMScheduler1D(**scheduler_kwargs(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    ckpt_dir = out_dir / "checkpoints"
    log_path = out_dir / "train_log.csv"
    best_loss = float("inf")
    logger.info("Training MFRMD MHTA-DDPM on %d samples with device=%s", len(dataset), device)
    logger.info("Class names: %s", class_names)
    logger.info("Run directory: %s", out_dir)

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "best_loss"])
        writer.writeheader()
        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            losses = []
            pbar = tqdm(dataloader, desc=f"mfrmd-mhta epoch {epoch:03d}", leave=False)
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
                loss = compute_loss(pred_noise, noise, x0, x_t, timesteps, scheduler, config)

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
                save_checkpoint(ckpt_dir / "best_model.pt", model, optimizer, epoch, epoch_loss, config, class_names)
            save_checkpoint(ckpt_dir / "latest_model.pt", model, optimizer, epoch, epoch_loss, config, class_names)
            if epoch % int(config["save_every"]) == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, epoch_loss, config, class_names)

            writer.writerow({"epoch": epoch, "loss": epoch_loss, "best_loss": best_loss})
            f.flush()
            logger.info("epoch=%03d loss=%.6f best=%.6f", epoch, epoch_loss, best_loss)

    return ckpt_dir / "best_model.pt", class_names


def load_model(ckpt_path: Path, device: torch.device) -> tuple[ConditionalMHTAUNet1D, DDPMScheduler1D, dict]:
    payload = torch.load(ckpt_path, map_location=device)
    model = ConditionalMHTAUNet1D(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    scheduler = DDPMScheduler1D(**payload["scheduler_kwargs"]).to(device)
    return model, scheduler, payload


@torch.no_grad()
def sample_batch(
    scheduler: DDPMScheduler1D,
    model: ConditionalMHTAUNet1D,
    labels: torch.Tensor,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    shape = (labels.size(0), 1, int(config["signal_length"]))
    if str(config["sampler"]).lower() == "ddpm":
        return scheduler.sample(model=model, shape=shape, labels=labels, device=device)
    return scheduler.ddim_sample(
        model=model,
        shape=shape,
        labels=labels,
        num_inference_steps=int(config["num_inference_steps"]),
        eta=float(config["eta"]),
        device=device,
    )


def generate(config: dict, out_dir: Path, ckpt_path: Path, logger: logging.Logger) -> Path:
    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    model, scheduler, payload = load_model(ckpt_path, device)
    class_names = [str(v) for v in payload["class_names"]]
    generated_dir = out_dir / str(config["generated_subdir"])
    generated_dir.mkdir(parents=True, exist_ok=True)

    generated_x, generated_y, meta_rows = [], [], []
    samples_per_class = int(config["samples_per_class"])
    batch_size = int(config["generate_batch_size"])
    with torch.no_grad():
        for label, class_name in enumerate(class_names):
            logger.info("Generating %d MFRMD samples for %s", samples_per_class, class_name)
            left = samples_per_class
            start = 0
            progress = tqdm(total=samples_per_class, desc=f"generate {class_name}", leave=False)
            while left > 0:
                bsz = min(batch_size, left)
                labels = torch.full((bsz,), label, dtype=torch.long, device=device)
                fake = sample_batch(scheduler, model, labels, config, device)
                fake_np = fake.squeeze(1).cpu().numpy().astype(np.float32)
                generated_x.append(fake_np)
                generated_y.append(np.full(bsz, label, dtype=np.int64))
                for i in range(bsz):
                    meta_rows.append(
                        {
                            "class_name": class_name,
                            "label": label,
                            "generated_index": start + i,
                            "checkpoint": ckpt_path.as_posix(),
                            "sampler": str(config["sampler"]).lower(),
                            "num_inference_steps": int(config["num_inference_steps"]),
                            "eta": float(config["eta"]),
                        }
                    )
                start += bsz
                left -= bsz
                progress.update(bsz)
            progress.close()

    x_gen = np.concatenate(generated_x, axis=0).astype(np.float32)
    y_gen = np.concatenate(generated_y, axis=0).astype(np.int64)
    np.savez_compressed(generated_dir / "generated_only.npz", X=x_gen, y=y_gen, class_names=np.asarray(class_names))
    pd.DataFrame(meta_rows).to_csv(generated_dir / "metadata_generated.csv", index=False)
    logger.info("Saved generated samples to %s", generated_dir)
    return generated_dir / "generated_only.npz"


def load_xy(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(npz_path, allow_pickle=True)
    return np.asarray(data["X"], dtype=np.float32), np.asarray(data["y"], dtype=np.int64), [str(v) for v in data["class_names"].tolist()]


def rmse(real: np.ndarray, generated: np.ndarray, eps: float) -> float:
    return float(np.sqrt(np.mean(np.square(real - generated)) + eps))


def psnr(real: np.ndarray, generated: np.ndarray, eps: float) -> float:
    mse = float(np.mean(np.square(real - generated)))
    data_range = float(np.max(real) - np.min(real))
    data_range = max(data_range, float(np.max(np.abs(real))), eps)
    return float(10.0 * np.log10((data_range * data_range + eps) / (mse + eps)))


def fscs(real: np.ndarray, generated: np.ndarray, fs: int, max_freq: float, eps: float) -> float:
    real_spec = np.abs(np.fft.rfft(real))
    gen_spec = np.abs(np.fft.rfft(generated))
    freqs = np.fft.rfftfreq(real.shape[-1], d=1.0 / float(fs))
    mask = (freqs > 0) & (freqs <= float(max_freq))
    a = real_spec[mask].astype(np.float64)
    b = gen_spec[mask].astype(np.float64)
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def discrete_frechet_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    n, m = curve_a.shape[0], curve_b.shape[0]
    distances = np.linalg.norm(curve_a[:, None, :] - curve_b[None, :, :], axis=2)
    ca = np.empty((n, m), dtype=np.float64)
    ca[0, 0] = distances[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], distances[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], distances[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), distances[i, j])
    return float(ca[-1, -1])


def frechet_signal_distance(real: np.ndarray, generated: np.ndarray, downsample: int) -> float:
    x = np.linspace(0.0, 1.0, int(downsample), dtype=np.float64)
    src_real = np.linspace(0.0, 1.0, real.size)
    src_gen = np.linspace(0.0, 1.0, generated.size)
    curve_real = np.stack([x, np.interp(x, src_real, real)], axis=1)
    curve_gen = np.stack([x, np.interp(x, src_gen, generated)], axis=1)
    return discrete_frechet_distance(curve_real, curve_gen)


def evaluate(config: dict, out_dir: Path, logger: logging.Logger) -> None:
    rng = np.random.default_rng(int(config["seed"]))
    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    x_real, y_real, class_names = load_xy(Path(config["data_dir"]) / "train.npz")
    x_gen, y_gen, _ = load_xy(out_dir / str(config["generated_subdir"]) / "generated_only.npz")

    pair_rows = []
    for label, class_name in enumerate(class_names):
        real_idx = np.flatnonzero(y_real == label)
        gen_idx = np.flatnonzero(y_gen == label)
        pairs = min(len(real_idx), len(gen_idx), int(config["max_pairs_per_class"]))
        if pairs == 0:
            continue
        real_pick = rng.choice(real_idx, size=pairs, replace=False)
        gen_pick = rng.choice(gen_idx, size=pairs, replace=False)
        for pair_index, (ri, gi) in enumerate(zip(real_pick, gen_pick)):
            real = x_real[int(ri)]
            gen = x_gen[int(gi)]
            pair_rows.append(
                {
                    "class_name": class_name,
                    "label": label,
                    "pair_index": pair_index,
                    "real_index": int(ri),
                    "generated_index": int(gi),
                    "RMSE": rmse(real, gen, float(config["eps"])),
                    "PSNR": psnr(real, gen, float(config["eps"])),
                    "FSCS": fscs(real, gen, int(config["fs"]), float(config["max_freq"]), float(config["eps"])),
                    "FD": frechet_signal_distance(real, gen, int(config["fd_downsample"])),
                }
            )

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(eval_dir / "mfrmd_pair_metrics.csv", index=False)
    summary = (
        pair_df.groupby(["class_name", "label"], as_index=False)
        .agg(
            pairs=("RMSE", "size"),
            RMSE=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            PSNR=("PSNR", "mean"),
            PSNR_std=("PSNR", "std"),
            FSCS=("FSCS", "mean"),
            FSCS_std=("FSCS", "std"),
            FD=("FD", "mean"),
            FD_std=("FD", "std"),
        )
        .fillna(0.0)
    )
    summary.to_csv(eval_dir / "mfrmd_generated_quality_metrics.csv", index=False)
    if not summary.empty:
        weights = summary["pairs"].to_numpy(dtype=np.float64)
        weights = weights / (weights.sum() + float(config["eps"]))
        overall = {"class_name": "overall_weighted_by_pairs", "label": -1, "pairs": int(summary["pairs"].sum())}
        for key in ("RMSE", "PSNR", "FSCS", "FD"):
            overall[key] = float(np.sum(summary[key].to_numpy(dtype=np.float64) * weights))
        pd.DataFrame([overall]).to_csv(eval_dir / "mfrmd_overall_metrics.csv", index=False)
    logger.info("Saved MFRMD evaluation metrics to %s", eval_dir)


def plot_time_freq_pair(
    real: np.ndarray,
    generated: np.ndarray,
    fs: int,
    title: str,
    out_path: Path,
    max_freq: float,
) -> None:
    t = np.arange(real.size) / float(fs)
    freqs = np.fft.rfftfreq(real.size, d=1.0 / float(fs))
    real_amp = np.abs(np.fft.rfft(real)) / real.size * 2.0
    gen_amp = np.abs(np.fft.rfft(generated)) / generated.size * 2.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(t, real, label="real", linewidth=0.9)
    axes[0].plot(t, generated, label="generated", linewidth=0.9, alpha=0.8)
    axes[0].set_title(f"{title} - time domain")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(freqs, real_amp, label="real", linewidth=0.9)
    axes[1].plot(freqs, gen_amp, label="generated", linewidth=0.9, alpha=0.8)
    axes[1].set_title(f"{title} - frequency domain")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlim(0, float(max_freq))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_samples(config: dict, out_dir: Path, logger: logging.Logger) -> None:
    rng = np.random.default_rng(int(config["seed"]))
    plot_dir = out_dir / "plots" / "sample_time_freq"
    plot_dir.mkdir(parents=True, exist_ok=True)
    x_real, y_real, class_names = load_xy(Path(config["data_dir"]) / "train.npz")
    x_gen, y_gen, _ = load_xy(out_dir / str(config["generated_subdir"]) / "generated_only.npz")
    rows = []
    for label, class_name in enumerate(class_names):
        real_idx = np.flatnonzero(y_real == label)
        gen_idx = np.flatnonzero(y_gen == label)
        count = min(len(real_idx), len(gen_idx), int(config["plot_samples_per_class"]))
        if count == 0:
            continue
        real_pick = rng.choice(real_idx, size=count, replace=False)
        gen_pick = rng.choice(gen_idx, size=count, replace=False)
        for i, (ri, gi) in enumerate(zip(real_pick, gen_pick)):
            out_path = plot_dir / f"{class_name}_{i:02d}_real_vs_generated_time_freq.png"
            plot_time_freq_pair(
                x_real[int(ri)],
                x_gen[int(gi)],
                int(config["fs"]),
                f"{class_name} sample {i}",
                out_path,
                float(config["max_freq"]),
            )
            rows.append(
                {
                    "class_name": class_name,
                    "label": label,
                    "real_index": int(ri),
                    "generated_index": int(gi),
                    "plot_path": out_path.as_posix(),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "plots" / "mfrmd_sample_plot_indices.csv", index=False)
    logger.info("Saved MFRMD sample plots to %s", plot_dir)


def main(config: dict = CONFIG) -> None:
    run_config = dict(config)
    logger = setup_logger()
    out_dir = create_run_dir(Path(run_config["run_root"]), run_config.get("run_id"))
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(serializable_config(run_config), f, indent=2)

    ckpt_path, _ = train(run_config, out_dir, logger)
    if bool(run_config.get("generate_after_train", True)):
        generate(run_config, out_dir, ckpt_path, logger)
        if bool(run_config.get("evaluate_after_generate", True)):
            evaluate(run_config, out_dir, logger)
        if bool(run_config.get("plot_after_generate", True)):
            plot_samples(run_config, out_dir, logger)


if __name__ == "__main__":
    main()
