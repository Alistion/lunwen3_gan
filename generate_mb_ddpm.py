from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from timm.utils import ModelEmaV3
from tqdm import tqdm

from models.mb_ddpm_1d import DDPM_Scheduler_1D, UNET_1D
from utils.seed import set_seed
from utils.signal_utils import CLASS_NAMES, LABEL_MAP, order_template_for_labels, safe_copy_npz
from utils.run_paths import resolve_existing_run_dir


CONFIG = {
    "data_dir": Path("processed/mhta_base_dataset"),
    "run_root": Path("runs/mb_ddpm_lunwen3_v1"),
    "run_id": None,
    "ckpt": None,
    "out_dir": Path("processed/mb_augmented_dataset"),
    "samples_per_fault_class": 10,
    "signal_length": 2048,
    "num_time_steps": 1000,
    "num_classes": 5,
    "sampler": "ddpm",
    "ddim_steps": 50,
    "ema_decay": 0.9999,
    "rpm": 740.0,
    "seed": 42,
    "device": "cuda_if_available",
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("generate_mb_ddpm")


def get_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def remove_signal_mean(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)).astype(np.float32)


def load_model(config: dict, device: torch.device) -> tuple[UNET_1D, DDPM_Scheduler_1D, dict]:
    ckpt = resolve_checkpoint_path(config)
    if not ckpt.exists():
        raise FileNotFoundError(f"MB-DDPM checkpoint does not exist: {ckpt}. Please run python train_mb_ddpm.py first.")

    payload = torch.load(ckpt, map_location=device)
    model_kwargs = payload.get(
        "model_kwargs",
        {
            "in_feature": int(config["signal_length"]),
            "time_steps": int(config["num_time_steps"]),
            "num_classes": int(config["num_classes"]),
        },
    )
    model = UNET_1D(**model_kwargs).to(device)
    model.load_state_dict(payload["weights"])

    ema = ModelEmaV3(model, decay=float(config["ema_decay"]))
    ema.load_state_dict(payload["ema"])
    model = ema.module.eval()
    scheduler = DDPM_Scheduler_1D(num_time_steps=int(model_kwargs["time_steps"])).to(device)
    return model, scheduler, payload


def resolve_checkpoint_path(config: dict) -> Path:
    explicit_ckpt = config.get("ckpt")
    run_id = config.get("run_id")
    if not explicit_ckpt and isinstance(run_id, str) and run_id.endswith(".pt"):
        explicit_ckpt = run_id
        run_id = None
    run_dir = resolve_existing_run_dir(Path(config["run_root"]), run_id)
    if explicit_ckpt:
        ckpt_path = Path(explicit_ckpt)
        if ckpt_path.is_absolute() or ckpt_path.parent != Path("."):
            return ckpt_path
        return run_dir / "checkpoints" / ckpt_path.name
    return run_dir / "checkpoints" / "best_model.pt"


def ddpm_sample_one(
    model: UNET_1D,
    scheduler: DDPM_Scheduler_1D,
    label_tensor: torch.Tensor,
    signal_length: int,
    num_time_steps: int,
    device: torch.device,
) -> torch.Tensor:
    z = torch.randn(1, 1, signal_length, device=device)
    for t in reversed(range(1, num_time_steps)):
        t_tensor = torch.tensor([t], device=device)
        beta_t = scheduler.beta[t_tensor]
        alpha_t = scheduler.alpha[t_tensor]
        pred_noise = model(z, t_tensor, label_tensor)
        temp = beta_t / (torch.sqrt(1 - alpha_t) * torch.sqrt(1 - beta_t))
        z = z / torch.sqrt(1 - beta_t) - temp * pred_noise
        noise = torch.randn_like(z) if t > 1 else torch.zeros_like(z)
        z = z + noise * torch.sqrt(beta_t)

    beta_0 = scheduler.beta[0]
    alpha_0 = scheduler.alpha[0]
    temp = beta_0 / (torch.sqrt(1 - alpha_0) * torch.sqrt(1 - beta_0))
    return z / torch.sqrt(1 - beta_0) - temp * model(z, torch.tensor([0], device=device), label_tensor)


def ddim_sample_one(
    model: UNET_1D,
    scheduler: DDPM_Scheduler_1D,
    label_tensor: torch.Tensor,
    signal_length: int,
    num_time_steps: int,
    ddim_steps: int,
    device: torch.device,
) -> torch.Tensor:
    z = torch.randn(1, 1, signal_length, device=device)
    ddim_timesteps = np.linspace(0, num_time_steps - 1, int(ddim_steps), dtype=int)[::-1].copy().tolist()
    for i, t in enumerate(ddim_timesteps[:-1]):
        next_t = ddim_timesteps[i + 1]
        current_t_tensor = torch.tensor([t], device=device)
        next_t_tensor = torch.tensor([next_t], device=device)
        pred_noise = model(z, current_t_tensor, label_tensor)
        alpha_current = scheduler.alpha[current_t_tensor]
        alpha_next = scheduler.alpha[next_t_tensor]
        pred_x0 = (z - torch.sqrt(1 - alpha_current) * pred_noise) / torch.sqrt(alpha_current)
        z = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next) * pred_noise
    return z


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(config["data_dir"]) / "train.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train set: {train_path}. Please run python data_preprocess/preprocess.py first.")
    train = np.load(train_path, allow_pickle=True)
    x_train = np.asarray(train["X"], dtype=np.float32)
    y_train = np.asarray(train["y"], dtype=np.int64)

    model, scheduler, payload = load_model(config, device)
    model_kwargs = payload.get("model_kwargs", {})
    signal_length = int(model_kwargs.get("in_feature", config["signal_length"]))
    num_time_steps = int(model_kwargs.get("time_steps", config["num_time_steps"]))
    samples_per_class = int(config["samples_per_fault_class"])
    sampler = str(config["sampler"]).lower()
    rpm = float(config["rpm"])
    fr = rpm / 60.0

    generated_x, generated_y, meta_rows = [], [], []
    with torch.no_grad():
        for class_name in CLASS_NAMES[1:]:
            label = int(LABEL_MAP[class_name])
            label_tensor = torch.tensor([label], dtype=torch.long, device=device)
            logger.info("Generating %d MB-DDPM samples for %s", samples_per_class, class_name)
            class_samples = []
            for generated_index in tqdm(range(samples_per_class), desc=class_name, leave=False):
                if sampler == "ddpm":
                    fake = ddpm_sample_one(model, scheduler, label_tensor, signal_length, num_time_steps, device)
                elif sampler == "ddim":
                    fake = ddim_sample_one(
                        model,
                        scheduler,
                        label_tensor,
                        signal_length,
                        num_time_steps,
                        int(config["ddim_steps"]),
                        device,
                    )
                else:
                    raise ValueError(f"Unsupported sampler: {config['sampler']}")
                class_samples.append(fake.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32))
                meta_rows.append(
                    {
                        "class_name": class_name,
                        "label": label,
                        "source": "mb_ddpm_lunwen3_v1",
                        "generated_index": generated_index,
                        "rpm": rpm,
                        "fr": fr,
                        "checkpoint": resolve_checkpoint_path(config).as_posix(),
                        "sampler": sampler,
                        "num_inference_steps": num_time_steps if sampler == "ddpm" else int(config["ddim_steps"]),
                    }
                )
            class_array = remove_signal_mean(np.stack(class_samples, axis=0))
            generated_x.append(class_array)
            generated_y.append(np.full(samples_per_class, label, dtype=np.int64))

    x_gen = np.concatenate(generated_x, axis=0).astype(np.float32)
    y_gen = np.concatenate(generated_y, axis=0).astype(np.int64)

    np.savez_compressed(
        out_dir / "generated_only.npz",
        X=x_gen,
        y=y_gen,
        class_names=np.asarray(CLASS_NAMES),
        rpm=np.full(len(y_gen), rpm, dtype=np.float32),
        fr=np.full(len(y_gen), fr, dtype=np.float32),
        order_templates=order_template_for_labels(y_gen),
    )
    np.savez_compressed(
        out_dir / "train_augmented.npz",
        X=np.concatenate([x_train, x_gen], axis=0).astype(np.float32),
        y=np.concatenate([y_train, y_gen], axis=0).astype(np.int64),
        class_names=np.asarray(CLASS_NAMES),
    )
    write_metadata(out_dir / "metadata_generated.csv", meta_rows)
    safe_copy_npz(Path(config["data_dir"]) / "val.npz", out_dir / "val.npz")
    safe_copy_npz(Path(config["data_dir"]) / "test.npz", out_dir / "test.npz")
    logger.info("Saved MB-DDPM generated dataset to %s", out_dir)


if __name__ == "__main__":
    main()
