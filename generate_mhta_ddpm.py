from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.mhta_ddpm_1d import ConditionalMHTAUNet1D, DDPMScheduler1D
from utils.seed import set_seed
from utils.signal_utils import CLASS_NAMES, LABEL_MAP, order_template_for_labels, safe_copy_npz


CONFIG = {
    "data_dir": Path("processed/omc_tf_gan_dataset"),
    "ckpt": Path("runs/omc_tf_mhta_ddpm_v1/checkpoints/best_model.pt"),
    "out_dir": Path("processed/omc_tf_mhta_ddpm_dataset_augmented"),
    "samples_per_fault_class": 10,
    "batch_size": 32,
    "signal_length": 2048,
    "sampler": "ddpm",
    "num_inference_steps": 100,
    "eta": 0.0,
    "rpm": 740.0,
    "seed": 42,
    "device": "cuda_if_available",
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("generate_mhta_ddpm")


def get_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def load_model(config: dict, device: torch.device) -> tuple[ConditionalMHTAUNet1D, DDPMScheduler1D, dict]:
    ckpt = Path(config["ckpt"])
    if not ckpt.exists():
        raise FileNotFoundError(f"MHTA-DDPM checkpoint does not exist: {ckpt}. Please run python train_mhta_ddpm.py first.")
    payload = torch.load(ckpt, map_location=device)
    model = ConditionalMHTAUNet1D(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    scheduler = DDPMScheduler1D(**payload["scheduler_kwargs"]).to(device)
    return model, scheduler, payload


def remove_signal_mean(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)).astype(np.float32)


def write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_batch(
    scheduler: DDPMScheduler1D,
    model: ConditionalMHTAUNet1D,
    labels: torch.Tensor,
    signal_length: int,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    shape = (labels.size(0), 1, int(signal_length))
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


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(config["data_dir"]) / "train.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train set: {train_path}. Please run python data/preprocess.py first.")
    train = np.load(train_path, allow_pickle=True)
    x_train = np.asarray(train["X"], dtype=np.float32)
    y_train = np.asarray(train["y"], dtype=np.int64)

    model, scheduler, payload = load_model(config, device)
    signal_length = int(payload["model_kwargs"].get("signal_length", config["signal_length"]))
    batch_size = int(config["batch_size"])
    samples_per_class = int(config["samples_per_fault_class"])
    rpm = float(config["rpm"])
    fr = rpm / 60.0

    generated_x, generated_y, meta_rows = [], [], []
    with torch.no_grad():
        for class_name in CLASS_NAMES[1:]:
            label = int(LABEL_MAP[class_name])
            logger.info("Generating %d MHTA-DDPM samples for %s", samples_per_class, class_name)
            left = samples_per_class
            start = 0
            progress = tqdm(total=samples_per_class, desc=class_name, leave=False)
            while left > 0:
                bsz = min(batch_size, left)
                labels = torch.full((bsz,), label, dtype=torch.long, device=device)
                fake = sample_batch(scheduler, model, labels, signal_length, config, device)
                fake_np = remove_signal_mean(fake.squeeze(1).cpu().numpy().astype(np.float32))
                generated_x.append(fake_np)
                generated_y.append(np.full(bsz, label, dtype=np.int64))
                for i in range(bsz):
                    meta_rows.append(
                        {
                            "class_name": class_name,
                            "label": label,
                            "source": "omc_tf_mhta_ddpm_v1",
                            "generated_index": start + i,
                            "rpm": rpm,
                            "fr": fr,
                            "checkpoint": Path(config["ckpt"]).as_posix(),
                            "sampler": str(config["sampler"]).lower(),
                            "num_inference_steps": int(config["num_inference_steps"]),
                            "eta": float(config["eta"]),
                        }
                    )
                start += bsz
                left -= bsz
                progress.update(bsz)
            progress.close()

    if generated_x:
        x_gen = np.concatenate(generated_x, axis=0).astype(np.float32)
        y_gen = np.concatenate(generated_y, axis=0).astype(np.int64)
    else:
        x_gen = np.empty((0, x_train.shape[1]), dtype=np.float32)
        y_gen = np.empty((0,), dtype=np.int64)

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
    logger.info("Saved MHTA-DDPM generated dataset to %s", out_dir)


if __name__ == "__main__":
    main()
