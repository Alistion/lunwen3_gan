from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.omc_tf_gan import OrderGuidedGenerator
from utils.seed import set_seed
from utils.signal_utils import CLASS_NAMES, LABEL_MAP, order_template_for_labels, safe_copy_npz

CONFIG = {
    "data_dir": Path("processed/omc_tf_gan_dataset"),
    "ckpt": Path("runs/omc_tf_gan_v2/checkpoints/best_generator.pt"),
    "out_dir": Path("processed/omc_tf_gan_dataset_augmented"),
    "target_per_class": 100,
    "latent_dim": 128,
    "fs": 2048,
    "rpm": 740,
    "seed": 42,
    "device": "cuda_if_available",
    "use_tanh_output": False,
    "use_amplitude_calibration": False,
    "amp_calibration_min_scale": 0.5,
    "amp_calibration_max_scale": 2.5,
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("generate_augmented")


def get_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def load_generator(config: dict, device: torch.device) -> tuple[OrderGuidedGenerator, int]:
    ckpt = config["ckpt"]
    if not ckpt.exists():
        raise FileNotFoundError(f"Generator checkpoint does not exist: {ckpt}. Please run python train_gan.py first.")
    payload = torch.load(ckpt, map_location=device)
    latent_dim = int(payload.get("latent_dim", config["latent_dim"]))
    base_channels = int(payload.get("base_channels_g", 128))
    model = OrderGuidedGenerator(latent_dim=latent_dim, base_channels=base_channels, use_tanh_output=config["use_tanh_output"]).to(device)
    model.load_state_dict(payload["generator_state_dict"])
    model.eval()
    return model, latent_dim


def rms_np(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=1) + 1e-8)


def class_target_rms(x_train: np.ndarray, y_train: np.ndarray) -> dict[int, float]:
    targets = {}
    for label in range(len(CLASS_NAMES)):
        x_cls = x_train[y_train == label]
        targets[label] = float(np.mean(rms_np(x_cls))) if len(x_cls) else 1.0
    return targets


def calibrate_amplitude(fake: np.ndarray, label: int, target_rms: dict[int, float], config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rms_before = rms_np(fake)
    target = np.full(len(fake), target_rms[label], dtype=np.float32)
    scale = target / (rms_before + 1e-8)
    scale = np.clip(scale, config["amp_calibration_min_scale"], config["amp_calibration_max_scale"]).astype(np.float32)
    calibrated = fake * scale[:, None]
    rms_after = rms_np(calibrated)
    return calibrated.astype(np.float32), scale, rms_before.astype(np.float32), rms_after.astype(np.float32)


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    set_seed(int(config["seed"]))
    device = get_device(config["device"])
    out_dir = config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = config["data_dir"] / "train.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train set: {train_path}. Please run python data/preprocess.py first.")
    train = np.load(train_path, allow_pickle=True)
    x_train = np.asarray(train["X"], dtype=np.float32)
    y_train = np.asarray(train["y"], dtype=np.int64)
    target_rms = class_target_rms(x_train, y_train)
    generator, latent_dim = load_generator(config, device)

    generated_x, generated_y, meta_rows = [], [], []
    batch_size = 64
    with torch.no_grad():
        for class_name in CLASS_NAMES[1:]:
            label = LABEL_MAP[class_name]
            current = int(np.sum(y_train == label))
            need = max(0, int(config["target_per_class"]) - current)
            logger.info("%s current=%d need_generate=%d", class_name, current, need)
            left = need
            while left > 0:
                bsz = min(batch_size, left)
                y = torch.full((bsz,), label, dtype=torch.long, device=device)
                z = torch.randn(bsz, latent_dim, device=device)
                rpm = torch.full((bsz,), float(config["rpm"]), device=device)
                order = torch.from_numpy(order_template_for_labels(np.full(bsz, label))).to(device)
                fake = generator(z, y, rpm, order).squeeze(1).cpu().numpy().astype(np.float32)
                if config["use_amplitude_calibration"]:
                    fake, amp_scale, rms_before, rms_after = calibrate_amplitude(fake, label, target_rms, config)
                else:
                    rms_before = rms_np(fake).astype(np.float32)
                    rms_after = rms_before.copy()
                    amp_scale = np.ones(bsz, dtype=np.float32)
                generated_x.append(fake)
                generated_y.append(np.full(bsz, label, dtype=np.int64))
                start_idx = need - left
                for i in range(bsz):
                    meta_rows.append(
                        {
                            "class_name": class_name,
                            "label": label,
                            "source": "omc_tf_gan_v2",
                            "generated_index": start_idx + i,
                            "rpm": float(config["rpm"]),
                            "fr": float(config["rpm"]) / 60.0,
                            "checkpoint": config["ckpt"].as_posix(),
                            "amp_scale": float(amp_scale[i]),
                            "rms_before": float(rms_before[i]),
                            "rms_after": float(rms_after[i]),
                            "target_rms": float(target_rms[label]),
                        }
                    )
                left -= bsz

    if generated_x:
        x_gen = np.concatenate(generated_x, axis=0)
        y_gen = np.concatenate(generated_y, axis=0)
    else:
        x_gen = np.empty((0, x_train.shape[1]), dtype=np.float32)
        y_gen = np.empty((0,), dtype=np.int64)

    np.savez_compressed(
        out_dir / "generated_only.npz",
        X=x_gen,
        y=y_gen,
        class_names=np.asarray(CLASS_NAMES),
        rpm=np.full(len(y_gen), float(config["rpm"]), dtype=np.float32),
        fr=np.full(len(y_gen), float(config["rpm"]) / 60.0, dtype=np.float32),
        order_templates=order_template_for_labels(y_gen),
    )
    np.savez_compressed(
        out_dir / "train_augmented.npz",
        X=np.concatenate([x_train, x_gen], axis=0),
        y=np.concatenate([y_train, y_gen], axis=0),
        class_names=np.asarray(CLASS_NAMES),
    )
    pd.DataFrame(meta_rows).to_csv(out_dir / "metadata_generated.csv", index=False)
    safe_copy_npz(config["data_dir"] / "val.npz", out_dir / "val.npz")
    safe_copy_npz(config["data_dir"] / "test.npz", out_dir / "test.npz")
    logger.info("Saved augmented dataset to %s", out_dir)


if __name__ == "__main__":
    main()
