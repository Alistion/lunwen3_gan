from __future__ import annotations

import csv
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm


CLASS_NAMES = ["normal", "unbalance", "misalignment", "looseness"]
LABEL_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}

CONFIG = {
    "raw_dir": Path("raw_data/Mechanical faults in rotating machinery dataset (normal, unbalance, misalignment, looseness)"),
    "out_dir": Path("processed/mfrmd_base_data"),
    "fs_in": 25000,
    "fs_out": 2048,
    "window": 2048,
    "channel_index": 2,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "seed": 42,
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("preprocess_mfrmd")


def class_name_from_path(path: Path) -> str | None:
    name = path.parent.name.lower()
    if "normal" in name:
        return "normal"
    if "unbalance" in name:
        return "unbalance"
    if "misalignment" in name:
        return "misalignment"
    if "looseness" in name:
        return "looseness"
    return None


def read_third_channel(npy_path: Path, channel_index: int) -> np.ndarray:
    arr = np.load(npy_path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array with 4 channels in {npy_path}, got shape={arr.shape}")

    if arr.shape[0] == 4:
        signal = arr[int(channel_index), :]
    elif arr.shape[1] == 4:
        signal = arr[:, int(channel_index)]
    else:
        raise ValueError(f"Could not identify 4-channel axis in {npy_path}, got shape={arr.shape}")

    signal = np.nan_to_num(signal.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if signal.size == 0:
        raise ValueError(f"Empty channel in {npy_path}")
    return signal


def resample_signal(signal: np.ndarray, fs_in: int, fs_out: int, target_length: int) -> np.ndarray:
    if int(fs_in) == int(fs_out):
        y = np.asarray(signal, dtype=np.float32)
    else:
        gcd = math.gcd(int(fs_in), int(fs_out))
        y = resample_poly(
            np.asarray(signal, dtype=np.float32),
            up=int(fs_out) // gcd,
            down=int(fs_in) // gcd,
        ).astype(np.float32)

    target_length = int(target_length)
    if y.size > target_length:
        y = y[:target_length]
    elif y.size < target_length:
        y = np.pad(y, (0, target_length - y.size), mode="edge")
    return y.astype(np.float32)


def collect_samples(config: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    raw_dir = Path(config["raw_dir"])
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw dataset directory: {raw_dir}")

    npy_files = sorted(raw_dir.rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found under {raw_dir}")

    signals = []
    labels = []
    rows = []
    for npy_path in tqdm(npy_files, desc="Read MFRMD npy"):
        class_name = class_name_from_path(npy_path)
        if class_name is None:
            continue

        raw = read_third_channel(npy_path, int(config["channel_index"]))
        raw = raw - float(np.mean(raw))
        resampled = resample_signal(raw, int(config["fs_in"]), int(config["fs_out"]), int(config["window"]))
        label = int(LABEL_MAP[class_name])

        signals.append(resampled.astype(np.float32))
        labels.append(label)
        rows.append(
            {
                "class_name": class_name,
                "label": label,
                "source_npy": npy_path.as_posix(),
                "source_file": npy_path.name,
                "source_dir": npy_path.parent.name,
                "channel_index": int(config["channel_index"]),
                "channel_name": "third_channel",
                "fs_in": int(config["fs_in"]),
                "fs_out": int(config["fs_out"]),
                "start": 0,
                "end": int(config["window"]),
                "rpm": 0.0,
                "fr": 0.0,
            }
        )

    if not signals:
        raise RuntimeError(f"No usable MFRMD samples were produced from {raw_dir}")
    return np.stack(signals).astype(np.float32), np.asarray(labels, dtype=np.int64), rows


def split_indices(labels: np.ndarray, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    rng = np.random.default_rng(int(seed))
    train_idx, val_idx, test_idx = [], [], []
    for class_name in CLASS_NAMES:
        label = int(LABEL_MAP[class_name])
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)

        n_total = len(idx)
        n_train = int(round(n_total * float(train_ratio)))
        n_val = int(round(n_total * float(val_ratio)))
        n_train = min(n_train, n_total)
        n_val = min(n_val, n_total - n_train)

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())

    for arr in (train_idx, val_idx, test_idx):
        rng.shuffle(arr)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def zscore(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return ((x - float(mean)) / (float(std) + 1e-8)).astype(np.float32)


def save_metadata_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_split(path: Path, x: np.ndarray, y: np.ndarray, rows: list[dict]) -> None:
    rpm = np.asarray([float(row["rpm"]) for row in rows], dtype=np.float32)
    fr = np.asarray([float(row["fr"]) for row in rows], dtype=np.float32)
    np.savez_compressed(
        path,
        X=x.astype(np.float32),
        y=y.astype(np.int64),
        class_names=np.asarray(CLASS_NAMES),
        rpm=rpm,
        fr=fr,
    )


def print_counts(logger: logging.Logger, name: str, labels: np.ndarray) -> None:
    counter = Counter(labels.tolist())
    logger.info("%s samples: %d", name, len(labels))
    for class_name in CLASS_NAMES:
        logger.info("  %-12s %d", class_name, counter.get(LABEL_MAP[class_name], 0))


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    x_all, y_all, meta_all = collect_samples(config)
    train_idx, val_idx, test_idx = split_indices(
        y_all,
        float(config["train_ratio"]),
        float(config["val_ratio"]),
        float(config["test_ratio"]),
        int(config["seed"]),
    )

    mean_train = float(x_all[train_idx].mean())
    std_train = float(x_all[train_idx].std() + 1e-8)
    x_norm = zscore(x_all, mean_train, std_train)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for split, idx in splits.items():
        rows = [meta_all[int(i)] for i in idx]
        save_split(out_dir / f"{split}.npz", x_norm[idx], y_all[idx], rows)
        save_metadata_csv(out_dir / f"metadata_{split}.csv", rows)
        print_counts(logger, split, y_all[idx])

    scaler = {
        "mean_train": mean_train,
        "std_train": std_train,
        "normalization": "global_zscore_train_only",
        "raw_dir": Path(config["raw_dir"]).as_posix(),
        "selected_channel_index": int(config["channel_index"]),
        "selected_channel": "third_channel",
        "fs_in": int(config["fs_in"]),
        "fs_out": int(config["fs_out"]),
        "window": int(config["window"]),
        "split": {
            "train_ratio": float(config["train_ratio"]),
            "val_ratio": float(config["val_ratio"]),
            "test_ratio": float(config["test_ratio"]),
            "seed": int(config["seed"]),
        },
        "class_names": CLASS_NAMES,
        "label_map": LABEL_MAP,
    }
    (out_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    logger.info("Saved processed MFRMD dataset to %s", out_dir)


if __name__ == "__main__":
    main()
