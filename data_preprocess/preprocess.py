from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.signal_utils import (
    CLASS_NAMES,
    DEFAULT_ORDER_TEMPLATES,
    LABEL_MAP,
    iter_window_starts,
    order_template_for_labels,
    parse_rpm_fr,
    resample_signal,
    warn_short_file,
    zscore,
)

CONFIG = {
    # This directory also contains older synthetic CSVs. Only files whose names
    # start with "zd" are used below.
    "raw_dir": Path("raw_data/guss_output_fault_signals_fs24000_rpm740"),
    "out_dir": Path("processed/mhta_base_dataset"),
    "file_prefix": "zd",
    # The zd-prefixed recordings are still acquired at 27000 Hz even though the
    # parent directory name contains "fs24000".
    "fs_in": 27000,
    "fs_out": 2048,
    "window": 2048,
    "overlap": 0.5,
    "seed": 42,
    "train_counts": {
        "normal": 50,
        "unbalance": 150,
        "misalignment": 150,
        "crack": 150,
        "looseness": 150,
    },
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("preprocess")


def read_zd_z_axis_g_signal(csv_path: Path) -> np.ndarray:
    """Read z-axis acceleration from a zd CSV and convert mg to g."""
    try:
        # zd files have a leading comment line before the actual CSV header.
        df = pd.read_csv(csv_path, comment="#", skip_blank_lines=True, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Failed to read zd CSV {csv_path}: {exc}") from exc

    lower_to_col = {str(col).strip().lower(): col for col in df.columns}
    z_col = lower_to_col.get("acc_z[mg]")
    if z_col is None:
        raise ValueError(f"Missing acc_z[mg] column in zd CSV: {csv_path}")

    z_mg = (
        pd.to_numeric(df[z_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=np.float32)
    )
    if z_mg.size == 0:
        raise ValueError(f"No valid acc_z[mg] values found in zd CSV: {csv_path}")
    return (z_mg / 1000.0).astype(np.float32)


def collect_windows(
    raw_dir: Path,
    file_prefix: str,
    fs_in: int,
    fs_out: int,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = []
    signals = []
    labels = []
    for class_name in CLASS_NAMES:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        csv_files = sorted(path for path in class_dir.rglob("*.csv") if path.name.lower().startswith(file_prefix.lower()))
        if not csv_files:
            raise FileNotFoundError(f"No {file_prefix}*.csv files found under {class_dir}")
        for csv_path in tqdm(csv_files, desc=f"Read {class_name}"):
            raw = read_zd_z_axis_g_signal(csv_path)
            # zd files are z-axis recordings in mg. They are converted to g
            # above and remain 27000 Hz recordings before resampling.
            resampled = resample_signal(raw, fs_in, fs_out)
            resampled = resampled - np.mean(resampled)
            starts = iter_window_starts(len(resampled), window, stride)
            if len(starts) == 0:
                warn_short_file(csv_path, len(resampled), window)
                continue
            rpm, fr = parse_rpm_fr(csv_path)
            for start in starts:
                # 50% overlap increases small-sample coverage while keeping
                # adjacent windows different enough for validation/testing.
                signals.append(resampled[start : start + window].astype(np.float32))
                labels.append(LABEL_MAP[class_name])
                rows.append(
                    {
                        "class_name": class_name,
                        "label": LABEL_MAP[class_name],
                        "source_csv": csv_path.as_posix(),
                        "source_file": csv_path.name,
                        "axis": "z",
                        "unit": "g",
                        "fs_in": int(fs_in),
                        "start": int(start),
                        "end": int(start + window),
                        "rpm": float(rpm),
                        "fr": float(fr),
                    }
                )
    if not signals:
        raise RuntimeError(f"No windows were produced from {raw_dir}")
    return np.stack(signals).astype(np.float32), np.asarray(labels, dtype=np.int64), pd.DataFrame(rows)


def split_indices(labels: np.ndarray, train_counts: dict[str, int], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for class_name in CLASS_NAMES:
        label = LABEL_MAP[class_name]
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        need = int(train_counts[class_name])
        if len(idx) < need:
            raise ValueError(f"{class_name} has only {len(idx)} windows, but train requires {need}.")
        train_idx.extend(idx[:need].tolist())
        remaining = idx[need:]
        half = len(remaining) // 2
        val_idx.extend(remaining[:half].tolist())
        test_idx.extend(remaining[half:].tolist())
    for arr in (train_idx, val_idx, test_idx):
        rng.shuffle(arr)
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def save_split(path: Path, x: np.ndarray, y: np.ndarray, meta: pd.DataFrame, class_names: list[str]) -> None:
    rpm = meta["rpm"].to_numpy(dtype=np.float32)
    fr = meta["fr"].to_numpy(dtype=np.float32)
    order_templates = order_template_for_labels(y)
    np.savez_compressed(
        path,
        X=x.astype(np.float32),
        y=y.astype(np.int64),
        class_names=np.asarray(class_names),
        rpm=rpm,
        fr=fr,
        order_templates=order_templates,
    )


def print_counts(logger: logging.Logger, name: str, labels: np.ndarray) -> None:
    counter = Counter(labels.tolist())
    logger.info("%s windows: %d", name, len(labels))
    for class_name in CLASS_NAMES:
        logger.info("  %-12s %d", class_name, counter.get(LABEL_MAP[class_name], 0))


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    raw_dir = config["raw_dir"]
    out_dir = config["out_dir"]
    file_prefix = config["file_prefix"]
    fs_in = config["fs_in"]
    fs_out = config["fs_out"]
    window = config["window"]
    overlap = config["overlap"]
    seed = config["seed"]
    stride = int(round(window * (1.0 - overlap)))
    if stride <= 0:
        raise ValueError("OVERLAP must be < 1.0")
    out_dir.mkdir(parents=True, exist_ok=True)

    x_all, y_all, meta_all = collect_windows(raw_dir, file_prefix, fs_in, fs_out, window, stride)
    train_idx, val_idx, test_idx = split_indices(y_all, config["train_counts"], seed)

    mean_train = float(x_all[train_idx].mean())
    std_train = float(x_all[train_idx].std() + 1e-8)
    x_norm = zscore(x_all, mean_train, std_train)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for split, idx in splits.items():
        meta = meta_all.iloc[idx].reset_index(drop=True)
        save_split(out_dir / f"{split}.npz", x_norm[idx], y_all[idx], meta, CLASS_NAMES)
        meta.to_csv(out_dir / f"metadata_{split}.csv", index=False)
        print_counts(logger, split, y_all[idx])

    scaler = {
        "mean_train": mean_train,
        "std_train": std_train,
        "normalization": "global_zscore_train_only",
        "file_prefix": file_prefix,
        "axis": "z",
        "raw_unit": "mg",
        "processed_unit_before_normalization": "g",
        "fs_in": fs_in,
        "fs_out": fs_out,
        "window": window,
        "stride": stride,
        "overlap": overlap,
        "class_names": CLASS_NAMES,
        "label_map": LABEL_MAP,
        "order_templates": DEFAULT_ORDER_TEMPLATES,
    }
    (out_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    logger.info("Saved processed dataset to %s", out_dir)


if __name__ == "__main__":
    main()
