from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


LABEL_MAP = {
    "normal": 0,
    "unbalance": 1,
    "misalignment": 2,
    "crack": 3,
    "looseness": 4,
}
CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
ORDER_NAMES = ["0.5X", "1X", "1.5X", "2X", "3X", "broadband"]
DEFAULT_ORDER_TEMPLATES = {
    "normal": [0.05, 0.60, 0.03, 0.08, 0.03, 0.10],
    "unbalance": [0.03, 1.00, 0.03, 0.08, 0.03, 0.08],
    "misalignment": [0.03, 0.45, 0.03, 1.00, 0.18, 0.10],
    "crack": [0.03, 0.75, 0.05, 0.55, 0.30, 0.25],
    "looseness": [0.45, 0.40, 0.35, 0.30, 0.20, 0.90],
}
DEFAULT_TARGET_MASKS = {
    "normal": [0, 1, 0, 0, 0, 0],
    "unbalance": [0, 1, 0, 0, 0, 0],
    "misalignment": [0, 1, 0, 1, 0, 0],
    "crack": [0, 1, 0, 1, 1, 0],
    "looseness": [1, 1, 1, 1, 1, 1],
}

SIGNAL_COLUMN_CANDIDATES = (
    "noisy_signal_g",
    "noisy_signal",
    "clean_fault_g",
    "signal",
    "vibration",
    "acceleration",
    "acc",
    "value",
    "amplitude",
)
TIME_LIKE_COLUMNS = {"time", "timestamp", "t", "time_s"}


def numeric_values(series: pd.Series) -> np.ndarray:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=np.float32)
    )


def read_csv_signal(csv_path: Path) -> np.ndarray:
    """Read vibration signal, preferring the last numeric non-time column."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV {csv_path}: {exc}") from exc

    lower_to_col = {str(col).strip().lower(): col for col in df.columns}
    for candidate in SIGNAL_COLUMN_CANDIDATES:
        if candidate in lower_to_col:
            values = numeric_values(df[lower_to_col[candidate]])
            if values.size:
                return values

    numeric_candidates: list[np.ndarray] = []
    for col in df.columns:
        lower_name = str(col).strip().lower()
        if lower_name in TIME_LIKE_COLUMNS:
            continue
        values = numeric_values(df[col])
        if values.size:
            numeric_candidates.append(values)
    if numeric_candidates:
        return numeric_candidates[-1]

    try:
        df_no_header = pd.read_csv(csv_path, header=None, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Failed to re-read CSV without header {csv_path}: {exc}") from exc
    for col in reversed(df_no_header.columns):
        values = numeric_values(df_no_header[col])
        if values.size:
            return values
    raise ValueError(f"No numeric signal column found in {csv_path}")


def resample_signal(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Anti-aliased resampling from raw acquisition rate to model rate."""
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if x.size == 0 or int(fs_in) == int(fs_out):
        return x.astype(np.float32, copy=False)
    gcd = math.gcd(int(fs_in), int(fs_out))
    return resample_poly(x, up=int(fs_out) // gcd, down=int(fs_in) // gcd).astype(np.float32)


def iter_window_starts(length: int, window: int, stride: int) -> range:
    if length < window:
        return range(0)
    return range(0, length - window + 1, stride)


def parse_rpm_fr(path: Path, default_rpm: float = 740.0) -> tuple[float, float]:
    name = path.name
    rpm_match = re.search(r"rpm([0-9]+(?:\.[0-9]+)?)", name, flags=re.IGNORECASE)
    fr_match = re.search(r"fr([0-9]+(?:\.[0-9]+)?)", name, flags=re.IGNORECASE)
    rpm = float(rpm_match.group(1)) if rpm_match else float(default_rpm)
    fr = float(fr_match.group(1)) if fr_match else rpm / 60.0
    return rpm, fr


def order_template_for_labels(labels: np.ndarray | list[int]) -> np.ndarray:
    templates = np.asarray([DEFAULT_ORDER_TEMPLATES[name] for name in CLASS_NAMES], dtype=np.float32)
    return templates[np.asarray(labels, dtype=np.int64)]


def target_mask_for_labels(labels: np.ndarray | list[int]) -> np.ndarray:
    masks = np.asarray([DEFAULT_TARGET_MASKS[name] for name in CLASS_NAMES], dtype=np.float32)
    return masks[np.asarray(labels, dtype=np.int64)]


def zscore(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return ((x - float(mean)) / (float(std) + 1e-8)).astype(np.float32)


def safe_copy_npz(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")
    data = np.load(src, allow_pickle=True)
    np.savez_compressed(dst, **{key: data[key] for key in data.files})


def warn_short_file(csv_path: Path, length: int, window: int) -> None:
    warnings.warn(f"{csv_path} has {length} points after resampling, shorter than window={window}; skipped.")
