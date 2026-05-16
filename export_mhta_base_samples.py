from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.signal_utils import CLASS_NAMES, LABEL_MAP


CONFIG = {
    "data_dir": Path("processed/mhta_base_dataset"),
    "out_dir": Path("runs/mhta_base_dataset_preview"),
    "fs": 2048,
    "samples_per_class": 5,
    "skip_classes": {"normal"},
    "seed": 42,
    "denormalize_to_g": True,
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("export_mhta_base_samples")


def load_all_splits(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for split in ("train", "val", "test"):
        npz_path = data_dir / f"{split}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        xs.append(np.asarray(data["X"], dtype=np.float32))
        ys.append(np.asarray(data["y"], dtype=np.int64))
    if not xs:
        raise FileNotFoundError(
            f"No train.npz / val.npz / test.npz found under {data_dir}. "
            "Please run python data_preprocess/preprocess.py first."
        )
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def load_scaler(data_dir: Path) -> tuple[float, float]:
    scaler_path = data_dir / "scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler.json under {data_dir}")
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    return float(scaler["mean_train"]), float(scaler["std_train"])


def restore_g_unit(x: np.ndarray, mean_train: float, std_train: float) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) * float(std_train) + float(mean_train)).astype(np.float32)


def amplitude_spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    signal = signal - np.mean(signal)
    window = np.hanning(signal.shape[-1]).astype(np.float32)
    spec = np.fft.rfft(signal * window)
    freq = np.fft.rfftfreq(signal.shape[-1], d=1.0 / float(fs))
    amp = (2.0 / (np.sum(window) + 1e-8)) * np.abs(spec)
    return freq.astype(np.float32), amp.astype(np.float32)


def choose_indices(indices: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) < count:
        raise ValueError(f"Need at least {count} samples, but only found {len(indices)}.")
    return np.sort(rng.choice(indices, size=count, replace=False))


def save_class_csv(class_name: str, samples: np.ndarray, fs: int, out_dir: Path) -> None:
    n_samples, signal_length = samples.shape
    time_s = np.arange(signal_length, dtype=np.float32) / float(fs)

    freq_hz, first_amp = amplitude_spectrum(samples[0], fs)
    amplitudes = [first_amp]
    for i in range(1, n_samples):
        _, amp = amplitude_spectrum(samples[i], fs)
        amplitudes.append(amp)

    # Time-domain vectors have length 2048, while rFFT vectors have length 1025.
    # Keep the requested single-CSV layout by padding the shorter frequency
    # columns with NaN after the valid frequency bins.
    freq_col = np.full(signal_length, np.nan, dtype=np.float32)
    freq_col[: len(freq_hz)] = freq_hz
    amp_cols = []
    for amp in amplitudes:
        padded = np.full(signal_length, np.nan, dtype=np.float32)
        padded[: len(amp)] = amp
        amp_cols.append(padded)

    payload: dict[str, np.ndarray] = {"time_s": time_s}
    for i, sample in enumerate(samples, start=1):
        payload[f"time_sample_{i}_g"] = sample.astype(np.float32)
    payload["freq_hz"] = freq_col
    for i, amp in enumerate(amp_cols, start=1):
        payload[f"freq_amp_sample_{i}_g"] = amp

    pd.DataFrame(payload).to_csv(out_dir / f"{class_name}_samples.csv", index=False)


def save_time_frequency_grid(class_name: str, samples: np.ndarray, fs: int, out_dir: Path) -> None:
    time_s = np.arange(samples.shape[1], dtype=np.float32) / float(fs)
    fig, axes = plt.subplots(
        nrows=samples.shape[0],
        ncols=2,
        figsize=(14, 3.5 * samples.shape[0]),
        squeeze=False,
    )

    for row, sample in enumerate(samples, start=1):
        time_ax = axes[row - 1, 0]
        freq_ax = axes[row - 1, 1]

        time_ax.plot(time_s, sample, linewidth=0.9, color="tab:blue")
        time_ax.set_title(f"sample {row} - time domain")
        time_ax.set_xlabel("Time (s)")
        time_ax.set_ylabel("Acceleration (g)")
        time_ax.grid(alpha=0.25)

        freq_hz, amp = amplitude_spectrum(sample, fs)
        freq_ax.plot(freq_hz, amp, linewidth=0.9, color="tab:orange")
        freq_ax.set_title(f"sample {row} - frequency domain")
        freq_ax.set_xlabel("Frequency (Hz)")
        freq_ax.set_ylabel("Amplitude (g)")
        freq_ax.grid(alpha=0.25)

    fig.suptitle(f"{class_name} samples", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / f"{class_name}_time_frequency_grid.png", dpi=220)
    plt.close(fig)


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    data_dir = Path(config["data_dir"])
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    x_all, y_all = load_all_splits(data_dir)
    if bool(config["denormalize_to_g"]):
        mean_train, std_train = load_scaler(data_dir)
        x_all = restore_g_unit(x_all, mean_train, std_train)

    rng = np.random.default_rng(int(config["seed"]))
    sample_count = int(config["samples_per_class"])
    skip_classes = set(config["skip_classes"])
    fs = int(config["fs"])

    for class_name in CLASS_NAMES:
        if class_name in skip_classes:
            continue
        label = LABEL_MAP[class_name]
        class_indices = np.flatnonzero(y_all == label)
        selected_indices = choose_indices(class_indices, sample_count, rng)
        selected_samples = x_all[selected_indices]

        save_class_csv(class_name, selected_samples, fs, out_dir)
        save_time_frequency_grid(class_name, selected_samples, fs, out_dir)
        logger.info("Saved %s samples from indices %s", class_name, selected_indices.tolist())

    logger.info("Saved CSV files and plots to %s", out_dir)


if __name__ == "__main__":
    main()
