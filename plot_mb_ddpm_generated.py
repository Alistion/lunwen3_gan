from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.signal_utils import CLASS_NAMES
from utils.run_paths import resolve_existing_run_dir

CONFIG = {
    "generated_npz": Path("processed/mb_augmented_dataset/generated_only.npz"),
    "run_root": Path("runs/mb_ddpm_lunwen3_v1"),
    "run_id": None,
    "fs": 2048,
    "samples_per_class": 3,
    "max_freq": 300.0,
    "seed": 42,
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("plot_mb_ddpm_generated")


def load_generated(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing generated npz: {npz_path}. Please run python generate_mb_ddpm.py first.")
    data = np.load(npz_path, allow_pickle=True)
    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    class_names = [str(name) for name in data["class_names"].tolist()] if "class_names" in data.files else list(CLASS_NAMES)
    return x, y, class_names


def amplitude_spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    signal = signal - np.mean(signal)
    window = np.hanning(signal.shape[-1]).astype(np.float32)
    spec = np.fft.rfft(signal * window)
    freq = np.fft.rfftfreq(signal.shape[-1], d=1.0 / float(fs))
    amp = (2.0 / (np.sum(window) + 1e-8)) * np.abs(spec)
    return freq.astype(np.float32), amp.astype(np.float32)


def choose_indices(indices: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= count:
        return indices
    return np.sort(rng.choice(indices, size=count, replace=False))


def plot_class_grid(
    x: np.ndarray,
    selected: np.ndarray,
    label: int,
    class_name: str,
    out_dir: Path,
    fs: int,
    max_freq: float,
) -> None:
    if len(selected) == 0:
        return
    t = np.arange(x.shape[1], dtype=np.float32) / float(fs)
    fig, axes = plt.subplots(len(selected), 2, figsize=(14, 3.5 * len(selected)), squeeze=False)
    for row, idx in enumerate(selected):
        sample = x[idx]
        axes[row, 0].plot(t, sample, linewidth=0.9, color="tab:blue")
        axes[row, 0].set_title(f"sample {int(idx)} - time domain")
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 0].grid(alpha=0.25)

        freq, amp = amplitude_spectrum(sample, fs)
        mask = freq <= float(max_freq)
        axes[row, 1].plot(freq[mask], amp[mask], linewidth=0.9, color="tab:orange")
        axes[row, 1].set_title(f"sample {int(idx)} - frequency domain")
        axes[row, 1].set_xlabel("Frequency (Hz)")
        axes[row, 1].set_ylabel("Amplitude")
        axes[row, 1].grid(alpha=0.25)
    fig.suptitle(f"MB-DDPM generated {class_name}", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / f"{label}_{class_name}_time_frequency_grid.png", dpi=220)
    plt.close(fig)


def save_selected_samples_csv(
    x: np.ndarray,
    selected_by_class: dict[int, np.ndarray],
    class_names: list[str],
    fs: int,
    out_path: Path,
) -> None:
    export_columns: dict[str, np.ndarray] = {}
    signal_length = x.shape[1]
    time_s = np.arange(signal_length, dtype=np.float32) / float(fs)

    for label, selected in selected_by_class.items():
        if len(selected) == 0:
            continue
        class_name = class_names[label]
        export_columns[f"{class_name}_time_s"] = time_s

        spectra = []
        freq_hz = None
        for idx in selected:
            export_columns[f"{class_name}_sample{int(idx)}_generated_waveform"] = x[idx].astype(np.float32)
            freq_hz, amp = amplitude_spectrum(x[idx], fs)
            spectra.append(amp)

        freq_col = np.full(signal_length, np.nan, dtype=np.float32)
        freq_col[: len(freq_hz)] = freq_hz
        export_columns[f"{class_name}_freq_hz"] = freq_col

        for idx, amp in zip(selected, spectra):
            amp_col = np.full(signal_length, np.nan, dtype=np.float32)
            amp_col[: len(amp)] = amp
            export_columns[f"{class_name}_sample{int(idx)}_generated_fft_mag"] = amp_col

    if export_columns:
        pd.DataFrame(export_columns).to_csv(out_path, index=False)


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    run_dir = resolve_existing_run_dir(Path(config["run_root"]), config.get("run_id"))
    out_dir = run_dir / "generated_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, class_names = load_generated(Path(config["generated_npz"]))
    rng = np.random.default_rng(int(config["seed"]))
    selected_by_class = {
        label: choose_indices(np.flatnonzero(y == label), int(config["samples_per_class"]), rng)
        for label in range(len(class_names))
    }
    for label, class_name in enumerate(class_names):
        plot_class_grid(
            x=x,
            selected=selected_by_class[label],
            label=label,
            class_name=class_name,
            out_dir=out_dir,
            fs=int(config["fs"]),
            max_freq=float(config["max_freq"]),
        )
    save_selected_samples_csv(
        x=x,
        selected_by_class=selected_by_class,
        class_names=class_names,
        fs=int(config["fs"]),
        out_path=out_dir / "generated_preview_samples_data.csv",
    )
    logger.info("Saved MB-DDPM preview plots to %s", out_dir)


if __name__ == "__main__":
    main()
