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
    "run_root": Path("runs/wgan_v1"),
    "generated_subdir": "generated_dataset",
    "run_id": None,
    "fs": 2048,
    "samples_per_class": 5,
    "max_freq": 300.0,
    "seed": 42,
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("plot_wgan_generated")


def load_generated(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing generated npz: {npz_path}. Please run python generate_wgan.py first.")
    data = np.load(npz_path, allow_pickle=True)
    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    if "class_names" in data.files:
        class_names = [str(name) for name in data["class_names"].tolist()]
    else:
        class_names = list(CLASS_NAMES)
    return x, y, class_names


def amplitude_spectrum(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.mean(x)
    window = np.hanning(x.shape[-1]).astype(np.float32)
    spec = np.fft.rfft(x * window)
    freq = np.fft.rfftfreq(x.shape[-1], d=1.0 / float(fs))
    amp = (2.0 / np.sum(window)) * np.abs(spec)
    return freq.astype(np.float32), amp.astype(np.float32)


def choose_indices(indices: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= count:
        return indices
    return np.sort(rng.choice(indices, size=count, replace=False))


def plot_class_samples(
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

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    for idx in selected:
        axes[0].plot(t, x[idx], linewidth=0.9, alpha=0.85, label=f"sample {idx}")
    axes[0].set_title(f"WGAN generated {class_name} - time domain")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)

    for idx in selected:
        freq, amp = amplitude_spectrum(x[idx], fs)
        mask = freq <= float(max_freq)
        axes[1].plot(freq[mask], amp[mask], linewidth=0.9, alpha=0.85, label=f"sample {idx}")
    axes[1].set_title(f"WGAN generated {class_name} - frequency domain")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=min(len(selected), 5), fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_{class_name}_time_freq.png", dpi=220)
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


def plot_overview(x: np.ndarray, y: np.ndarray, class_names: list[str], out_dir: Path, fs: int, max_freq: float) -> None:
    labels = [label for label in range(len(class_names)) if np.any(y == label)]
    if not labels:
        return

    fig, axes = plt.subplots(len(labels), 2, figsize=(13, 3.2 * len(labels)))
    if len(labels) == 1:
        axes = np.asarray([axes])

    t = np.arange(x.shape[1], dtype=np.float32) / float(fs)
    for row, label in enumerate(labels):
        class_x = x[y == label]
        mean_wave = class_x.mean(axis=0)
        std_wave = class_x.std(axis=0)
        axes[row, 0].plot(t, mean_wave, color="tab:blue", linewidth=1.2)
        axes[row, 0].fill_between(t, mean_wave - std_wave, mean_wave + std_wave, color="tab:blue", alpha=0.18)
        axes[row, 0].set_title(f"{class_names[label]} mean waveform")
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 0].grid(alpha=0.25)

        spectra = []
        for sample in class_x:
            freq, amp = amplitude_spectrum(sample, fs)
            spectra.append(amp)
        spectra = np.stack(spectra, axis=0)
        mean_amp = spectra.mean(axis=0)
        std_amp = spectra.std(axis=0)
        mask = freq <= float(max_freq)
        axes[row, 1].plot(freq[mask], mean_amp[mask], color="tab:orange", linewidth=1.2)
        axes[row, 1].fill_between(
            freq[mask],
            mean_amp[mask] - std_amp[mask],
            mean_amp[mask] + std_amp[mask],
            color="tab:orange",
            alpha=0.18,
        )
        axes[row, 1].set_title(f"{class_names[label]} mean spectrum")
        axes[row, 1].set_xlabel("Frequency (Hz)")
        axes[row, 1].set_ylabel("Amplitude")
        axes[row, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "overview_mean_time_freq.png", dpi=220)
    plt.close(fig)


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    run_dir = resolve_existing_run_dir(Path(config["run_root"]), config.get("run_id"))
    out_dir = run_dir / "generated_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_npz = run_dir / str(config.get("generated_subdir", "generated_dataset")) / "generated_only.npz"
    x, y, class_names = load_generated(generated_npz)
    rng = np.random.default_rng(int(config["seed"]))

    logger.info("Loaded generated samples: X=%s y=%s", x.shape, y.shape)
    selected_by_class = {
        label: choose_indices(np.where(y == label)[0], int(config["samples_per_class"]), rng)
        for label in range(len(class_names))
    }
    for label, class_name in enumerate(class_names):
        plot_class_samples(
            x=x,
            selected=selected_by_class[label],
            label=label,
            class_name=class_name,
            out_dir=out_dir,
            fs=int(config["fs"]),
            max_freq=float(config["max_freq"]),
        )
    plot_overview(x, y, class_names, out_dir, int(config["fs"]), float(config["max_freq"]))
    save_selected_samples_csv(
        x=x,
        selected_by_class=selected_by_class,
        class_names=class_names,
        fs=int(config["fs"]),
        out_path=out_dir / "generated_preview_samples_data.csv",
    )
    logger.info("Saved WGAN preview plots to %s", out_dir)


if __name__ == "__main__":
    main()
