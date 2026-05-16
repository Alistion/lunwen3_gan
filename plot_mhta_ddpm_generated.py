from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.signal_utils import CLASS_NAMES


CONFIG = {
    "generated_npz": Path("processed/omc_tf_mhta_ddpm_dataset_augmented/generated_only.npz"),
    "out_dir": Path("runs/omc_tf_mhta_ddpm_v1/generated_preview"),
    "fs": 2048,
    "samples_per_class": 5,
    "max_freq": 300.0,
    "seed": 42,
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("plot_mhta_ddpm_generated")


def load_generated(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing generated npz: {npz_path}. Please run python generate_mhta_ddpm.py first.")
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
    y: np.ndarray,
    label: int,
    class_name: str,
    out_dir: Path,
    fs: int,
    samples_per_class: int,
    max_freq: float,
    rng: np.random.Generator,
) -> None:
    indices = np.where(y == label)[0]
    if len(indices) == 0:
        return
    selected = choose_indices(indices, samples_per_class, rng)
    t = np.arange(x.shape[1], dtype=np.float32) / float(fs)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    for idx in selected:
        axes[0].plot(t, x[idx], linewidth=0.9, alpha=0.85, label=f"sample {idx}")
    axes[0].set_title(f"MHTA-DDPM generated {class_name} - time domain")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)

    for idx in selected:
        freq, amp = amplitude_spectrum(x[idx], fs)
        mask = freq <= float(max_freq)
        axes[1].plot(freq[mask], amp[mask], linewidth=0.9, alpha=0.85, label=f"sample {idx}")
    axes[1].set_title(f"MHTA-DDPM generated {class_name} - frequency domain")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=min(len(selected), 5), fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_{class_name}_time_freq.png", dpi=220)
    plt.close(fig)


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
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, class_names = load_generated(Path(config["generated_npz"]))
    rng = np.random.default_rng(int(config["seed"]))

    logger.info("Loaded generated samples: X=%s y=%s", x.shape, y.shape)
    for label, class_name in enumerate(class_names):
        plot_class_samples(
            x=x,
            y=y,
            label=label,
            class_name=class_name,
            out_dir=out_dir,
            fs=int(config["fs"]),
            samples_per_class=int(config["samples_per_class"]),
            max_freq=float(config["max_freq"]),
            rng=rng,
        )
    plot_overview(x, y, class_names, out_dir, int(config["fs"]), float(config["max_freq"]))
    logger.info("Saved MHTA-DDPM preview plots to %s", out_dir)


if __name__ == "__main__":
    main()
