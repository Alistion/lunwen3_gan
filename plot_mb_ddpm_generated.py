from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    y: np.ndarray,
    label: int,
    class_name: str,
    out_dir: Path,
    fs: int,
    samples_per_class: int,
    max_freq: float,
    rng: np.random.Generator,
) -> None:
    indices = np.flatnonzero(y == label)
    if len(indices) == 0:
        return
    selected = choose_indices(indices, samples_per_class, rng)
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


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    run_dir = resolve_existing_run_dir(Path(config["run_root"]), config.get("run_id"))
    out_dir = run_dir / "generated_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, class_names = load_generated(Path(config["generated_npz"]))
    rng = np.random.default_rng(int(config["seed"]))
    for label, class_name in enumerate(class_names):
        plot_class_grid(
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
    logger.info("Saved MB-DDPM preview plots to %s", out_dir)


if __name__ == "__main__":
    main()
