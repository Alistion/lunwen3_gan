from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.signal_utils import CLASS_NAMES, LABEL_MAP

CONFIG = {
    "raw_processed_dir": Path("processed/omc_tf_gan_dataset"),
    "augmented_dir": Path("processed/omc_tf_gan_dataset_augmented"),
    "npz_name": "generated_only.npz",
    "out_dir": Path("runs/omc_tf_gan_v2/augmented_plots"),
    "fs": 2048,
    "rpm": 740,
    "max_freq": 200,
    "samples_per_class": 5,
    "seed": 42,
}


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data: {path}")
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["X"], dtype=np.float32), np.asarray(data["y"], dtype=np.int64)


def load_real(config: dict) -> tuple[np.ndarray, np.ndarray]:
    return load_npz(config["raw_processed_dir"] / "train.npz")


def load_augmented(config: dict) -> tuple[np.ndarray, np.ndarray]:
    npz_path = config["augmented_dir"] / config["npz_name"]
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing augmented data: {npz_path}. Please run python generate_augmented.py first.")
    return load_npz(npz_path)


def spectrum(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    spec = np.abs(np.fft.rfft(x, axis=1))
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / float(fs))
    return freqs, spec


def plot_class_samples(class_name: str, x_class: np.ndarray, out_path: Path, config: dict, rng: np.random.Generator) -> None:
    n = min(int(config["samples_per_class"]), len(x_class))
    if n == 0:
        return
    idx = rng.choice(len(x_class), size=n, replace=False)
    samples = x_class[idx]
    fs = int(config["fs"])
    max_freq = float(config["max_freq"])
    t = np.arange(samples.shape[1]) / float(fs)
    freqs, spec = spectrum(samples, fs)
    freq_mask = freqs <= max_freq
    fr = float(config["rpm"]) / 60.0

    fig, axes = plt.subplots(n, 2, figsize=(12, 2.4 * n))
    axes = np.asarray(axes).reshape(n, 2)
    for i in range(n):
        axes[i, 0].plot(t, samples[i], linewidth=0.8)
        axes[i, 0].set_title(f"{class_name} augmented #{i + 1} time")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(alpha=0.25)

        axes[i, 1].plot(freqs[freq_mask], spec[i, freq_mask], linewidth=0.8)
        for mul, name in [(1.0, "1X"), (2.0, "2X"), (3.0, "3X")]:
            axes[i, 1].axvline(fr * mul, color="tab:red", linestyle="--", linewidth=0.8)
            axes[i, 1].text(fr * mul, axes[i, 1].get_ylim()[1] * 0.9, name, rotation=90, va="top", ha="right", fontsize=8)
        axes[i, 1].set_xlim(0, max_freq)
        axes[i, 1].set_title(f"{class_name} augmented #{i + 1} spectrum")
        axes[i, 1].set_xlabel("Frequency (Hz)")
        axes[i, 1].set_ylabel("Magnitude")
        axes[i, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def sample_rows(x_class: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x_class) == 0:
        return np.empty((0, 0), dtype=np.float32)
    count = min(n, len(x_class))
    idx = rng.choice(len(x_class), size=count, replace=False)
    return x_class[idx]


def plot_real_fake_class_comparison(
    class_name: str,
    real_class: np.ndarray,
    fake_class: np.ndarray,
    out_path: Path,
    config: dict,
    rng: np.random.Generator,
) -> None:
    n = int(config["samples_per_class"])
    real_samples = sample_rows(real_class, n, rng)
    fake_samples = sample_rows(fake_class, n, rng)
    if len(real_samples) == 0 and len(fake_samples) == 0:
        return

    fs = int(config["fs"])
    max_freq = float(config["max_freq"])
    fr = float(config["rpm"]) / 60.0
    sample_len = real_samples.shape[1] if len(real_samples) else fake_samples.shape[1]
    t = np.arange(sample_len) / float(fs)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    panels = [
        (axes[0, 0], real_samples, f"{class_name} real time", "time"),
        (axes[0, 1], fake_samples, f"{class_name} fake time", "time"),
        (axes[1, 0], real_samples, f"{class_name} real spectrum", "spectrum"),
        (axes[1, 1], fake_samples, f"{class_name} fake spectrum", "spectrum"),
    ]
    for ax, samples, title, mode in panels:
        if len(samples) == 0:
            ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.grid(alpha=0.25)
            continue
        if mode == "time":
            for row in samples:
                ax.plot(t, row, linewidth=0.8, alpha=0.85)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
        else:
            freqs, spec = spectrum(samples, fs)
            mask = freqs <= max_freq
            for row in spec:
                ax.plot(freqs[mask], row[mask], linewidth=0.8, alpha=0.85)
            for mul, name in [(1.0, "1X"), (2.0, "2X"), (3.0, "3X")]:
                ax.axvline(fr * mul, color="tab:red", linestyle="--", linewidth=0.8)
                ax.text(fr * mul, ax.get_ylim()[1] * 0.9, name, rotation=90, va="top", ha="right", fontsize=8)
            ax.set_xlim(0, max_freq)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_mean_spectrum(x: np.ndarray, y: np.ndarray, out_path: Path, config: dict) -> None:
    fs = int(config["fs"])
    max_freq = float(config["max_freq"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for class_name in CLASS_NAMES:
        label = LABEL_MAP[class_name]
        x_class = x[y == label]
        if len(x_class) == 0:
            continue
        freqs, spec = spectrum(x_class, fs)
        mask = freqs <= max_freq
        ax.plot(freqs[mask], spec[:, mask].mean(axis=0), linewidth=1.0, label=class_name)
    fr = float(config["rpm"]) / 60.0
    for mul, name in [(1.0, "1X"), (2.0, "2X"), (3.0, "3X")]:
        ax.axvline(fr * mul, color="tab:red", linestyle="--", linewidth=0.8)
        ax.text(fr * mul, ax.get_ylim()[1] * 0.9, name, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlim(0, max_freq)
    ax.set_title("Augmented data mean spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_real_fake_mean_spectrum(x_real: np.ndarray, y_real: np.ndarray, x_fake: np.ndarray, y_fake: np.ndarray, out_path: Path, config: dict) -> None:
    fs = int(config["fs"])
    max_freq = float(config["max_freq"])
    fig, axes = plt.subplots(len(CLASS_NAMES), 1, figsize=(10, 2.4 * len(CLASS_NAMES)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, class_name in zip(axes, CLASS_NAMES):
        label = LABEL_MAP[class_name]
        real_class = x_real[y_real == label]
        fake_class = x_fake[y_fake == label]
        if len(real_class):
            freqs, spec = spectrum(real_class, fs)
            mask = freqs <= max_freq
            ax.plot(freqs[mask], spec[:, mask].mean(axis=0), linewidth=1.0, label="real")
        if len(fake_class):
            freqs, spec = spectrum(fake_class, fs)
            mask = freqs <= max_freq
            ax.plot(freqs[mask], spec[:, mask].mean(axis=0), linewidth=1.0, label="fake")
        ax.set_title(class_name)
        ax.set_ylabel("Magnitude")
        ax.grid(alpha=0.25)
    fr = float(config["rpm"]) / 60.0
    for ax in axes:
        for mul, name in [(1.0, "1X"), (2.0, "2X"), (3.0, "3X")]:
            ax.axvline(fr * mul, color="tab:red", linestyle="--", linewidth=0.8)
            ax.text(fr * mul, ax.get_ylim()[1] * 0.9, name, rotation=90, va="top", ha="right", fontsize=8)
        ax.set_xlim(0, max_freq)
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def main(config: dict = CONFIG) -> None:
    x_real, y_real = load_real(config)
    x, y = load_augmented(config)
    out_dir = config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))

    for class_name in CLASS_NAMES:
        label = LABEL_MAP[class_name]
        x_class = x[y == label]
        real_class = x_real[y_real == label]
        plot_class_samples(class_name, x_class, out_dir / f"{class_name}_augmented_waveform_spectrum.png", config, rng)
        plot_real_fake_class_comparison(class_name, real_class, x_class, out_dir / f"{class_name}_real_vs_fake_waveform_spectrum.png", config, rng)

    plot_mean_spectrum(x, y, out_dir / "augmented_mean_spectrum_each_class.png", config)
    plot_real_fake_mean_spectrum(x_real, y_real, x, y, out_dir / "real_vs_fake_mean_spectrum_each_class.png", config)
    print(f"Saved augmented waveform and spectrum plots to: {out_dir}")


if __name__ == "__main__":
    main()
