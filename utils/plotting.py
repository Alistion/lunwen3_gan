from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.signal_utils import CLASS_NAMES


def plot_generated_preview(samples: dict[str, np.ndarray], out_dir: Path, fs: int = 2048) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name, arr in samples.items():
        np.save(out_dir / f"{class_name}_fake.npy", arr.astype(np.float32))
        n = min(3, len(arr))
        if n == 0:
            continue
        t = np.arange(arr.shape[1]) / float(fs)
        fig, axes = plt.subplots(n, 2, figsize=(10, 2.6 * n))
        axes = np.asarray(axes).reshape(n, 2)
        for i in range(n):
            x = arr[i]
            axes[i, 0].plot(t, x, linewidth=0.8)
            axes[i, 0].set_title(f"{class_name} fake #{i + 1} time")
            axes[i, 0].set_xlabel("Time (s)")
            axes[i, 0].set_ylabel("Amplitude")
            spec = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
            mask = freqs <= 200
            axes[i, 1].plot(freqs[mask], spec[mask], linewidth=0.8)
            axes[i, 1].set_title("Spectrum 0-200 Hz")
            axes[i, 1].set_xlabel("Frequency (Hz)")
            axes[i, 1].set_ylabel("Magnitude")
        fig.tight_layout()
        fig.savefig(out_dir / f"{class_name}_preview.png", dpi=220)
        plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_training_curve(history, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="train_loss")
    ax.plot(history["epoch"], history["val_loss"], label="val_loss")
    ax2 = ax.twinx()
    ax2.plot(history["epoch"], history["val_acc"], color="tab:green", label="val_acc")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Validation accuracy")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
