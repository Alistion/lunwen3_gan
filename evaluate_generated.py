from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from utils.metrics import beds_similarity, cosine_similarity_mean, kurtosis_np, order_energy_np, peak, rms, spectrum, spectrum_corr_mean
from utils.signal_utils import CLASS_NAMES, LABEL_MAP, ORDER_NAMES

CONFIG = {
    "data_dir": Path("processed/omc_tf_gan_dataset"),
    "generated_dir": Path("processed/omc_tf_gan_dataset_augmented"),
    "run_dir": Path("runs/omc_tf_gan_v2"),
    "fs": 2048,
    "rpm": 740,
    "use_tanh_output": False,
    "seed": 42,
    "env_segments": 16,
    "acf_max_lag": 512,
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("evaluate_generated")


def compute_saturation_ratio(x: np.ndarray, use_tanh_output: bool = False) -> float:
    if len(x) == 0:
        return float("nan")
    threshold = 0.98 if use_tanh_output else 3.0
    return float(np.mean(np.abs(x) > threshold))


def normalized_log_spectrum_np(x: np.ndarray, fs: int, max_freq: float = 100.0) -> np.ndarray:
    freqs, spec = spectrum(x, fs)
    mag = np.log1p(spec)
    mag = mag[:, freqs <= max_freq]
    return mag / (mag.sum(axis=1, keepdims=True) + 1e-8)


def spectrum_shape_error(real: np.ndarray, fake: np.ndarray, fs: int) -> float:
    if len(real) == 0 or len(fake) == 0:
        return float("nan")
    real_s = normalized_log_spectrum_np(real, fs).mean(axis=0)
    fake_s = normalized_log_spectrum_np(fake, fs).mean(axis=0)
    return float(np.mean((real_s - fake_s) ** 2))


def ptp_np(x: np.ndarray) -> np.ndarray:
    return np.max(x, axis=1) - np.min(x, axis=1)


def fft_peak_np(x: np.ndarray, fs: int, max_freq: float = 100.0) -> np.ndarray:
    if len(x) == 0:
        return np.asarray([], dtype=np.float32)
    freqs, spec = spectrum(x, fs)
    return spec[:, freqs <= max_freq].max(axis=1)


def amplitude_summary(x: np.ndarray, fs: int) -> np.ndarray:
    if len(x) == 0:
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
    return np.asarray(
        [
            float(np.mean(rms(x))),
            float(np.mean(ptp_np(x))),
            float(np.mean(fft_peak_np(x, fs))),
        ],
        dtype=np.float32,
    )


def amplitude_template_error(real: np.ndarray, fake: np.ndarray, fs: int) -> float:
    if len(real) == 0 or len(fake) == 0:
        return float("nan")
    real_amp = np.log1p(amplitude_summary(real, fs))
    fake_amp = np.log1p(amplitude_summary(fake, fs))
    return float(np.mean(np.abs(fake_amp - real_amp)))


def envelope_rms_np(x: np.ndarray, segments: int = 16) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, segments), dtype=np.float32)
    seg_len = x.shape[1] // int(segments)
    x_seg = x[:, : seg_len * int(segments)].reshape(len(x), int(segments), seg_len)
    return np.sqrt(np.mean(np.square(x_seg), axis=-1) + 1e-8).astype(np.float32)


def envelope_error(real: np.ndarray, fake: np.ndarray, segments: int = 16) -> float:
    if len(real) == 0 or len(fake) == 0:
        return float("nan")
    real_env = np.log1p(envelope_rms_np(real, segments).mean(axis=0))
    fake_env = np.log1p(envelope_rms_np(fake, segments).mean(axis=0))
    return float(np.mean(np.abs(fake_env - real_env)))


def temporal_energy_cv(x: np.ndarray, segments: int = 16) -> float:
    if len(x) == 0:
        return float("nan")
    env = envelope_rms_np(x, segments)
    cv = env.std(axis=1) / (env.mean(axis=1) + 1e-8)
    return float(np.mean(cv))


def acf_np(x: np.ndarray, max_lag: int = 512) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, max_lag), dtype=np.float32)
    x0 = x - x.mean(axis=1, keepdims=True)
    n = x0.shape[1]
    spec = np.fft.rfft(x0, n=2 * n, axis=1)
    acf = np.fft.irfft(spec * np.conj(spec), n=2 * n, axis=1)[:, :n]
    acf = acf / (acf[:, :1] + 1e-8)
    return acf[:, 1 : int(max_lag) + 1].astype(np.float32)


def acf_error(real: np.ndarray, fake: np.ndarray, max_lag: int = 512) -> float:
    if len(real) == 0 or len(fake) == 0:
        return float("nan")
    real_acf = acf_np(real, max_lag).mean(axis=0)
    fake_acf = acf_np(fake, max_lag).mean(axis=0)
    return float(np.mean(np.abs(fake_acf - real_acf)))


def feature_matrix(x: np.ndarray, fs: int, rpm: float) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, 12), dtype=np.float32)
    r = rms(x)
    ptp = np.max(x, axis=1) - np.min(x, axis=1)
    k = kurtosis_np(x)
    crest = np.max(np.abs(x), axis=1) / (r + 1e-8)
    order = order_energy_np(x, fs=fs, rpm=rpm)
    freqs, spec = spectrum(x, fs)
    mask = freqs <= 100
    mag = spec[:, mask] + 1e-8
    f = freqs[mask]
    centroid = (mag * f[None, :]).sum(axis=1) / mag.sum(axis=1)
    prob = mag / mag.sum(axis=1, keepdims=True)
    entropy = -(prob * np.log(prob + 1e-8)).sum(axis=1) / np.log(prob.shape[1])
    features = np.stack([r, ptp, k, crest, *[order[:, i] for i in range(order.shape[1])], centroid, entropy], axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def save_tsne(items: list[np.ndarray], labels: list[str], out_path: Path, seed: int, title: str) -> None:
    x = np.concatenate([arr for arr in items if len(arr)], axis=0)
    if len(x) < 5:
        return
    labels_arr = np.asarray(labels)
    perplexity = min(30, max(2, len(x) // 4))
    emb = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(x)
    fig, ax = plt.subplots(figsize=(9, 7))
    for name in sorted(set(labels)):
        mask = labels_arr == name
        ax.scatter(emb[mask, 0], emb[mask, 1], s=14, alpha=0.75, label=name)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    out_dir = config["run_dir"] / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = int(config["fs"])
    rpm = float(config["rpm"])
    env_segments = int(config["env_segments"])
    acf_max_lag = int(config["acf_max_lag"])

    train = np.load(config["data_dir"] / "train.npz", allow_pickle=True)
    gen = np.load(config["generated_dir"] / "generated_only.npz", allow_pickle=True)
    x_real, y_real = np.asarray(train["X"], dtype=np.float32), np.asarray(train["y"], dtype=np.int64)
    x_fake, y_fake = np.asarray(gen["X"], dtype=np.float32), np.asarray(gen["y"], dtype=np.int64)

    rows = []
    order_plot = []
    for class_name in CLASS_NAMES:
        label = LABEL_MAP[class_name]
        real_c = x_real[y_real == label]
        fake_c = x_fake[y_fake == label]
        if len(fake_c) == 0:
            continue
        real_ord = order_energy_np(real_c, fs=fs, rpm=rpm).mean(axis=0)
        fake_ord = order_energy_np(fake_c, fs=fs, rpm=rpm).mean(axis=0)
        real_amp = amplitude_summary(real_c, fs)
        fake_amp = amplitude_summary(fake_c, fs)
        order_plot.append((class_name, real_ord, fake_ord))
        rows.append(
            {
                "class_name": class_name,
                "real_n": len(real_c),
                "fake_n": len(fake_c),
                "real_rms": float(np.mean(rms(real_c))),
                "fake_rms": float(np.mean(rms(fake_c))),
                "real_ptp": float(real_amp[1]),
                "fake_ptp": float(fake_amp[1]),
                "real_fft_peak_0_100": float(real_amp[2]),
                "fake_fft_peak_0_100": float(fake_amp[2]),
                "rms_error": float(abs(fake_amp[0] - real_amp[0])),
                "ptp_error": float(abs(fake_amp[1] - real_amp[1])),
                "fft_peak_error": float(abs(fake_amp[2] - real_amp[2])),
                "amplitude_template_error": amplitude_template_error(real_c, fake_c, fs),
                "envelope_error": envelope_error(real_c, fake_c, env_segments),
                "acf_error": acf_error(real_c, fake_c, acf_max_lag),
                "real_temporal_energy_cv": temporal_energy_cv(real_c, env_segments),
                "fake_temporal_energy_cv": temporal_energy_cv(fake_c, env_segments),
                "temporal_energy_cv": temporal_energy_cv(fake_c, env_segments),
                "real_peak": float(np.mean(peak(real_c))),
                "fake_peak": float(np.mean(peak(fake_c))),
                "real_kurtosis": float(np.nanmean(kurtosis_np(real_c))),
                "fake_kurtosis": float(np.nanmean(kurtosis_np(fake_c))),
                "SC": spectrum_corr_mean(real_c, fake_c),
                "BEDS": beds_similarity(real_c, fake_c),
                "CS": cosine_similarity_mean(real_c, fake_c),
                "saturation_ratio_real": compute_saturation_ratio(real_c, config["use_tanh_output"]),
                "saturation_ratio_fake": compute_saturation_ratio(fake_c, config["use_tanh_output"]),
                "order_energy_error": float(np.mean(np.abs(real_ord - fake_ord))),
                "spectrum_shape_error": spectrum_shape_error(real_c, fake_c, fs),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "generated_quality_metrics.csv", index=False)

    if order_plot:
        fig, axes = plt.subplots(len(order_plot), 1, figsize=(9, 2.4 * len(order_plot)), sharex=True)
        axes = np.asarray(axes).reshape(-1)
        x = np.arange(len(ORDER_NAMES))
        for ax, (class_name, real_ord, fake_ord) in zip(axes, order_plot):
            ax.bar(x - 0.18, real_ord, width=0.36, label="real")
            ax.bar(x + 0.18, fake_ord, width=0.36, label="fake")
            ax.set_title(class_name)
            ax.set_ylabel("Energy ratio")
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(ORDER_NAMES)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "order_energy_comparison.png", dpi=260)
        plt.close(fig)

    fig, axes = plt.subplots(len(CLASS_NAMES), 1, figsize=(9, 2.2 * len(CLASS_NAMES)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, class_name in zip(axes, CLASS_NAMES):
        label = LABEL_MAP[class_name]
        real_c = x_real[y_real == label]
        fake_c = x_fake[y_fake == label]
        if len(real_c):
            freqs, spec = spectrum(real_c, fs)
            ax.plot(freqs[freqs <= 200], spec.mean(axis=0)[freqs <= 200], label="real", linewidth=0.9)
        if len(fake_c):
            freqs, spec = spectrum(fake_c, fs)
            ax.plot(freqs[freqs <= 200], spec.mean(axis=0)[freqs <= 200], label="fake", linewidth=0.9)
        ax.set_title(class_name)
        ax.set_ylabel("Magnitude")
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spectrum_comparison_each_class.png", dpi=260)
    plt.close(fig)

    rng = np.random.default_rng(config["seed"])
    time_items, feature_items, labels = [], [], []
    for source, x_arr, y_arr in [("real", x_real, y_real), ("fake", x_fake, y_fake)]:
        for class_name in CLASS_NAMES:
            if source == "fake" and class_name == "normal":
                continue
            label = LABEL_MAP[class_name]
            idx = np.flatnonzero(y_arr == label)
            idx = rng.choice(idx, size=min(80, len(idx)), replace=False) if len(idx) else []
            samples = x_arr[idx]
            time_items.append(samples)
            feature_items.append(feature_matrix(samples, fs=fs, rpm=rpm))
            labels.extend([f"{source} {class_name}"] * len(samples))
    save_tsne(time_items, labels, out_dir / "tsne_time_real_fake.png", config["seed"], "t-SNE Time Real vs Fake")
    save_tsne(feature_items, labels, out_dir / "tsne_feature_real_fake.png", config["seed"], "t-SNE Feature Real vs Fake")
    logger.info("Saved evaluation outputs to %s", out_dir)


if __name__ == "__main__":
    main()
