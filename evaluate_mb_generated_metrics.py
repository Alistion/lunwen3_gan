from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.signal_utils import CLASS_NAMES, LABEL_MAP
from utils.run_paths import resolve_existing_run_dir


CONFIG = {
    "real_npz": Path("processed/mhta_base_dataset/train.npz"),
    "generated_npz": Path("processed/mb_augmented_dataset/generated_only.npz"),
    "run_root": Path("runs/mb_ddpm_lunwen3_v1"),
    "run_id": None,
    "fs": 2048,
    "max_freq": 300.0,
    "fd_downsample": 256,
    "max_pairs_per_class": 100,
    "plot_max_freq": 300.0,
    "plot_samples_per_class": 3,
    "seed": 42,
    "eps": 1e-8,
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("evaluate_mb_generated_metrics")


def load_xy(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return np.asarray(data["X"], dtype=np.float32), np.asarray(data["y"], dtype=np.int64)


def rmse(real: np.ndarray, generated: np.ndarray, eps: float) -> float:
    return float(np.sqrt(np.mean(np.square(real - generated)) + eps))


def psnr(real: np.ndarray, generated: np.ndarray, eps: float) -> float:
    mse = float(np.mean(np.square(real - generated)))
    data_range = float(np.max(real) - np.min(real))
    if data_range <= eps:
        data_range = float(np.max(np.abs(real)))
    data_range = max(data_range, eps)
    return float(10.0 * np.log10((data_range * data_range + eps) / (mse + eps)))


def fscs(real: np.ndarray, generated: np.ndarray, fs: int, max_freq: float | None, eps: float) -> float:
    real_spec = np.abs(np.fft.rfft(real))
    gen_spec = np.abs(np.fft.rfft(generated))
    freqs = np.fft.rfftfreq(real.shape[-1], d=1.0 / float(fs))

    mask = np.ones_like(freqs, dtype=bool)
    mask[0] = False
    if max_freq is not None and float(max_freq) > 0:
        mask &= freqs <= float(max_freq)

    a = real_spec[mask].astype(np.float64)
    b = gen_spec[mask].astype(np.float64)
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def resample_curve(signal: np.ndarray, length: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size == int(length):
        y = signal
    else:
        src_x = np.linspace(0.0, 1.0, signal.size)
        dst_x = np.linspace(0.0, 1.0, int(length))
        y = np.interp(dst_x, src_x, signal)
    x = np.linspace(0.0, 1.0, int(length), dtype=np.float64)
    return np.stack([x, y], axis=1)


def discrete_frechet_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    n = curve_a.shape[0]
    m = curve_b.shape[0]
    distances = np.linalg.norm(curve_a[:, None, :] - curve_b[None, :, :], axis=2)
    ca = np.empty((n, m), dtype=np.float64)

    ca[0, 0] = distances[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], distances[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], distances[0, j])
    for i in range(1, n):
        prev_row = ca[i - 1]
        row = ca[i]
        for j in range(1, m):
            row[j] = max(min(prev_row[j], prev_row[j - 1], row[j - 1]), distances[i, j])
    return float(ca[-1, -1])


def frechet_signal_distance(real: np.ndarray, generated: np.ndarray, downsample: int) -> float:
    curve_real = resample_curve(real, int(downsample))
    curve_gen = resample_curve(generated, int(downsample))
    return discrete_frechet_distance(curve_real, curve_gen)


def amplitude_spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    signal = signal - np.mean(signal)
    window = np.hanning(signal.shape[-1]).astype(np.float32)
    spec = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(signal.shape[-1], d=1.0 / float(fs))
    amp = (2.0 / (np.sum(window) + 1e-8)) * np.abs(spec)
    return freqs.astype(np.float32), amp.astype(np.float32)


def plot_one_class_comparison(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    class_name: str,
    label: int,
    real_indices: np.ndarray,
    generated_indices: np.ndarray,
    out_dir: Path,
    config: dict,
) -> None:
    fs = int(config["fs"])
    max_freq = float(config["plot_max_freq"])
    t = np.arange(real_samples.shape[-1], dtype=np.float32) / float(fs)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    for i, (real, real_index) in enumerate(zip(real_samples, real_indices)):
        axes[0].plot(t, real, color="tab:blue", linewidth=0.85, alpha=0.45 + 0.15 * i, label=f"train real #{int(real_index)}")
    for i, (generated, generated_index) in enumerate(zip(generated_samples, generated_indices)):
        axes[0].plot(
            t,
            generated,
            color="tab:orange",
            linewidth=0.85,
            alpha=0.45 + 0.15 * i,
            label=f"MB-DDPM generated #{int(generated_index)}",
        )
    axes[0].set_title(f"{class_name} time domain")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    for i, (real, real_index) in enumerate(zip(real_samples, real_indices)):
        real_freqs, real_amp = amplitude_spectrum(real, fs)
        real_mask = real_freqs <= max_freq
        axes[1].plot(
            real_freqs[real_mask],
            real_amp[real_mask],
            color="tab:blue",
            linewidth=0.85,
            alpha=0.45 + 0.15 * i,
            label=f"train real #{int(real_index)}",
        )
    for i, (generated, generated_index) in enumerate(zip(generated_samples, generated_indices)):
        gen_freqs, gen_amp = amplitude_spectrum(generated, fs)
        gen_mask = gen_freqs <= max_freq
        axes[1].plot(
            gen_freqs[gen_mask],
            gen_amp[gen_mask],
            color="tab:orange",
            linewidth=0.85,
            alpha=0.45 + 0.15 * i,
            label=f"MB-DDPM generated #{int(generated_index)}",
        )
    axes[1].set_title(f"{class_name} frequency domain")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_{class_name}_real_vs_mb_time_freq.png", dpi=220)
    plt.close(fig)


def save_class_sample_plots(
    x_real: np.ndarray,
    y_real: np.ndarray,
    x_gen: np.ndarray,
    y_gen: np.ndarray,
    out_dir: Path,
    config: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    plot_dir = out_dir / "sample_time_freq"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for class_name in CLASS_NAMES:
        label = int(LABEL_MAP[class_name])
        real_indices = np.where(y_real == label)[0]
        gen_indices = np.where(y_gen == label)[0]
        if len(real_indices) == 0 or len(gen_indices) == 0:
            continue
        sample_count = int(config["plot_samples_per_class"])
        real_selected = rng.choice(real_indices, size=min(sample_count, len(real_indices)), replace=False)
        gen_selected = rng.choice(gen_indices, size=min(sample_count, len(gen_indices)), replace=False)
        plot_one_class_comparison(
            real_samples=x_real[real_selected],
            generated_samples=x_gen[gen_selected],
            class_name=class_name,
            label=label,
            real_indices=real_selected,
            generated_indices=gen_selected,
            out_dir=plot_dir,
            config=config,
        )
        plot_path = (plot_dir / f"{label}_{class_name}_real_vs_mb_time_freq.png").as_posix()
        max_rows = max(len(real_selected), len(gen_selected))
        for i in range(max_rows):
            rows.append(
                {
                    "class_name": class_name,
                    "label": label,
                    "sample_slot": i,
                    "real_global_index": int(real_selected[i]) if i < len(real_selected) else -1,
                    "generated_global_index": int(gen_selected[i]) if i < len(gen_selected) else -1,
                    "plot_path": plot_path,
                }
            )
    return pd.DataFrame(rows)


def paired_indices(real_n: int, gen_n: int, max_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = min(real_n, gen_n, int(max_pairs))
    real_idx = rng.choice(real_n, size=n, replace=real_n < n)
    gen_idx = rng.choice(gen_n, size=n, replace=gen_n < n)
    return real_idx, gen_idx


def evaluate_pairs_for_class(
    real_c: np.ndarray,
    gen_c: np.ndarray,
    class_name: str,
    label: int,
    config: dict,
    rng: np.random.Generator,
) -> list[dict]:
    real_idx, gen_idx = paired_indices(len(real_c), len(gen_c), int(config["max_pairs_per_class"]), rng)
    rows = []
    for pair_id, (ri, gi) in enumerate(zip(real_idx, gen_idx)):
        real = real_c[int(ri)]
        generated = gen_c[int(gi)]
        rows.append(
            {
                "class_name": class_name,
                "label": int(label),
                "pair_id": int(pair_id),
                "real_index_in_class": int(ri),
                "generated_index_in_class": int(gi),
                "RMSE": rmse(real, generated, float(config["eps"])),
                "PSNR": psnr(real, generated, float(config["eps"])),
                "FSCS": fscs(real, generated, int(config["fs"]), config["max_freq"], float(config["eps"])),
                "FD": frechet_signal_distance(real, generated, int(config["fd_downsample"])),
            }
        )
    return rows


def summarize_by_class(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()
    grouped = pair_df.groupby(["class_name", "label"], as_index=False)
    summary = grouped.agg(
        pairs=("pair_id", "count"),
        RMSE=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        PSNR=("PSNR", "mean"),
        PSNR_std=("PSNR", "std"),
        FSCS=("FSCS", "mean"),
        FSCS_std=("FSCS", "std"),
        FD=("FD", "mean"),
        FD_std=("FD", "std"),
    )
    return summary.fillna(0.0)


def add_topsis_ci(summary: pd.DataFrame, eps: float) -> pd.DataFrame:
    if summary.empty:
        summary["CI"] = []
        return summary

    rmse_values = summary["RMSE"].to_numpy(dtype=np.float64)
    fd_values = summary["FD"].to_numpy(dtype=np.float64)
    rmse_pos = np.max(rmse_values) - rmse_values
    fd_pos = np.max(fd_values) - fd_values
    x = np.stack(
        [
            rmse_pos,
            summary["PSNR"].to_numpy(dtype=np.float64),
            summary["FSCS"].to_numpy(dtype=np.float64),
            fd_pos,
        ],
        axis=1,
    )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    v = x / np.sqrt(np.sum(np.square(x), axis=0, keepdims=True) + eps)
    ideal_best = np.max(v, axis=0)
    ideal_worst = np.min(v, axis=0)
    d_plus = np.sqrt(np.sum(np.square(v - ideal_best[None, :]), axis=1))
    d_minus = np.sqrt(np.sum(np.square(v - ideal_worst[None, :]), axis=1))

    out = summary.copy()
    out["RMSE_pos"] = rmse_pos
    out["FD_pos"] = fd_pos
    out["d_plus"] = d_plus
    out["d_minus"] = d_minus
    out["CI"] = d_minus / (d_plus + d_minus + eps)
    return out


def overall_row(summary: pd.DataFrame) -> dict:
    if summary.empty:
        return {}
    weights = summary["pairs"].to_numpy(dtype=np.float64)
    weights = weights / (np.sum(weights) + 1e-8)
    row = {
        "class_name": "overall_weighted_by_pairs",
        "label": -1,
        "pairs": int(summary["pairs"].sum()),
    }
    for key in ("RMSE", "PSNR", "FSCS", "FD", "CI"):
        row[key] = float(np.sum(summary[key].to_numpy(dtype=np.float64) * weights))
    return row


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    run_dir = resolve_existing_run_dir(Path(config["run_root"]), config.get("run_id"))
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))

    x_real, y_real = load_xy(Path(config["real_npz"]))
    x_gen, y_gen = load_xy(Path(config["generated_npz"]))
    logger.info("Loaded real X=%s, generated X=%s", x_real.shape, x_gen.shape)

    plot_df = save_class_sample_plots(x_real, y_real, x_gen, y_gen, out_dir, config, rng)
    plot_df.to_csv(out_dir / "mb_sample_plot_indices.csv", index=False)

    pair_rows = []
    for class_name in CLASS_NAMES:
        label = int(LABEL_MAP[class_name])
        real_c = x_real[y_real == label]
        gen_c = x_gen[y_gen == label]
        if len(real_c) == 0 or len(gen_c) == 0:
            logger.info("Skip %s: real_n=%d generated_n=%d", class_name, len(real_c), len(gen_c))
            continue
        logger.info("Evaluate %s: real_n=%d generated_n=%d", class_name, len(real_c), len(gen_c))
        pair_rows.extend(evaluate_pairs_for_class(real_c, gen_c, class_name, label, config, rng))

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out_dir / "mb_pair_metrics.csv", index=False)

    summary = summarize_by_class(pair_df)
    summary = add_topsis_ci(summary, float(config["eps"]))
    summary.to_csv(out_dir / "mb_generated_quality_metrics.csv", index=False)

    overall = overall_row(summary)
    if overall:
        pd.DataFrame([overall]).to_csv(out_dir / "mb_overall_metrics.csv", index=False)

    logger.info("Saved pair metrics to %s", out_dir / "mb_pair_metrics.csv")
    logger.info("Saved class metrics with CI to %s", out_dir / "mb_generated_quality_metrics.csv")
    logger.info("Saved sample time/frequency plots to %s", out_dir / "sample_time_freq")
    if overall:
        logger.info("Saved overall metrics to %s", out_dir / "mb_overall_metrics.csv")


if __name__ == "__main__":
    main()
