from __future__ import annotations

import csv
import logging
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from generate_mhta_ddpm import get_device, load_model, sample_batch
from utils.run_paths import resolve_existing_run_dir
from utils.seed import set_seed
from utils.signal_utils import CLASS_NAMES, LABEL_MAP, order_template_for_labels


CONFIG = {
    "data_dir": Path("processed/mhta_base_dataset"),
    "run_root": Path("runs/omc_tf_mhta_ddpm_v1"),
    "run_id": None,
    "generated_root_subdir": "generated_by_epoch",
    "plot_subdir": "epoch_progress_preview",
    "epochs": [500, 1000, 3000, 5000],  # None 表示自动扫描当前 run 下所有 epoch_*.pt
    "plot_after_generate": True,
    "plot_class_name": "looseness",
    "plot_generated_index": 0,
    "plot_best_match_epoch": 5000,  # None 表示所有 epoch 都使用 plot_generated_index
    "plot_best_match_fft_weight": 0.70,
    "plot_best_match_time_weight": 0.30,
    "plot_include_real": True,
    "plot_real_split": "train.npz",
    "plot_real_index_within_class": 0,
    "plot_real_auto_clean": True,
    "plot_fs": 2048,
    "plot_max_seconds": None,
    "plot_max_freq": 300.0,
    "plot_ncols": 2,
    "plot_line_color": "red",
    "plot_real_line_color": "blue",
    "plot_dpi": 220,
    "samples_per_fault_class": 100,
    "batch_size": 32,
    "signal_length": 2048,
    "sampler": "ddpm",
    "num_inference_steps": 200,
    "eta": 0.0,
    "rpm": 740.0,
    "seed": 42,
    "device": "cuda_if_available",
}

EPOCH_PATTERN = re.compile(r"^epoch_(\d+)\.pt$")


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("generate_mhta_ddpm_by_epoch")


def write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def discover_epoch_checkpoints(run_dir: Path, requested_epochs: list[int] | tuple[int, ...] | None) -> list[tuple[int, Path]]:
    ckpt_dir = run_dir / "checkpoints"
    if requested_epochs is not None:
        checkpoints = [(int(epoch), ckpt_dir / f"epoch_{int(epoch)}.pt") for epoch in requested_epochs]
        missing = [path for _, path in checkpoints if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing requested epoch checkpoints: {missing}")
        return checkpoints

    checkpoints: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("epoch_*.pt"):
        match = EPOCH_PATTERN.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_*.pt checkpoints found under {ckpt_dir}")
    return sorted(checkpoints, key=lambda item: item[0])


def generate_one_epoch(
    run_dir: Path,
    epoch: int,
    checkpoint: Path,
    config: dict,
    device: torch.device,
    logger: logging.Logger,
) -> Path:
    # 每个 epoch 使用同一随机种子，便于做训练过程横向比较。
    set_seed(int(config["seed"]))
    epoch_out_dir = run_dir / str(config["generated_root_subdir"]) / f"epoch_{epoch:04d}"
    epoch_out_dir.mkdir(parents=True, exist_ok=True)

    load_config = {
        "run_root": Path(config["run_root"]),
        "run_id": run_dir.name,
        "ckpt": checkpoint,
        "signal_length": int(config["signal_length"]),
        "sampler": str(config["sampler"]),
        "num_inference_steps": int(config["num_inference_steps"]),
        "eta": float(config["eta"]),
    }
    model, scheduler, payload = load_model(load_config, device)
    signal_length = int(payload["model_kwargs"].get("signal_length", config["signal_length"]))
    batch_size = int(config["batch_size"])
    samples_per_class = int(config["samples_per_fault_class"])
    rpm = float(config["rpm"])
    fr = rpm / 60.0

    generated_x, generated_y, meta_rows = [], [], []
    with torch.no_grad():
        for class_name in CLASS_NAMES[1:]:
            label = int(LABEL_MAP[class_name])
            logger.info("epoch=%d | generating %d samples for %s", epoch, samples_per_class, class_name)
            left = samples_per_class
            start = 0
            progress = tqdm(total=samples_per_class, desc=f"epoch {epoch} {class_name}", leave=False)
            while left > 0:
                bsz = min(batch_size, left)
                labels = torch.full((bsz,), label, dtype=torch.long, device=device)
                fake = sample_batch(scheduler, model, labels, signal_length, config, device)
                fake_np = fake.squeeze(1).cpu().numpy().astype(np.float32)
                generated_x.append(fake_np)
                generated_y.append(np.full(bsz, label, dtype=np.int64))
                for i in range(bsz):
                    meta_rows.append(
                        {
                            "epoch": int(epoch),
                            "class_name": class_name,
                            "label": label,
                            "source": "omc_tf_mhta_ddpm_v1",
                            "generated_index": start + i,
                            "rpm": rpm,
                            "fr": fr,
                            "checkpoint": checkpoint.as_posix(),
                            "sampler": str(config["sampler"]).lower(),
                            "num_inference_steps": int(config["num_inference_steps"]),
                            "eta": float(config["eta"]),
                        }
                    )
                start += bsz
                left -= bsz
                progress.update(bsz)
            progress.close()

    x_gen = np.concatenate(generated_x, axis=0).astype(np.float32)
    y_gen = np.concatenate(generated_y, axis=0).astype(np.int64)
    if not np.isfinite(x_gen).all():
        raise RuntimeError(f"epoch={epoch} generated samples contain NaN or Inf")

    np.savez_compressed(
        epoch_out_dir / "generated_only.npz",
        X=x_gen,
        y=y_gen,
        class_names=np.asarray(CLASS_NAMES),
        epoch=np.full(len(y_gen), int(epoch), dtype=np.int64),
        rpm=np.full(len(y_gen), rpm, dtype=np.float32),
        fr=np.full(len(y_gen), fr, dtype=np.float32),
        order_templates=order_template_for_labels(y_gen),
    )
    write_metadata(epoch_out_dir / "metadata_generated.csv", meta_rows)
    logger.info(
        "epoch=%d | saved %d samples to %s | mean=%.4f std=%.4f min=%.4f max=%.4f",
        epoch,
        len(y_gen),
        epoch_out_dir,
        float(x_gen.mean()),
        float(x_gen.std()),
        float(x_gen.min()),
        float(x_gen.max()),
    )
    return epoch_out_dir


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def discover_generated_epoch_dirs(generated_root: Path, requested_epochs: list[int] | tuple[int, ...] | None) -> list[tuple[int, Path]]:
    if requested_epochs is not None:
        pairs = [(int(epoch), generated_root / f"epoch_{int(epoch):04d}") for epoch in requested_epochs]
        missing = [path for _, path in pairs if not (path / "generated_only.npz").exists()]
        if missing:
            raise FileNotFoundError(f"Missing generated epoch folders: {missing}")
        return pairs

    pairs: list[tuple[int, Path]] = []
    for path in generated_root.glob("epoch_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("epoch_")
        if suffix.isdigit() and (path / "generated_only.npz").exists():
            pairs.append((int(suffix), path))
    if not pairs:
        raise FileNotFoundError(f"No generated epoch folders found under {generated_root}")
    return sorted(pairs, key=lambda item: item[0])


def load_generated_sample(epoch_dir: Path, label: int, generated_index: int) -> np.ndarray:
    data = np.load(epoch_dir / "generated_only.npz", allow_pickle=True)
    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    class_indices = np.where(y == int(label))[0]
    if len(class_indices) == 0:
        raise ValueError(f"No generated samples for label={label} in {epoch_dir}")
    if generated_index >= len(class_indices):
        raise IndexError(
            f"generated_index={generated_index} is out of range for label={label}; "
            f"only {len(class_indices)} samples available in {epoch_dir}"
        )
    return x[class_indices[int(generated_index)]]


def load_generated_class_samples(epoch_dir: Path, label: int) -> np.ndarray:
    data = np.load(epoch_dir / "generated_only.npz", allow_pickle=True)
    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    class_indices = np.where(y == int(label))[0]
    if len(class_indices) == 0:
        raise ValueError(f"No generated samples for label={label} in {epoch_dir}")
    return x[class_indices]


def load_real_sample(config: dict, label: int) -> np.ndarray:
    data = np.load(Path(config["data_dir"]) / str(config["plot_real_split"]), allow_pickle=True)
    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    class_indices = np.where(y == int(label))[0]
    if len(class_indices) == 0:
        raise ValueError(f"No real samples for label={label}")
    index_within_class = int(config["plot_real_index_within_class"])
    if index_within_class >= len(class_indices):
        raise IndexError(
            f"plot_real_index_within_class={index_within_class} is out of range for label={label}; "
            f"only {len(class_indices)} real samples available"
        )
    return x[class_indices[index_within_class]]


def load_real_class_samples(config: dict, label: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(config["data_dir"]) / str(config["plot_real_split"]), allow_pickle=True)
    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    class_indices = np.where(y == int(label))[0]
    if len(class_indices) == 0:
        raise ValueError(f"No real samples for label={label}")
    return x[class_indices], class_indices


def impulse_score(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    centered = signal - float(np.median(signal))
    rms = float(np.sqrt(np.mean(centered**2)))
    if rms <= 1e-12:
        return 0.0
    abs_signal = np.abs(centered)
    crest_factor = float(abs_signal.max() / rms)
    p99_to_rms = float(np.percentile(abs_signal, 99.0) / rms)
    return crest_factor + 0.5 * p99_to_rms


def select_clean_real_sample(config: dict, label: int, fs: int) -> tuple[int, int, np.ndarray, float]:
    samples, absolute_indices = load_real_class_samples(config, label)
    best: tuple[int, int, np.ndarray, float] | None = None
    for index_within_class, (absolute_index, signal) in enumerate(zip(absolute_indices, samples)):
        cropped = crop_signal(signal, fs, config.get("plot_max_seconds"))
        score = impulse_score(cropped)
        if best is None or score < best[3]:
            best = (index_within_class, int(absolute_index), signal, score)
    if best is None:
        raise ValueError(f"No real samples available for label={label}")
    return best


def crop_signal(signal: np.ndarray, fs: int, max_seconds: float | None) -> np.ndarray:
    if max_seconds is None:
        return signal
    n = min(len(signal), int(round(float(fs) * float(max_seconds))))
    return signal[:n]


def amplitude_spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32)
    window = np.hanning(signal.shape[-1]).astype(np.float32)
    spec = np.fft.rfft(signal * window)
    freq = np.fft.rfftfreq(signal.shape[-1], d=1.0 / float(fs))
    amp = np.abs(spec) * 2.0 / max(float(window.sum()), 1.0)
    return freq, amp.astype(np.float32)


def pearson_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def select_best_generated_sample(
    epoch_dir: Path,
    label: int,
    real_signal: np.ndarray,
    fs: int,
    max_seconds: float | None,
    max_freq: float,
    fft_weight: float,
    time_weight: float,
) -> tuple[int, np.ndarray, float, float, float]:
    generated = load_generated_class_samples(epoch_dir, label)
    real = crop_signal(real_signal, fs, max_seconds)
    real_freq, real_amp = amplitude_spectrum(real, fs)
    freq_mask = real_freq <= float(max_freq)

    best: tuple[int, np.ndarray, float, float, float] | None = None
    for generated_index, signal in enumerate(generated):
        signal = crop_signal(signal, fs, max_seconds)
        n = min(len(signal), len(real))
        _, gen_amp = amplitude_spectrum(signal, fs)
        m = min(len(gen_amp), len(real_amp))
        fft_score = pearson_similarity(gen_amp[:m][freq_mask[:m]], real_amp[:m][freq_mask[:m]])
        time_score = pearson_similarity(signal[:n], real[:n])
        score = float(fft_weight) * fft_score + float(time_weight) * time_score
        if best is None or score > best[2]:
            best = (generated_index, signal, score, fft_score, time_score)

    if best is None:
        raise ValueError(f"No generated samples available for best-match selection in {epoch_dir}")
    return best


def write_selected_samples_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_column_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text.strip().lower()).strip("_")


def write_epoch_plot_data_csv(path: Path, panels: list[tuple[str, np.ndarray, str]], fs: int) -> None:
    if not panels:
        return
    spectra = [amplitude_spectrum(signal, fs) for _, signal, _ in panels]
    max_rows = max(max(len(signal), len(freq)) for (_, signal, _), (freq, _) in zip(panels, spectra))
    fieldnames = ["sample_index", "time_s", "freq_hz"]
    panel_columns: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for idx, ((title, signal, _), (_, amp)) in enumerate(zip(panels, spectra), start=1):
        column_prefix = f"panel{idx:02d}_{safe_column_name(title)}"
        signal_col = f"{column_prefix}_signal"
        fft_col = f"{column_prefix}_fft_mag"
        fieldnames.extend([signal_col, fft_col])
        panel_columns.append((signal_col, fft_col, np.asarray(signal, dtype=np.float32), amp))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(max_rows):
            row: dict[str, int | float | str] = {"sample_index": i}
            row["time_s"] = i / float(fs) if i < max(len(signal) for _, signal, _ in panels) else ""
            row["freq_hz"] = spectra[0][0][i] if i < len(spectra[0][0]) else ""
            for signal_col, fft_col, signal, amp in panel_columns:
                row[signal_col] = float(signal[i]) if i < len(signal) else ""
                row[fft_col] = float(amp[i]) if i < len(amp) else ""
            writer.writerow(row)


def plot_epoch_grid(run_dir: Path, config: dict, logger: logging.Logger) -> Path:
    generated_root = run_dir / str(config["generated_root_subdir"])
    epoch_dirs = discover_generated_epoch_dirs(generated_root, config.get("epochs"))
    class_name = str(config["plot_class_name"])
    if class_name not in LABEL_MAP:
        raise ValueError(f"Unknown class_name={class_name!r}; choices={list(LABEL_MAP)}")
    label = int(LABEL_MAP[class_name])
    fs = int(config["plot_fs"])
    real_signal = None
    real_index_within_class: int | str = ""
    real_absolute_index: int | str = ""
    real_selection_score: float | str = ""
    real_selected_by = ""
    if bool(config["plot_include_real"]):
        if bool(config.get("plot_real_auto_clean", False)):
            real_index_within_class, real_absolute_index, real_signal, real_selection_score = select_clean_real_sample(
                config, label, fs
            )
            real_selected_by = "cleanest_real"
            logger.info(
                "selected clean real %s sample index_within_class=%d absolute_index=%d | impulse_score=%.4f",
                class_name,
                real_index_within_class,
                real_absolute_index,
                real_selection_score,
            )
        else:
            real_signal = load_real_sample(config, label)
            real_index_within_class = int(config["plot_real_index_within_class"])
            real_selected_by = "configured_real_index"

    panels: list[tuple[str, np.ndarray, str]] = []
    rows: list[dict] = []
    best_match_epoch = config.get("plot_best_match_epoch")
    for epoch, epoch_dir in epoch_dirs:
        generated_index = int(config["plot_generated_index"])
        selected_by = "fixed_index"
        best_score: float | str = ""
        best_fft_score: float | str = ""
        best_time_score: float | str = ""
        if best_match_epoch is not None and int(epoch) == int(best_match_epoch):
            if real_signal is None:
                raise ValueError("plot_best_match_epoch requires plot_include_real=True")
            generated_index, signal, best_score, best_fft_score, best_time_score = select_best_generated_sample(
                epoch_dir=epoch_dir,
                label=label,
                real_signal=real_signal,
                fs=fs,
                max_seconds=config.get("plot_max_seconds"),
                max_freq=float(config["plot_max_freq"]),
                fft_weight=float(config["plot_best_match_fft_weight"]),
                time_weight=float(config["plot_best_match_time_weight"]),
            )
            selected_by = "best_match_to_real"
            logger.info(
                "epoch=%d | selected best %s sample index=%d | score=%.4f fft=%.4f time=%.4f",
                epoch,
                class_name,
                generated_index,
                best_score,
                best_fft_score,
                best_time_score,
            )
        else:
            signal = load_generated_sample(epoch_dir, label, generated_index)
            signal = crop_signal(signal, fs, config.get("plot_max_seconds"))
        panels.append((f"Epoch = {epoch}", signal, str(config["plot_line_color"])))
        rows.append(
            {
                "panel": len(panels),
                "kind": "generated",
                "epoch": int(epoch),
                "class_name": class_name,
                "label": label,
                "generated_index": generated_index,
                "real_index_within_class": "",
                "real_absolute_index": "",
                "real_selected_by": "",
                "real_impulse_score": "",
                "selected_by": selected_by,
                "best_match_score": best_score,
                "best_match_fft_score": best_fft_score,
                "best_match_time_score": best_time_score,
                "source_path": (epoch_dir / "generated_only.npz").as_posix(),
            }
        )

    if real_signal is not None:
        real_signal = crop_signal(real_signal, fs, config.get("plot_max_seconds"))
        panels.append(("Real signal", real_signal, str(config["plot_real_line_color"])))
        rows.append(
            {
                "panel": len(panels),
                "kind": "real",
                "epoch": "",
                "class_name": class_name,
                "label": label,
                "generated_index": "",
                "real_index_within_class": real_index_within_class,
                "real_absolute_index": real_absolute_index,
                "real_selected_by": real_selected_by,
                "real_impulse_score": real_selection_score,
                "selected_by": "reference",
                "best_match_score": "",
                "best_match_fft_score": "",
                "best_match_time_score": "",
                "source_path": (Path(config["data_dir"]) / str(config["plot_real_split"])).as_posix(),
            }
        )

    ncols = int(config["plot_ncols"])
    panel_rows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(panel_rows * 2, ncols, figsize=(6.8 * ncols, 4.8 * panel_rows), squeeze=False)
    max_freq = float(config["plot_max_freq"])

    for idx, (title, signal, color) in enumerate(panels):
        row = idx // ncols
        col = idx % ncols
        time_ax = axes[row * 2, col]
        freq_ax = axes[row * 2 + 1, col]

        time_s = np.arange(len(signal), dtype=np.float32) / float(fs)
        time_ax.plot(time_s, signal, color=color, linewidth=0.8)
        time_ax.set_title(title)
        time_ax.set_xlabel("Time (s)")
        time_ax.set_ylabel("Amplitude")
        time_ax.grid(alpha=0.2)

        freq, amp = amplitude_spectrum(signal, fs)
        mask = freq <= max_freq
        freq_ax.plot(freq[mask], amp[mask], color=color, linewidth=0.85)
        freq_ax.set_xlabel("Frequency (Hz)")
        freq_ax.set_ylabel("FFT magnitude")
        freq_ax.grid(alpha=0.2)
        freq_ax.text(0.5, -0.36, f"({chr(97 + idx)})", transform=freq_ax.transAxes, ha="center", va="top", fontsize=11)

    for idx in range(len(panels), panel_rows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row * 2, col].axis("off")
        axes[row * 2 + 1, col].axis("off")

    fig.suptitle(f"MHTA-DDPM {class_name} samples across epochs: time and spectrum", y=1.005)
    fig.tight_layout()
    out_dir = run_dir / str(config["plot_subdir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}_{class_name}_epoch_progress_time_freq.png"
    fig.savefig(out_path, dpi=int(config["plot_dpi"]), bbox_inches="tight")
    plt.close(fig)
    write_selected_samples_csv(out_dir / f"{label}_{class_name}_epoch_progress_sources.csv", rows)
    data_path = out_dir / f"{label}_{class_name}_epoch_progress_time_freq_data.csv"
    write_epoch_plot_data_csv(data_path, panels, fs)
    logger.info("Saved epoch progression plot to %s", out_path)
    logger.info("Saved epoch progression time/frequency data to %s", data_path)
    return out_path


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    device = get_device(str(config["device"]))
    run_dir = resolve_existing_run_dir(Path(config["run_root"]), config.get("run_id"))
    checkpoints = discover_epoch_checkpoints(run_dir, config.get("epochs"))
    logger.info("Using MHTA run directory: %s", run_dir)
    logger.info("Will generate for epochs: %s", [epoch for epoch, _ in checkpoints])

    manifest_rows = []
    for epoch, checkpoint in checkpoints:
        out_dir = generate_one_epoch(run_dir, epoch, checkpoint, config, device, logger)
        manifest_rows.append(
            {
                "epoch": int(epoch),
                "checkpoint": checkpoint.as_posix(),
                "generated_dir": out_dir.as_posix(),
                "samples_per_fault_class": int(config["samples_per_fault_class"]),
                "sampler": str(config["sampler"]).lower(),
                "num_inference_steps": int(config["num_inference_steps"]),
                "seed": int(config["seed"]),
            }
        )

    generated_root = run_dir / str(config["generated_root_subdir"])
    write_manifest(generated_root / "manifest.csv", manifest_rows)
    logger.info("Saved all epoch-wise generations under %s", generated_root)
    if bool(config.get("plot_after_generate", True)):
        plot_epoch_grid(run_dir, config, logger)


if __name__ == "__main__":
    main()
