from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.signal_utils import CLASS_NAMES


CONFIG = {
    "real_npz": Path("processed/mhta_base_dataset/train.npz"),
    "out_dir": Path("runs/tsne_real_vs_generated"),
    "labels_to_plot": [1, 2, 3, 4],
    "samples_per_class": 50,
    "seed": 42,
    "fs": 2048,
    "max_freq": 300.0,
    "feature_mode": "time_log_fft",  # choices: log_fft, time, time_log_fft
    "time_feature_weight": 1.0,
    "freq_feature_weight": 1.0,
    "tsne_perplexity": 30,
    "tsne_learning_rate": "auto",
    "tsne_iter": 1500,
    "knn_k": 10,
    "point_size": 12,
    "dpi": 260,
    "models": [
        {
            "name": "MHTA-DDPM",
            "npz": Path("runs/omc_tf_mhta_ddpm_v1/20260517_222057/generated_dataset/generated_only.npz"),
        },
        {
            "name": "Baseline-DDPM",
            "npz": Path("runs/baseline_ddpm_v1/20260518_182449/generated_dataset/generated_only.npz"),
        },
        {
            "name": "DCGAN",
            "npz": Path("runs/dcgan_v1/20260518_194930/generated_dataset/generated_only.npz"),
        },
        {
            "name": "WGAN",
            "npz": Path("runs/wgan_v1/20260518_182415/generated_dataset/generated_only.npz"),
        },
    ],
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("plot_tsne_real_vs_generated_models")


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing npz file: {path}")
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["X"], dtype=np.float32), np.asarray(data["y"], dtype=np.int64)


def balanced_indices(
    y: np.ndarray,
    labels_to_plot: list[int] | tuple[int, ...],
    samples_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = []
    for label in labels_to_plot:
        class_indices = np.where(y == int(label))[0]
        if len(class_indices) == 0:
            continue
        count = min(int(samples_per_class), len(class_indices))
        selected.append(rng.choice(class_indices, size=count, replace=False))
    if not selected:
        return np.asarray([], dtype=np.int64)
    return np.sort(np.concatenate(selected).astype(np.int64))


def zscore_per_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def log_fft_features(x: np.ndarray, fs: int, max_freq: float) -> np.ndarray:
    x = zscore_per_signal(x)
    window = np.hanning(x.shape[1]).astype(np.float32)
    spec = np.fft.rfft(x * window[None, :], axis=1)
    freq = np.fft.rfftfreq(x.shape[1], d=1.0 / float(fs))
    mask = freq <= float(max_freq)
    amp = np.abs(spec[:, mask]) * 2.0 / max(float(window.sum()), 1.0)
    return np.log1p(amp).astype(np.float32)


def make_features(x: np.ndarray, config: dict) -> np.ndarray:
    mode = str(config["feature_mode"]).lower()
    if mode == "log_fft":
        return log_fft_features(x, int(config["fs"]), float(config["max_freq"]))
    if mode == "time":
        return zscore_per_signal(x)
    if mode == "time_log_fft":
        time = standardize_features(zscore_per_signal(x)) * float(config["time_feature_weight"])
        freq = standardize_features(log_fft_features(x, int(config["fs"]), float(config["max_freq"]))) * float(
            config["freq_feature_weight"]
        )
        return np.concatenate([time, freq], axis=1).astype(np.float32)
    raise ValueError(f"Unknown feature_mode={mode!r}")


def standardize_features(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def run_tsne(features: np.ndarray, config: dict) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for t-SNE. Install it in your active environment, "
            "for example: conda install scikit-learn"
        ) from exc

    perplexity = min(float(config["tsne_perplexity"]), max(5.0, (len(features) - 1) / 3.0))
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "learning_rate": config["tsne_learning_rate"],
        "init": "pca",
        "random_state": int(config["seed"]),
    }
    try:
        return TSNE(max_iter=int(config["tsne_iter"]), **kwargs).fit_transform(features)
    except TypeError:
        return TSNE(n_iter=int(config["tsne_iter"]), **kwargs).fit_transform(features)


def centroid_distance(features: np.ndarray, source: np.ndarray, labels: np.ndarray) -> float:
    distances = []
    for label in sorted(np.unique(labels).tolist()):
        real = features[(source == "real") & (labels == label)]
        generated = features[(source == "generated") & (labels == label)]
        if len(real) == 0 or len(generated) == 0:
            continue
        distances.append(float(np.linalg.norm(real.mean(axis=0) - generated.mean(axis=0))))
    return float(np.mean(distances)) if distances else float("nan")


def pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    dists = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(dists, 0.0)


def rbf_mmd(features: np.ndarray, source: np.ndarray) -> float:
    real = features[source == "real"].astype(np.float64)
    generated = features[source == "generated"].astype(np.float64)
    if len(real) == 0 or len(generated) == 0:
        return float("nan")
    combined = np.concatenate([real, generated], axis=0)
    sq = pairwise_sq_dists(combined, combined)
    positive = sq[sq > 0]
    gamma = 1.0 / max(float(np.median(positive)), 1e-8) if positive.size else 1.0

    xx = np.exp(-gamma * pairwise_sq_dists(real, real)).mean()
    yy = np.exp(-gamma * pairwise_sq_dists(generated, generated)).mean()
    xy = np.exp(-gamma * pairwise_sq_dists(real, generated)).mean()
    return float(xx + yy - 2.0 * xy)


def knn_real_fraction_for_generated(features: np.ndarray, source: np.ndarray, k: int) -> float:
    generated_indices = np.where(source == "generated")[0]
    if len(generated_indices) == 0 or len(features) <= 1:
        return float("nan")
    sq = pairwise_sq_dists(features, features)
    np.fill_diagonal(sq, np.inf)
    k = min(int(k), len(features) - 1)
    nearest = np.argpartition(sq[generated_indices], kth=k - 1, axis=1)[:, :k]
    return float(np.mean(source[nearest] == "real"))


def write_points_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_panels(model_results: list[dict], out_path: Path, config: dict) -> None:
    n = len(model_results)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 4.4 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    labels_to_plot = [int(label) for label in config["labels_to_plot"]]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(labels_to_plot)))
    label_to_color = {label: colors[i] for i, label in enumerate(labels_to_plot)}

    for ax, result in zip(flat_axes, model_results):
        embedding = result["embedding"]
        labels = result["labels"]
        source = result["source"]
        for label in labels_to_plot:
            color = label_to_color[label]
            real_mask = (labels == label) & (source == "real")
            gen_mask = (labels == label) & (source == "generated")
            ax.scatter(
                embedding[real_mask, 0],
                embedding[real_mask, 1],
                s=float(config["point_size"]),
                facecolors="none",
                edgecolors=[color],
                linewidths=0.55,
                alpha=0.85,
            )
            ax.scatter(
                embedding[gen_mask, 0],
                embedding[gen_mask, 1],
                s=float(config["point_size"]),
                c=[color],
                marker="x",
                linewidths=0.55,
                alpha=0.75,
            )

        metric_text = (
            f"centroid={result['centroid_distance']:.3f}, "
            f"MMD={result['mmd']:.4f}, "
            f"kNN-real={result['knn_real_fraction']:.3f}"
        )
        ax.set_title(f"{result['model_name']}\n{metric_text}", fontsize=10)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(alpha=0.18)

    for ax in flat_axes[n:]:
        ax.axis("off")

    class_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=label_to_color[label],
            label=CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label),
            markersize=5,
        )
        for label in labels_to_plot
    ]
    source_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="black", label="Real"),
        plt.Line2D([0], [0], marker="x", linestyle="", color="black", label="Generated"),
    ]
    fig.legend(handles=class_handles, title="Class", loc="center right", bbox_to_anchor=(1.02, 0.58))
    fig.legend(handles=source_handles, title="Source", loc="center right", bbox_to_anchor=(1.02, 0.24))
    fig.suptitle("t-SNE: Real vs Generated Samples by Model", y=1.01)
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    fig.savefig(out_path, dpi=int(config["dpi"]), bbox_inches="tight")
    plt.close(fig)


def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    rng = np.random.default_rng(int(config["seed"]))
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    x_real_all, y_real_all = load_xy(Path(config["real_npz"]))
    model_results = []
    point_rows = []
    metric_rows = []

    for model in config["models"]:
        model_name = str(model["name"])
        model_npz = Path(model["npz"])
        if not model_npz.exists():
            logger.warning("Skipping %s because %s does not exist", model_name, model_npz)
            continue

        x_gen_all, y_gen_all = load_xy(model_npz)
        labels_to_plot = [int(label) for label in config["labels_to_plot"]]
        real_indices = balanced_indices(y_real_all, labels_to_plot, int(config["samples_per_class"]), rng)
        gen_indices = balanced_indices(y_gen_all, labels_to_plot, int(config["samples_per_class"]), rng)
        x = np.concatenate([x_real_all[real_indices], x_gen_all[gen_indices]], axis=0)
        labels = np.concatenate([y_real_all[real_indices], y_gen_all[gen_indices]], axis=0)
        source = np.asarray(["real"] * len(real_indices) + ["generated"] * len(gen_indices), dtype=object)
        sample_indices = np.concatenate([real_indices, gen_indices], axis=0)

        features = standardize_features(make_features(x, config))
        embedding = run_tsne(features, config)
        centroid = centroid_distance(features, source, labels)
        mmd = rbf_mmd(features, source)
        knn_real = knn_real_fraction_for_generated(features, source, int(config["knn_k"]))

        model_results.append(
            {
                "model_name": model_name,
                "embedding": embedding,
                "labels": labels,
                "source": source,
                "centroid_distance": centroid,
                "mmd": mmd,
                "knn_real_fraction": knn_real,
            }
        )
        metric_rows.append(
            {
                "model": model_name,
                "generated_npz": model_npz.as_posix(),
                "feature_mode": str(config["feature_mode"]),
                "labels_to_plot": " ".join(str(label) for label in labels_to_plot),
                "samples_per_class": int(config["samples_per_class"]),
                "centroid_distance_lower_better": centroid,
                "mmd_lower_better": mmd,
                "knn_real_fraction_for_generated_higher_better": knn_real,
            }
        )

        for i, (xy, label, src, sample_index) in enumerate(zip(embedding, labels, source, sample_indices)):
            point_rows.append(
                {
                    "model": model_name,
                    "source": src,
                    "class_name": CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else str(label),
                    "label": int(label),
                    "sample_index": int(sample_index),
                    "point_index": i,
                    "tsne_1": float(xy[0]),
                    "tsne_2": float(xy[1]),
                }
            )

        logger.info(
            "%s | points=%d | centroid=%.4f | MMD=%.5f | kNN-real=%.4f",
            model_name,
            len(labels),
            centroid,
            mmd,
            knn_real,
        )

    if not model_results:
        raise RuntimeError("No model npz files were found. Please check CONFIG['models'].")

    plot_panels(model_results, out_dir / "tsne_real_vs_generated_models.png", config)
    write_points_csv(out_dir / "tsne_real_vs_generated_models_points.csv", point_rows)
    write_points_csv(out_dir / "tsne_real_vs_generated_models_metrics.csv", metric_rows)
    logger.info("Saved t-SNE outputs to %s", out_dir)


if __name__ == "__main__":
    main()
