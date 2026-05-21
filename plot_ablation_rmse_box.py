from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


CONFIG = {
    "input_csv": Path("runs/omc_tf_mhta_ddpm_v1/ablations/ablation_pair_metrics_summary_20260520_164205.csv"),
    "real_npz": Path("processed/mhta_base_dataset/train.npz"),
    "out_dir": Path("runs/omc_tf_mhta_ddpm_v1/ablations"),
    "output_prefix": "ablation_rmse_box",
    "metric": "RMSE",
    # Choose one:
    #   None: use CONFIG["metric"] directly.
    #   "nrmse_std": RMSE / std(real signal)
    #   "nrmse_range": RMSE / (max(real signal) - min(real signal))
    "derived_metric": "nrmse_std",
    # Set sample_count to None to use all rows after optional trimming.
    "sample_count": 50,
    "random_seed": 42,
    # Set trim_quantiles to None to disable trimming.
    # Examples:
    #   None
    #   (0.05, 0.95)
    #   (0.02, 0.98)
    "trim_quantiles": None,
    "ablation_order": [
        "without_temporal_attention",
        "without_mhta_unet_only",
        "without_qkv_dwconv",
        "without_fft_loss",
        "input_kernel_3",
    ],
    "ablation_labels": {
        "without_temporal_attention": "w/o TA",
        "without_mhta_unet_only": "w/o MHTA",
        "without_qkv_dwconv": "w/o DWConv",
        "without_fft_loss": "w/o FFT",
        "input_kernel_3": "kernel=3",
    },
    "title": None,
    "ylabel": "NRMSE",
    "xlabel": "Ablation Setting",
    "dpi": 300,
    "figsize": (10.5, 6.2),
    "point_size": 14,
    "point_jitter": 0.045,
    "point_offset": 0.28,
    "box_width": 0.42,
}


def suffix_from_config(config: dict) -> str:
    metric = active_metric_name(config).lower()
    sample_count = config.get("sample_count")
    sample_part = "all" if sample_count is None else f"sample{int(sample_count)}"
    trim = config.get("trim_quantiles")
    if trim is None:
        trim_part = "notrim"
    else:
        low, high = trim
        trim_part = f"trim{int(round(float(low) * 100))}_{int(round(float(high) * 100))}"
    return f"{metric}_{trim_part}_{sample_part}"


def active_metric_name(config: dict) -> str:
    derived = config.get("derived_metric")
    if derived is None:
        return str(config["metric"])
    return str(derived).upper()


def add_derived_metric(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    derived = config.get("derived_metric")
    if derived is None:
        return df

    derived = str(derived).lower()
    if derived not in {"nrmse_std", "nrmse_range"}:
        raise ValueError(f"Unknown derived_metric={derived!r}. Use None, 'nrmse_std', or 'nrmse_range'.")

    real_npz = Path(config["real_npz"])
    data = np.load(real_npz, allow_pickle=True)
    x_real = np.asarray(data["X"], dtype=np.float32)
    y_real = np.asarray(data["y"], dtype=np.int64)
    eps = float(config.get("eps", 1e-8))

    denominators: list[float] = []
    for row in df.itertuples(index=False):
        label = int(getattr(row, "label"))
        real_index_in_class = int(getattr(row, "real_index_in_class"))
        class_indices = np.flatnonzero(y_real == label)
        if real_index_in_class >= len(class_indices):
            raise IndexError(
                f"real_index_in_class={real_index_in_class} is out of range for label={label} "
                f"with {len(class_indices)} real samples."
            )
        real = x_real[int(class_indices[real_index_in_class])]
        if derived == "nrmse_std":
            denom = float(np.std(real))
        else:
            denom = float(np.max(real) - np.min(real))
        denominators.append(max(denom, eps))

    out = df.copy()
    out["real_signal_denominator"] = np.asarray(denominators, dtype=np.float64)
    metric_name = active_metric_name(config)
    out[metric_name] = out[str(config["metric"])].to_numpy(dtype=np.float64) / out["real_signal_denominator"].to_numpy(dtype=np.float64)
    return out


def select_points(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric = active_metric_name(config)
    order = list(config["ablation_order"])
    trim = config.get("trim_quantiles")
    sample_count = config.get("sample_count")
    rng = np.random.default_rng(int(config["random_seed"]))

    selected_parts = []
    info_rows = []
    for name in order:
        group = df[df["ablation_name"] == name].copy()
        if group.empty:
            continue

        if trim is None:
            low_value = float(group[metric].min())
            high_value = float(group[metric].max())
            trimmed = group
        else:
            low_q, high_q = trim
            low_value = float(group[metric].quantile(float(low_q)))
            high_value = float(group[metric].quantile(float(high_q)))
            trimmed = group[(group[metric] >= low_value) & (group[metric] <= high_value)].copy()

        if sample_count is None:
            chosen = trimmed.index.to_numpy()
        else:
            take = min(int(sample_count), len(trimmed))
            chosen = rng.choice(trimmed.index.to_numpy(), size=take, replace=False)

        selected_parts.append(trimmed.loc[chosen])
        info_rows.append(
            {
                "ablation_name": name,
                "original_count": len(group),
                "selected_count": len(chosen),
                "trim_low_value": low_value,
                "trim_high_value": high_value,
                "trim_quantiles": "" if trim is None else f"{trim[0]}-{trim[1]}",
            }
        )

    if not selected_parts:
        raise RuntimeError("No ablation rows were selected. Check CONFIG['input_csv'] and CONFIG['ablation_order'].")
    return pd.concat(selected_parts, axis=0).reset_index(drop=True), pd.DataFrame(info_rows)


def plot_box_scatter(selected: pd.DataFrame, config: dict, out_path: Path) -> pd.DataFrame:
    metric = active_metric_name(config)
    order = [name for name in config["ablation_order"] if name in set(selected["ablation_name"])]
    labels = dict(config["ablation_labels"])
    plot_data = [selected.loc[selected["ablation_name"] == name, metric].dropna().to_numpy() for name in order]
    positions = np.arange(1, len(order) + 1)
    means = np.array([values.mean() for values in plot_data])

    fig, ax = plt.subplots(figsize=tuple(config["figsize"]), constrained_layout=True)
    box_color = "#d8cdeb"
    edge_color = "#6f4aa8"
    point_color = "#f4a261"
    mean_line_color = "#8fcac0"

    ax.boxplot(
        plot_data,
        positions=positions,
        widths=float(config["box_width"]),
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "*", "markerfacecolor": edge_color, "markeredgecolor": edge_color, "markersize": 8},
        medianprops={"color": edge_color, "linewidth": 1.2},
        boxprops={"facecolor": box_color, "edgecolor": edge_color, "linewidth": 1.2},
        whiskerprops={"color": edge_color, "linewidth": 1.1},
        capprops={"color": edge_color, "linewidth": 1.1},
        flierprops={
            "marker": "D",
            "markerfacecolor": edge_color,
            "markeredgecolor": edge_color,
            "markersize": 3.5,
            "alpha": 0.95,
        },
    )

    rng = np.random.default_rng(int(config["random_seed"]))
    for pos, values in zip(positions, plot_data):
        jitter = rng.normal(loc=0.0, scale=float(config["point_jitter"]), size=len(values))
        ax.scatter(
            np.full(len(values), pos + float(config["point_offset"])) + jitter,
            values,
            s=float(config["point_size"]),
            color=point_color,
            alpha=0.72,
            edgecolors="none",
            zorder=2,
        )

    ax.plot(positions, means, color=mean_line_color, linewidth=1.8, marker="x", markersize=6, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels.get(name, name) for name in order], rotation=15, ha="right")
    ax.set_ylabel(str(config["ylabel"]))
    ax.set_xlabel(str(config["xlabel"]))
    title = config.get("title")
    if title is None:
        trim = config.get("trim_quantiles")
        sample_count = config.get("sample_count")
        trim_text = "No Trim" if trim is None else f"{int(trim[0] * 100)}%-{int(trim[1] * 100)}% Trimmed"
        sample_text = "All Pairs" if sample_count is None else f"{int(sample_count)} Random Pairs Each"
        title = f"{metric} Distribution Across Ablations ({trim_text}, {sample_text})"
    ax.set_title(str(title))
    ax.grid(axis="y", linestyle=(0, (5, 5)), color="#bdbdbd", alpha=0.75)
    ax.set_xlim(0.5, len(order) + 0.9)

    legend_handles = [
        Patch(facecolor=box_color, edgecolor=edge_color, label="25%-75%"),
        Line2D([0], [0], color=edge_color, linewidth=1.1, label="1.5 Interquartile Range"),
        Line2D([0], [0], color=edge_color, linewidth=1.2, label="Median line"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=edge_color, markeredgecolor=edge_color, markersize=8, label="Average value"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=edge_color, markeredgecolor=edge_color, markersize=4, label="Outlier"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=point_color, markeredgecolor="none", markersize=5, label="Data point"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(out_path, dpi=int(config["dpi"]), bbox_inches="tight")
    plt.close(fig)

    return selected.groupby("ablation_name")[metric].agg(["count", "mean", "std", "median", "min", "max"]).loc[order]


def main(config: dict = CONFIG) -> None:
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = suffix_from_config(config)
    prefix = str(config["output_prefix"]).strip()
    stem = f"{prefix}_{suffix}" if prefix else suffix

    df = pd.read_csv(Path(config["input_csv"]))
    df = add_derived_metric(df, config)
    selected, info = select_points(df, config)
    selected_path = out_dir / f"{stem}_points.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    info_path = out_dir / f"{stem}_selection_info.csv"
    fig_path = out_dir / f"{stem}.png"

    selected.to_csv(selected_path, index=False)
    info.to_csv(info_path, index=False)
    summary = plot_box_scatter(selected, config, fig_path)
    summary.to_csv(summary_path)

    print(fig_path)
    print(selected_path)
    print(summary_path)
    print(info_path)


if __name__ == "__main__":
    main()
