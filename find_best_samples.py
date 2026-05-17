import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import pearsonr
from pathlib import Path

# 从你的工具包中导入类别名
# 如果报错，可以直接替换为 CLASS_NAMES = ["Normal", "Unbalance", "Misalignment", "Crack", "Looseness"]
from utils.run_paths import resolve_existing_run_dir
from utils.signal_utils import CLASS_NAMES 


CONFIG = {
    "run_root": Path("runs/omc_tf_mhta_ddpm_v1"),
    "run_id": None,
    "generated_subdir": "generated_dataset",
    "out_subdir": "best_sample_pairs",
}


def compute_fft(signals, sample_rate=2048):
    """计算单边幅度谱"""
    # 动态判断：如果有3个维度且中间是1，才squeeze
    if signals.ndim == 3 and signals.shape[1] == 1:
        signals = signals.squeeze(1) 
        
    fft_complex = np.fft.rfft(signals, axis=-1, norm="ortho")
    fft_mag = np.abs(fft_complex)
    freqs = np.fft.rfftfreq(signals.shape[-1], d=1/sample_rate)
    return freqs, fft_mag


def standardize(signal):
    """标准化信号，避免幅值尺度主导形状相似度。"""
    signal = np.asarray(signal, dtype=np.float64)
    std = np.std(signal)
    if std < 1e-12:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def max_normalized_xcorr(signal_a, signal_b):
    """计算允许时移的最大归一化互相关。"""
    a = standardize(signal_a)
    b = standardize(signal_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    corr = correlate(a, b, mode="full", method="fft") / denom
    return float(np.max(corr))


def diff_similarity(signal_a, signal_b):
    """将一阶差分 MSE 转成越大越好的相似度分数。"""
    diff_a = np.diff(standardize(signal_a))
    diff_b = np.diff(standardize(signal_b))
    mse = float(np.mean((diff_a - diff_b) ** 2))
    return 1.0 / (1.0 + mse), mse

def main(config: dict = CONFIG):
    # 1. 配置文件路径 (请根据你的实际路径修改)
    real_data_path = Path("processed/mhta_base_dataset/train.npz")
    run_dir = resolve_existing_run_dir(Path(config["run_root"]), config.get("run_id"))
    gen_data_path = run_dir / str(config.get("generated_subdir", "generated_dataset")) / "generated_only.npz"
    out_dir = run_dir / str(config.get("out_subdir", "best_sample_pairs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not real_data_path.exists() or not gen_data_path.exists():
        print("未找到数据文件，请检查路径设置！")
        return

    # 2. 加载数据
    real_data = np.load(real_data_path)
    X_real, y_real = real_data["X"], real_data["y"]
    
    gen_data = np.load(gen_data_path)
    X_gen, y_gen = gen_data["X"], gen_data["y"]
    
    # 获取采样率和时间轴
    sample_rate = 2048
    length = X_real.shape[-1]
    t = np.arange(length) / sample_rate
    freqs, _ = compute_fft(X_real[:1], sample_rate)

    # 只展示故障类别，排除 Normal
    selected_classes = [(idx, name) for idx, name in enumerate(CLASS_NAMES) if name != "Normal"]

    # 准备画图的画布：每个排名占两行（时域 + 频域），每一列对应一个类别
    num_classes = len(selected_classes)
    top_k = 3
    fig, axes = plt.subplots(2 * top_k, num_classes, figsize=(4 * num_classes, 4 * top_k))
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    # 用于保存最终入选配对的原始时域/频域数据；按类别分块组织列
    export_columns = {}

    # 3. 遍历每个故障类别，寻找最像的前 3 个真实-生成配对
    for plot_idx, (class_idx, class_name) in enumerate(selected_classes):
        # 筛选当前类别的数据
        real_mask = (y_real == class_idx)
        gen_mask = (y_gen == class_idx)
        
        X_real_c = X_real[real_mask]
        X_gen_c = X_gen[gen_mask]
        
        if len(X_real_c) == 0 or len(X_gen_c) == 0:
            for row in range(2 * top_k):
                axes[row, plot_idx].axis('off')
            continue
            
        # 计算真实样本和生成样本的频域
        _, fft_real_all = compute_fft(X_real_c, sample_rate)
        _, fft_gen_all = compute_fft(X_gen_c, sample_rate)

        # 第一阶段：先按低频频谱相关性筛候选，只保留最像的一批配对
        fft_corr_scores = np.empty((len(fft_real_all), len(fft_gen_all)), dtype=np.float64)
        for real_idx, real_fft in enumerate(fft_real_all):
            for gen_idx, gen_fft in enumerate(fft_gen_all):
                fft_corr_scores[real_idx, gen_idx] = pearsonr(real_fft[:300], gen_fft[:300])[0]

        total_pair_count = fft_corr_scores.size
        sorted_candidate_flat_indices = np.argsort(fft_corr_scores.ravel())[::-1]
        target_pair_count = min(top_k, len(X_real_c), len(X_gen_c))
        initial_pool_size = min(total_pair_count, max(top_k * 20, int(np.ceil(total_pair_count * 0.05))))

        # 第二阶段 + 第三阶段：先从最优低频谱候选开始，如果去重后不够 3 对，就逐步扩大候选池
        top_pairs = []
        pool_size = initial_pool_size
        while True:
            candidate_flat_indices = sorted_candidate_flat_indices[:pool_size]
            candidate_pair_indices = np.dstack(np.unravel_index(candidate_flat_indices, fft_corr_scores.shape))[0]

            candidate_pairs = []
            for real_idx, gen_idx in candidate_pair_indices:
                real_idx = int(real_idx)
                gen_idx = int(gen_idx)
                real_signal = X_real_c[real_idx].squeeze()
                gen_signal = X_gen_c[gen_idx].squeeze()

                fft_corr = float(fft_corr_scores[real_idx, gen_idx])
                time_xcorr = max_normalized_xcorr(real_signal, gen_signal)
                diff_sim, diff_mse = diff_similarity(real_signal, gen_signal)

                # 综合排序。频域优先，其次时域互相关，再用差分形状约束冲击。
                fft_sim = (fft_corr + 1.0) / 2.0
                time_sim = (time_xcorr + 1.0) / 2.0
                final_score = 0.50 * fft_sim + 0.35 * time_sim + 0.15 * diff_sim
                candidate_pairs.append((real_idx, gen_idx, final_score, fft_corr, time_xcorr, diff_mse))

            candidate_pairs.sort(key=lambda item: item[2], reverse=True)
            top_pairs = []
            used_real_indices = set()
            used_gen_indices = set()
            for pair in candidate_pairs:
                real_idx, gen_idx = pair[0], pair[1]
                if real_idx in used_real_indices or gen_idx in used_gen_indices:
                    continue
                top_pairs.append(pair)
                used_real_indices.add(real_idx)
                used_gen_indices.add(gen_idx)
                if len(top_pairs) == target_pair_count:
                    break

            if len(top_pairs) == target_pair_count or pool_size == total_pair_count:
                break
            pool_size = min(total_pair_count, max(pool_size * 2, pool_size + top_k * 20))

        score_text = ", ".join(
            (
                f"#{rank + 1}: real[{real_idx}] ↔ gen[{gen_idx}] "
                f"score={score:.4f}, fft={fft_corr:.4f}, xcorr={time_xcorr:.4f}, diff_mse={diff_mse:.4f}"
            )
            for rank, (real_idx, gen_idx, score, fft_corr, time_xcorr, diff_mse) in enumerate(top_pairs)
        )
        print(f"[{class_name}] Top-{len(top_pairs)} 综合相似配对: {score_text}")

        # 先为当前类别建立一个连续的数据块：时间轴 -> 波形 -> 频率轴 -> FFT
        export_columns[f"{class_name}_time_s"] = t
        for rank, (real_idx, gen_idx, *_metrics) in enumerate(top_pairs):
            prefix = f"{class_name}_top{rank + 1}"
            export_columns[f"{prefix}_real_waveform"] = X_real_c[real_idx].squeeze()
            export_columns[f"{prefix}_generated_waveform"] = X_gen_c[gen_idx].squeeze()

        export_columns[f"{class_name}_freq_hz"] = np.pad(
            freqs,
            (0, len(t) - len(freqs)),
            constant_values=np.nan,
        )
        for rank, (real_idx, gen_idx, *_metrics) in enumerate(top_pairs):
            prefix = f"{class_name}_top{rank + 1}"
            export_columns[f"{prefix}_real_fft_mag"] = np.pad(
                fft_real_all[real_idx],
                (0, len(t) - len(fft_real_all[real_idx])),
                constant_values=np.nan,
            )
            export_columns[f"{prefix}_generated_fft_mag"] = np.pad(
                fft_gen_all[gen_idx],
                (0, len(t) - len(fft_gen_all[gen_idx])),
                constant_values=np.nan,
            )

        # --- 开始画图 ---
        max_freq_plot = 200
        freq_mask = freqs <= max_freq_plot

        for rank, (real_idx, gen_idx, score, fft_corr, time_xcorr, diff_mse) in enumerate(top_pairs):
            real_signal = X_real_c[real_idx].squeeze()
            gen_signal = X_gen_c[gen_idx].squeeze()
            real_fft = fft_real_all[real_idx]
            gen_fft = fft_gen_all[gen_idx]

            ax_time = axes[2 * rank, plot_idx]
            ax_freq = axes[2 * rank + 1, plot_idx]

            # 画时域
            ax_time.plot(t, real_signal, color='tab:blue', alpha=0.7, label=f'Real [{real_idx}]')
            ax_time.plot(t, gen_signal, color='tab:orange', alpha=0.8, label=f'Generated [{gen_idx}]')
            ax_time.set_title(
                f"{class_name} | Pair Top {rank + 1}\n"
                f"Score: {score:.3f} | FFT: {fft_corr:.3f} | XCorr: {time_xcorr:.3f}"
            )
            ax_time.set_xlabel("Time (s)")
            ax_time.set_ylabel("Amplitude")
            if plot_idx == 0:
                ax_time.legend(loc="upper right", fontsize=8)

            # 画频域（只展示低频部分，更便于观察关键峰值）
            ax_freq.plot(freqs[freq_mask], real_fft[freq_mask], color='tab:blue', alpha=0.7)
            ax_freq.plot(freqs[freq_mask], gen_fft[freq_mask], color='tab:orange', alpha=0.8)
            ax_freq.set_xlabel("Frequency (Hz)")
            ax_freq.set_ylabel("Magnitude")

        # 如果某一类可用配对不足 3 个，把剩余子图隐藏掉
        for rank in range(len(top_pairs), top_k):
            axes[2 * rank, plot_idx].axis('off')
            axes[2 * rank + 1, plot_idx].axis('off')

    # 4. 保存所有入选配对的原始数据
    export_path = out_dir / "best_generated_sample_pairs_data.csv"
    with export_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(export_columns.keys())
        for row in zip(*export_columns.values()):
            writer.writerow(row)

    figure_path = out_dir / "best_generated_samples_comparison.png"
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    print(f"对比图已保存为 {figure_path}")
    print(f"原始数据已保存为 {export_path}")
    plt.show()

if __name__ == "__main__":
    main()