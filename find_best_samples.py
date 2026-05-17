import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

# 从你的工具包中导入类别名
# 如果报错，可以直接替换为 CLASS_NAMES = ["Normal", "Unbalance", "Misalignment", "Crack", "Looseness"]
from utils.signal_utils import CLASS_NAMES 

def compute_fft(signals, sample_rate=2048):
    """计算单边幅度谱"""
    # 动态判断：如果有3个维度且中间是1，才squeeze
    if signals.ndim == 3 and signals.shape[1] == 1:
        signals = signals.squeeze(1) 
        
    fft_complex = np.fft.rfft(signals, axis=-1, norm="ortho")
    fft_mag = np.abs(fft_complex)
    freqs = np.fft.rfftfreq(signals.shape[-1], d=1/sample_rate)
    return freqs, fft_mag

def main():
    # 1. 配置文件路径 (请根据你的实际路径修改)
    real_data_path = Path("processed/mhta_base_dataset/train.npz")
    gen_data_path = Path("processed/mhta_augmented_dataset/generated_only.npz")
    
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

    # 准备画图的画布 (行：时域和频域，列：类别)
    num_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(2, num_classes, figsize=(4 * num_classes, 6))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # 3. 遍历每个类别，寻找最像的生成样本
    for c_idx, class_name in enumerate(CLASS_NAMES):
        # 筛选当前类别的数据
        real_mask = (y_real == c_idx)
        gen_mask = (y_gen == c_idx)
        
        X_real_c = X_real[real_mask]
        X_gen_c = X_gen[gen_mask]
        
        if len(X_real_c) == 0 or len(X_gen_c) == 0:
            continue
            
        # 计算真实数据的频域平均模板
        _, fft_real_all = compute_fft(X_real_c, sample_rate)
        fft_real_template = np.mean(fft_real_all, axis=0) # 该类别的标准频域特征
        
        # 计算生成的每个样本的频域
        _, fft_gen_all = compute_fft(X_gen_c, sample_rate)
        
        # 计算相似度评估指标
        best_score = -1.0
        best_gen_idx = 0
        
        # 遍历该类别的所有生成样本，打分！
        for i in range(len(fft_gen_all)):
            # 评估指标：频域皮尔逊相关系数
            score, _ = pearsonr(fft_gen_all[i], fft_real_template)
            
            if score > best_score:
                best_score = score
                best_gen_idx = i
                
        # 拿到了最像的那个生成信号
        best_gen_signal = X_gen_c[best_gen_idx].squeeze()
        best_gen_fft = fft_gen_all[best_gen_idx]
        
        # 为了对比，我们也挑一个最具有代表性的真实信号 (和模板最像的)
        real_scores = [pearsonr(fft, fft_real_template)[0] for fft in fft_real_all]
        best_real_idx = np.argmax(real_scores)
        rep_real_signal = X_real_c[best_real_idx].squeeze()
        rep_real_fft = fft_real_all[best_real_idx]

        print(f"[{class_name}] 最佳生成样本相似度得分 (Freq Correlation): {best_score:.4f}")

        # --- 开始画图 ---
        ax_time = axes[0, c_idx]
        ax_freq = axes[1, c_idx]
        
        # 画时域
        ax_time.plot(t, rep_real_signal, color='tab:blue', alpha=0.7, label='Real (Representative)')
        ax_time.plot(t, best_gen_signal, color='tab:orange', alpha=0.8, label='Generated (Best)')
        ax_time.set_title(f"{class_name}\nSim Score: {best_score:.3f}")
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Amplitude")
        if c_idx == 0:
            ax_time.legend(loc="upper right", fontsize=8)
            
        # 画频域 (只展示低频部分，比如前 100 Hz，更能看清 1倍频/2倍频)
        max_freq_plot = 200 
        freq_mask = freqs <= max_freq_plot
        
        ax_freq.plot(freqs[freq_mask], rep_real_fft[freq_mask], color='tab:blue', alpha=0.7)
        ax_freq.plot(freqs[freq_mask], best_gen_fft[freq_mask], color='tab:orange', alpha=0.8)
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude")

    plt.tight_layout()
    plt.savefig("best_generated_samples_comparison.png", dpi=300)
    print("对比图已保存为 best_generated_samples_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()