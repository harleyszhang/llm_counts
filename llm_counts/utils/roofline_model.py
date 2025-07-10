from dataclasses import dataclass
from typing import Sequence, List, Dict
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 全局常量：最大算术强度和最大 TFLOPS，用于坐标轴范围
MAX_OI = 1400
MAX_TFLOPS = 2000

@dataclass(frozen=True)
class GPU:
    """GPU 硬件参数：FP16 峰值算力（TFLOPS）和带宽（TB/s）。"""
    name: str
    fp16_tflops: float  # TFLOPS
    hbm_bw: float       # TB/s

@dataclass(frozen=True)
class ModelConfig:
    """模型配置参数"""
    name: str
    total_flops: float   # 总 FLOP (以 TeraFLOP 为单位)
    total_bytes: float   # 总内存访问 (以 TeraByte 为单位)
    color: str           # 绘图颜色

def roofline_analysis(
    peak_flops: float,
    bw: float,
    total_flops: float,
    total_mac_bytes: float
) -> tuple[float, float, str]:
    """
    Analyzes the roofline model and returns the arithmetic intensity, 
    attainable FLOPs, and the bounding factor (memory or compute).
    """
    if total_mac_bytes == 0:  # 防止除以零
        return 0, peak_flops, "compute"
    
    ai = total_flops / total_mac_bytes
    turning_point = peak_flops / bw

    if ai < turning_point:
        return ai, ai * bw, "memory"
    else:
        return ai, peak_flops, "compute"

def plot_roofline(
    models: Sequence[ModelConfig],
    gpus: Sequence[GPU],
    output_file: str = "roofline_optimized.png"
) -> None:
    """
    绘制经过优化的、用于比较的 Roofline 曲线。

    主要优化点:
    1. 使用 Log-Log 坐标轴，符合行业标准。
    2. 采用智能图例管理，避免图例冗长。
    3. 使用 adjust_text 自动防止文本标签重叠。
    4. 优化视觉设计，突出重点信息。
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_colors = ['red', 'blue', 'green', 'orange', 'purple']
    # --- 1. 绘制 GPU 屋顶线 (作为背景) ---
    # 使用对数坐标轴，范围从 0.1 到 10000
    oi_range = np.logspace(-1, 4, 200)
    gpu_linestyles = ['-', '--', '-.', ':']

    for i, gpu in enumerate(gpus):
        roofline = np.minimum(oi_range * gpu.hbm_bw, gpu.fp16_tflops)
        linestyle = gpu_linestyles[i % len(gpu_linestyles)]
        # 使用统一的灰色系，不同线型来区分，作为背景不干扰主要数据
        ax.plot(
            oi_range,
            roofline,
            linestyle=linestyle,
            linewidth=2,
            label=f"{gpu.name} Roof (Turn @ {gpu.fp16_tflops / gpu.hbm_bw:.1f})",
            color=plot_colors[i % len(plot_colors)],
            alpha=0.9
        )

    # --- 2. 绘制模型性能点并收集文本标签 ---
    text_labels = []

    for model in models:
        # **智能图例技巧**: 为每个模型创建一个“虚拟”的图例条目，
        # 这样图例中每个模型只显示一次。
        ax.scatter([], [], color=model.color, marker='o', s=120, label=f"{model.name}")

        for gpu in gpus:
            ai, attainable, bound = roofline_analysis(
                gpu.fp16_tflops,
                gpu.hbm_bw,
                model.total_flops,
                model.total_bytes
            )

            ax.scatter(
                ai,
                attainable,
                s=120,
                marker='o', # 使用统一标记，用颜色区分模型
                color=model.color,
                edgecolors='black',
                zorder=5  # 确保点在最上层
            )

            # 准备文本标签，稍后由 adjust_text 统一处理
            label_text = f"{gpu.name}\n{attainable:.0f} TFLOPS ({bound[:3]}.)"
            text_labels.append(
                ax.text(ai, attainable, label_text, fontsize=9, ha='center')
            )

    # --- 3. 图表美化与最终处理 ---
    # 切换到对数坐标轴
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel("Arithmetic Intensity (FLOPs / Bytes) [log scale]", fontsize=12)
    ax.set_ylabel("Attainable Performance (TFLOPS) [log scale]", fontsize=12)
    ax.set_title("Comparative Roofline Analysis", fontsize=16, fontweight='bold')
    
    # 使用 'both' 在主次刻度上都显示网格，对 log 尺度很友好
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # 自动调整坐标轴范围，并留出一些边距
    ax.autoscale(True)
    ax.set_xlim(left=max(ax.get_xlim()[0], 0.5))
    ax.set_ylim(bottom=max(ax.get_ylim()[0], 10))

    # **关键步骤**: 调用 adjust_text 来智能地防止标签重叠
    # 它会自动移动标签，并可以用箭头指向原始数据点
    from adjustText import adjust_text
    adjust_text(
        text_labels,
        ax=ax,
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)
    )

    # 图例现在很简洁，可以优雅地放在图内
    ax.legend(fontsize=10, loc='lower right')

    fig.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Optimized roofline plot saved to {output_file}")
    plt.close(fig)


def main():
    # 预定义 GPU 配置
    GPUS = [
        GPU("H100", 989, 3.35),
        GPU("A100", 312, 2.039),
        GPU("RTX4090", 330, 1.008),
        GPU("MI300X", 1150, 5.2),
        GPU("L40S", 363, 0.864),
    ]
    
    # 预定义模型配置
    MODELS = {
        "gpt3": ModelConfig(
            "GPT-3 (175B)", 
            total_flops=314000,  # TFLOPs (3.14e14 FLOPs)
            total_bytes=1000,    # TB (1e15 bytes)
            color='red'
        ),
        "llama2-70b": ModelConfig(
            "LLaMA2-70B", 
            total_flops=70000,   # TFLOPs (7e13 FLOPs)
            total_bytes=200,     # TB (2e14 bytes)
            color='blue'
        ),
        "qwen2.5-3b": ModelConfig(
            "Qwen2.5-3B", 
            total_flops=3000,    # TFLOPs (3e12 FLOPs)
            total_bytes=10,      # TB (1e13 bytes)
            color='green'
        ),
    }
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Roofline Model Analysis Tool")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=["gpt3", "llama2-70b", "qwen2.5-3b"],
                        help="Models to analyze (default: all)")
    parser.add_argument("--gpus", nargs="+", 
                        default=["H100", "A100", "RTX4090"],
                        help="GPUs to analyze (default: H100, A100, RTX4090)")
    parser.add_argument("--output", default="roofline_analysis.png",
                        help="Output filename (default: roofline_analysis.png)")
    
    args = parser.parse_args()
    
    # 获取选中的模型和GPU
    selected_models = [MODELS[model] for model in args.models]
    selected_gpus = [gpu for gpu in GPUS if gpu.name in args.gpus]
    
    if not selected_gpus:
        print("Error: No valid GPUs selected. Available options:")
        for gpu in GPUS:
            print(f"  - {gpu.name}")
        return
    
    # 生成屋顶线图
    plot_roofline(selected_models, selected_gpus, args.output)

if __name__ == "__main__":
    main()