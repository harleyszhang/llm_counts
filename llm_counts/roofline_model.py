import matplotlib.pyplot as plt

MAX_OI = 1400  # 硬件运算强度最大值
MAX_TFLOPS = 1400  # 单个 GPU 最大算力

# nvdia gpus 硬件参数，单位都是 10e12
HW_910B = {"fp16_tflops": 376, "hbm_bw": 0.46}
A40_SXM_GPU = {"fp16_tflops": 149.7, "hbm_bw": 0.696}  # TFLOPS, TB/s
V100_SXM_GPU = {"fp16_tflops": 125, "hbm_bw": 0.9}
RTX4090_PCIE_GPU = {"fp16_tflops": 330, "hbm_bw": 1.008}
A100_SXM_GPU = {"fp16_tflops": 312, "hbm_bw": 2.039}
H100_SXM_GPU = {"fp16_tflops": 989, "hbm_bw": 3.35}  # 不开启稀疏计算
H20_SXM_GPU = {"fp16_tflops": 148, "hbm_bw": 4.0}


def roofline_analysis(peak_flops, bandwidth, flops, mac_bytes):
    """
    Analyzes the roofline model and returns the arithmetic intensity, attainable FLOPs,
    and the bounding factor (memory or compute).

    Parameters:
        - bandwidth: Peak memory bandwidth in bytes per second.
        - peak_flops: Peak floating-point performance in FLOP/s.
        - flops: Total floating-point operations.
        - mac_bytes: Total memory access in bytes.

    Returns:
        - arithmetic_intensity: Operations per byte.
        - attainable_flops: The attainable FLOPs based on the roofline model.
        - bound: The limiting factor ('memory' or 'compute').
    """
    # Calculate arithmetic intensity and turning point
    arithmetic_intensity = flops / mac_bytes
    turning_point = peak_flops / bandwidth

    # Determine the bound and attainable FLOPs
    if arithmetic_intensity < turning_point:
        bound = "memory"
        attainable_flops = arithmetic_intensity * bandwidth
    else:
        bound = "compute"
        attainable_flops = peak_flops

    return arithmetic_intensity, attainable_flops, bound


def plot_model_roofline_graph(model_data, gpus, colors, labels):
    """
    coef = peak_perf / mem_bw
    Parameters:
        - model_data: data: 这是一个包含算术强度 (Arithmetic Intensity, AI)、FLOPs 和模型名称的列表,
            形如 [(ai1, gflops1, model1), (ai2, gflops2, model2), ...]。
        - peak_perf: Peak Floating Point Performance, TFLOPS
        - mem_bw:Peak Memory Bandwidth, tb/s
    """
    flops, mac, model_name = (
        model_data[0] / 1e12,
        model_data[1] / 1e12,
        model_data[2],
    )

    for gpu, color, label in zip(gpus, colors, labels):
        peak_flops = gpu["fp16_tflops"]
        bw = gpu["hbm_bw"]
        oi_intervals = range(MAX_OI)  # 硬件运算强度范围
        tflops_intervals = [min(oi_value * bw, peak_flops) for oi_value in oi_intervals]
        plt.plot(
            oi_intervals,
            tflops_intervals,
            color=color,
            label=f"{label}, OI: {peak_flops / bw: .1f}",
        )

        ai, attainable_flops, bound = roofline_analysis(
            peak_flops, bw, flops, mac
        )
        attainable_tflops = attainable_flops
        plt.scatter(
            ai,
            attainable_tflops,
            color=color,
            label=f"{model_name} (AI: {ai:.1f}, TFlops: {attainable_tflops:.1f}, Bound: {bound})",
            s=50,
        )
        plt.text(
            ai,
            attainable_tflops,
            f" {model_name}",
            color=color,
            va="bottom",
            ha="right",
        )

    plt.xlim(0, MAX_OI)
    plt.ylim(0, MAX_TFLOPS)
    plt.xlabel("Operationl Intensity (FLOPs/Byte)")  # Arithmetic Intensity
    plt.ylabel("Attainable TFlops/s")
    plt.title("Roofline Model of NVIDIA GPUs")
    plt.grid(True)
    plt.legend(fontsize="small", loc="best")
    plt.show()

