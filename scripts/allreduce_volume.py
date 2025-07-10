import math
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 默认按 fp16 两字节计算
BYTES_FP16 = 2

ModelConfig = dict[str, int]  # {"hidden_size": int, "num_layers": int, "tp_size": int}

# ------------------------------ 模型配置 ------------------------------
# 如果后续需要其它规模，只需在此处填入 h 与 L
MODELS: dict[str, ModelConfig] = {
    # 基于公开资料的大致参数
    # Llama-4 系列
    "Llama-3.1-70B": {"hidden_size": 8192, "num_layers": 80},
    "Llama-3.1-405B": {"hidden_size": 16384, "num_layers": 126},
    # Qwen-3 系列
    "Qwen3-235B-A22B": {"hidden_size": 4096, "num_layers": 94},
    "Qwen3-32b": {"hidden_size": 5120, "num_layers": 64},
}

# ---------------------------- 计算公式 ----------------------------

def allreduce_bytes(bs: int, seq_len: int, h: int, tp: int, bytes_per_param: int = BYTES_FP16) -> int:
    """根据论文公式计算一次 layer 前向需要的 AllReduce 数据量(字节)。

    公式摘自 DeepSpeed/Pytorch TP 论文: 一层有两次 AllReduce，每次数据量 ~2*b*s*h。
    """
    phi = bs * seq_len * h * (tp - 1) / tp  # 总参数量 (unit: element)
    return int(phi * bytes_per_param)


# ---------------------------- 主流程 ----------------------------

def run( prefill_batch_sizes: list[int], decode_batch_sizes: list[int], seq_lens: list[int], out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    tp_size = 4 # tp 8 通信量差别不大
    ################### prefill stage ###################
    for model_name, cfg in MODELS.items():
        rows: list[tuple[int, int, int]] = []  # (bs, seq_len, bytes)
        for bs in prefill_batch_sizes:
            for s in seq_lens:
                al_reduce_size = allreduce_bytes(bs, s, cfg["hidden_size"], tp_size)
                rows.append((bs, s, al_reduce_size))

        df = pd.DataFrame(rows, columns=["batch_size", "seq_len", "allreduce_bytes"])
        csv_path = os.path.join(out_dir, f"{model_name}_prefill_allreduce.csv")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] CSV saved to {csv_path}")

        # 可视化——用 heatmap，x=seq_len, y=batch_size
        pivot = df.pivot_table(index="batch_size", columns="seq_len", values="allreduce_bytes", aggfunc="mean") / (1024 ** 2)  # 转 MB
        plt.figure(figsize=(len(seq_lens) * 2, len(prefill_batch_sizes) * 1.5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title(f"{model_name} Prefill AllReduce Data Volume (MB) ")
        plt.ylabel("Batch Size")
        plt.xlabel("Sequence Length")
        png_path = os.path.join(out_dir, f"{model_name}_prefill_allreduce.png")
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        print(f"[INFO] Figure saved to {png_path}")

    ################### decode stage ###################
    for model_name, cfg in MODELS.items():
        rows: list[tuple[int, int, int]] = []  # (bs, seq_len, bytes)
        for bs in decode_batch_sizes:
            al_reduce_size = allreduce_bytes(bs, 1, cfg["hidden_size"], tp_size)
            rows.append((bs, 1, al_reduce_size))

        df = pd.DataFrame(rows, columns=["batch_size", "seq_len", "allreduce_bytes"])
        csv_path = os.path.join(out_dir, f"{model_name}_decode_allreduce.csv")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] CSV saved to {csv_path}")

        # 可视化——用 heatmap，x=seq_len, y=batch_size
        pivot = df.pivot_table(index="batch_size", columns="seq_len", values="allreduce_bytes", aggfunc="mean") / (1024 ** 2)  # 转 MB
        plt.figure(figsize=(len(seq_lens) * 2, len(decode_batch_sizes) * 1.5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title(f"{model_name} Decode AllReduce Data Volume (MB) ")
        plt.ylabel("Batch Size")
        plt.xlabel("Sequence Length")
        png_path = os.path.join(out_dir, f"{model_name}_decode_allreduce.png")
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        print(f"[INFO] Figure saved to {png_path}")

if __name__ == "__main__":
    # 可自行调整
    prefill_batch_sizes = [1, 2, 4, 8, 16]
    decode_batch_sizes = [8, 16, 32, 64, 128, 256]
    seq_lens = [128, 256, 512, 1024, 2048, 4096]
    run(prefill_batch_sizes, decode_batch_sizes, seq_lens) 