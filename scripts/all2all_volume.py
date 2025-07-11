# -*- coding: utf-8 -*-
"""All-to-All 通信量分析脚本

功能：
1. 计算 MoE 训练 / 推理中，各种 EP 规模的 All-to-All 通信量。
2. 同时输出 prefill 与 decode（seq_len = 1）阶段结果。
3. 结果保存为 CSV 及热力图（单位 MB）。

使用：
    python all2all_volume.py  # 直接运行，结果输出到当前目录

后续如需改模型 / EP 配置，可修改下方 `ModelArgs` 与 `EP_CONFIGS`。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ------------------------- 参数定义 -------------------------
class ModelArgs:
    """简化版模型参数，仅保留本脚本需要的字段。"""

    dim: int = 7168  # hidden size
    n_routed_experts: int = 256  # 总专家数
    n_activated_experts: int = 8  # 每 token 激活专家数


class MoEGroup:
    """某种 EP 配置下的通信统计封装。"""

    def __init__(self, args: ModelArgs, ep_num: int, redundant_exp: int):
        self.args = args
        self.ep_num = ep_num
        self.redundant_exp = redundant_exp

        # 每卡持有的专家个数
        self.expert_per_gpu: float = (args.n_routed_experts + redundant_exp) / ep_num

    # ---------- 关键计算 ----------
    def comm_tokens(self, bs: int) -> float:
        """给定每卡 batch_size，返回需要经过 All-to-All 的 token 数。"""

        frac_local = self.expert_per_gpu / self.args.n_routed_experts
        return bs * self.args.n_activated_experts * (1 - frac_local)


# ------------------------- 核心逻辑 -------------------------


def comm_volume_mb(moe: MoEGroup, bs: int, seq_len: int, dtype_bytes: int, hidden: int) -> float:
    """计算单卡通信量 (MB)。"""

    tokens = bs * seq_len
    comm_tokens = moe.comm_tokens(tokens)
    return comm_tokens * hidden * dtype_bytes / 1024 ** 2


def analyse_stage(
    stage: str,
    moe_dict: Dict[str, MoEGroup],
    batch_sizes: List[int],
    seq_lens: List[int],
    dtype_bytes: int,
    hidden: int,
    out_dir: Path,
):
    """生成某个阶段 (prefill / decode) 的表格和热力图。"""

    for name, moe in moe_dict.items():
        rows = []  # (seq_len, bs, MB)
        for s in seq_lens:
            for bs in batch_sizes:
                mb = comm_volume_mb(moe, bs, s, dtype_bytes, hidden)
                rows.append((s, bs, mb))

        df = pd.DataFrame(rows, columns=["seq_len", "batch_size", "comm_MB"])
        pivot = df.pivot_table("comm_MB", index="seq_len", columns="batch_size", aggfunc="mean")

        # 保存 CSV
        csv_path = out_dir / f"all2all_{stage}_{name}.csv"
        pivot.to_csv(csv_path)
        print(f"[INFO] CSV saved to {csv_path}")

        # 绘图
        plt.figure(figsize=(pivot.shape[1] * 2, pivot.shape[0] * 1.5))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
        plt.title(f"{name} {stage.capitalize()} All-to-All per-GPU (MB)")
        plt.xlabel("Per-GPU Batch Size")
        plt.ylabel("Sequence Length")
        png_path = out_dir / f"all2all_{stage}_{name}.png"
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        print(f"[INFO] Figure saved to {png_path}")


# ------------------------- 主入口 -------------------------
def main():
    args = ModelArgs()

    # EP 配置字典: 名称 -> (ep_num, redundant_exp)
    EP_CONFIGS = {
        "EP34": (34, 16),
        "EP72": (72, 32),
        "EP144": (144, 32),
        "EP320": (320, 64),
    }

    moe_dict = {name: MoEGroup(args, ep, red) for name, (ep, red) in EP_CONFIGS.items()}

    # 输入维度 / 数据类型大小
    hidden = args.dim
    dtype_bytes = 2  # bf16

    # ---------------- Prefill ----------------
    prefill_bs = [1, 2, 4, 8, 16, 32]
    prefill_seq = [256, 1024, 2048, 4096, 8192, 16384]

    # ---------------- Decode -----------------
    decode_bs = [8, 16, 32, 64, 128, 256]
    decode_seq = [1]  # 逐 token

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    analyse_stage("prefill", moe_dict, prefill_bs, prefill_seq, dtype_bytes, hidden, out_dir)
    analyse_stage("decode", moe_dict, decode_bs, decode_seq, dtype_bytes, hidden, out_dir)


if __name__ == "__main__":
    main()
