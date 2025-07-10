# -*- coding: utf-8 -*-
"""cli_perf_visual.py

LLM 性能可视化 CLI 脚本
==========================

改进要点：
1. 删除通配符导入，避免潜在命名冲突。
2. 使用 `logging` 替代 `print`，便于日志等级控制。
3. 引入 `argparse` 作为命令行接口，提升脚本通用性。
4. 添加类型提示与中文文档字符串，提高可读性及 IDE 体验。
5. 封装 `llm_profile` 调用，统一异常处理。
"""

from __future__ import annotations

import argparse
import logging
import math
from pprint import pprint
from typing import Dict, List

from llm_counts.benchmark_analyzer import llm_profile

# -----------------------------------------------------------------------------
# 日志配置
# -----------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 默认配置常量
# -----------------------------------------------------------------------------

MODEL_NAME_LIST: List[str] = [
    "llama-7B",
    "llama-13B",
    "llama-65B",
    "llama2-70B",
    "internlm-20B",
]

GPU_NAME_LIST: List[str] = [
    "a30-sxm-24gb",
    "a40-pcie-48gb",
    "a100-sxm-40gb",
    "a100-sxm-80gb",
    "910b-64gb",
    "v100-sxm-32gb",
    "t4-pcie-15gb",
]

TP_NUMS_LIST: List[int] = [1, 2, 4, 8]

# -----------------------------------------------------------------------------
# 内部工具函数
# -----------------------------------------------------------------------------

def _safe_llm_profile(**kwargs):
    """包装 `llm_profile`，捕获所有异常。

    返回 `(result, error)`，其中 result 为 None 表示失败。
    """

    try:
        result, _ = llm_profile(**kwargs)
        return result, None
    except Exception as exc:  # pylint: disable=broad-except
        return None, exc

# -----------------------------------------------------------------------------
# 公开接口
# -----------------------------------------------------------------------------

def generate_tgi_service_dict_list(
    model_name_list: List[str] = MODEL_NAME_LIST,
    gpu_name_list: List[str] = GPU_NAME_LIST,
    tp_nums_list: List[int] = TP_NUMS_LIST,
    seq_len: int = 1024,
    generate_len: int = 1024,
) -> List[Dict]:
    """生成所有模型/GPU/TP 组合的服务参数列表。"""

    service_dict_list: List[Dict] = []

    for model_name in model_name_list:
        # 如需对特定模型设置不同默认值可在此处理
        for gpu_name in gpu_name_list:
            for tp_size in tp_nums_list:
                args = dict(
                    model_name=model_name,
                    gpu_name=gpu_name,
                    tp_size=tp_size,
                    seq_len=seq_len,
                    generate_len=generate_len,
                    print_flag=False,
                    visual_flag=False,
                )

                result, error = _safe_llm_profile(**args)
                if error is not None or result is None:
                    logger.warning(
                        "llm_profile 失败: model=%s, gpu=%s, tp=%s, error=%s",
                        model_name,
                        gpu_name,
                        tp_size,
                        error,
                    )
                    continue

                max_batch_total_tokens = int(result["max_batch_total_tokens"])
                service_dict = {
                    "model_name": model_name,
                    "gpu_name": gpu_name,
                    "tp_size": tp_size,
                    "max_batch_total_tokens": max_batch_total_tokens,
                    "max_bs": math.floor(max_batch_total_tokens / (seq_len + generate_len)),
                }
                service_dict_list.append(service_dict)

    return service_dict_list


def print_all_llm_analyzer() -> None:
    """打印所有组合的分析结果。"""

    service_dict_list = generate_tgi_service_dict_list()
    logger.info(
        "================ TGI+LightLLM service max_batch_total_tokens params list ================"
    )
    pprint(service_dict_list)

# -----------------------------------------------------------------------------
# CLI 入口
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="LLM performance visualizer")
    parser.add_argument("--model-name", type=str, default="Qwen3-30B-A3B", help="模型名称")
    parser.add_argument("--gpu-name", type=str, default="a100-sxm-40gb", help="GPU 型号")
    parser.add_argument("--tp", type=int, default=8, help="张量并行大小")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--seq-len", type=int, default=1024, help="输入序列长度")
    parser.add_argument("--gen-len", type=int, default=128, help="生成长度")
    parser.add_argument("--print-all", action="store_true", help="是否打印全部组合结果")

    return parser.parse_args()


def main() -> None:
    """脚本主入口。"""

    args = parse_args()

    # 单次 profile
    llm_profile(
        model_name=args.model_name,
        gpu_name=args.gpu_name,
        tp_size=args.tp,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        generate_len=args.gen_len,
        print_flag=True,
        visual_flag=True,
    )

    # 可选打印全部分析
    if args.print_all:
        print_all_llm_analyzer()


if __name__ == "__main__":
    main()

"""
python cli_perf_visual.py \
    --model-name Qwen3-30B-A3B \
    --gpu-name a100-sxm-40gb \
    --tp 1 --batch-size 16 \
    --seq-len 1024 --gen-len 128 \
    --print-all
"""