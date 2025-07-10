import pprint
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any
import pandas as pd
from .constants import *

class GPU_perf():
    def __init__(self,gpu_type, sm,comm_sm, gpu_per_node,fp16_flops,fp8_flops,fp4_flops,mem,mem_bw, nvlink_bw,pcie_bw, discount_rate):
        self.gpu_type = gpu_type
        self.sm = sm
        self.gpu_per_node = gpu_per_node
        self.comm_sm = comm_sm
        self.fp16_flops = fp16_flops
        self.fp8_flops = fp8_flops
        self.fp4_flops = fp4_flops
        self.mem = mem
        self.mem_bw = mem_bw
        self.nvlink_bw = nvlink_bw
        self.pcie_bw = pcie_bw
        self.discount_rate = discount_rate

    def get_fp16_flops(self):
        return self.fp16_flops * self.discount_rate  * ( self.sm  - self.comm_sm) / self.sm

    def get_fp8_flops(self):
        return self.fp8_flops *  self.discount_rate * ( self.sm  - self.comm_sm) / self.sm

    def get_fp4_flops(self):
        return self.fp4_flops *  self.discount_rate * ( self.sm  - self.comm_sm) / self.sm

    def get_mem_bw(self):
        return self.mem_bw *  self.discount_rate

    def get_nvlink_bw(self):
        return self.nvlink_bw *  self.discount_rate

    def get_pcie_bw(self):
        return self.pcie_bw *  self.discount_rate

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
# 辅助函数：格式化整数序列为字符串
def format_int_sequence(obj: Any) -> Any:
    """
    递归地格式化整数列表，并对 dict 与 OrderedDict 递归处理，
    保留映射类型。
    """
    # 1. 先处理所有映射类型（dict、OrderedDict 等）
    if isinstance(obj, Mapping):
        # 用 type(obj) 保留原始类型
        return type(obj)(
            (key, format_int_sequence(value))
            for key, value in obj.items()
        )

    # 2. 处理 tuple
    if isinstance(obj, tuple):
        # 如果全是 int，就直接拼成 "[...]"；否则递归每个元素
        if all(isinstance(x, int) for x in obj):
            return f"[{', '.join(map(str, obj))}]"
        return tuple(format_int_sequence(x) for x in obj)

    # 3. 处理 list
    if isinstance(obj, list):
        if all(isinstance(x, int) for x in obj):
            return f"[{', '.join(map(str, obj))}]"
        return [format_int_sequence(x) for x in obj]

    # 4. 其它类型原样返回
    return obj


class Formatter(object):
    @classmethod
    def format_value(cls, value, category):
        """根据类别统一格式化 value."""
        if category == "params" or category == "flops":
            return num_to_string(value)
        elif category == "latency":
            return latency_to_string(value)
        elif category == "memory":
            return f"{num_to_string(value)}B"
        return value  # 如果没有匹配，返回原值

    @classmethod
    def print_format_summary_dict(
        cls,
        summary_dict: dict,
        depth: int,
        category: str | None = None,
    ) -> str:
        """
        打印时对 params / flops / latency / memory 等进行统一转换显示。
        If *category* is provided, apply that formatting to every leaf value that is
        not a nested dict; otherwise fall back to key‑based inference.
        """
        if category is not None and not isinstance(summary_dict, dict):
            # Safety bail‑out (shouldn't happen)
            return summary_dict
        for key, value in summary_dict.items():
            # If category is explicitly provided, ignore key‑name heuristics
            explicit_cat = category
            if (explicit_cat == "params" or explicit_cat == "flops") or ("params" in key or "flops" in key):
                if not isinstance(value, dict):
                    summary_dict.update({key: num_to_string(value)})
                else:
                    self.print_format_summary_dict(
                        value, get_dict_depth(value) - 1, category
                    )  # 递归
            if explicit_cat == "latency" or "latency" in key:
                if not isinstance(value, dict):
                    summary_dict.update({key: latency_to_string(value)})
                else:
                    self.print_format_summary_dict(value, get_dict_depth(value) - 1, category)
            if explicit_cat == "memory" or "memory" in key:
                if not isinstance(value, dict):
                    summary_dict.update({key: f"{num_to_string(value)}B"})
                else:
                    self.print_format_summary_dict(value, get_dict_depth(value) - 1, category)
        if depth >= 1:
            pprint.pprint(summary_dict, indent=4, sort_dicts=False)


def print_list(lst):
    """print one-dimensional list

    :param list: List[int]
    :return: None
    """
    for i, x in enumerate(lst):
        print(x, end="\n")


def get_dict_depth(d, depth=0):
    if not isinstance(d, dict):
        return depth
    if not d:
        return depth

    return max(get_dict_depth(v, depth + 1) for v in d.values())


def latency_to_string(latency_in_s, precision=2, return_type="string"):
    if latency_in_s is None:
        return "None" if return_type == "string" else None

    day = 24 * 60 * 60
    hour = 60 * 60
    minute = 60
    ms = 1 / 1000
    us = 1 / 1000000

    if latency_in_s // day > 0:
        value = round(latency_in_s / day, precision)
        unit = "days"
    elif latency_in_s // hour > 0:
        value = round(latency_in_s / hour, precision)
        unit = "hours"
    elif latency_in_s // minute > 0:
        value = round(latency_in_s / minute, precision)
        unit = "minutes"
    elif latency_in_s > 1:
        value = round(latency_in_s, precision)
        unit = "s"
    elif latency_in_s > ms:
        value = round(latency_in_s / ms, precision)
        unit = "ms"
    else:
        value = round(latency_in_s / us, precision)
        unit = "us"

    if return_type == "string":
        return f"{value} {unit}"
    elif return_type == "float":
        return value
    else:
        return (value, unit)


def num_to_string(num, precision=2, return_type="string"):
    if num is None:
        return "None" if return_type == "string" else None

    if num // 10**12 > 0:
        value = round(num / 10.0**12, precision)
        unit = "T"
    elif num // 10**9 > 0:
        value = round(num / 10.0**9, precision)
        unit = "G"
    elif num // 10**6 > 0:
        value = round(num / 10.0**6, precision)
        unit = "M"
    elif num // 10**3 > 0:
        value = round(num / 10.0**3, precision)
        unit = "K"
    else:
        value = num
        unit = ""

    if return_type == "string":
        return f"{value} {unit}".strip()
    elif return_type == "float":
        return value
    else:
        return (value, unit)


def get_readable_summary_dict(summary_dict: dict, title="Summary") -> str:
    log_str = f"\n{title.center(PRINT_LINE_WIDTH, '-')}\n"
    for key, value in summary_dict.items():
        if "num_tokens" in key or "num_params" in key or "flops" in key:
            log_str += f"{key}: {num_to_string(value)}\n"
        elif "gpu_hours" == key:
            log_str += f"{key}: {int(value)}\n"
        elif "memory" in key and "efficiency" not in key:
            log_str += f"{key}: {num_to_string(value)}B\n"
        elif "latency" in key:
            log_str += f"{key}: {latency_to_string(value)}\n"
        else:
            log_str += f"{key}: {value}\n"
    log_str += f"{'-' * PRINT_LINE_WIDTH}\n"
    return log_str


def within_range(val, target, tolerance):
    if target == 0:
        raise ValueError("Target cannot be zero")
    return abs(val - target) / target < tolerance


def average(lst):
    if not lst:
        return None
    return sum(lst) / len(lst)


def max_value(lst):
    if not lst:
        return None
    return max(lst)
