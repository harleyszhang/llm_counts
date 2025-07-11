# -*- coding  : utf-8 -*-
# Description : gpu, model, Parallelism, data, train and inference config definition

import math, json
from .constants import *
from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering
from transformers import AutoConfig
import os


class ActivationRecomputation(Enum):
    NONE = 0
    """No activation recomputation; requires the most amount of memory."""

    SELECTIVE = 1
    """Selectively checkpoints and recomputes only parts of each transformer
    layer that take up a considerable amount of memory but are not
    computationally expensive to recompute, i.e. Q K V matrix multiplies, 
    QK^T matrix multiply, softmax, softmax dropout, and attention over V."""

    FULL = 2
    """Full activation recomputation stores the input to EVERY transformer
    layer, which is sharded across the tensor parallel group, thus requiring an
    extra all-gather (ignored for now) per layer and add communication
    overhead; requires the lease amount of memory; requires an extra forward
    pass."""


@total_ordering
class DSZeRO(Enum):
    NONE = 0
    """No DeepSPeed ZeRO; requires the most amount of memory."""

    STAGE_1 = 1
    """ZeRO stage 1 shards the optimizer states across the data parallel
    group."""

    STAGE_2 = 2
    """ZeRO stage 2 shards the optimizer states and gradients across the data
    parallel group."""

    STAGE_3 = 3
    """ZeRO stage 3 shards the optimizer states, gradients, and model weights
    across the data parallel group."""

    def __lt__(self, other):
        # 炫技写法
        if other.__class__ is self.__class__:
            return self.value < other.value  # Enum 枚举类自动赋值
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, DSZeRO):
            return self.value == other.value
        return NotImplemented


@dataclass
class GPUEfficiencyConfig:
    flops_efficiency: float = 1.0
    hbm_memory_efficiency: float = 1.0
    intra_node_memory_efficiency: float = 1.0
    inter_node_memory_efficiency: float = 1.0


@dataclass
class InferenceConfig:
    """Inference configuration dataclass."""

    bs: int = None  # batch size
    seq_len: int = 522  # input sequence length
    generate_len: int = 1526  # number of tokens to generate
    context_len: int = None  # context length
    bytes_per_param: int = BYTES_FP16  # model weight bytes
    act_dtype_bytes: int = BYTES_FP16  # activation data type bytes
    kv_cache_bytes: int = BYTES_FP16  # key/value cache data type bytes

    def __post_init__(self):
        if self.context_len is None:
            self.context_len = self.seq_len + self.generate_len


@dataclass
class ParallelismConfig:
    """Configuration for various parallelism strategies."""

    tp_size: int = 1  # tensor parallelism size
    pp_size: int = 1  # pipeline parallelism size
    dp_size: int = 1  # data parallelism size
    sp_size: int = 1  # sequence parallelism size

@dataclass
class DeepseekConfig:
    """Deepseek configuration dataclass."""
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_layers: int = 61  # number of transformer layers (blocks)
    num_dense_layers: int = 3  # number of dense layers (non-MoE layers)
    num_moe_layers: Optional[int] = None 
    num_heads: int = 128
    num_kv_heads: int = 128
    
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5

    # mla config
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_head_dim: Optional[int] = None 

    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    # 模型类型标识
    model_type: str = "deepseek_v3"
    model_name: str = "DeepseekV3"

    def __post_init__(self):
        if self.qk_nope_head_dim and self.qk_rope_head_dim:
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.num_moe_layers is None:
            self.num_moe_layers = self.num_layers - self.num_dense_layers

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, trust_remote_code: bool = True
    ):
        """
        Load a Hugging Face model configuration and map it to DeepseekConfig.
        自动兼容 Deepseek V3 模型。
        """
        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )

        # 检查是否为 Deepseek 模型
        model_type = getattr(hf_config, "model_type", None)
        if model_type != "deepseek_v3":
            raise ValueError(f"Expected deepseek_v3 model type, got {model_type}")

        return cls(
            num_layers=getattr(hf_config, "num_hidden_layers", None),
            num_dense_layers=getattr(hf_config, "num_dense_layers", None),
            num_moe_layers=getattr(hf_config, "num_moe_layers", None),
            num_heads=getattr(hf_config, "num_attention_heads", None),
            num_kv_heads=getattr(hf_config, "num_kv_heads", None),
            hidden_size=getattr(hf_config, "hidden_size", None),
            intermediate_size=getattr(hf_config, "intermediate_size", None),
            moe_intermediate_size=getattr(hf_config, "moe_intermediate_size", None),
            vocab_size=getattr(hf_config, "vocab_size", None),
            max_seq_len=getattr(hf_config, "max_position_embeddings", None),
            n_routed_experts=getattr(hf_config, "n_routed_experts", None),
            n_shared_experts=getattr(hf_config, "n_shared_experts", None),
            num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", None),
            q_lora_rank=getattr(hf_config, "q_lora_rank", None),
            kv_lora_rank=getattr(hf_config, "kv_lora_rank", None),
            qk_nope_head_dim=getattr(hf_config, "qk_nope_head_dim", None),
            qk_rope_head_dim=getattr(hf_config, "qk_rope_head_dim", None),
            v_head_dim=getattr(hf_config, "v_head_dim", None),
            original_seq_len=getattr(hf_config, "original_seq_len", None),
            rope_theta=getattr(hf_config, "rope_theta", None),
            rope_factor=getattr(hf_config, "rope_factor", None),
            beta_fast=getattr(hf_config, "beta_fast", None),
            beta_slow=getattr(hf_config, "beta_slow", None),
            mscale=getattr(hf_config, "mscale", None),
        )

@dataclass
class ModelConfig:
    num_layers: Optional[int] = None  # number of transformer layers (blocks)
    num_heads: Optional[int] = None  # number of attention heads
    head_dim: Optional[int] = None          # <— 新增：允许显式传入
    hidden_size: Optional[int] = None  # hidden dimension
    vocab_size: Optional[int] = None  # vocabulary size
    num_kv_heads: Optional[int] = None
    max_seq_len: Optional[int] = None  # max sequence length
    intermediate_size: Optional[int] = None  # hidden dimension of FFN, default to 4 * hidden_size
    
    model_type: str = None 
    model_name: str = None

    # 新增 MoE 相关参数
    moe_intermediate_size: Optional[int] = None  # MoE FFN hidden dimension
    num_experts: Optional[int] = None  # MoE 专家数
    moe_layer_distribution: Optional[list] = None  # MoE层分布
    num_experts_per_tok: int = 8 # 每个 token 选择 top-k 个专家,这里假设k=8

    def __post_init__(self) -> None:
        # ① KV-heads 默认 = Q-heads
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        # ② FFN 维度默认 = 4×hidden_size
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4

        # ③ head_dim 计算
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads

        # ④ MoE 相关参数处理
        if self.moe_intermediate_size is not None:
            self.intermediate_size = self.moe_intermediate_size

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, trust_remote_code: bool = True
    ):
        """
        Load a Hugging Face model configuration and map it to ModelConfig.
        自动兼容 Qwen3 MoE 及常规模型。
        """
        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )

        model_type = getattr(hf_config, "model_type", None)
        num_layers = getattr(hf_config, "num_hiddenum_layers", None)
        num_heads = getattr(hf_config, "num_attention_heads", None)
        num_kv_heads = getattr(hf_config, "num_kv_heads", None)
        head_dim = getattr(hf_config, "head_dim", None)
        moe_num_experts = getattr(hf_config, "num_experts", None) # 兼容 qwen3 moe 模型专家数量字段
        moe_layer_distribution = getattr(hf_config, "moe_layer_distribution", None)

        return cls(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=getattr(hf_config, "hidden_size", None),
            vocab_size=getattr(hf_config, "vocab_size", None),
            num_kv_heads=num_kv_heads,
            max_seq_len=getattr(hf_config, "max_position_embeddings", None),
            intermediate_size=getattr(hf_config, "intermediate_size", None),
            model_type=model_type,
            head_dim=head_dim,
            num_experts=moe_num_experts,
            moe_layer_distribution=moe_layer_distribution,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=getattr(hf_config, "hidden_size", None),
            vocab_size=getattr(hf_config, "vocab_size", None),
            num_kv_heads=num_kv_heads,
            max_seq_len=getattr(hf_config, "max_position_embeddings", None),
            intermediate_size=getattr(hf_config, "intermediate_size", None),
            model_type=model_type,
            head_dim=head_dim,
            num_experts=moe_num_experts,
            moe_layer_distribution=moe_layer_distribution,
        )
    
    
@dataclass
class GPUConfig:
    # 1, gpu 型号和显存大小
    name: str  # GPU config name
    memory_in_GB: float  # memory per GPU in GB
    onchip_buffer: float = None  # on-chip buffer size in bytes, e.g., register file size
    gpu_per_node: int = 8
    sm: int = 160
    comm_sm: int = 20
    
    # 2, gpu 显存带宽、节点内带宽、节点间带宽
    hbm_bw: float=None  # GPU HBM bw in GB/s
    intra_node_bw: float=None # intra node GPU bw in GB/s.(PCIE/NVLINK)
    intra_node_min_message_latency: float=None # minimum intra node message latency in seconds
    
    inter_node_bw: float = 50 # inter node bw in GB/s, assuming Mellanox 400Gbps HDR Infiniband
    discount_rate: float = 0.85

    # 3, 不同精度的 Tensor core 的计算性能
    peak_tf32_TFLOPS: float = None  # peak Tensor TFLOPS for FP32
    peak_fp16_TFLOPS: float = None  # peak Tensor TFLOPS for FP16
    peak_fp8_TFLOPS: float = None  # peak Tensor TFLOPS for FP16
    peak_fp4_TFLOPS: float = None  # peak Tensor TFLOPS for FP16
    peak_int8_TFLOPS: float = None  # peak Tensor TFLOPS for INT8
    peak_int4_TFLOPS: float = None  # peak Tensor TFLOPS for INT4

    FLOPS_EFFICIENCY = 1.0
    HBM_MEMORY_EFFICIENCY = 0.9
    INTRA_NODE_bw_EFFICIENCY = 0.9

    def __post_init__(self):
        """
        Post-initialization processing to compute missing values and apply efficiencies.
        """

        # Apply FLOPS efficiency and round to nearest integer
        if self.FLOPS_EFFICIENCY:
            self.actual_peak_tf32_TFLOPS = math.ceil(
                self.peak_tf32_TFLOPS * self.FLOPS_EFFICIENCY
            )
            self.actual_peak_fp16_TFLOPS = math.ceil(
                self.peak_fp16_TFLOPS * self.FLOPS_EFFICIENCY
            )
            self.actual_peak_int8_TFLOPS = math.ceil(
                self.peak_int8_TFLOPS * self.FLOPS_EFFICIENCY
            )

    def get_fp16_tflops(self):
        return self.peak_fp16_TFLOPS * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp8_tflops(self):
        return self.peak_fp8_TFLOPS * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp4_tflops(self):
        return self.peak_fp4_TFLOPS * self.discount_rate * (self.sm - self.comm_sm) / self.sm
    
    def get_hbm_bw(self):
        return self.hbm_bw * self.discount_rate
    
    def get_nvlink_bw(self):
        return self.intra_node_bw * self.discount_rate // 2 # 单向 nvlink 带宽

    def get_pcie_bw(self):
        return self.inter_node_bw * self.discount_rate
    
class LLMConfigs(object):
    """LLMConfigs is a dataclass that contains all the configurations for the LLM model."""

    def __init__(
        self,
        gpu_config: GPUConfig,
        model_config: Union[ModelConfig, DeepseekConfig],
        parallelism_config: ParallelismConfig = ParallelismConfig(),
        inference_config: InferenceConfig = InferenceConfig(),
        gpu_efficiency_config: GPUEfficiencyConfig = GPUEfficiencyConfig(),
    ) -> None:
        self.model_config = model_config
        self.gpu_config = gpu_config
        self.parallelism_config = parallelism_config
        self.inference_config = inference_config  # 用户自行指定配置
        self.gpu_efficiency_config = gpu_efficiency_config  # 用户自行指定配置


def get_model_and_gpu_config_by_name(
    model_name="llama-13b", gpu_name="v100-pcie-32gb"
) -> dict:
    """Read model and gpu configs from a json file."""
    current_dir = os.path.dirname(__file__)
    model_config_path = os.path.join(current_dir, "../configs/model_configs.json")
    gpu_config_path = os.path.join(current_dir, "../configs/gpu_configs.json")

    with open(model_config_path, "r") as f:
        config_json = json.load(f)  # 类似于 dict 类型
        if model_name in config_json:
            config_dict = config_json[model_name]
            
            # 检查是否为 Deepseek 模型
            if config_dict.get("model_type") == "deepseek_v3":
                model_config = DeepseekConfig(**config_dict)
            else:
                model_config = ModelConfig(**config_dict)
        else:
            print(
                f"model name {model_name} is not found in {model_config_path} so need to apply transformers AutoConfig"
            )
            # 加载模型配置
            try:
                # 先尝试加载为 Deepseek 模型
                model_config = DeepseekConfig.from_pretrained(model_name, trust_remote_code=True)
            except (ValueError, AttributeError):
                # 如果不是 Deepseek 模型，则加载为普通模型
                model_config = ModelConfig.from_pretrained(model_name, trust_remote_code=True)

    with open(gpu_config_path, "r") as f:
        config_json = json.load(f)
        if gpu_name not in config_json:
            raise ValueError(f"gpu name {gpu_name} not found in {gpu_config_path}")
        gpu_config = GPUConfig(**config_json[gpu_name])

    return model_config, gpu_config


def get_flops(
    gpu_config: GPUConfig, data_type="fp16", flops_efficiency=FLOPS_EFFICIENCY
) -> float:
    """Get the expected TFLOPS per GPU for the specified data type
    configuration/GPU (adjusted by flops_efficiency)

    Returns:
        float: TFLOPS per GPU and unit is T.
    """
    if data_type == "int8":
        gemm_TFOPS = gpu_config.peak_int8_TFLOPS
    elif data_type == "fp16":
        gemm_TFOPS = gpu_config.peak_fp16_TFLOPS
    elif data_type == "fp8":
        gemm_TFOPS = gpu_config.peak_fp8_TFLOPS
    elif data_type == "fp4":
        gemm_TFOPS = gpu_config.peak_fp4_TFLOPS
    else:
        raise ValueError("data_type must be 'fp16' or 'int8'")

    return gemm_TFOPS * flops_efficiency


def get_gpu_hbm_bw(
    gpu_config: GPUConfig, hbm_memory_efficiency=HBM_MEMORY_EFFICIENCY
) -> list:
    return gpu_config.hbm_bw * hbm_memory_efficiency, gpu_config.onchip_buffer


def get_intra_node_bw(
    gpu_config: GPUConfig, intra_node_memory_efficiency=INTRA_NODE_MEMORY_EFFICIENCY
) -> float:
    return gpu_config.intra_node_bw * intra_node_memory_efficiency


def get_inter_node_bw(
    gpu_config: GPUConfig, inter_node_memory_efficiency=INTER_NODE_MEMORY_EFFICIENCY
) -> float:
    return gpu_config.inter_node_bw * inter_node_memory_efficiency
