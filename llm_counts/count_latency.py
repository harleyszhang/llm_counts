from .utils.config import (
    LLMConfigs, 
    DeepseekConfig,
    get_gpu_hbm_bw,
    get_intra_node_bw,
    get_inter_node_bw,
    get_flops,
)
from .utils.constants import *
from .utils.utils import latency_to_string, num_to_string

from .count_flops import CountCausalLMFlops, CountDeepseekV3Flops
from .count_params import CountCausalLMParams
from .count_memory import CountCausalLMMemory, CountDeepseekV3Memory

import math

def n_pow2_range(x):
    """将数值映射到最接近的2的幂次方"""
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))

class CountCausalLMLatency(object):
    """Count latency by roof-line performance model."""

    def __init__(self, llm_configs: LLMConfigs) -> None:
        self.model_config = llm_configs.model_config
        self.gpu_config = llm_configs.gpu_config
        self.inference_config = llm_configs.inference_config
        self.parallelism_config = llm_configs.parallelism_config

        self.h = self.model_config.hidden_size
        self.l = self.model_config.num_layers
        self.V = self.model_config.vocab_size

        self.bs = llm_configs.inference_config.bs
        self.s = llm_configs.inference_config.seq_len
        self.o = llm_configs.inference_config.generate_len
        self.bsytes_per_param = llm_configs.inference_config.bytes_per_param

        self.tp_size = self.parallelism_config.tp_size
        self.pp_size = self.parallelism_config.pp_size
        self.num_layers_per_gpu = int(self.l / self.parallelism_config.pp_size)

        self.gpu_hbm_bw, self.onchip_buffer = (
            get_gpu_hbm_bw(self.gpu_config, HBM_MEMORY_EFFICIENCY)
        )  
        self.gpu_hbm_bw *= 10**9 
        
        self.gpu_intra_node_bw = (
            get_intra_node_bw(self.gpu_config, INTRA_NODE_MEMORY_EFFICIENCY)
            * 10**9
        )  # intra node bw，GB/s
        self.gpu_TFLOPS = (
            get_flops(self.gpu_config, flops_efficiency=FLOPS_EFFICIENCY)
            * 10**12
        )  # TFLOPS

        self.llm_params = CountCausalLMParams(self.model_config)
        self.llm_memory = CountCausalLMMemory(llm_configs)
        self.llm_flops = CountCausalLMFlops(self.model_config)

    @staticmethod
    def print_kernel_bound_info(stage, memory_latency, compute_latency, ops_type):
        """Print the kernel bound information for the given stage."""
        if memory_latency > compute_latency:
            print(
                f"{stage} stage: memory_latency {latency_to_string(memory_latency)} \
                > compute_latency {latency_to_string(compute_latency)}, this {ops_type} layer is memory bound!"
            )
        else:
            print(
                f"{stage} stage: memory_latency {latency_to_string(memory_latency)} \
                <= compute_latency {latency_to_string(compute_latency)}, this {ops_type} layer is compute bound!"
            )

    def common_latency_for_ops(
        self,
        bs: int,
        seq_len: int,
        generate_len: int = 0,
        ops_type: str = "qkvo_proj",
        stage="decode_",
        print_bound: bool = False,
    ) -> float:
        """Count the latency for the forward layer or model, assuming the compute and memory operations are perfectly overlapped.

        Args:
            flops (float): flops of the forward layer or model
            memory (float): r/w memory(bytes) of the forward layer or model
            tp_size (float): tensor parallelism size
            gpu_TFLOPS (float): GPU TFLOPS in T(10^12)FLOPS
            gpu_hbm_bw (float): GPU HBM bw in GB/s(10^9)

        Returns:
            float: the latency in seconds for the forward pass
        """
        ops_type = ops_type.lower()

        if ops_type == "qkvo_proj":
            flops = (
                self.llm_flops.flops_per_layer_qkvo_proj(bs, seq_len)
                / self.tp_size
            )
            weight_memory = (
                self.llm_params.params_per_layer_mha()
                * self.bsytes_per_param
                / self.tp_size
            ) * BYTES_FP16
            mac = self.llm_memory.mac_per_layer_qkvo_proj(bs, seq_len)[1] / self.tp_size
            
            memory = weight_memory + mac
        elif ops_type == "attn_kernel":
            flops = (
                self.llm_flops.flops_per_layer_attn_kernel(bs, seq_len, generate_len)
                / self.tp_size
            )
            weight_memory = 0
            mac = self.llm_memory.mac_per_layer_attn_kernel(bs, seq_len, generate_len, kv_cache_bytes=BYTES_FP16)[1] / self.tp_size
            memory = weight_memory + mac

        elif ops_type == "mlp":
            flops = self.llm_flops.flops_per_layer_moe_mlp(bs, seq_len) / self.tp_size
            weight_memory = (
                self.llm_params.params_per_layer_moe_mlp()
                * self.bsytes_per_param
                / self.tp_size
            ) * BYTES_FP16
            mac = (self.llm_memory.mac_per_layer_moe_mlp(bs, seq_len)[1] / self.tp_size)
            memory = weight_memory + mac

        elif ops_type == "rmsnorm":
            # Two RMSNorm operations (pre‑attention & pre‑MLP) share the same
            # vector weight, replicated across TP ranks.
            weight_memory = 2 * self.llm_params.params_per_layer_norm() * BYTES_FP16
            flops = self.llm_flops.flops_per_layer_norm(bs, seq_len)
            mac = self.llm_memory.mac_per_layer_norm(bs, seq_len)[1]
            memory = weight_memory + mac
        else:
            raise ValueError(f"Unsupported ops_type: {ops_type}")

        compute_latency = flops / (self.gpu_TFLOPS)  # 单位秒
        memory_latency = memory / (self.gpu_hbm_bw)

        if print_bound:
            self.print_kernel_bound_info(stage, memory_latency, compute_latency, ops_type)

        return max(compute_latency, memory_latency)

    def latency_per_layer_tp_comm(self, bs: int, seq_len: int) -> float:
        """Count the latency of a single allreduce communication across the
        tensor parallel group in the forward pass of a transformer layer.
        The latency is the max of the latency for the allreduce and the minimum
        message latency through intra-node connect.
        """

        if self.tp_size == 1:
            return 0

        # 一次 AllReduce 产生的通讯量为 \phi = 2bsh
        # Self-Attention 和 MLP 部分的计算各需要进行一次 All-Reduce 操作, 即每层做 2 次 All-Reduce操作
        # if tp_size is large enough num_data_per_all_reduce can be 4bsh
        num_data_per_all_reduce = (
            6 * bs * seq_len * self.h * (self.tp_size - 1) / (self.tp_size)
        )

        latency_per_layer_tp_comm = (
            num_data_per_all_reduce
            * self.bsytes_per_param
            / self.gpu_intra_node_bw
        )

        # intra_node_min_message_latency: 节点内连接的最小消息延迟
        return max(
            latency_per_layer_tp_comm,
            self.gpu_config.intra_node_min_message_latency,
        )

    def latency_per_layer(
        self,
        bs: int,
        seq_len: int,
        generate_len: int = 0,
        flash_attn=False,
        kv_cache_bytes: int = BYTES_FP16,
    ) -> tuple:
        kernel_latency_per_layer = 0.0
        dict_latency_per_layer = dict()
        ops_list = ["qkvo_proj", "attn_kernel", "mlp", "rmsnorm"]

        for ops_name in ops_list:
            kernel_latency = self.common_latency_for_ops(
                bs, seq_len, generate_len, ops_name,
            )
            dict_latency_per_layer[ops_name] = kernel_latency
            kernel_latency_per_layer += kernel_latency

        latency_per_layer_tp_comm = self.latency_per_layer_tp_comm(bs, seq_len)
        kv_cache_latency = self.latency_kv_cache_per_layer(
            bs, seq_len, generate_len, flash_attn, kv_cache_bytes
        )

        latency_per_layer = (
            kernel_latency_per_layer
            + latency_per_layer_tp_comm
            + kv_cache_latency
        )

        dict_latency_per_layer["tp_comm"] = latency_per_layer_tp_comm
        dict_latency_per_layer["kv_cache_rw"] = kv_cache_latency

        return latency_per_layer, dict_latency_per_layer

    def latency_input_embedding(self, bs: int, seq_len: int) -> float:
        """Get the latency for the forward pass of the input embedding layer,
        given the batch size, sequence length, and data type of the embedding
        weight.

        Args:
            bs (int): batch size
            seq_len (int): sequence length

        Returns:
            float: the latency in seconds for the forward pass of the input embedding layer
        """
        memory_latency = (
            self.model_config.vocab_size
            * self.model_config.hidden_size
            * self.bsytes_per_param
            / (self.gpu_hbm_bw)
        )
        comm_latency = self.latency_per_layer_tp_comm(bs, seq_len)
        return memory_latency + comm_latency

    def latency_output_embedding(self, bs: int, seq_len: int) -> float:
        """Get the latency for the forward pass of the output embedding layer (computing the logits). 
        The operation is compute bound. With tensor parallelism size > 1, 
        an allgather communicates `bs * seq_len` elements, 
        which is ignored here. Refer to https://arxiv.org/abs/1909.08053 for more details.

        Args:
            bs (int): batch size
            seq_len (int): sequence length
        """

        compute_latency = (
            2 * bs * seq_len * self.h * self.V / self.tp_size / self.gpu_TFLOPS
        )

        return compute_latency

    def latency_kv_cache_per_layer(
        self,
        bs: int,
        seq_len: int,
        generate_len: int,
        flash_attn: bool = False,
        kv_cache_bytes: int = BYTES_FP16,
    ) -> tuple:
        """Get the latency for the forward pass of the key and value cache in a transformer layer,
        given the batch size, sequence length, and whether the key and value cache is used.

        Args:
            bs (int): batch size
            seq_len (int): sequence length
            generate_len (int): number of tokens to generate
        """
        kv_cache_mac = (
            self.llm_memory.mac_per_layer_kv_cache(
                bs, seq_len, generate_len, flash_attn, kv_cache_bytes
            )
            / self.tp_size
        )

        memory_latency = kv_cache_mac / (self.gpu_hbm_bw)

        return memory_latency

    def latency_model(
        self,
        bs: int,
        seq_len: int,
        generate_len: int,
        flash_attn: bool = False,
        kv_cache_bytes: int = BYTES_FP16,
        breakdown_prefix: str = "",
    ) -> tuple:
        latency_per_layer, breakdown_per_layer = self.latency_per_layer(
            bs,
            seq_len,
            generate_len,
            flash_attn,
            kv_cache_bytes,
        )
        num_layers_per_gpu = self.num_layers_per_gpu

        latency_all_layers = latency_per_layer * self.num_layers_per_gpu
        latency_input_embedding = self.latency_input_embedding(bs, seq_len)
        latency_output_embedding = self.latency_output_embedding(bs, seq_len)

        model_latency = (
            latency_all_layers + latency_input_embedding + latency_output_embedding
        )

        model_latency_breakdown = {
            breakdown_prefix + "qkvo_proj": (
                breakdown_per_layer["qkvo_proj"] * num_layers_per_gpu
            ),
            breakdown_prefix + "attn_kernel": (
                breakdown_per_layer["attn_kernel"] * num_layers_per_gpu
            ),
            breakdown_prefix + "mlp": (breakdown_per_layer["mlp"] * num_layers_per_gpu),
            breakdown_prefix + "rmsnorm": (
                breakdown_per_layer["rmsnorm"] * num_layers_per_gpu
            ),
            breakdown_prefix + "tp_comm": (
                breakdown_per_layer["tp_comm"] * num_layers_per_gpu
            ),
            breakdown_prefix + "kv_cache_rw": (
                breakdown_per_layer["kv_cache_rw"] * num_layers_per_gpu
            ),
        }

        return model_latency, model_latency_breakdown

    def latency(
        self,
        bs: int,
        seq_len: int,
        generate_len: int,
        flash_attn: bool = False,
        kv_cache_bytes: int = BYTES_FP16,
    ) -> tuple:
        # 1, 预填充阶段
        prefill_latency, prefill_latency_breakdown = self.latency_model(
            bs,
            seq_len,
            generate_len=0,
            flash_attn=flash_attn,
            kv_cache_bytes=kv_cache_bytes,
            breakdown_prefix="prefill_",
        )

        prefill_latency_breakdown.update(
            {
                "TTFT": prefill_latency,
            }
        )

        # 2, 解码阶段
        kv_cache_latency = self.latency_kv_cache_per_layer(
            bs, seq_len, generate_len, flash_attn, kv_cache_bytes
        ) * self.num_layers_per_gpu

        decode_model_latency, decode_latency_breakdown = self.latency_model(
            bs,
            1,
            generate_len=generate_len,
            flash_attn=flash_attn,
            kv_cache_bytes=kv_cache_bytes,
            breakdown_prefix="decode_",
        )

        decode_latency = decode_model_latency + kv_cache_latency

        decode_latency_breakdown.update(
            {
                "kv_cache_latency": kv_cache_latency,
                "TTOT": (decode_latency),
            }
        )
        return prefill_latency_breakdown, decode_latency_breakdown
