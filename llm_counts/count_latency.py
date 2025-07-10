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


    def __init__(self, model_cfg, gpu_config,
                 tp=[2, 4, 8, 16,32], kv_cache_rate=0.563,
                 enable_gemm_fp4=True, 
                 seq_len=256, generate_len=128, 
                 min_ar_time=0.015, mla_discount=0.7, mla_kernel_static_time=0.05) -> None:
        """
        # Allreduce 的静态延迟 based on FlashMLA result on H800
        Prefill：Prefill 集群采用 4 台节点部署, 路由专家 EP32、MLA 和共享专家 DP32，一个部署单元是 4 节点，32 个冗余路由专家，每张卡 9 个路由专家和 1 个共享专家
        Decode：Decoding 集群采用 18 台部署, 路由专家 EP144、MLA 和共享专家 DP144，一个部署单元是 18 节点，32 个冗余路由专家，每张卡 2 个路由专家和 1 个共享专家
        """
        self.cfg = model_cfg
        self.gpu = gpu_config

        self.V = model_cfg.vocab_size
        self.h = model_cfg.hidden_size

        self.num_layers = model_cfg.num_layers
        self.num_dense_layers = model_cfg.num_dense_layers
        self.num_moe_layers = model_cfg.num_moe_layers

        self.n_routed_experts = model_cfg.n_routed_experts
        self.n_activated_experts = model_cfg.num_experts_per_tok

        self.s = seq_len
        self.o = generate_len

        self.tp = tp
        self.enable_gemm_fp4 = enable_gemm_fp4

        self.llm_flops = CountDeepseekV3Flops(model_cfg, seq_len, generate_len, kv_cache_rate)
        self.llm_memory = CountDeepseekV3Memory(model_cfg)
     
        self.min_ar_time = min_ar_time  # Allreduce 的静态延迟
        self.mla_discount = mla_discount  # MLA kernel 的折扣率
        self.mla_kernel_static_time = mla_kernel_static_time  # MLA kernel 的静态延时，单位 ms

    def _mla_time(self, batch_size=1, decoding_mode=True, 
                  print_console=True, kv_cache_rate=0.563):
        if decoding_mode:
            # Decoding 时计算为 qlen=1, kv_cache_rate = 1
            _, gemm_flops, attn_fp16_tflops = self.llm_flops.mla_matabsob_flops(1, self.s, kv_cache_rate=1)
            gemm_flops *= batch_size
            attn_fp16_tflops *= batch_size
        else:
            # prefill 阶段使用非吸收的版本
            _, gemm_flops, attn_fp16_tflops = self.llm_flops.mla_flops(self.s, self.s, kv_cache_rate)

        gemm_fp8_time = (gemm_flops / 1e9) / self.gpu.get_fp8_tflops() / self.mla_discount # 单位 ms
        attn_fp16_time = (attn_fp16_tflops / 1e9) / self.gpu.get_fp16_tflops() / self.mla_discount # 单位 ms
        load_weight_time = (self.llm_memory.mla_mem() / 1e6) / self.gpu.get_hbm_bw() # 单位 ms
        mla_time = (gemm_fp8_time + attn_fp16_time) + load_weight_time # 单位 ms

        if self.enable_gemm_fp4:
            if self.gpu.get_fp4_tflops() != 0:
                gemm_fp4_time = (gemm_flops / 1e9)  / (self.gpu.get_fp4_tflops())
                mla_time = gemm_fp4_time + attn_fp16_time + load_weight_time # 单位 ms

        tokens_len = batch_size if decoding_mode else self.s
        all_reduce_comm_size = tokens_len * self.h * 2 # fp16 take 2Bytes, 单位 MB
        all_reduce_time = (all_reduce_comm_size/1024/1024) / self.gpu.get_nvlink_bw() +  self.min_ar_time # 单位 ms
        
        tp_time = {}
        for v in self.tp:
            if v == 1:
                tp_time[v] = mla_time + self.mla_kernel_static_time # 单位 ms
            else:
                tp_time[v] = mla_time / v + all_reduce_time + (self.mla_kernel_static_time) # 单位 ms
    
        if print_console:
            print("[%7s] [%8s]All reduce comm size: %s" % 
              ("Decode" if decoding_mode else "Prefill", self.gpu.name, num_to_string(all_reduce_comm_size)))
            
            print("[%7s] [%8s]Load weight Elapsed time(ms): %.4f" % 
                  ("Decode" if decoding_mode else "Prefill", self.gpu.name, load_weight_time))
            if self.enable_gemm_fp4 and (self.gpu.get_fp4_tflops() != 0) and gemm_fp4_time is not None:
                print("[%7s] [%8s]GEMM_FP4 Elapsed time(ms): %.3f" % 
                      ("Decode" if decoding_mode else "Prefill", self.gpu.name, gemm_fp4_time))
            print("[%7s] [%8s]GEMM_FP8 Elapsed time(ms): %.3f" % 
                  ("Decode" if decoding_mode else "Prefill", self.gpu.name, gemm_fp8_time))
            print("[%7s] [%8s]ATTN_FP16 Elapsed time(ms): %.3f" %
                ("Decode" if decoding_mode else "Prefill", self.gpu.name, attn_fp16_time))
            print("[%7s] [%8s]Total Elapsed time(ms): %.3f" % ("Decode" if decoding_mode else "Prefill", self.gpu.name, mla_time))
            print("[%7s] [%8s]All reduce Elapsed time(ms): %.3f" % ("Decode" if decoding_mode else "Prefill", self.gpu.name, all_reduce_time))
            
            for v in self.tp:
                print("[%7s] [%8s]TP[%2d] Elapsed time(ms): %.3f" %
                    ("Decode" if decoding_mode else "Prefill", self.gpu.name, v, tp_time[v]))

        return mla_time, tp_time # 单位 ms
    
    def _prefill_dense_mlp_time(self, seq_len, print_console=False):
        gemm_flops = self.llm_flops.dense_mlp_flops(seq_len) / 1e9
        if self.enable_gemm_fp4 and self.gpu.get_fp4_tflops() != 0:
            gpu_gemm_flops = self.gpu.peak_fp4_TFLOPS
        else:
            gpu_gemm_flops = self.gpu.get_fp8_tflops() # 单位 TFLIOPs
        
        gemm_time = gemm_flops / gpu_gemm_flops

        load_time = self.llm_memory.dense_mlp_mem() /(1024 * 1024) / self.gpu.get_hbm_bw()
        gemm_time = gemm_time + load_time
        
        if print_console:
            print("[%8s]Elapsed time(ms): %s" % (self.gpu.name, gemm_time))
        
        return gemm_time # 单位 ms
    
    def _prefill_moe_expert_time(self, seq_len, tp: int=4, dp: int=8):
        load_expert_weight_time = self.llm_memory.moe_expert_mem() / 1024 / 1024 / self.gpu.get_hbm_bw()
        if self.enable_gemm_fp4 and self.gpu.get_fp4_tflops() != 0:
            gpu_gemm_flops = self.gpu.peak_fp4_TFLOPS
        else:
            gpu_gemm_flops = self.gpu.get_fp8_tflops() # 单位 TFLIOPs

        num_devices = tp * dp
        num_shared_tokens = dp * seq_len / num_devices
        shared_flops = self.llm_flops.moe_expert_flops(num_shared_tokens) / 1e9
        shared_expert_time = shared_flops / gpu_gemm_flops + load_expert_weight_time

        num_routed_token = seq_len * dp * self.n_activated_experts / num_devices
        routed_flops = self.llm_flops.moe_expert_flops(num_routed_token) / 1e9
        experts_num = math.ceil(self.n_routed_experts / num_devices)

        routed_expert_time = routed_flops / (gpu_gemm_flops) + load_expert_weight_time * experts_num

        return shared_expert_time, routed_expert_time
    
    def _prefill_alltoall_time(self, seq_len, tp=4, dispatch_node=4, static_latency=0.05):
        """
        DeepSeek V3 和 R1 的所有服务均使用 H800 GPU，使用和训练一致的精度，即矩阵计算和 dispatch 传输采用和训练一致的 FP8 格式，
        core-attention 计算和 combine 传输采用和训练一致的 BF16，最大程度保证了服务效果.
        """
        # h200 节点机器
        if self.gpu.gpu_per_node == 8:
            dp = self.gpu.gpu_per_node / tp
            dispatch_size = (dispatch_node - 1) * dp * seq_len * self.n_activated_experts / self.gpu.gpu_per_node * self.h /1024/1024 # MB fp16 take 2Bytes
            comm_bw = self.gpu.get_pcie_bw() * self.gpu.gpu_per_node
        else: # NVL72
            expert_num = math.ceil(self.n_routed_experts / self.gpu.gpu_per_node)
            dispatch_prob = (self.n_routed_experts - expert_num) / self.n_routed_experts
            dispatch_size = dispatch_prob * self.n_activated_experts * seq_len/tp * self.h / 1024/1024
            comm_bw = self.gpu.get_nvlink_bw()
        
        combine_size = dispatch_size * 2
        combine_size = 2 * dispatch_size  # fp16

        if self.gpu.get_fp4_tflops() != 0:
            dispatch_size = dispatch_size / 2

        dispatch_time = dispatch_size / comm_bw + static_latency
        combine_time = combine_size / comm_bw + static_latency

        return dispatch_time, combine_time # 单位 ms

    def _prefill_latency(self, batch_size=1, seq_len=4383, tp=4, dp=2, kv_cache_rate=0.563,dispatch_node=4, decoding_mode=False):
        mla_t, tp_mla_t = self._mla_time(batch_size=1, decoding_mode=decoding_mode, kv_cache_rate=kv_cache_rate)
        dense_mlp_t = self._prefill_dense_mlp_time(seq_len)
        shared_t, routed_t = self._prefill_moe_expert_time(seq_len, tp, dp)
        dispatch_t, combine_t = self._prefill_alltoall_time(seq_len, tp, dispatch_node)

        return mla_t, tp_mla_t[tp], dense_mlp_t, shared_t, routed_t, dispatch_t, combine_t
    
    def prefill_latency(self, batch_size=1, seq_len=4383, tp=4, dp=2, dispatch_node=4, print_console=False):
        """
        累计耗时, 非Overlap计算
            3x(MLA_tp1 + DenseMLP) + 58x(MLA_tpN + Shared Expert + Routed Expert +Dispatch + Combine)
        完全Overlap计算
            3x(MLA_tp1 + DenseMLP) + 58x(MLA_tpN + Shared Expert + Routed Expert)
        """
        dense_mla, tp_mla, dense_mlp, shared, routed, dispatch, combine = self._prefill_latency(seq_len, tp, dp, dispatch_node)
        
        sum_overlap = self.num_dense_layers * (dense_mla + dense_mlp) + self.num_moe_layers * (tp_mla + shared + routed)
        sum_non_overlap = sum_overlap + self.num_moe_layers * (dispatch + combine) # non_overlap need add all_toall time
        latency_dict = {
            "dense_mla": dense_mla,
            "tp_mla": tp_mla,
            "dense_mlp": dense_mlp,
            "shared_expert": shared,
            "routed_expert": routed,
            "dispatch": dispatch,
            "combine": combine,
            "sum_overlap": sum_overlap, 
            "sum_non_overlap": sum_non_overlap
        }
        if print_console:
            print(f"Prefill latency for {self.cfg.name} with seq_len={seq_len},\
                   tp={tp}, dp={dp}, dispatch_node={dispatch_node}")
            print(f"Sum Overlap Latency: {latency_to_string(sum_overlap)}")
            print(f"Sum Non-Overlap Latency: {latency_to_string(sum_non_overlap)}")
            for k, v in latency_dict.items():
                print(f"{k}: {latency_to_string(v)}")

        return sum_overlap, sum_non_overlap, latency_dict
    
    def _decode_moe_time(self, batch_size, device_num):
        load_time = self.llm_memory.moe_expert_mem() / 1024 / 1024 / self.gpu.hbm_bw
        if self.gpu.get_fp4_tflops() != 0:
            load_time = load_time / 2
        gpu_flops = self.gpu.get_fp4_tflops() if self.gpu.get_fp4_tflops() != 0 else self.gpu.get_fp8_tflops()
        
        gemm_group_per_device = math.ceil(self.n_routed_experts / device_num)
        total_expert = gemm_group_per_device * device_num
        m_per_group = batch_size * self.n_activated_experts * device_num / total_expert

        #data from hs's profiling result
        flops_discounts = {
            1: 0.05,
            2: 0.05,
            4: 0.05,
            8: 0.05,
            16: 0.08,
            32: 0.1,
            64: 0.2,
            128: 0.35,
            256: 0.4,
            512: 0.6,
            1024: 0.7,
            2048: 0.7,
            4096: 0.7,
            8192: 0.7,
            16384: 0.7,
            32768: 0.7,
            65536: 0.7
        }

        # H20 exception based on hs's result
        if self.gpu.name.find('H20')!= -1 :
            flops_discounts = {
            1: 0.06,
            2: 0.06,
            4: 0.06,
            8: 0.12,
            16: 0.25,
            32: 0.45,
            64: 0.8,
            128: 0.9,
            256: 1.0,
            512: 1.0,
            1024: 1.0,
            2048: 1.0,
            4096: 1.0,
            8192: 1.0,
            16384: 1.0,
            32768: 1.0,
            65536: 1.0
        }

        gpu_flops = gpu_flops * flops_discounts[n_pow2_range(int(m_per_group))]
        
        shared_flops = self.llm_flops.moe_expert_flops(batch_size) / 1e9
        shared_time = shared_flops / gpu_flops + load_time

        num_routed_token = batch_size * self.n_activated_experts
        routed_flops = self.llm_flops.moe_expert_flops(num_routed_token) / 1e9
        routed_time = routed_flops / gpu_flops + load_time * gemm_group_per_device
        
        return shared_time, routed_time # 单位 ms
    
    def _decode_all2alltime(self, batch_size, expert_num, device_num, fp8_combine=False, static_latency=0.005, mbs=2):
        dispatch_size = batch_size * self.h * self.n_activated_experts / 1024/1024 # MB/s
        if fp8_combine & (self.gpu.get_fp4_tflops() != 0):  # 支持FP4GPU才能开启FP8 Combine
            combine_size = dispatch_size
        else:
            combine_size = dispatch_size * 2  # FP16
        if self.gpu.gpu_per_node == 8:
            comm_bw = self.gpu.get_pcie_bw()
            # single host deployment
            if self.n_routed_experts / (expert_num - 1) == self.gpu.gpu_per_node:
                comm_bw = self.gpu.get_nvlink_bw()
        # NVL72 /144 / 576
        elif (self.gpu.gpu_per_node in NVL_GPU_LIST) & (device_num > self.gpu.gpu_per_node):
            comm_bw = self.gpu.get_pcie_bw()
        else:
            comm_bw = self.gpu.get_nvlink_bw()

        dispatch_t = dispatch_size / comm_bw + static_latency * mbs
        combine_t = combine_size / comm_bw + static_latency * mbs
        return dispatch_t, combine_t
    
    def _decode_latency(self, batch_size, device_num,  mbs=2, fp8_combine=False, print_console=False):
        gemm_group_per_device = math.ceil(self.n_routed_experts / device_num)
        expert_per_device = gemm_group_per_device + 1  # add shared expert[冗余专家]
        mla_t, tp_mla_t = self._mla_time(decoding_mode=True)
        dense_mlp_t = self._prefill_dense_mlp_time(batch_size)
        shared_t, routed_t = self._decode_moe_time(batch_size, device_num)
        dispatch_t, combine_t = self._decode_all2alltime(expert_per_device, device_num, fp8_combine, mbs=mbs)

        return mla_t, tp_mla_t, dense_mlp_t, shared_t, routed_t, dispatch_t, combine_t