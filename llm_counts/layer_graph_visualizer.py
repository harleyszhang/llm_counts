"""
cli entry point for LayerAnalyzer, which analyzes the memory access and FLOPs of a model.
Usage:
    ```bash
    python -m llm_counts.layer_graph_visualizer \
    --result-json results.json \
    --model-type qwen3 \
    --output my_layer_graph
```
"""
from __future__ import annotations
from .utils.constants import BYTES_FP16
from .utils.config import *
from .utils.utils import num_to_string
from .utils.roofline_model import roofline_analysis
from copy import deepcopy
from pathlib import Path
from graphviz import Digraph
import argparse
from typing import Union

import json
import math

class LayerAnalyzer(object):
    """Count memory access of the model and layers."""
    STAGES = ("prefill", "decode")  # pipeline stages

    def __init__(self, model_config: 'ModelConfig', gpu_config: 'GPUConfig', tp_size: int) -> None:
        self.tp_size = tp_size
        self.bw, self.onchip_buffer = get_gpu_hbm_bw(gpu_config) # GB/s
        self.bw *= 10**9 
        self.gpu_max_ops = get_flops(gpu_config, data_type="fp16") * 10**12  # TFLOPs
    
        self.model_type = model_config.model_type
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        self.num_heads = model_config.num_heads
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = model_config.head_dim
        self.V = model_config.vocab_size

        self.is_moe = (getattr(model_config, "num_experts", None) is not None)
        self.is_qwen3moe = "qwen3_moe" == self.model_type
        
        if self.is_moe:
            # Default to 1 expert if not specified
            self.num_experts = getattr(model_config, "num_experts", 1)  
            self.num_experts_per_tok = getattr(model_config, "num_experts_per_tok", 1)

        # attention linear layers
        if self.is_qwen3moe:
            self.linear_layers = {
                "q_proj": [self.hidden_size, self.num_heads * self.head_dim],
                "k_proj": [self.hidden_size, self.num_kv_heads * self.head_dim],
                "v_proj": [self.hidden_size, self.num_kv_heads * self.head_dim],
                "out_proj": [self.num_heads * self.head_dim, self.hidden_size],
                # MoE结构
                "router_gate": [self.hidden_size, self.num_experts],
                "expert_gate_proj": [self.hidden_size, self.intermediate_size],
                "expert_up_proj": [self.hidden_size, self.intermediate_size],
                "expert_down_proj": [self.intermediate_size, self.hidden_size],
            }
        else:
            self.linear_layers = {
                "q_proj": [self.hidden_size, self.num_heads * self.head_dim], # type: ignore
                "k_proj": [self.hidden_size, self.num_kv_heads * self.head_dim],
                "v_proj": [self.hidden_size, self.num_kv_heads * self.head_dim],
                "out_proj": [self.num_heads * self.head_dim, self.hidden_size],
                "gate_proj": [self.hidden_size, self.intermediate_size],
                "up_proj": [self.hidden_size, self.intermediate_size],
                "down_proj": [self.intermediate_size, self.hidden_size],
            }

        self.results = {"decode": {}, "prefill": {}}

    def _analyze_to_results(
        self,
        stage,
        kernel_name,
        flops,
        load_weight,
        load_act,
        store_act,
        load_kv_cache,
        store_kv_cache,
    ):
        memory_access = (load_weight + load_act + store_act  + load_kv_cache + store_kv_cache)
        a_intensity, att_flops, bound = roofline_analysis(self.gpu_max_ops, 
                                                          self.bw, 
                                                          flops, memory_access) # Arithmetic Intensity

        self.results[stage][kernel_name] = {
            "flops": flops,                       # store raw integer for arithmetic use
            "memory_access": memory_access,       # raw bytes for later computation
            "arithmetic_intensity": int(a_intensity),
            "att_flops": att_flops,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
        }

        return self.results

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    def analyze_linear_layers(
        self,
        bs: int,
        seq_len: int,
        linear_weight_bytes: int = BYTES_FP16,
        act_byte: int = BYTES_FP16,
        kv_byte: int = BYTES_FP16,
    ):
        """
        Compute FLOPs and memory accesses of projection / MLP linear layers
        for both pipeline stages in a single loop.
        """
        for stage in self.STAGES:
            cur_seq = seq_len if stage == "prefill" else 1
            for name, (in_ch, out_ch) in self.linear_layers.items():
                is_kv = name in {"k_proj", "v_proj"}
                flops = 2 * bs * cur_seq * in_ch * out_ch // self.tp_size
                load_weight = in_ch * out_ch * linear_weight_bytes // self.tp_size
                load_act = in_ch * bs * cur_seq * act_byte // self.tp_size
                store_act = 0 if is_kv else bs * cur_seq * out_ch * act_byte // self.tp_size
                load_kv_cache = 0
                store_kv_cache = (
                    0 if not is_kv else out_ch * bs * cur_seq * kv_byte // self.tp_size
                )

                self._analyze_to_results(
                    stage,
                    name,
                    flops=flops,
                    load_weight=load_weight,
                    load_act=load_act,
                    store_act=store_act,
                    load_kv_cache=load_kv_cache,
                    store_kv_cache=store_kv_cache,
                )

    def analyze_experts(
        self,
    ):
        for stage in self.STAGES:
            expert_flops = (
                self.results[stage]["expert_down_proj"]["flops"] + 
                self.results[stage]["expert_up_proj"]["flops"] + 
                self.results[stage]["expert_gate_proj"]["flops"]
            )
            expert_load_weight = (
                self.results[stage]["expert_down_proj"]["load_weight"] +
                self.results[stage]["expert_up_proj"]["load_weight"] +
                self.results[stage]["expert_gate_proj"]["load_weight"]
            )
            expert_load_act = (
                self.results[stage]["expert_down_proj"]["load_act"] +
                self.results[stage]["expert_up_proj"]["load_act"] +
                self.results[stage]["expert_gate_proj"]["load_act"]
            )
            expert_store_act = (
                self.results[stage]["expert_down_proj"]["store_act"] +
                self.results[stage]["expert_up_proj"]["store_act"] +
                self.results[stage]["expert_gate_proj"]["store_act"]
            )

            for i in range(0, self.num_experts_per_tok):
                expert_node = f"moe_mlp_{i}"
                self._analyze_to_results(
                    stage,
                    expert_node,
                    flops=expert_flops,
                    load_weight=expert_load_weight,
                    load_act=expert_load_act,
                    store_act=expert_store_act,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )

    def analyze_self_atten_kernel(
        self, 
        bs: int,
        seq_len: int,
        generate_len: int,
        num_kv_heads: int,
        num_heads: int,
        head_dim: int,
        flash_attn: bool = False,
        act_byte: int = BYTES_FP16,
        kv_byte: int = BYTES_FP16,
    ):
        """
        Count and save the FLOPs and memory access of self-attention kernels.
        This function is used to analyze the self-attention kernels in the model.
        """
        if not flash_attn:
            ##########################prefill stage##########################
            # 1, qkt kernel analysis
            name = "qk_matmul"
            load_q_mem = bs * self.num_heads * seq_len * self.head_dim
            load_k_mem = bs * self.num_kv_heads * seq_len * self.head_dim
            qk_store_mem = bs * self.num_heads * seq_len * seq_len
            self._analyze_to_results(
                "prefill",
                name,
                flops=2 * seq_len * seq_len * self.head_dim * bs * self.num_heads,
                load_weight=0,
                load_act=(load_q_mem + load_k_mem) * act_byte, # load q and k act, shape is [s, h]
                store_act=qk_store_mem * act_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
            # 2, softmax kernel analysis
            name = f"softmax"
            load_softmax_mem = qk_store_mem
            softmax_store_mem = bs * self.num_heads * seq_len * seq_len
            self._analyze_to_results(
                "prefill",
                name,
                flops= (bs * num_heads * seq_len * seq_len * 1 * 5),
                load_weight=0,
                load_act=load_softmax_mem * act_byte,
                store_act=softmax_store_mem * act_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
            # 3, sv kernel analysis
            name = "sv_matmul"
            load_s_mem = softmax_store_mem
            load_v_mem = bs * self.num_kv_heads * seq_len * self.head_dim
            sv_store_mem = bs * self.num_heads * seq_len * self.head_dim
            self._analyze_to_results(
                "prefill",
                name,
                flops=bs * 2 * seq_len * seq_len * head_dim * num_heads,
                load_weight=0,
                load_act=load_s_mem * act_byte, # load score(qkt) act, shape is [s, s]
                store_act=sv_store_mem * act_byte,
                load_kv_cache=load_v_mem,
                store_kv_cache=0,
            )
            ##########################decode stage##########################
            name = "qk_matmul"
            # load q and k, k is form kv cache
            qk_matmul_flops = 2 * self.num_heads * self.head_dim * (seq_len + generate_len)
            load_q_mem = bs * self.num_heads * 1  * self.head_dim
            load_k_mem = bs * self.num_kv_heads * (seq_len + generate_len) * self.head_dim
            qk_store_mem = bs * self.num_heads * (seq_len + generate_len) * (seq_len + generate_len)
            self._analyze_to_results(
                "decode",
                name,
                flops=qk_matmul_flops,
                load_weight=0,
                load_act=load_q_mem * act_byte,
                store_act=qk_store_mem * act_byte,
                load_kv_cache=load_k_mem * kv_byte,
                store_kv_cache=0,
            )
            # 2, softmax kernel analysis
            name = f"softmax"
            load_softmax_mem = qk_store_mem
            softmax_store_mem = bs * self.num_heads * (seq_len + generate_len) * (seq_len + generate_len)
            self._analyze_to_results(
                "decode",
                name,
                flops= (bs * num_heads * seq_len * seq_len * 1 * 5),
                load_weight=0,
                load_act=load_softmax_mem * act_byte,
                store_act=softmax_store_mem * act_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
            # 3, sv kernel analysis
            name = "sv_matmul"
            load_s_mem = softmax_store_mem
            load_v_mem = bs * self.num_kv_heads * (seq_len + generate_len) * self.head_dim
            sv_store_mem = bs * self.num_heads * (seq_len + generate_len) * self.head_dim
            self._analyze_to_results(
                "decode",
                name,
                flops=qk_matmul_flops,
                load_weight=0,
                load_act=load_s_mem * act_byte, # load score(qkt) act, shape is [s, s]
                store_act=sv_store_mem * act_byte,
                load_kv_cache=load_v_mem,
                store_kv_cache=0,
            )
        else:
            name = f"fused_attention" # flash_attn2
            qk_matmul_OPs = seq_len * seq_len * head_dim * num_heads * bs * 2
            sv_matmul_OPs = seq_len * head_dim * seq_len * num_heads * bs * 2
            softmax_OPs = bs * num_heads * seq_len * seq_len * 5

            block_size_r = min(math.ceil(self.onchip_buffer / (kv_byte * head_dim)), head_dim)
            n_blocks_r = math.ceil(seq_len / block_size_r)
            q_numel = seq_len * head_dim * bs * num_heads * act_byte
            o_numel = seq_len * seq_len * bs * num_heads * act_byte

            self._analyze_to_results(
                "prefill",
                name,
                flops=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seq_len) * head_dim * bs * num_kv_heads * kv_byte * 2,
                store_kv_cache=0,
            )

            qk_matmul_OPs = seq_len * head_dim * num_heads * bs * 2
            sv_matmul_OPs = 1 * head_dim * seq_len * num_heads * bs * 2
            softmax_OPs = bs * num_heads * seq_len * 1 * 5

            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = (1) * head_dim * bs * num_heads * act_byte
            o_numel = 1 * seq_len * bs * num_heads * act_byte
            self._analyze_to_results(
                "decode",
                name,
                flops=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seq_len) * head_dim * bs * num_kv_heads * kv_byte * 2,
                store_kv_cache=0,
            )

        if self.model_type in ["qwen3", "qwen3_moe"]:
            kernel_names = ["q_norm", "k_norm"]
            # qwen3 模型中 rms_norm 计算中使用了一个额外的线性变换
            q_norm_flops = bs * 4 * seq_len * self.head_dim
            q_norm_load_weight = self.head_dim * BYTES_FP16
            q_norm_load_act = bs * seq_len * self.head_dim * BYTES_FP16 # equal k_norm_load_act
            q_norm_store_act = bs * seq_len * self.head_dim * BYTES_FP16

            # prefill/decode 阶段
            for stage in ["prefill", "decode"]:
                if stage == "decode":
                    q_norm_flops = int(q_norm_flops // seq_len)
                    q_norm_load_act = int(q_norm_load_act // seq_len)
                    q_norm_store_act = int(q_norm_store_act // seq_len)

                for _, kernel_name in enumerate(kernel_names):
                    self._analyze_to_results(
                        stage,
                        kernel_name,
                        flops=q_norm_flops // self.tp_size,
                        load_weight=q_norm_load_weight // self.tp_size,
                        load_act=q_norm_load_act // self.tp_size,
                        store_act=q_norm_store_act // self.tp_size,
                        load_kv_cache=0,
                        store_kv_cache=0,
                    )
    
    # ------------------------------------------------------------------ #
    # Router kernels (Qwen‑3 MoE)                                        #
    # ------------------------------------------------------------------ #
    def analyze_router_kernels(
        self,
        bs: int,
        seq_len: int,
        act_byte: int = BYTES_FP16,
        weight_byte: int = BYTES_FP16,
    ):
        """
        Profile FLOPs and memory transactions for the router part of
        Qwen‑3 MoE.  A compact table eliminates repeated code while the
        `token_scale` factor transparently handles *prefill* vs *decode*
        stages (the latter processes a single token).
        """
        # Guard: router kernels only exist in Qwen‑3 MoE
        if not getattr(self, "is_qwen3moe", False):
            return

        rows = [
            # (kernel_name, flops_fn, load_w_fn, load_act_fn, store_act_fn)
            ("router_gate",
             lambda: 2 * bs * seq_len * self.hidden_size * self.num_experts,
             lambda: self.hidden_size * self.num_experts * weight_byte,
             lambda: bs * seq_len * self.hidden_size * act_byte,
             lambda: bs * seq_len * self.num_experts * act_byte),
            ("router_softmax",
             lambda: bs * seq_len * self.num_experts * 5,
             lambda: 0,
             lambda: bs * seq_len * self.num_experts * act_byte,
             lambda: bs * seq_len * self.num_experts * act_byte),
            ("router_top-k",
             lambda: bs * seq_len * self.num_experts,
             lambda: 0,
             lambda: bs * seq_len * self.num_experts * act_byte,
             lambda: bs * seq_len * self.num_experts * act_byte),
            ("router_norm_topk_prob",
             lambda: bs * seq_len * self.hidden_size,
             lambda: 0,
             lambda: bs * seq_len * self.hidden_size * act_byte,
             lambda: bs * seq_len * self.hidden_size * act_byte),
        ]

        for stage in self.STAGES:  # ("prefill", "decode")
            token_scale = 1.0 if stage == "prefill" else 1.0 / seq_len  # 1‑token decode
            for name, flops_fn, w_fn, la_fn, sa_fn in rows:
                flops       = int(flops_fn() * token_scale)
                load_act    = int(la_fn()   * token_scale)
                store_act   = int(sa_fn()   * token_scale)
                load_weight = w_fn()

                self._analyze_to_results(
                    stage,
                    name,
                    flops=flops // self.tp_size,
                    load_weight=load_weight // self.tp_size,
                    load_act=load_act // self.tp_size,
                    store_act=store_act // self.tp_size,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
        

    def analyze_other_kernels(
        self,
        bs: int,
        seq_len: int,
        act_byte: int = BYTES_FP16,
    ):
        embedding_weight = self.hidden_size * self.V * BYTES_FP16
        emebdding_flops = 0
        embedding_load_act = bs * seq_len * act_byte
        embedding_store_act = bs * seq_len * self.hidden_size * act_byte

        norm_flops = bs * seq_len * 4 * self.hidden_size  # post_attention_layernorm, input_layernorm
        norm_load_weight = self.hidden_size * BYTES_FP16
        norm_load_act = bs * seq_len * self.hidden_size * BYTES_FP16
        norm_store_act = bs * seq_len * self.hidden_size * BYTES_FP16

        # silu 和 dot * 都是纯逐元素操作算子
        silu_dot_flops = (bs * 4 * seq_len * self.intermediate_size)  # 每个张量元素执行 4 次操作
        silu_dot_load_act = bs * 2 * seq_len * self.intermediate_size * act_byte
        silu_dot_store_act = (bs * 2 * seq_len * self.intermediate_size * act_byte)

        mlp_add_flops = bs * seq_len * self.hidden_size
        mlp_add_load_act = 2 * bs * seq_len * self.hidden_size * act_byte
        mlp_add_store_act = bs * seq_len * self.hidden_size * act_byte

        attn_add_flops = mlp_add_flops
        attn_add_load_act = mlp_add_load_act
        attn_add_store_act = mlp_add_store_act

        rope_flops = 2 * bs * seq_len * self.num_heads * self.head_dim
        rope_load_act = bs * seq_len * self.num_heads * self.head_dim * act_byte
        rope_store_act = rope_load_act

        # other kernels (memory bound)
        kernel_names = ["embedding", 
                        "input_layernorm", "post_attention_layernorm", "qwen3_moe_norm", 
                        "mlp_silu_dot", "attn_add", "mlp_add",
                        "rope", "lm_head"]
        flops_list = [emebdding_flops, 
                      norm_flops, norm_flops, norm_flops, 
                      silu_dot_flops, attn_add_flops, mlp_add_flops,
                      rope_flops, emebdding_flops]
        
        load_act_list = [embedding_load_act,
                        norm_load_act, norm_load_act, norm_load_act, 
                        silu_dot_load_act, attn_add_load_act, mlp_add_load_act, 
                        rope_load_act, embedding_load_act]
        store_act_list = [embedding_store_act, 
                          norm_store_act, norm_store_act, norm_store_act, 
                          silu_dot_store_act, attn_add_store_act, mlp_add_store_act, 
                          rope_store_act, embedding_store_act]

        # prefill/decode 阶段
        for stage in ["prefill", "decode"]:
            for i, kernel_name in enumerate(kernel_names):
                load_act = load_act_list[i]
                store_act = store_act_list[i]
                flops = flops_list[i]
            
                if kernel_name == "embedding":
                    load_weight = embedding_weight // self.tp_size
                else:
                    load_weight = (0 if (kernel_name not in ["qwen3_moe_norm", "input_layernorm", "post_attention_layernorm"]) else norm_load_weight)

                if kernel_name == "update_kv_cache" and stage == "prefill":
                    # prefill 阶段不需要更新 kv cache
                    load_kv_cache = 0
                    store_kv_cache = 0

                if stage == "decode":
                    flops = int(flops // seq_len)
                    load_act = int(load_act // seq_len)
                    store_act = int(store_act // seq_len)
                    
                self._analyze_to_results(
                    stage,
                    kernel_name,
                    flops=flops // self.tp_size,
                    load_weight=load_weight // self.tp_size,
                    load_act=load_act // self.tp_size,
                    store_act=store_act // self.tp_size,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
    
    def analyze_model(
        self,
        bs: int,
        seq_len: int,
        generate_len: int = 0,
        flash_attn: bool = False,
        act_byte: int = BYTES_FP16,
        kv_byte: int = BYTES_FP16,
    ):
        """
        Analyze the model and save the results.
        This function is used to analyze the model and save the results.
        """
        # 1. analyze linear layers
        self.analyze_linear_layers(bs, seq_len, act_byte=act_byte, kv_byte=kv_byte)

        # 2. analyze experts
        if self.is_qwen3moe:
            self.analyze_experts()
            # 3. analyze router kernels
            self.analyze_router_kernels(bs, seq_len, act_byte=act_byte)
        
        # 4. analyze self attention kernels
        self.analyze_self_atten_kernel(
            bs, seq_len, generate_len, 
            num_kv_heads=self.num_kv_heads, 
            num_heads=self.num_heads, 
            head_dim=self.head_dim, 
            flash_attn=flash_attn, 
            act_byte=act_byte, 
            kv_byte=kv_byte
        )

        # 5. analyze other kernels
        self.analyze_other_kernels(
            bs, seq_len,
        )

        return self.results

from collections import OrderedDict

BASE_DEPENDENCIES = OrderedDict({
    "input": [],
    "embedding": ["input"],
    "input_layernorm": ["embedding"],
    "q_proj": ["input_layernorm"],
    "k_proj": ["input_layernorm"],
    "v_proj": ["input_layernorm"],
    "qk_matmul": ["q_proj", "k_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
    "out_proj": ["sv_matmul"],
    "attn_add": ["input", "out_proj"],
    "post_attention_layernorm": ["attn_add"],
    "gate_proj": ["post_attention_layernorm"],
    "up_proj": ["post_attention_layernorm"],
    "mlp_silu_dot": ["up_proj", "gate_proj"],
    "down_proj": ["mlp_silu_dot"],
    "mlp_add": ["attn_add", "down_proj"],
    "output": ["mlp_add"],
})

QWEN3_MOE_DEPS = OrderedDict({
    "input": [],
    "embedding": ["input"],
    "input_layernorm": ["embedding"],
    "q_proj": ["input_layernorm"],
    "k_proj": ["input_layernorm"],
    "v_proj": ["input_layernorm"],
    
    "qk_matmul": ["rope", "rope"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
    "out_proj": ["sv_matmul"],
    "attn_add": ["embedding", "out_proj"],
    "post_attention_layernorm": ["attn_add"],

    "router_gate": ["post_attention_layernorm"],
    "router_softmax": ["router_gate"],
    "router_top-k": ["router_softmax"],
    "router_norm_topk_prob": ["router_top-k"],

    "mlp_add": ["attn_add", "index_add_"],
    "qwen3_moe_norm": ["mlp_add"],
    "lm_head": ["qwen3_moe_norm"],
    "output": ["lm_head"],
})


def _expand_moe_deps_with_moe_mlp(num_act_experts: int) -> dict[str, list[str]]:
    """
    Expand Qwen‑3 MoE dependencies:
    • Experts 0 … num_experts_per_tok appear as single placeholder nodes `moe_mlp_<i>`
      that depend on `router_norm_topk_prob` and feed directly into
      `index_add_`.

    This gives a detailed view of one expert while still showing the
    presence of the others in the graph.
    """
    deps = deepcopy(QWEN3_MOE_DEPS)

    moe_mlp_nodes: list[str] = []

    # ---------- Placeholder nodes for remaining experts ----------
    for i in range(0, num_act_experts):
        node = f"moe_mlp_{i}"
        deps[node] = ["router_norm_topk_prob"]
        moe_mlp_nodes.append(node)

    # index_add_ gathers outputs from expert‑0 down_proj + all placeholders
    deps["index_add_"] = moe_mlp_nodes
    return deps
            

# --------------------------------------------------------------------------- #
# Color rules (exact → prefix → substring)
# --------------------------------------------------------------------------- #
COLOR_RULES = {
    "exact": {
        "router_gate": "gold",
        "mlp_add": "lightcoral",
        "attn_add": "lightcoral",
        "index_add_": "orange",
    },
    "prefix": {
        "moe_mlp_": "lightgreen",
    },
    "contains": {
        "proj": "lightblue",
        "matmul": "plum",
        "router": "gold",
        "norm": "lightyellow",
    },
}

# --------------------------------------------------------------------------- #
# Logical clusters – nodes will be grouped into these subgraphs for clarity
# --------------------------------------------------------------------------- #
CLUSTER_DEFS: dict[str, list[str]] = {
    "Self Attention": [
        "qk_matmul", "softmax", "sv_matmul",
    ],
    # Router group is only used for MoE‑style models
    "Router": [
        "router_gate", "router_softmax", "router_top-k", "router_norm_topk_prob",
    ],
}

class LayerGraphVisualizer:
    """Render a transformer‑layer roofline graph (Graphviz PNG)."""

    # ------------------------------ Construction --------------------------- #
    def __init__(self, model_cfg: Union[str, object],
                 results: dict, num_experts: int | None = None):
        self.results = results
        self.model_type = getattr(model_cfg, "model_type", model_cfg)

        if self.model_type == "qwen3_moe":
            self.num_experts = getattr(model_cfg, "num_experts_per_tok", 8)
            self.deps = _expand_moe_deps_with_moe_mlp(self.num_experts)
        elif self.model_type == "qwen3":
            self.deps = QWEN3_MOE_DEPS.copy()
        else:
            self.deps = BASE_DEPENDENCIES.copy()

        # q_norm / k_norm exist for Qwen3 variants
        if self.model_type in {"qwen3", "qwen3_moe"}:            
            self.deps.update({"q_norm": ["q_proj"], "k_norm": ["k_proj"]})
            self.deps.update({"rope": ["q_norm", "k_norm"]})
            self.deps.update({"qk_matmul": ["rope", "rope"]})

        # build cluster definitions on‑the‑fly to keep constants minimal
        self.cluster_map = {
            "Self Attention": {"qk_matmul", "softmax", "sv_matmul"},
        }
        if self.model_type == "qwen3_moe":
            self.cluster_map["Router"] = set(CLUSTER_DEFS["Router"])

    # ------------------------------ Node utils ----------------------------- #
    def _node_color(self, node: str) -> str:
        if node in COLOR_RULES["exact"]:
            return COLOR_RULES["exact"][node]
        for p, c in COLOR_RULES["prefix"].items():
            if node.startswith(p):
                return c
        return next((c for k, c in COLOR_RULES["contains"].items() if k in node),
                    "lightcyan")

    def _node_label(self, node: str, d: dict) -> str:
        title = f"Expert {node.split('_')[-1]}" if node.startswith("moe_mlp_") else node
        return "\\n".join((
            title,
            f"Flops:{num_to_string(d.get('flops', 0))}"
            f"  Access:{num_to_string(d.get('memory_access', 0))}",
            f"Params:{num_to_string(d.get('load_weight', 0))}"
            f"  Bound:{d.get('bound', 'N/A')}",
        ))

    def _make_dot(self) -> Digraph:
        return Digraph(
            format="png",
            node_attr=dict(style="rounded,filled", shape="box",
                           fontname="Arial", fontsize="16"),
            graph_attr=dict(rankdir="TB", splines="ortho", dpi="300",
                            center="true", nodesep="0.4", ranksep="0.3",
                            pack="true"),
        )

    # ----------------------------- Drawing logic --------------------------- #
    def _add_nodes_and_clusters(self, dot: Digraph, res: dict) -> set[str]:
        use_nodes = set(self.deps)                     # always draw full graph
        remaining = use_nodes.copy()

        # ---- Pre‑defined clusters ----
        for label, members in self.cluster_map.items():
            group = members & remaining
            if not group:
                continue
            remaining -= group
            
            with dot.subgraph(name=f"cluster_{label.lower()}") as sub:
                sub.attr(label=label, style="dashed", margin="8")
                for n in sorted(group):
                    sub.node(n, label=self._node_label(n, res.get(n, {})),
                             fillcolor=self._node_color(n))

        # ---- Experts cluster ----
        experts = {n for n in remaining
                   if n.startswith(("moe_mlp_", "expert_"))}
        if experts:
            remaining -= experts
            with dot.subgraph(name="cluster_experts") as sub:
                sub.attr(label="Experts", style="dashed", color="lightgreen",
                         margin="8")
                for n in sorted(experts):
                    sub.node(n, label=self._node_label(n, res.get(n, {})),
                             fillcolor=self._node_color(n))
            # invisible edges to stabilise layout
            if "router_norm_topk_prob" in use_nodes:
                dot.edge("router_norm_topk_prob", sorted(experts)[0],
                         style="invis", weight="5")
            if "index_add_" in use_nodes:
                dot.edge(sorted(experts)[-1], "index_add_", style="invis",
                         weight="5")

        # ---- Remaining nodes ----
        for n in sorted(remaining):
            dot.node(n, label=self._node_label(n, res.get(n, {})),
                     fillcolor=self._node_color(n))
        return use_nodes

    def _add_edges(self, dot: Digraph, use_nodes: set[str]) -> None:
        """
        Draw edges according to self.deps.  If a parent is listed multiple
        times for the same child, emit that many parallel edges.  This is
        needed for Qwen‑3 where the single “rope” node feeds *both* the
        query and key paths into `qk_matmul`.
        """
        from collections import Counter

        for child, parents in self.deps.items():
            if child not in use_nodes:
                continue

            parent_counts = Counter(parents)
            for parent, cnt in parent_counts.items():
                if parent not in use_nodes:
                    continue

                # Decide styling once per parent→child pair
                style = {}
                if "moe_mlp_" in (child + parent):
                    style = {"color": "green", "style": "dashed"}
                elif "index_add_" in (child, parent):
                    style = {"color": "orange", "style": "bold"}

                # Emit edges, duplicating where required
                for idx in range(cnt):
                    # For the special Rope→QK case, annotate the twin edges
                    if parent == "rope" and child == "qk_matmul":
                        style = {**style, "label": "Q" if idx == 0 else "K",
                                 "fontsize": "10"}
                    dot.edge(parent, child, **style)

    # ----------------------------- Public API ----------------------------- #
    def render(self, out_prefix: str = "layer_graph") -> None:
        out_dir = Path("figures")
        out_dir.mkdir(exist_ok=True)

        for stage, res in self.results.items():
            dot = self._make_dot()
            use_nodes = self._add_nodes_and_clusters(dot, res)
            self._add_edges(dot, use_nodes)

            png_path = out_dir / f"graph_{stage}_{out_prefix}"
            dot.render(str(png_path), cleanup=True)
            print(f"[LayerGraphVisualizer] Saved to {png_path}.png")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate transformer layer graph")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--output", default="layer_graph")
    args = parser.parse_args()

    with args.result_json.open() as fp:
        results = json.load(fp)

    LayerGraphVisualizer(
        model_cfg=args.model_type,
        results=results,
        num_experts=args.num_experts,
    ).render(args.output)


if __name__ == "__main__":
    _main()
