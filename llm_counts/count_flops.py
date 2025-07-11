from .utils.config import ModelConfig, DeepseekConfig
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
import pandas as pd  # 移至顶部，避免函数内重复导入


class CountCausalLMFlops(object):
    """CountCausalLMFlops is a class that counts the number of floating point operations (FLOPs) 
    for a causal language model (LLM) during the forward passes, 支持 MoE 结构。"""

    def __init__(
        self,
        model_config: ModelConfig,
    ) -> None:
        self.model_type = model_config.model_type
        self.num_heads = model_config.num_heads
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = model_config.head_dim
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        self.l = model_config.num_layers
        self.V = model_config.vocab_size
        
        # Determine if this is an MoE model
        self.is_qwen3moe = "qwen3_moe" == self.model_type
        
        if self.is_qwen3moe:
            self.n_routed_experts = getattr(model_config, "num_experts", 1)  # Default to 1 expert if not specified
            self.num_experts_per_tok = getattr(model_config, "num_experts_per_tok", 1)

    def flops_per_layer_qkvo_proj(self, bs: int, seq_len: int) -> int:
        """Get the number of floating point operations (flops) for the forward
        pass of the attention linear layers, given the batch size and sequence length.

        flops_qkvo_proj = flops_q + flops_k + flops_v + flops_output

        Arguments:
            bs (int): batch size
            seq_len (int): sequence length
        """
        q_proj_flops = 2 * bs * seq_len * self.hidden_size * self.num_heads * self.head_dim
        k_proj_flops = 2 * bs * seq_len * self.hidden_size * self.num_kv_heads * self.head_dim
        v_proj_flops = 2 * bs * seq_len * self.hidden_size * self.num_kv_heads * self.head_dim
        o_proj_flops = 2 * bs * seq_len * self.hidden_size * self.num_heads * self.head_dim
        qkvo_proj_flops = q_proj_flops + k_proj_flops + v_proj_flops + o_proj_flops

        return qkvo_proj_flops

    def flops_per_layer_moe_mlp(self, bs: int, seq_len: int) -> int:
        """Count the number of floating point operations (flops) for the moe_mlp layer forward    
        """
        if self.is_qwen3moe:
            flops_router = 2 * bs * seq_len * self.hidden_size * self.n_routed_experts
            flops_experts = self.num_experts_per_tok * 3 * 2 * bs * seq_len * self.hidden_size * self.intermediate_size
            moe_flops_per_layer = flops_router + flops_experts
            return moe_flops_per_layer
        else:
            flops_gate_proj = 2 * bs * seq_len * self.hidden_size * self.intermediate_size
            flops_up_proj = 2 * bs * seq_len * self.hidden_size * self.intermediate_size
            flops_down_proj = 2 * bs * seq_len * self.intermediate_size * self.hidden_size
            return flops_gate_proj + flops_up_proj + flops_down_proj
    
    def flops_per_layer_attn_kernel(self, bs: int, seq_len: int, generate_len: int) -> int:
        q_norm_flops = bs * 4 * seq_len * self.head_dim
        k_norm_flops = q_norm_flops
        # e^x / sum(e^x); bs = 1 和 seq_len = 1 时 flops 为 3d-1, 张量中每个元素约执行 3 次操作
        softmax_flops =  bs * 3 * seq_len * self.num_heads * self.head_dim

        if seq_len != 1:
            qk_matmul_flops = bs * 2 * seq_len * seq_len * self.num_heads * self.head_dim
            sv_matmul_flops = qk_matmul_flops
   
        else:
            qk_matmul_flops = 2 * self.num_heads * self.head_dim * (seq_len + generate_len)
            sv_matmul_flops = qk_matmul_flops
            
        flops_self_attention_kernel = q_norm_flops + k_norm_flops + qk_matmul_flops + sv_matmul_flops + softmax_flops

        return flops_self_attention_kernel

    def flops_per_layer_norm(self, bs: int, seq_len: int) -> int:
        """flops of 2 rmsnrom per layer"""
        return bs * 4 * seq_len * self.hidden_size

    def flops_per_layer(self, bs: int, seq_len: int, generate_len:int) -> tuple:
        flops_per_layer_qkvo_proj = self.flops_per_layer_qkvo_proj(bs, seq_len)
        flops_per_layer_mlp = self.flops_per_layer_moe_mlp(bs, seq_len)

        flops_per_layer_attention_kernel = self.flops_per_layer_attn_kernel(
            bs, seq_len, generate_len,
        )
        flops_per_layer_rmsnorm = self.flops_per_layer_norm(bs, seq_len) * 2

        flops_positional_embedding = self.flops_positional_embedding(bs, seq_len)

        flops_per_layer = (
            flops_per_layer_qkvo_proj
            + flops_per_layer_mlp
            + flops_per_layer_rmsnorm
            + flops_per_layer_attention_kernel
            + flops_positional_embedding
        )

        dict_flops_per_layer = {
            "attention_kernel": flops_per_layer_attention_kernel,
            "qkvo_proj": flops_per_layer_qkvo_proj,
            "mlp": flops_per_layer_mlp,
            "rmsnorm": flops_per_layer_rmsnorm,
            "positional_embedding": flops_positional_embedding,
            "input_embedding": 0,
        }

        return flops_per_layer, dict_flops_per_layer

    def flops_positional_embedding(
        self,
        bs:int,
        seq_len:int,
    ) -> int:
        """flops of output token logits layer"""
        return 2 * bs * seq_len * self.hidden_size

    def flops_model(self, bs: int, seq_len: int, generate_len: int) -> int:
        """Count flops of the forward pass of the transformer model, 
        given the batch size and sequence length.
        """
        num_flops_model = self.flops_per_layer(bs, seq_len, generate_len)[0] * self.l
    
        return num_flops_model

    def flops_bwd_model(self, bs: int, seq_len: int, generate_len:int) -> int:
        """Get the number of floating point operations (flops) for the backward
        pass of the entire transformer model, given the batch size and sequence
        """
        return 2 * self.flops_model(bs, seq_len, generate_len)

