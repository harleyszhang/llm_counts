from .utils.config import ModelConfig, DeepseekConfig
from .utils.constants import *
from .utils.utils import num_to_string
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any
import numpy as np


class CountCausalLMParams(object):
    def __init__(self, model_config: ModelConfig) -> None:
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        self.num_layers = model_config.num_layers
        self.V = model_config.vocab_size

        self.num_heads = model_config.num_heads
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = model_config.head_dim
        self.model_type = model_config.model_type

        self.is_moe = (getattr(model_config, "num_experts", None) is not None)
        self.is_qwen3moe = "qwen3_moe" == self.model_type

        if self.is_moe:
            self.num_experts = getattr(model_config, "num_experts", 1)  # Default to 1 expert if not specified
            self.num_experts_per_tok = getattr(model_config, "num_experts_per_tok", 1)

    def params_embedding(self, shared_embedding: bool = True) -> int:
        """Get the number of parameters in the embedding layer. 
        params_te = vocab_size * d_model
        Args:
            shared_embedding (bool, optional):  whether the output embedding 
            shares weights with the input embedding. Defaults to True.

        Returns: 
            int: the number of parameters in the embedding layer.
        """
        num_params_input_embedding = self.V * self.hidden_size
        num_params_output_embedding = (
            self.V * self.hidden_size if not shared_embedding else 0
        )

        return num_params_input_embedding + num_params_output_embedding

    def params_per_layer_mha(self) -> int:
        """Get the number of parameters per layer in the attention module
        which include 4 linear layer: q/k/v/o linear layers.

        Returns:
            int: the number of parameters per layer in the attention module(mha)
        """
        params_qo_proj = 2 * self.hidden_size * self.num_heads * self.head_dim
        params_kv_proj = 2 * self.hidden_size * self.num_kv_heads * self.head_dim
        return params_qo_proj + params_kv_proj

    def params_per_layer_moe_mlp(self, is_active: bool = True) -> int:
        """Get the number of parameters in the MoE MLP linear layers.
        params of mlp = gate_proj + up_proj + down_proj
        gate_proj / up_proj / down_proj params: hidden_size * intermediate_size
        """
        if self.is_qwen3moe:
            params_router = self.hidden_size * self.num_experts
            expert_counts = self.num_experts_per_tok if is_active else self.num_experts
            params_experts = expert_counts * 3 * self.hidden_size * self.intermediate_size
            return params_router + params_experts
        else:
            params_mlp = 3 * self.hidden_size * self.intermediate_size
            return params_mlp

    def params_per_layer_norm(self) -> int:
        """Get the number of atten_norm and mlp_norm parameters per layer.
        """
        # q_norm、k_norm、atten_norm、mlp_norm
        if self.model_type == "qwen3":
            return 2 * self.hidden_size + 2 * self.head_dim
        else:
            return 2 * self.hidden_size

    def params_per_layer(self, norm_ignore=False) -> tuple:
        """Get the number of params per layer mainly including the attention and MLP layers.

        params_per_layer = params_mha + params_mlp + params_norm

        """
        params_per_layer_mha = self.params_per_layer_mha()
        params_per_layer_mlp = self.params_per_layer_moe_mlp()
        params_per_layer_norm = 0 if norm_ignore else self.params_per_layer_norm()
        params_input_embedding = self.params_embedding()

        params_per_layer = (
            params_per_layer_mha + params_per_layer_mlp + params_per_layer_norm
        )

        dict_params_per_layer = {
            "qkvo_proj": params_per_layer_mha,
            "mlp": params_per_layer_mlp,
            "rmsnorm": params_per_layer_norm,
            "input_embedding": params_input_embedding,
            "output_embedding": params_input_embedding,
        }

        return params_per_layer, dict_params_per_layer

    def params_model(self) -> int:
        """Get the total number of parameters in the model 
        including all layers and token embedding layer.
        params_model = params_embedding + params_per_layer * num_layers
                    = V * d_model + 12 * d_model**2 * num_layers
        Returns:
            int: the total number of parameters in the model
        """
        params_per_layer, _ = self.params_per_layer()
        params_model = (
            params_per_layer * self.num_layers + self.params_embedding()
        )

        return params_model