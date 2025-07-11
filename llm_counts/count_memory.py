from .utils.config import LLMConfigs, DeepseekConfig, GPUConfig
from .utils.constants import BYTES_FP16
from .count_params import CountCausalLMParams

from functools import reduce
import operator as _op

def _B(*dims):
    """Utility: multiply arbitrary dimensions to get a byte count."""
    return reduce(_op.mul, dims, 1)


class CountCausalLMMemory(object):
    """Count memory of the model and layers."""

    def __init__(self, llm_configs: LLMConfigs) -> None:
        self.model_config = llm_configs.model_config
        self.model_type = self.model_config.model_type
        self.hidden_size = self.model_config.hidden_size
        self.intermediate_size = self.model_config.intermediate_size

        self.num_heads = self.model_config.num_heads
        self.num_kv_heads = self.model_config.num_kv_heads
        self.head_dim = self.model_config.head_dim
        self.num_layers = self.model_config.num_layers
        self.V = self.model_config.vocab_size

        self.bytes_per_param = llm_configs.inference_config.bytes_per_param
        self.act_dtype_bytes = BYTES_FP16

        self.tp_size = llm_configs.parallelism_config.tp_size
        self.pp_size = llm_configs.parallelism_config.pp_size
        self.num_layers_per_gpu = int(self.num_layers / self.pp_size)

        self.gpu_memory = llm_configs.gpu_config.memory_in_GB * 10**9
        self.llm_params = CountCausalLMParams(self.model_config)

        self.is_moe = (getattr(self.model_config, "num_experts", None) is not None)
        self.is_qwen3moe = "qwen3_moe" == self.model_type
        
        if self.is_moe:
            # Default to 1 expert if not specified
            self.num_experts = getattr(self.model_config, "num_experts", 1)  
            self.num_experts_per_tok = getattr(self.model_config, "num_experts_per_tok", 1)

    def memory_weight_per_gpu(self, ):
        """Get the memory of the model weights"""
        params_model = self.llm_params.params_model()
        memory_weight_per_gpu = params_model * self.bytes_per_param / self.tp_size

        return memory_weight_per_gpu
    
    def mac_per_layer_attn_kernel(
        self,
        bs: int,
        seq_len,
        generate_len: int = 0,
        flash_attn: bool = False,
        kv_cache_bytes: int = BYTES_FP16,
    ):
        if self.model_type == "qwen3":
            norm_bytes = 2 * (
                _B(self.head_dim, BYTES_FP16)                    # load γ
                + 2 * _B(bs, seq_len, self.head_dim, BYTES_FP16) # load + store acts
            )
        else:
            norm_bytes = 0

        if not flash_attn:
            if seq_len != 1:
                # dim changge: (bs, seq_len, hidden_size) -> (bs, seq_len, num_heads, head_dim)
                # (bs, seq_len, num_heads, head_dim) -> (bs, num_heads, seq_len, head_dim)
                # qk^t: (bs, num_heads, seq_len, head_dim) *  (bs, num_kv_heads, seq_len, head_sim) -> (bs, num_heads, seq_len, seq_len)
                # sv: (bs, num_heads, seq_len, seq_len) * (bs, num_kv_heads, seq_len, head_dim) -> (bs, num_heads, seq_len, head_dim)

                load_q_mem = bs * self.num_heads * seq_len * self.head_dim
                load_k_mem = bs * self.num_kv_heads * seq_len * self.head_dim
                qk_store_mem = bs * self.num_heads * seq_len * seq_len

                load_softmax_mem = qk_store_mem
                softmax_store_mem = bs * self.num_heads * seq_len * seq_len

                load_s_mem = softmax_store_mem
                load_v_mem = bs * self.num_kv_heads * seq_len * self.head_dim
                sv_store_mem = bs * self.num_heads * seq_len * self.head_dim

                self_atten_mac = (load_q_mem + load_k_mem + qk_store_mem
                                  + load_softmax_mem + softmax_store_mem
                                  + load_s_mem + load_v_mem + sv_store_mem)
                max_act = max(load_q_mem, load_k_mem, qk_store_mem,
                              load_softmax_mem, softmax_store_mem,
                              load_s_mem, load_v_mem, sv_store_mem) * self.act_dtype_bytes
                return max_act, self_atten_mac * kv_cache_bytes + norm_bytes

            else:
                # dim changge: (bs, 1, hidden_size) -> (bs, 1, num_heads, head_dim)
                # (bs, 1, num_heads, head_dim) -> (bs, num_heads, 1, head_dim)
                # qk^t: (bs, num_heads, seq_len + generate_len, head_dim) *  (bs, num_kv_heads, seq_len + generate_len, head_sim) -> (bs, num_heads, seq_len + generate_len, seq_len + generate_len)
                # sv: (bs, num_heads, seq_len + generate_len, seq_len + generate_len) * (bs, num_kv_heads, seq_len + generate_len, head_dim) -> (bs, num_heads, seq_len + generate_len, head_dim)

                load_q_mem = bs * self.num_heads * 1  * self.head_dim
                load_k_mem = bs * self.num_kv_heads * (seq_len + generate_len) * self.head_dim
                qk_store_mem = bs * self.num_heads * (seq_len + generate_len) * (seq_len + generate_len)

                load_softmax_mem = qk_store_mem
                softmax_store_mem = bs * self.num_heads * (seq_len + generate_len) * (seq_len + generate_len)

                load_s_mem = softmax_store_mem
                load_v_mem = bs * self.num_kv_heads * (seq_len + generate_len) * self.head_dim
                sv_store_mem = bs * self.num_heads * (seq_len + generate_len) * self.head_dim

                max_act = max(load_q_mem, load_k_mem, qk_store_mem,
                              load_softmax_mem, softmax_store_mem,
                              load_s_mem, load_v_mem, sv_store_mem) * self.act_dtype_bytes
                self_atten_mac = (load_q_mem + load_k_mem + qk_store_mem
                                  + load_softmax_mem + softmax_store_mem
                                  + load_s_mem + load_v_mem + sv_store_mem)
                
                return max_act, self_atten_mac * kv_cache_bytes + norm_bytes
        
    def mac_per_layer_kv_cache(
        self,
        bs,
        seq_len,
        generate_len: int = 0,
        flash_attn: bool = False,
        kv_cache_bytes: int = BYTES_FP16,
    ):
        if not flash_attn:
            store_k_cache = (
                self.num_kv_heads * self.head_dim * bs * seq_len * kv_cache_bytes
            )
            store_v_cache = (
                self.num_kv_heads * self.head_dim * bs * seq_len * kv_cache_bytes
            )
            if seq_len != 1:
                return store_k_cache + store_v_cache
            else:
                qk_matmul_load_k_cache = (
                    (seq_len + generate_len)
                    * self.head_dim
                    * bs
                    * self.num_kv_heads
                    * kv_cache_bytes
                )
                sv_matmul_load_v_cache = (
                    (seq_len + generate_len)
                    * self.head_dim
                    * bs
                    * self.num_kv_heads
                    * kv_cache_bytes
                )
            
                kv_cache_mac = (
                    qk_matmul_load_k_cache
                    + sv_matmul_load_v_cache
                    + store_k_cache
                    + store_v_cache
                )
                return kv_cache_mac
        else:
            # FlashAttention path: compute attention on‑the‑fly; only new K/V cache entries are stored
            kv_cache_mac = (
                self.num_kv_heads
                * self.head_dim
                * bs
                * seq_len
                * 2  # K + V
                * kv_cache_bytes
            )

        return kv_cache_mac

    def mac_per_layer_qkvo_proj(
        self,
        bs: int,
        seq_len: int,
        qkvo_weight_dtype_bytes=BYTES_FP16,
    ) -> int:
        """
        Count memory access cost for Q/K/V/O projection layers.
        """
        atten_linear_layers = {
            "q_proj": [self.hidden_size, self.num_heads * self.head_dim],
            "k_proj": [self.hidden_size, self.num_kv_heads * self.head_dim],
            "v_proj": [self.hidden_size, self.num_kv_heads * self.head_dim],
            "out_proj": [self.num_heads * self.head_dim, self.hidden_size],
        }

        atten_linear_layers_mac = 0
        max_act = 0
        for name, (in_ch, out_ch) in atten_linear_layers.items():
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj

            load_weight = in_ch * out_ch
            load_act = in_ch * bs * seq_len
            store_act = 0 if is_kv_proj else bs * seq_len * out_ch
            load_kv_cache = 0
            store_kv_cache = 0 if is_normal_proj else out_ch * bs * seq_len

            max_act = max(max_act, load_weight, load_act, store_act, store_kv_cache) 

            mac = load_weight + load_act + store_act + load_kv_cache + store_kv_cache
            atten_linear_layers_mac += mac
        
        return max_act * self.act_dtype_bytes, atten_linear_layers_mac * qkvo_weight_dtype_bytes

    def mac_per_layer_moe_mlp(
        self,
        bs: int,
        seq_len: int,
        mlp_weight_dtype_bytes=BYTES_FP16,
    ) -> float:
        """The `mlp` acts include the input to the two linear layers.
        Refer to https://arxiv.org/abs/2205.05198 for details. 
        The two linear layers store their inputs with size 2bsh and 8bsh
        """
        mlp_linear_layers = {
            "gate_proj": [self.hidden_size, self.intermediate_size],
            "up_proj": [self.hidden_size, self.intermediate_size],
            "down_proj": [self.intermediate_size, self.hidden_size],
        }
        num_experts_per_tok = getattr(self.model_config, "num_experts_per_tok", 1)

        if self.is_qwen3moe:
            # Qwen3-MoE has an additional linear layer[router] for the expert selection
            mlp_linear_layers["gate"] = [self.hidden_size, self.head_dim]
            
        num_experts_per_tok = getattr(self.model_config, "num_experts_per_tok", 1)

        if self.is_qwen3moe:
            # Qwen3-MoE has an additional linear layer[router] for the expert selection
            mlp_linear_layers["gate"] = [self.hidden_size, self.head_dim]
            
        mlp_linear_layers_mac = 0
        max_act = 0
        for _, (in_ch, out_ch) in mlp_linear_layers.items():
            load_weight = in_ch * out_ch
            load_act = in_ch * bs * seq_len
            store_act = bs * seq_len * out_ch

            max_act = max(max_act, load_weight, load_act, store_act) 
            mac = load_weight + load_act + store_act
            mlp_linear_layers_mac += mac
        
        return max_act * self.act_dtype_bytes, mlp_linear_layers_mac * mlp_weight_dtype_bytes

    def mac_per_layer_norm(
        self,
        bs: int,
        seq_len: int,
    ) -> float:
        """Get the memory (in bytes) required to store the acts of a single layernorm in a transformer layer."""
        rmsnorm_load_weight = self.hidden_size * self.act_dtype_bytes
        rmsnorm_load_act = bs * seq_len * self.hidden_size * self.act_dtype_bytes
        rmsnorm_store_act = bs * seq_len * self.hidden_size * self.act_dtype_bytes

        norm_mac_per_gpu = (
            rmsnorm_load_weight + rmsnorm_load_act + rmsnorm_store_act
        )
        max_act = max(rmsnorm_load_weight, rmsnorm_load_act, rmsnorm_store_act) * self.act_dtype_bytes
        return max_act, norm_mac_per_gpu

    def mac_input_embedding(self, bs: int, seq_len: int) -> float:
        input_embedding_load_act = bs * seq_len * self.act_dtype_bytes
        input_embedding_store_act = (
            bs * seq_len * self.hidden_size * self.act_dtype_bytes
        )
        input_embedding_mac_per_gpu = (
            input_embedding_load_act + input_embedding_store_act
        )

        return input_embedding_mac_per_gpu

    def memory_kv_cache_per_layer(
        self,
        bs: int,
        seq_len: int,
        generate_len: int,
        kv_cache_bytes: int = BYTES_FP16,
    ) -> float:
        """Get the memory (in bytes) required to store the key and value cache
        for a transformer layer in inference, given the batch size, sequence
        length, act data type, and tensor parallelism size.

        memory_kv_cache = 4blh(s+o) unit is byte
        Args:
            bs (int): batch size
            context_len (int): seq_len + generate_len

        Returns:
            float: the memory (in bytes) required  to store the key and value cache 
            for a transformer layer in inference.
        """

        # At least on attention head on each tensor-parallel GPU
        num_kv_heads_per_gpu = max(self.num_kv_heads, 1)
        memory_kv_cache_per_layer = (
            bs
            * (seq_len + generate_len)
            * num_kv_heads_per_gpu
            * self.head_dim
            * 2
            * kv_cache_bytes
        )

        return memory_kv_cache_per_layer

    def max_act_per_layer(
        self,
        bs: int,
        seq_len_ctx: int,
        generate_len: int = 0,  # used only for decode stage
        *,
        stage: str = "prefill",        # "prefill" | "decode"
        flash_attn: bool = False,
        qkvo_weight_dtype_bytes: int = BYTES_FP16,
        mlp_weight_dtype_bytes: int = BYTES_FP16,
    ) -> float:
        assert stage in {"prefill", "decode"}

        # For decode stage each step handles just **one token**.
        tokens = 1 if stage == "decode" else seq_len_ctx

        act_per_layer_self_atten, _ = self.mac_per_layer_attn_kernel(
            bs,
            tokens,
            generate_len=generate_len,
            flash_attn=flash_attn,
            kv_cache_bytes=qkvo_weight_dtype_bytes,
        )
        act_per_layer_qkvo_proj, _ = self.mac_per_layer_qkvo_proj(
            bs,
            tokens,
            qkvo_weight_dtype_bytes=qkvo_weight_dtype_bytes,
        )
        act_per_layer_mlp, _ = self.mac_per_layer_moe_mlp(
                bs,
                tokens,
                mlp_weight_dtype_bytes=mlp_weight_dtype_bytes,
            )
        act_per_layer_rn, _ = self.mac_per_layer_norm(bs, tokens) 

        act_per_layer = max(act_per_layer_self_atten, act_per_layer_qkvo_proj, act_per_layer_mlp, act_per_layer_rn)

        return act_per_layer
    
    def memory_per_gpu(
        self,
        bs: int,
        seq_len: int,
        generate_len: int,
        flash_attn: bool = True,
        qkvo_weight_dtype_bytes: int = BYTES_FP16,
        mlp_weight_dtype_bytes=BYTES_FP16,
        kv_cache_bytes: int = BYTES_FP16,
    ) -> tuple:
        # 1, prefill stage count memory and max_bs
        weight_memory_per_gpu = self.memory_weight_per_gpu()  # count model weights memory
        memory_left_per_gpu = self.gpu_memory - weight_memory_per_gpu

        # --- 1) PREFILL stage ----------------------------------------- #
        prefill_act_bs_1 = self.max_act_per_layer(
            1,
            seq_len,
            generate_len=generate_len,
            stage="prefill",
            flash_attn=flash_attn,
            qkvo_weight_dtype_bytes=qkvo_weight_dtype_bytes,
            mlp_weight_dtype_bytes=mlp_weight_dtype_bytes,
        ) // self.tp_size

        prefill_max_bs = int(memory_left_per_gpu / prefill_act_bs_1)
        prefill_act_per_gpu = bs * prefill_act_bs_1

        # --- 2) DECODE stage ------------------------------------------ #
        kv_cache_memory_bs_1_per_gpu = (self.memory_kv_cache_per_layer(1, seq_len, generate_len, kv_cache_bytes) * self.num_layers_per_gpu) / self.tp_size
        decode_act_bs_1_per_gpu = self.max_act_per_layer(
            1,
            seq_len,
            generate_len=generate_len,
            stage="decode",
            flash_attn=flash_attn,
            qkvo_weight_dtype_bytes=qkvo_weight_dtype_bytes,
            mlp_weight_dtype_bytes=mlp_weight_dtype_bytes,
        ) // self.tp_size
        decode_max_bs = memory_left_per_gpu // (decode_act_bs_1_per_gpu + kv_cache_memory_bs_1_per_gpu)

        kv_cache_memory_per_gpu = bs * kv_cache_memory_bs_1_per_gpu
        decode_act_per_gpu = decode_act_bs_1_per_gpu * bs
        max_batch_total_tokens = decode_max_bs * (seq_len + generate_len)

        assert bs <= decode_max_bs, (
            f"For context length: {seq_len + generate_len}, bs {bs} is too large to fit"
            " in GPU memory, decode_max_bs:"
            f" {decode_max_bs}"
        )

        assert memory_left_per_gpu > (
            kv_cache_memory_per_gpu + decode_act_per_gpu
        ), (
            "kv_cache and act memory with bs ="
            f" {bs} is too large to fit in GPU memory"
        )
        
        consume_memory_per_gpu = (
            weight_memory_per_gpu + decode_act_per_gpu + kv_cache_memory_per_gpu
        )

        # memory summary
        memory_prefill_summary_dict = {
            "weight_memory_per_gpu": weight_memory_per_gpu,
            "prefill_max_bs": prefill_max_bs,
            "prefill_act_per_gpu": prefill_act_per_gpu,
        }

        memory_decode_summary_dict = {
            "decode_act_per_gpu": decode_act_per_gpu,
            "kv_cache_memory_per_gpu": kv_cache_memory_per_gpu,
            "consume_memory_per_gpu": consume_memory_per_gpu,
            "decode_max_bs": decode_max_bs,
            "max_batch_total_tokens": int(max_batch_total_tokens * 0.97),
        }

        return memory_prefill_summary_dict, memory_decode_summary_dict

