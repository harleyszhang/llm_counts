# llm_profiler

llm theoretical performance analysis tools and support params, flops, memory and latency analysis.

## 主要功能

- 支持 qwen2.5、qwen3 dense 系列模型。
- 支持张量并行推理模式。
- 支持 `A100`、`V100`、`T4` 等硬件以及主流 decoder-only 的自回归模型，可自行在配置文件中增加。
- 支持分析性能瓶颈，不同 `layer` 是 `memory bound` 还是 `compute bound`，以及 `kv_cache` 的性能瓶颈。
- 支持输出每层和整个模型的参数量、计算量，内存和 `latency`。
- 推理时支持预填充和解码阶段分别计算内存和 latency、以及理论支持的最大 `bs` 等等。
- 支持设置计算效率、内存读取效率（不同推理框架可能不一样，这个设置好后，可推测输出实际值）。
- 推理性能理论分析结果的格式化输出。

## 如何使用

使用方法，直接调用 `llm_profiler/llm_profiler.py` 文件中函数 `llm_profile()` 函数并输入相关参数即可。

```python
def llm_profile(model_name="llama-13b",
                gpu_name: str = "v100-sxm-32gb",
                bytes_per_param: int = BYTES_FP16,
                bs: int = 1,
                seq_len: int = 522,
                generate_len=1526,
                ds_zero: int = 0,
                dp_size: int = 1,
                tp_size: int = 1,
                pp_size: int = 1,
                sp_size: int = 1,
                layernorm_dtype_bytes: int = BYTES_FP16,
                kv_cache_bytes: int = BYTES_FP16,
                flops_efficiency: float = FLOPS_EFFICIENCY,
                hbm_memory_efficiency: float = HBM_MEMORY_EFFICIENCY,
                intra_node_memory_efficiency=INTRA_NODE_MEMORY_EFFICIENCY,
                inter_node_memory_efficiency=INTER_NODE_MEMORY_EFFICIENCY,
                mode: str = "inference",
            ) -> dict:

    """format print dicts of the total floating-point operations, MACs, parameters and latency of a llm.

    Args:
        model_name (str, optional): model name to query the pre-defined `model_configs.json`. Defaults to "llama-13b".
        gpu_name (str, optional): gpu name to query the pre-defined `model_configs.json`. Defaults to "v100-sxm2-32gb".
        bs (int, optional): _description_. Defaults to 1.
        seq_len (int, optional): batch size per GPU.. Defaults to 522.
        generate_len (int, optional): The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. Defaults to 1526.
        dp_size (int, optional): data parallelism size. Defaults to 1.
        tp_size (int, optional): tensor parallelism size. Defaults to 1.
        pp_size (int, optional): pipeline parallelism size. Defaults to 1.
        sp_size (int, optional): sequence parallelism size. Defaults to 1.
            speed up decoding. Defaults to True.
        layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations.. Defaults to BYTES_FP16.
        kv_cache_bytes (int, optional): number of bytes in the data type for the kv_cache. Defaults to None.
        flops_efficiency (float, optional): flops efficiency, ranging from 0 to 1. Defaults to None.
        hbm_memory_efficiency (float, optional): GPU HBM memory efficiency, ranging from 0 to 1. Defaults to HBM_MEMORY_EFFICIENCY.
        intra_node_memory_efficiency (_type_, optional): intra-node memory efficiency, ranging from 0 to 1.. Defaults to INTRA_NODE_MEMORY_EFFICIENCY.
        inter_node_memory_efficiency (_type_, optional): inter-node memory efficiency, ranging from 0 to 1.. Defaults to INTER_NODE_MEMORY_EFFICIENCY.

    Returns:
        None: format print some summary dictionary of the inference analysis
    """
```

`llama2-70` 模型，tp_size = 8 和 bs = 20，输出示例信息如下所示：

```bash
-------------------------- LLM main infer config --------------------------
{   'inference_config': {   'model_name': 'llama2-70b',
                            'num_attention_heads': 64,
                            'num_kv_heads': 8,
                            'head_dim': 128,
                            'hidden_size': 8192,
                            'intermediate_size': 28672,
                            'vocab_size': 32000,
                            'max_seq_len': 4096,
                            'bs': 32,
                            'seq_len': 1024,
                            'tp_size': 8,
                            'pp_size': 1,
                            'generate_len': 128},
    'gpu_config': {   'name': 'a100-sxm-40gb',
                      'memory_in_GB': '40 GB',
                      'gpu_hbm_bw': '1555 GB/s',
                      'gpu_intra_node_bw': '600 GB/s',
                      'gpu_fp16_TFLOPS': '312 TFLOPS'}}

-------------------------- LLM infer performance analysis --------------------------
{   'weight_memory_per_gpu': '17.18 GB',
    'consume_memory_per_gpu': '20.57 GB',
    'prefill_flops': '4574.25 T',
    'decode_flops_per_step': '4.38 T',
    'TTFT': 2.7060724961666294,
    'TTOT': 0.040541745771914876,
    'kv_cache_latency': '959.04 us',
    'total_infer_latency': '7.9 s',
    'support_max_batch_total_tokens': 240249}

---------------------------- LLM Params per_layer analysis ----------------------------
{   'qkvo_proj': '150.99 M',
    'mlp': '704.64 M',
    'rmsnorm': '16.38 K',
    'input_embedding': '262.14 M',
    'output_embedding': '262.14 M'}
{'params_model': '68.71 G'}

---------------------------- LLM Prefill Flops per_layer analysis ----------------------------
{   'attention_kernel': '1.1 T',
    'qkvo_proj': '9.9 T',
    'mlp': '46.18 T',
    'rmsnorm': '4.29 G',
    'positional_embedding': '536.87 M',
    'input_embedding': '0'}
{'prefill flops_model': '4574.25 T'}

---------------------------- LLM Memory analysis (Prefill) ----------------------------
{   'weight_memory_per_gpu': '17.18 GB',
    'prefill_max_bs': '388B',
    'prefill_act_per_gpu': '1.88 GB'}

---------------------------- LLM Memory analysis (Decode) ----------------------------
{   'decode_act_per_gpu': '1.88 GB',
    'kv_cache_memory_per_gpu': '1.51 GB',
    'consume_memory_per_gpu': '20.57 GB',
    'decode_max_bs': '215.0B',
    'max_batch_total_tokens': '240.25 KB'}

---------------------------- LLM Latency analysis (Prefill) ----------------------------
{   'prefill_qkvo_proj': '352.41 ms',
    'prefill_attn_kernel': '131.39 ms',
    'prefill_mlp': '1.64 s',
    'prefill_rmsnorm': '61.38 ms',
    'prefill_tp_comm': '501.08 ms',
    'prefill_kv_cache_rw': '959.04 us',
    'prefill_latency': '2.71 s'}

---------------------------- LLM Latency analysis (Decode) ----------------------------
{   'decode_qkvo_proj': '6.5 ms',
    'decode_attn_kernel': '2.56 ms',
    'decode_mlp': '30.26 ms',
    'decode_rmsnorm': '64.62 us',
    'decode_tp_comm': '640.0 us',
    'decode_kv_cache_rw': '121.75 us',
    'kv_cache_latency': '959.04 us',
    'decode_latency': '40.54 ms'}
```

## 模型结构可视化

llama2-70b 模型，A100-SXM40GB，tp_size = 8 和 bs = 20，prefill 阶段:

<div align="center">
<img src="images/grpah_prefill_llama2-70b_tp8_bs32_seqlen1024_genlen128.png" width="50%" alt="prefill 阶段">
</div>

llama2-70b 模型，A100-SXM40GB，tp_size = 8 和 bs = 20， decode 阶段:

<div align="center">
<img src="images/grpah_decode_llama2-70b_tp8_bs32_seqlen1024_genlen128.png" width="50%" alt="decode 阶段">
</div>

qwen3 moe 模型结构可视化, expert 结构其实就是 moe_mlp，下图没有展开。graph_prefil_Qwen3-30B-A3B_tp1_bs16_seqlen600_genlen128:

![graph_prefill_Qwen3-30B-A3B](./images/Qwen3-30B-A3B/graph_prefill_Qwen3-30B-A3B_tp1_bs16_seqlen600_genlen128.png)

graph_decode_Qwen3-30B-A3B_tp1_bs16_seqlen600_genlen128:

![decode_Qwen3-30B-A3B](./images/Qwen3-30B-A3B/graph_decode_Qwen3-30B-A3B_tp1_bs16_seqlen600_genlen128.png)

## 模型参数量、计算量、latency 分布

llama2-70b 模型，A100-SXM40GB，tp_size = 8 和 bs = 20，参数量统计分布:

<div align="center">
<img src="images/params_llama2-70b_tp8_bs32_seqlen1024_genlen128.png" width="50%" alt="prefill 阶段">
</div>

llama2-70b 模型，A100-SXM40GB，tp_size = 8 和 bs = 20，prefill 阶段计算量统计分布:

<div align="center">
<img src="images/flops_prefill_llama2-70b_tp8_bs32_seqlen1024_genlen128.png" width="50%" alt="prefill 阶段计算量统计分布">
</div>

llama2-70b 模型，A100-SXM40GB，tp_size = 8 和 bs = 20，generate_len = 128, decode 阶段计算量统计分布:

<div align="center">
<img src="images/flops_decode_llama2-70b_tp8_bs32_seqlen1024_genlen128.png" width="50%" alt="decode 阶段计算量统计分布">
</div>

llama2-70b 模型，A100-SXM40GB，tp_size = 8 和 bs = 20，prefill 阶段 latency 统计分布:

<div align="center">
<img src="images/latency_prefill_llama2-70b_tp8_bs32_seqlen1024_genlen128.png" width="50%" alt="prefill 阶段 latency 统计分布">
</div>

llama2-70b 模型，A100-SXM40GB，tp_size = 8 和 bs = 20，decode 阶段 latency 统计分布:

<div align="center">
<img src="images/latency_decode_llama2-70b_tp8_bs32_seqlen1024_genlen128.png" width="50%" alt="decode 阶段 latency 统计分布">
</div>

## 参考链接
- [Transformer 性能分析理论基础](https://github.com/HarleysZhang/dl_note/blob/main/6-llm_note/transformer_basic/Transformer%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.md)
- [llm_analysis](https://github.com/cli99/llm-analysis)
- [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/)
- [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer.git)