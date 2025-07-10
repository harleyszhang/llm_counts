# -*- coding  : utf-8 -*-
# author      : honggao.zhang
# Create      : 2024-7-19
# Update      : 2025-01-05
# Version     : 0.2.0
# Description : transformer model(llm) profiling tools,
#               with latency, memory, flops, and params distribution analysis.

import logging
import pprint
import matplotlib.pyplot as plt
import argparse
import json
import copy

from .utils.config import *
from .utils.utils import *

from .count_flops import CountCausalLMFlops
from .count_params import CountCausalLMParams
from .count_memory import CountCausalLMMemory
from .count_latency import CountCausalLMLatency
from .layer_graph_visualizer import LayerAnalyzer, LayerGraphVisualizer

logger = logging.getLogger()


class LayerAnalyzerVisual(object):
    """Measures the latency, memory, number of estimated floating-point operations,
    and parameters of each module in a PyTorch model.
    """

    def __init__(self, llm_configs: LLMConfigs) -> None:
        self.llm_configs = llm_configs
        self.model_config = llm_configs.model_config
        self.gpu_config = llm_configs.gpu_config
        self.inference_config = llm_configs.inference_config
        self.parallelism_config = llm_configs.parallelism_config
        self.gpu_efficiency_config = llm_configs.gpu_efficiency_config

        self.h = self.model_config.hidden_size
        self.l = self.model_config.num_layers
        self.V = self.model_config.vocab_size

        self.b = llm_configs.inference_config.bs
        self.s = llm_configs.inference_config.seq_len
        self.o = llm_configs.inference_config.generate_len
        self.bytes_per_param = llm_configs.inference_config.bytes_per_param

        self.tp_size = self.parallelism_config.tp_size
        self.pp_size = self.parallelism_config.pp_size
        self.num_layers_per_gpu = int(self.l / self.parallelism_config.pp_size)

        self.gpu_memory_in_GB = (
            llm_configs.gpu_config.memory_in_GB * 10**9
        ) 

        self.llm_params = CountCausalLMParams(self.model_config)
        self.llm_flops = CountCausalLMFlops(self.model_config)
        self.llm_memory = CountCausalLMMemory(llm_configs)
        self.llm_latency = CountCausalLMLatency(llm_configs)

    def infer_profile(
        self,
        bs: int = 1,
        seq_len: int = 522,
        generate_len: int = 1526,
        act_dtype_bytes: int = BYTES_FP16,
        kv_cache_bytes: int = BYTES_FP16,
        qkvo_weight_dtype_bytes: int = BYTES_FP16,
        mlp_weight_dtype_bytes=BYTES_FP16,
        flops_efficiency: float = None,
        hbm_memory_efficiency: float = HBM_MEMORY_EFFICIENCY,
        intra_node_memory_efficiency=INTRA_NODE_MEMORY_EFFICIENCY,
        inter_node_memory_efficiency=INTER_NODE_MEMORY_EFFICIENCY,
        print_flag=False,
        visual_flag=False,
    ) -> dict:
        """LLM inference analysis given the llm configs and inputs."""

        if self.model_config.max_seq_len is not None:
            assert seq_len + generate_len <= self.model_config.max_seq_len, (
                f"seq_len {seq_len} + generate_len {generate_len} Exceeding the model max_seq_len {self.model_config.max_seq_len}"
            )

        if self.l % self.pp_size != 0:
            logger.warning(
                "Warning: the number of layers is not divisible by pp_size, please taking the floor!"
            )

        infer_config_dict = {
            "inference_config": {
                "model_name": self.model_config.model_name,
                "num_attention_heads": self.model_config.num_heads,
                "num_kv_heads": self.model_config.num_kv_heads,
                "head_dim": self.model_config.head_dim,
                "hidden_size": self.model_config.hidden_size,
                "intermediate_size": self.model_config.intermediate_size,
                "vocab_size": self.model_config.vocab_size,
                "max_seq_len": self.model_config.max_seq_len,
                "bs": bs,
                "seq_len": seq_len,
                "tp_size": self.tp_size,
                "pp_size": self.pp_size,
                "generate_len": generate_len,
            },
            "gpu_config": {
                "name": self.gpu_config.name,
                "memory_in_GB": f"{self.gpu_config.memory_in_GB} GB",
                "gpu_hbm_bw": f"{self.gpu_config.hbm_bw} GB/s",
                "gpu_intra_node_bw": f"{self.gpu_config.intra_node_bw} GB/s",
                "gpu_fp16_TFLOPS": f"{self.gpu_config.peak_fp16_TFLOPS} TFLOPS",
            },
        }

        # -------------------------- 1. Params --------------------------
        params_per_layer, dict_params_per_layer = (
            self.llm_params.params_per_layer()
        )
        num_params_model = self.llm_params.params_model()

        # -------------------------- 2. FLOPs ---------------------------
        prefill_flops_per_layer, prefill_dict_flops_per_layer = (
            self.llm_flops.flops_per_layer(bs, seq_len, generate_len)
        )
        decode_flops_per_layer, decode_dict_flops_per_layer = (
            self.llm_flops.flops_per_layer(bs, 1, generate_len)
        )

        prefill_num_flops_model = self.llm_flops.flops_model(bs, seq_len, generate_len)
        decode_num_flops_model = self.llm_flops.flops_model(bs, 1, generate_len)

        # -------------------------- 3. Memory --------------------------
        memory_prefill_summary_dict, memory_decode_summary_dict = (
            self.llm_memory.memory_per_gpu(
                bs,
                seq_len,
                generate_len,
                flash_attn=False,
                qkvo_weight_dtype_bytes=qkvo_weight_dtype_bytes,
                mlp_weight_dtype_bytes=mlp_weight_dtype_bytes,
                kv_cache_bytes=kv_cache_bytes,
            )
        )

        # -------------------------- 4. Latency -------------------------
        prefill_latency_per_layer, prefill_dict_latency_per_layer = (
            self.llm_latency.latency_per_layer(bs, seq_len, 0)
        )
        decode_latency_per_layer, decode_dict_latency_per_layer = (
            self.llm_latency.latency_per_layer(bs, 1, generate_len)
        )
        prefill_latency_breakdown, decode_latency_breakdown = (
            self.llm_latency.latency(
                bs,
                seq_len,
                generate_len,
                kv_cache_bytes=kv_cache_bytes,
            )
        )

        infer_result_dict = {
            "weight_memory_per_gpu": memory_prefill_summary_dict["weight_memory_per_gpu"],
            "consume_memory_per_gpu": memory_decode_summary_dict["consume_memory_per_gpu"],
            "prefill_flops": prefill_num_flops_model,
            "decode_flops_per_step": decode_num_flops_model,
            "TTFT": prefill_latency_breakdown["TTFT"],
            "TTOT": decode_latency_breakdown["TTOT"],
            "kv_cache_latency": decode_latency_breakdown["kv_cache_latency"],
            "total_infer_latency": prefill_latency_breakdown["TTFT"] + decode_latency_breakdown["TTOT"] * generate_len,
            "max_batch_total_tokens": memory_decode_summary_dict["max_batch_total_tokens"],
        }

        # --------------------------- 5. Memory Access ----------------------
        if visual_flag:
            model_type = self.model_config.model_type
            llm_analyzer = LayerAnalyzer(self.model_config, self.gpu_config, tp_size=self.tp_size)
            results = llm_analyzer.analyze_model(bs=bs, seq_len=seq_len, generate_len=generate_len)

            # # -------------------------- 绘图：模型 graph 图示例 --------------------------
            llm_visualizer = LayerGraphVisualizer(self.model_config, results)
            base_path = f"_{self.model_config.model_name}_tp{self.tp_size}_bs{self.b}_seqlen{self.s}_genlen{self.o}.png"
            llm_visualizer.render(base_path)

            # -------------------------- 绘图：Pie 图示例 --------------------------
            prefill_latency_pie_save_path = f"./figures/latency_prefill" + base_path
            decode_latency_pie_save_path = f"./figures/latency_decode" + base_path
            prefill_flops_pie_save_path = f"./figures/flops_prefill" + base_path
            decode_flops_pie_save_path = f"./figures/flops_decode" + base_path
            params_pie_save_path = f"./figures/params" + base_path

            pie_tasks = [
                (dict_params_per_layer, "Params Distribution", params_pie_save_path),
                (prefill_dict_flops_per_layer, "Prefill FLOPS Distribution", prefill_flops_pie_save_path),
                (decode_dict_flops_per_layer, "Decode FLOPS Distribution", decode_flops_pie_save_path),
                (prefill_dict_latency_per_layer, "Prefill Latency Distribution", prefill_latency_pie_save_path),
                (decode_dict_latency_per_layer, "Decode Latency Distribution", decode_latency_pie_save_path),
            ]
            for data, title, path in pie_tasks:
                self.plot_distribution_pie(data, title, path)

        # ------------------------- 6. pretty‑print report --------------------
        if print_flag:
            self._print_report(
                infer_config_dict,
                copy.deepcopy(infer_result_dict),
                dict_params_per_layer,
                num_params_model,
                prefill_dict_flops_per_layer,
                prefill_num_flops_model,
                memory_prefill_summary_dict,
                memory_decode_summary_dict,
                prefill_latency_breakdown,
                decode_latency_breakdown,
            )

        return infer_result_dict

    def plot_distribution_pie(
        self,
        data: dict[str, float],
        title: str,
        save_path: str,
        *,
        explode_small_pct: float = 4.0,   # explode slices whose pct < this value
        label_pct_threshold: float = 0.5, # display "<x%" for very small slices
        label_display_threshold: float = 2.0,  # hide outer label below this pct
    ):
        """
        Pie chart styled similar to the user's sample:

        • Solid pie (no donut) with white borders between slices.
        • Slice label placed *outside*; percentage text inside.
        • Slices whose share < ``explode_small_pct`` are exploded.
        • Title large, bold, perfectly centred horizontally.
        """
        if not data:
            return

        labels = list(data.keys())
        sizes = list(data.values())
        total = float(sum(sizes)) or 1.0

        pct_list = [100 * s / total for s in sizes]
        labels_display = [
            lbl if pct >= label_display_threshold else "" for lbl, pct in zip(labels, pct_list)
        ]

        # colour palette
        cmap = plt.get_cmap("tab20" if len(labels) > 9 else "tab10")
        colors = [cmap(i % cmap.N) for i in range(len(labels))]

        # proportional explode: smaller share → larger offset (capped at 0.18)
        explode = [
            min(0.18, 0.04 + (explode_small_pct - pct) / explode_small_pct * 0.10)
            if (pct := 100 * s / total) < explode_small_pct
            else 0
            for s in sizes
        ]

        # formatting tiny percentage
        def _autopct(pct: float) -> str:
            return (
                f"<{label_pct_threshold:.1f}%" if pct < label_pct_threshold else f"{pct:.1f}%"
            )

        # high‑dpi for clarity
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels_display,
            labeldistance=1.18,
            autopct=_autopct,
            pctdistance=0.78,
            startangle=140,
            colors=colors,
            explode=explode,
            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
            textprops={"fontsize": 10, "color": "black"},
        )
        # inner % text style
        plt.setp(autotexts, size=9, weight="bold", color="white")

        # keep legend for color reference but remove title to save space
        ax.legend(
            wedges,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=min(len(labels), 5),
            fontsize=9,
            frameon=False,
        )

        ax.axis("equal")  # perfect circle

        # Title
        fig.suptitle(
            title,
            fontsize=18,
            weight="bold",
            y=0.98,
            color="#2c3e50",
        )

        # tidy layout – adjust bottom for legend
        fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.25)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.06, dpi=300)
        plt.close(fig)

    # ------------------------- Pretty‑print helpers -------------------- #
    def _print_section(self, title, summary_dict, category, extra_totals=None):
        """Print a single analysis section with optional totals."""
        print(f"\n---------------------------- {title} ----------------------------")
        Formatter.print_format_summary_dict(
            summary_dict=summary_dict,
            depth=get_dict_depth(summary_dict),
            category=category,
        )
        if extra_totals:
            pprint.pprint(extra_totals, indent=4, sort_dicts=False)

    def _print_report(
        self,
        infer_config_dict,
        infer_result_dict,
        dict_params_per_layer,
        num_params_model,
        prefill_dict_flops_per_layer,
        prefill_num_flops_model,
        memory_prefill_summary_dict,
        memory_decode_summary_dict,
        prefill_latency_breakdown,
        decode_latency_breakdown,
    ):
        """Pretty‑print a full performance report."""
        print("\n-------------------------- LLM main infer config --------------------------")
        pprint.pprint(infer_config_dict, indent=4, sort_dicts=False)

        print("\n-------------------------- LLM infer performance analysis --------------------------")
        Formatter.print_format_summary_dict(
            infer_result_dict, get_dict_depth(infer_result_dict)
        )

        sections = [
            (
                "LLM Params per_layer analysis",
                dict_params_per_layer,
                "params",
                {"params_model": num_to_string(num_params_model)},
            ),
            (
                "LLM Prefill Flops per_layer analysis",
                prefill_dict_flops_per_layer,
                "flops",
                {"prefill flops_model": num_to_string(prefill_num_flops_model)},
            ),
            (
                "LLM Memory analysis (Prefill)",
                memory_prefill_summary_dict,
                "memory",
                None,
            ),
            (
                "LLM Memory analysis (Decode)",
                memory_decode_summary_dict,
                "memory",
                None,
            ),
            (
                "LLM Latency analysis (Prefill)",
                prefill_latency_breakdown,
                "latency",
                None,
            ),
            (
                "LLM Latency analysis (Decode)",
                decode_latency_breakdown,
                "latency",
                None,
            ),
        ]

        for title, summary_dict, category, extra in sections:
            self._print_section(title, summary_dict, category, extra)

def llm_profile(
    model_name,
    gpu_name: str = "a100-sxm-40gb",
    bytes_per_param: int = BYTES_FP16,
    batch_size: int = 20,
    seq_len: int = 1024,
    generate_len=1024,
    dp_size: int = 1,
    tp_size: int = 8,
    pp_size: int = 1,
    sp_size: int = 1,
    act_dtype_bytes: int = BYTES_FP16,
    kv_cache_bytes: int = BYTES_FP16,
    flops_efficiency: float = FLOPS_EFFICIENCY,
    hbm_memory_efficiency: float = HBM_MEMORY_EFFICIENCY,
    intra_node_memory_efficiency=INTRA_NODE_MEMORY_EFFICIENCY,
    inter_node_memory_efficiency=INTER_NODE_MEMORY_EFFICIENCY,
    print_flag: bool = False,
    visual_flag: bool = False,
) -> dict:
    """Returns dict of the total floating-point operations, MACs, parameters and latency of a llm.
    It now returns a dictionary containing FLOPs, latency, HBM memory usage and max_batch_total_tokens.

    Args:
        model_name (str, optional): model name to query the pre-defined `model_configs.json`. 
            Defaults to "llama-13b".
        gpu_name (str, optional): gpu name to query the pre-defined `model_configs.json`. 
            Defaults to "v100-sxm2-32gb".
        batch_size (int, optional): _description_. Defaults to 1.
        seq_len (int, optional): batch size per GPU.. Defaults to 522.
        generate_len (int, optional): The maximum numbers of tokens to generate, 
            ignoring the number of tokens in the prompt. Defaults to 1526.
        dp_size (int, optional): data parallelism size. Defaults to 1.
        tp_size (int, optional): tensor parallelism size. Defaults to 1.
        pp_size (int, optional): pipeline parallelism size. Defaults to 1.
        sp_size (int, optional): sequence parallelism size. Defaults to 1.
            past last key/values attentions (if applicable to the model) to speed up decoding. Defaults to True.
        layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations..
            Defaults to BYTES_FP16.
        kv_cache_bytes (int, optional): number of bytes in the data type for the kv_cache. Defaults to None.
        flops_efficiency (float, optional): flops efficiency, ranging from 0 to 1. Defaults to None.
        hbm_memory_efficiency (float, optional): GPU HBM memory efficiency, ranging from 0 to 1. 
            Defaults to HBM_MEMORY_EFFICIENCY.
        intra_node_memory_efficiency (_type_, optional): intra-node memory efficiency, ranging from 0 to 1.. 
            Defaults to INTRA_NODE_MEMORY_EFFICIENCY.
        inter_node_memory_efficiency (_type_, optional): inter-node memory efficiency, ranging from 0 to 1.. 
            Defaults to INTER_NODE_MEMORY_EFFICIENCY.

    Returns:
        dict: a summary dictionary of the inference analysis
    """
    model_config, gpu_config = get_model_and_gpu_config_by_name(model_name, gpu_name)

    parallelism_config = ParallelismConfig(
        tp_size=tp_size, pp_size=pp_size, dp_size=dp_size, sp_size=sp_size
    )

    inference_config = InferenceConfig(
        bs=batch_size,
        seq_len=seq_len,
        generate_len=generate_len,
        bytes_per_param=bytes_per_param,
        act_dtype_bytes=act_dtype_bytes,
        kv_cache_bytes=kv_cache_bytes,
    )

    gpu_efficiency_config = GPUEfficiencyConfig(
        flops_efficiency=flops_efficiency,
        hbm_memory_efficiency=hbm_memory_efficiency,
        intra_node_memory_efficiency=intra_node_memory_efficiency,
        inter_node_memory_efficiency=inter_node_memory_efficiency,
    )

    llm_configs = LLMConfigs(
        model_config=model_config,
        gpu_config=gpu_config,
        parallelism_config=parallelism_config,
        inference_config=inference_config,
        gpu_efficiency_config=gpu_efficiency_config,
    )

    profiler = LayerAnalyzerVisual(llm_configs)

    infer_result_dict = profiler.infer_profile(
        bs=batch_size,
        seq_len=seq_len,
        generate_len=generate_len,
        act_dtype_bytes=act_dtype_bytes,
        flops_efficiency=flops_efficiency,
        hbm_memory_efficiency=hbm_memory_efficiency,
        print_flag=print_flag,
        visual_flag=visual_flag,
    )

    # ---------------------------------------------------------------------
    # Collect summary metrics (keep raw numbers for downstream maths)      #
    # ---------------------------------------------------------------------
    weight_memory_per_gpu = infer_result_dict.get("weight_memory_per_gpu", None)
    consume_memory_per_gpu = infer_result_dict.get("consume_memory_per_gpu", None)
    prefill_flops = infer_result_dict.get("prefill_flops", None)
    
    table_results = {
        "seq_len": seq_len,
        "generate_len": generate_len,
        "prefill_flops": num_to_string(prefill_flops),           
        "weight_memory_per_gpu": num_to_string(weight_memory_per_gpu),
        "consume_memory_per_gpu": num_to_string(consume_memory_per_gpu),       
        "TTFT": infer_result_dict.get("TTFT", None),
        "TTOT": infer_result_dict.get("TTOT", None),
        "Total_latency": infer_result_dict.get("total_infer_latency", None),
        "max_batch_total_tokens": infer_result_dict.get("max_batch_total_tokens", None),
    }
    visual_results = {
        "seq_len": seq_len,
        "generate_len": generate_len,
        "prefill_flops": prefill_flops,                   # raw number
        "weight_memory_per_gpu": weight_memory_per_gpu,
        "consume_memory_per_gpu": consume_memory_per_gpu, # raw bytes
        "TTFT": infer_result_dict.get("TTFT", None),
        "TTOT": infer_result_dict.get("TTOT", None),
        "Total_latency": infer_result_dict.get("total_infer_latency", None),
    }
    return table_results, visual_results


# ----------------------------- Command‑line interface ----------------------------- #
def _cli():
    """Command‑line wrapper for quick profiling."""
    parser = argparse.ArgumentParser(description="LLMCounts – quick model inference profiler")
    parser.add_argument("--model_name", required=True, help="Model name defined in model_configs.json")
    parser.add_argument("--gpu_name", default="a100-sxm-40gb", help="GPU name defined in model_configs.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--generate_len", type=int, default=1024)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--visual", action="store_true", help="Generate pie‑charts and layer graph")
    parser.add_argument("--print", dest="print_flag", action="store_true", help="Pretty‑print verbose breakdown")
    parser.add_argument("--json", dest="json_flag", action="store_true", help="Output raw results as JSON")
    args = parser.parse_args()

    table_results, visual_results = llm_profile(
        model_name=args.model_name,
        gpu_name=args.gpu_name,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        generate_len=args.generate_len,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        dp_size=args.dp_size,
        sp_size=args.sp_size,
        print_flag=args.print_flag,
        visual_flag=args.visual,
    )

    if args.json_flag:
        print(json.dumps(visual_results, indent=2))
    else:
        import pprint
        pprint.pprint(table_results, indent=2)


if __name__ == "__main__":
    _cli()