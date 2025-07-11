# usage: python cli_benchmark.py --model_name Qwen3-30B-A3B --gpu_name a100-sxm-80gb --batch_size 16 --generate_len 1024 --tp_size 4
import pandas as pd
import argparse
from llm_counts.utils.config import *
from llm_counts.benchmark_analyzer import llm_profile
from llm_counts.utils.visualizer import SeqLenVisualizer


def sweep_seq_len(model_name, gpu_name="h100-sxm-80gb", batch_size=16, generate_len=1024, tp_size=2, seq_len_list=None, **kwargs):
    """Profile a model over several sequence lengths and print / plot a table.

    Args:
        model_name (str): name of the LLM
        gpu_name (str): target GPU
        seq_len_list (List[int]): list of sequence lengths to test
        **kwargs: forwarded to llm_profile
    Returns:
        pandas.DataFrame: one row per sequence length with metrics
    """
    if seq_len_list is None:
        seq_len_list = [128, 256, 512, 1024, 1334, 1567, 1678, 2567, 3072, 
                        4096, 5120, 6144, 8192, 10240, 12288, 16384,
                        21472, 24576, 30346, 32768, 33792, 34980, 36790]

    records1 = []
    records2 = []
    for seq in seq_len_list:
        res1, res2 = llm_profile(
            model_name=model_name,
            gpu_name=gpu_name,
            batch_size=batch_size,
            seq_len=seq,
            generate_len=generate_len,
            tp_size=tp_size,
            print_flag=False,
            visual_flag=False,
        )
        print("=" * 80)
        print(f"model_name: {model_name}, gpu_name: {gpu_name}, tp_size: {tp_size}, "
              f"batch_size: {batch_size}, seq_len: {seq}, generate_len: {generate_len}")

        records1.append(res1)
        records2.append(res2)

    df1 = pd.DataFrame(records1)
    print("=" * 80)
    print(df1.to_string(index=False))
    print("=" * 80)

    df2 = pd.DataFrame(records2)
    # Derive throughput in tokens / second for visualisation
    if "TTFT" in df2.columns:
        df2["throughput_tok_per_second"] = df2["seq_len"] * batch_size / df2["TTFT"].replace(0, float("nan"))
    # Visualise the results using *plot_seq_len_sweep*
    if kwargs.get("visual_flag", True):
        viz = SeqLenVisualizer(df2, model_name, gpu_name, show=True)
        viz.visualize()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep sequence lengths, profile an LLM, and generate visualisations."
    )
    parser.add_argument("--model_name", required=True, help="LLM model name, e.g. Qwen3-32B")
    parser.add_argument("--gpu_name", default="h100-sxm-80gb", help="Target GPU name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--generate_len", type=int, default=1024, help="Generation length")
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor‑parallel size")
    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="*",
        default=None,
        help="Space‑separated list of sequence lengths (tokens) to sweep",
    )
    parser.add_argument(
        "--no_visual",
        action="store_true",
        help="Disable visualisation (figures will not be generated)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sweep_seq_len(
        model_name=args.model_name,
        gpu_name=args.gpu_name,
        batch_size=args.batch_size,
        generate_len=args.generate_len,
        tp_size=args.tp_size,
        seq_len_list=args.seq_lens,
        visual_flag=not args.no_visual,
    )

"""
python cli_benchmark.py \
    --model_name Qwen3-30B-A3B \
    --gpu_name a100-sxm-80gb \
    --batch_size 16 \
    --generate_len 1024 \
    --tp_size 4
"""