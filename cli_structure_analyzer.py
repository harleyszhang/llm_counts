from llm_counts.layer_graph_visualizer import LayerAnalyzer, LayerGraphVisualizer
from llm_counts.utils.utils import *
from llm_counts.utils.config import get_model_and_gpu_config_by_name
import argparse

def test_llm_analyzer(
        model_name: str = "Qwen3-30B-A3B",
        gpu_name="a100-sxm-80gb",
        bs: int = 4,
        seq_len: int = 600,
        generate_len: int = 128,
        tp_size: int = 1,
        tade: bool = True
    ):
    # Load configurations
    model_config, gpu_config = get_model_and_gpu_config_by_name(model_name, gpu_name)

    # Select appropriate analyzer and visualizer
    Analyzer = LayerAnalyzer
    Visualizer = LayerGraphVisualizer

    # Analyze model
    analyzer = Analyzer(model_config, gpu_config, tp_size=tp_size)
    results = analyzer.analyze_model(bs=bs, seq_len=seq_len, generate_len=generate_len)

    # Generate graph
    base_fname = f"{model_name.replace('/', '_')}_tp{tp_size}_bs{bs}_seqlen{seq_len}_genlen{generate_len}"
    print(f"Rendering graph to: {base_fname}")
    Visualizer(model_config, results).render(base_fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TadeLayerAnalyzer, print a formatted summary, "
                    "and generate per‑stage layer‑graph PNGs."
    )
    parser.add_argument("--model-name", default="Qwen3-32B")
    parser.add_argument("--gpu-name", default="a100-sxm-80gb")
    parser.add_argument("--bs",          type=int, default=4)
    parser.add_argument("--seq-len",     type=int, default=600)
    parser.add_argument("--generate-len",type=int, default=128)
    parser.add_argument("--tp-size",     type=int, default=1)
    parser.add_argument(
        "--no-tade",
        dest="tade",
        action="store_false",
        help="Use standard analyzer and visualizer instead of Tade."
    )
    args = parser.parse_args()

    test_llm_analyzer(
        model_name=args.model_name,
        gpu_name=args.gpu_name,
        bs=args.bs,
        seq_len=args.seq_len,
        generate_len=args.generate_len,
        tp_size=args.tp_size,
        tade=args.tade
    )

""""
python cli_structure_analyzer.py \
  --model-name Qwen3-30B-A3B \
  --gpu-name  a100-sxm-80gb \
  --bs 16 \
  --seq-len 600 \
  --generate-len 128 \
  --tp-size 1 \
  --no-tade
"""