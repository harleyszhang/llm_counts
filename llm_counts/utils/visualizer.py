import pandas as pd

# ====================================================================== #
# Refactored visualisation as a lightweight class                        #
# ====================================================================== #
class SeqLenVisualizer:
    """Encapsulates all plots & tables for a sequence‑length sweep."""

    def __init__(
        self,
        df: pd.DataFrame,
        model: str,
        gpu: str,
        *,
        out_dir: str = "figures",
        flops_unit: str = "TFLOPs",
        mem_unit: str = "GiB",
        dpi: int = 300,
        show: bool = False,
    ):
        from pathlib import Path
        import matplotlib.pyplot as plt

        self.df = df.sort_values("seq_len")
        self.model = model
        self.gpu = gpu
        self.out_dir = Path(out_dir)
        self.flops_unit = flops_unit
        self.mem_unit = mem_unit
        self.dpi = dpi
        self.show = show

        self.out_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-paper")
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.grid": True,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "axes.edgecolor": "#cccccc",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": 12,
                "axes.titleweight": "bold",
            }
        )

        # Pre‑compute unit scaling divisors
        _scale = {
            "GFLOPs": 1e9,
            "TFLOPs": 1e12,
            "PFLOPs": 1e15,
            "MiB": 2**20,
            "GiB": 2**30,
        }
        self.flops_div = _scale.get(self.flops_unit, 1.0)
        self.mem_div = _scale.get(self.mem_unit, 1.0)

    # -------------------------- helpers ------------------------------ #
    def _save(self, fig, suffix: str):
        fig.savefig(
            self.out_dir
            / f"{self.model}_{self.gpu}_{suffix}.png",
            dpi=self.dpi,
            bbox_inches="tight",
        )
        if self.show:
            import webbrowser, os, matplotlib.pyplot as _plt
            _plt.show()
        import matplotlib.pyplot as plt
        plt.close(fig)

    @staticmethod
    def _line_scatter(ax, x, y, y_label, title, cmap="viridis_r"):
        sc = ax.scatter(
            x,
            y,
            c=y,
            cmap=cmap,
            s=70,
            edgecolor="black",
            linewidths=0.4,
        )
        ax.plot(x, y, linewidth=1.2, alpha=0.75)
        ax.set_xlabel("Sequence length (tokens)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)
        return sc

    # ----------------------- public interface ------------------------ #
    def visualize(self):
        self._metric_figs()
        self._latency_fig()
        self._composite_fig()
        self._interactive_html()
        self._print_table()

    # ------------------ individual plot generators ------------------ #
    def _metric_figs(self):
        import matplotlib.pyplot as plt

        metrics = [
            ("flops", "prefill_flops", self.flops_div, f"Prefill {self.flops_unit}"),
            ("memory", "consume_memory_per_gpu", self.mem_div, f"HBM ({self.mem_unit})"),
        ]
        if "throughput_tok_per_second" in self.df.columns:
            metrics.append(("throughput", "throughput_tok_per_second", 1.0, "Throughput (tok/s)"))

        for suffix, col, div, label in metrics:
            if col not in self.df:
                continue
            y = (self.df[col] / div) if div != 1.0 else self.df[col]
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            sc = self._line_scatter(
                ax,
                self.df["seq_len"],
                y,
                label,
                f"{self.model} on {self.gpu}\n{label} vs seq_len",
            )
            plt.colorbar(sc, ax=ax, label=label)
            self._save(fig, f"{suffix}_vs_seq_len")

    def _latency_fig(self):
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(7, 4), constrained_layout=True)
        if "TTFT" in self.df.columns:
            ax1.plot(
                self.df["seq_len"],
                self.df["TTFT"],
                "s-.",
                linewidth=1.5,
                label="TTFT (s)",
            )
            ax1.set_ylabel("TTFT (s)")
        ax2 = ax1.twinx()
        if "TTOT" in self.df.columns:
            ax2.plot(
                self.df["seq_len"],
                self.df["TTOT"] * 1000.0,
                "^:",
                linewidth=1.5,
                color="tab:red",
                label="TTOT (ms)",
            )
            ax2.set_ylabel("TTOT (ms)")
        ax1.set_xlabel("Sequence length (tokens)")
        ax1.set_title(f"{self.model} on {self.gpu}\nTTFT & TTOT vs seq_len")
        handles, labels = [], []
        for ax in (ax1, ax2):
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        ax1.legend(handles, labels, loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.3)
        self._save(fig, "latency_vs_seq_len")

    def _composite_fig(self):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        mem_norm = self.df["consume_memory_per_gpu"] / self.mem_div
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        axes = {
            "FLOPs": fig.add_subplot(gs[0, 0]),
            "Latency": fig.add_subplot(gs[0, 1]),
            "Memory": fig.add_subplot(gs[1, 0]),
            "Throughput": fig.add_subplot(gs[1, 1]),
        }

        self._line_scatter(
            axes["FLOPs"],
            self.df["seq_len"],
            self.df["prefill_flops"] / self.flops_div,
            f"Prefill {self.flops_unit}",
            "FLOPs",
        )

        if "TTFT" in self.df.columns:
            axes["Latency"].plot(self.df["seq_len"], self.df["TTFT"], "s-.", label="TTFT (s)")
        if "TTOT" in self.df.columns:
            axes["Latency"].plot(self.df["seq_len"], self.df["TTOT"] * 1000.0, "^:", label="TTOT (ms)")
        axes["Latency"].set_title("Latency"); axes["Latency"].legend()

        self._line_scatter(
            axes["Memory"],
            self.df["seq_len"],
            mem_norm,
            f"HBM ({self.mem_unit})",
            "Memory",
        )

        if "throughput_tok_per_second" in self.df.columns:
            self._line_scatter(
                axes["Throughput"],
                self.df["seq_len"],
                self.df["throughput_tok_per_second"],
                "Throughput (tok/s)",
                "Throughput",
            )

        fig.suptitle(f"{self.model} on {self.gpu}\nOverview", fontsize=14)
        self._save(fig, "overview")

    def _interactive_html(self):
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot as psave
            from pathlib import Path
            figs = []

            # basic metrics
            meta = [
                ("TTFT (s)", "TTFT", 1.0),
                ("TTOT (ms)", "TTOT", 1000.0),
                (f"Prefill {self.flops_unit}", "prefill_flops", self.flops_div),
                (f"HBM ({self.mem_unit})", "consume_memory_per_gpu", self.mem_div),
                ("Throughput (tok/s)", "throughput_tok_per_second", 1.0),
            ]
            for name, col, div in meta:
                if col not in self.df:
                    continue
                y = self.df[col] * div if div != 1.0 else self.df[col]
                f = go.Figure(go.Scatter(x=self.df["seq_len"], y=y, mode="lines+markers"))
                f.update_layout(title=f"{name} vs seq_len", template="seaborn")
                figs.append((name, f))

            html_path = Path(self.out_dir) / f"{self.model}_{self.gpu}_interactive.html"
            with open(html_path, "w") as fhtml:
                for i, (title, fig) in enumerate(figs):
                    fhtml.write(f"<h2>{title}</h2>")
                    fhtml.write(psave(fig, include_plotlyjs="cdn" if i == 0 else False, output_type="div"))
            if self.show:
                import webbrowser, os
                webbrowser.open("file://" + os.path.abspath(html_path))
        except ImportError:
            print("[INFO] plotly missing – skipped interactive output.")

    def _print_table(self):
        summary = self.df.copy()
        summary["prefill_flops"] = (summary["prefill_flops"] / self.flops_div).map("{:,.2f}".format)
        summary["consume_memory_per_gpu"] = (summary["consume_memory_per_gpu"] / self.mem_div).map("{:,.2f}".format)
        if "throughput_tok_per_second" in summary.columns:
            summary["throughput_tok_per_second"] = summary["throughput_tok_per_second"].map("{:,.2f}".format)
        print("=" * 80)
        print(summary.to_string(index=False))
        print("=" * 80)