from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1f2933"
MUTED = "#9aa5b1"
ACCENT = "#d9480f"
BASELINE = "#486581"
PALETTE = {"name_match": "#bcccdc", "bm25": "#9fb3c8", "dense": "#829ab1", "hybrid": BASELINE,
           "lsp (inherited)": "#f7a072", "lsp-ranked (default)": "#f0803c", "lsp-ranked (tuned)": ACCENT,
           "lsp-novelty (tuned)": "#8d2b0b"}


def style(ax, title: str, ylabel: str):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color="#e4e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def method_bars(run: dict, level: str, path: Path):
    methods = list(run["methods"])
    multi_file = run["manifest"]["selection"].get("min_files", 1) > 1
    # With several gold files, Acc@1 is zero by construction; partial coverage is what differs.
    metrics = ["recall@5", "recall@10", "acc@10", "mrr"] if multi_file else ["acc@1", "acc@5", "acc@10", "mrr"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        values = [run["methods"][method][level].get(m, 0) for m in metrics]
        positions = [x + i * width for x in range(len(metrics))]
        ax.bar(positions, values, width, label=method, color=PALETTE.get(method, MUTED))
    labels = [m.replace("acc", "Acc").replace("recall", "Recall").replace("mrr", "MRR") for m in metrics]
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(metrics))], labels)
    ax.set_ylim(0, 1.05)
    n = next(iter(run["methods"].values()))[level].get("n", 0)
    style(ax, f"{level.capitalize()}-level localization ({run['run']}, n={n})", "score")
    ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def ablation_tradeoff(run: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    points: dict[tuple[float, float], list[str]] = {}
    for name, values in run["ablations"].items():
        if name.startswith("depth") or name.startswith("seeds"):
            key = (round(values["latency"].get("expansion_mean_s", 0), 6), round(values["file"]["mrr"], 6))
            points.setdefault(key, []).append(name)
    for (x, y), names in points.items():
        ax.scatter(x, y, s=60, color=ACCENT if any(n.startswith("depth") for n in names) else BASELINE, zorder=3)
        ax.annotate(" / ".join(names), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
    hybrid = run["methods"]["hybrid"]["file"]["mrr"]
    ax.axhline(hybrid, color=MUTED, linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[1], hybrid, " hybrid", va="center", fontsize=8, color=MUTED)
    style(ax, f"Expansion cost vs file MRR ({run['run']})", "file-level MRR")
    ax.set_xlabel("mean LSP expansion time per query (s)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def edge_ablation(run: dict, path: Path):
    names = [n for n in run["ablations"] if n.startswith("only") or n.startswith("without")]
    values = [run["ablations"][n]["file"]["mrr"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [ACCENT if n.startswith("only") else BASELINE for n in names]
    ax.barh(names, values, color=colors)
    hybrid = run["methods"]["hybrid"]["file"]["mrr"]
    ax.axvline(hybrid, color=MUTED, linestyle="--", linewidth=1, label=f"hybrid ({hybrid:.3f})")
    ax.set_xlim(min(values + [hybrid]) - 0.03, max(values + [hybrid]) + 0.015)
    for i, value in enumerate(values):
        ax.text(value + 0.0005, i, f"{value:.3f}", va="center", fontsize=8, color=INK)
    style(ax, f"Edge-type ablation, file MRR ({run['run']})", "")
    ax.grid(axis="x", color="#e4e7eb")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def headroom_chart(run: dict, path: Path):
    configs = list(run["headroom@10"])
    misses = [run["headroom@10"][c]["hybrid_misses"] for c in configs]
    symbols = [run["headroom@10"][c]["gold_symbol_discovered"] for c in configs]
    files = [run["headroom@10"][c]["gold_file_discovered"] for c in configs]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(configs))
    ax.bar([i - 0.27 for i in x], misses, 0.27, color=MUTED, label="hybrid misses gold file in top-10")
    ax.bar(x, files, 0.27, color=BASELINE, label="gold file reached by expansion")
    ax.bar([i + 0.27 for i in x], symbols, 0.27, color=ACCENT, label="gold symbol reached by expansion")
    ax.set_xticks(list(x), configs, fontsize=8)
    style(ax, f"Headroom for expansion ({run['run']})", "queries")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def latency_chart(run: dict, path: Path):
    methods = [m for m, v in run["methods"].items() if v["latency"]]
    retrieval = [run["methods"][m]["latency"]["retrieval_mean_s"] for m in methods]
    expansion = [run["methods"][m]["latency"]["expansion_mean_s"] for m in methods]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(methods, retrieval, color=BASELINE, label="seed retrieval (BM25 + dense)")
    ax.barh(methods, expansion, left=retrieval, color=ACCENT, label="LSP expansion")
    for i, (r, e) in enumerate(zip(retrieval, expansion)):
        ax.text(r + e + 0.05, i, f"{r + e:.2f}s", va="center", fontsize=8, color=INK)
    style(ax, "Mean per-query latency (fresh LSP session per checkout)", "")
    ax.set_xlabel("seconds", color=INK)
    ax.grid(axis="x", color="#e4e7eb")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_figures(report: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, run in report["runs"].items():
        run = {**run, "run": name}
        method_bars(run, "file", out_dir / f"{name}_file_methods.png")
        method_bars(run, "function", out_dir / f"{name}_function_methods.png")
        ablation_tradeoff(run, out_dir / f"{name}_tradeoff.png")
        edge_ablation(run, out_dir / f"{name}_edges.png")
        headroom_chart(run, out_dir / f"{name}_headroom.png")
        latency_chart(run, out_dir / f"{name}_latency.png")
