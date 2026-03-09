"""Plot results from eval_e2e.py.

Reads results.json and produces:
1. A 1x3 figure: one subplot per eval mode (cascading / parallel / single-MLP),
   showing CE degradation vs top_k for each model type.
2. A grouped bar chart comparing all models at each k.

Usage:
    python experiments/exp_019_eval_e2e/plot_e2e.py
    python experiments/exp_019_eval_e2e/plot_e2e.py --results path/to/results.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_DIR = Path("experiments/exp_019_eval_e2e/output")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_STYLES = {
    "tc_cascading":    dict(marker="o", color="#1f77b4", linewidth=1.8, markersize=7),
    "tc_parallel":     dict(marker="s", color="#ff7f0e", linewidth=1.8, markersize=7),
    "tc_independent":  dict(marker="^", color="#2ca02c", linewidth=1.8, markersize=7),
    "clt_cascading":   dict(marker="D", color="#9467bd", linewidth=1.8, markersize=7),
    "clt_parallel":    dict(marker="X", color="#17becf", linewidth=1.8, markersize=8),
}

EVAL_MODES = [
    ("ce_cascading", "All-replace cascading"),
    ("ce_parallel", "All-replace parallel"),
    ("ce_single", "Single-MLP (avg)"),
]

LABEL_MAP = {
    "tc_cascading": "TC cascading",
    "tc_parallel": "TC parallel",
    "tc_independent": "TC independent",
    "clt_cascading": "CLT cascading",
    "clt_parallel": "CLT parallel",
}


SPD_STYLES = {
    "SPD (CI>0.5)": dict(color="#d62728", marker="P", markersize=11),
    "SPD (CI>0.0)": dict(color="#d62728", marker="D", markersize=9),
    "Jose (CI>0.5)": dict(color="#8c564b", marker="P", markersize=11),
    "Jose (CI>0.0)": dict(color="#8c564b", marker="D", markersize=9),
}


def load_results(path: Path) -> tuple[float, list[dict], list[dict], list[dict], float | None]:
    with open(path) as f:
        data = json.load(f)
    return (
        data["baseline_ce"],
        data["results"],
        data.get("spd_results", []),
        data.get("jose_results", []),
        data.get("jose_baseline_ce"),
    )


def group_by_model(results: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for r in results:
        key = f"{r['type']}_{r['mode']}"
        groups.setdefault(key, []).append(r)
    for v in groups.values():
        v.sort(key=lambda x: x["top_k"])
    return groups


def plot_three_panel(baseline_ce: float, results: list[dict], save_path: Path,
                     spd_results: list[dict] | None = None,
                     jose_results: list[dict] | None = None,
                     jose_baseline_ce: float | None = None):
    """1x3 subplot: one per eval mode, x = L0 per module, y = CE degradation."""
    groups = group_by_model(results)

    spd_ce_keys = {
        "ce_cascading": "ce_all",
        "ce_parallel": "ce_all",
        "ce_single": "ce_single",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    panel_labels = ["(a)", "(b)", "(c)"]

    for ax, (ce_key, title), label in zip(axes, EVAL_MODES, panel_labels):
        for model_name in MODEL_STYLES:
            if model_name not in groups:
                continue
            pts = groups[model_name]
            l0s = [p["l0"] for p in pts]
            deltas = [p[ce_key] - baseline_ce for p in pts]
            style = MODEL_STYLES[model_name]
            ax.plot(l0s, deltas, label=LABEL_MAP[model_name],
                    markeredgecolor="white", markeredgewidth=0.8, **style)

        # SPD points
        def _plot_spd_points(spd_list, base_ce, name_prefix):
            if not spd_list:
                return
            spd_ce_key = spd_ce_keys[ce_key]
            for spd_r in spd_list:
                if "l0" not in spd_r:
                    continue
                spd_label = f"{name_prefix} ({spd_r['mode']})"
                delta = spd_r[spd_ce_key] - base_ce
                style = SPD_STYLES.get(spd_label, dict(color="#d62728", marker="P", markersize=11))
                ax.plot(spd_r["l0"], delta, label=spd_label,
                        marker=style.get("marker", "P"), color=style["color"],
                        markersize=style.get("markersize", 11),
                        markeredgecolor="white", markeredgewidth=0.8,
                        linestyle="none", zorder=6)

        _plot_spd_points(spd_results, baseline_ce, "SPD")
        if jose_results and jose_baseline_ce is not None:
            _plot_spd_points(jose_results, jose_baseline_ce, "Jose")

        ax.set_xlabel("L0 (active features per module)")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x)}" if x == int(x) else f"{x:g}"
        ))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.tick_params(direction="in", which="both")
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

    axes[0].set_ylabel("CE degradation (Δ from baseline)")
    axes[0].set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate legend entries (SPD appears in each panel)
    seen = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)

    fig.legend(unique_handles, unique_labels, loc="upper center",
               ncol=min(len(unique_labels), 7),
               frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95,
               bbox_to_anchor=(0.5, 1.03), fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_matched_vs_mismatched(baseline_ce: float, results: list[dict], save_path: Path):
    """Bar chart: for each model, show CE delta in matched vs mismatched eval mode."""
    groups = group_by_model(results)

    # For each model, its "matched" eval mode
    matched_mode = {
        "tc_cascading": "ce_cascading",
        "tc_parallel": "ce_parallel",
        "tc_independent": "ce_single",
        "clt_cascading": "ce_cascading",
        "clt_parallel": "ce_parallel",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    ks = [16, 32, 64]
    bar_colors = {
        "ce_cascading": "#1f77b4",
        "ce_parallel": "#ff7f0e",
        "ce_single": "#2ca02c",
    }
    eval_labels = {
        "ce_cascading": "Cascading",
        "ce_parallel": "Parallel",
        "ce_single": "Single-MLP",
    }

    for ax_idx, k in enumerate(ks):
        ax = axes[ax_idx]
        model_names = [m for m in MODEL_STYLES if m in groups]
        x = np.arange(len(model_names))
        width = 0.25

        for i, (ce_key, color) in enumerate([
            ("ce_cascading", bar_colors["ce_cascading"]),
            ("ce_parallel", bar_colors["ce_parallel"]),
            ("ce_single", bar_colors["ce_single"]),
        ]):
            vals = []
            for m in model_names:
                pt = [p for p in groups[m] if p["top_k"] == k]
                assert len(pt) == 1
                vals.append(pt[0][ce_key] - baseline_ce)
            bars = ax.bar(x + i * width, vals, width, color=color, label=eval_labels[ce_key],
                          edgecolor="white", linewidth=0.5)

            # Highlight matched bars
            for j, m in enumerate(model_names):
                if matched_mode[m] == ce_key:
                    bars[j].set_edgecolor("black")
                    bars[j].set_linewidth(1.5)

        ax.set_xticks(x + width)
        ax.set_xticklabels([LABEL_MAP[m] for m in model_names], rotation=30, ha="right", fontsize=9)
        ax.set_title(f"k = {k}", fontsize=11, pad=8)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
        ax.tick_params(direction="in", which="both")

    axes[0].set_ylabel("CE degradation (Δ from baseline)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95,
               bbox_to_anchor=(0.5, 1.03), fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_heatmap(baseline_ce: float, results: list[dict], save_path: Path):
    """Heatmap: rows = models (type_mode @ k), columns = eval modes."""
    groups = group_by_model(results)
    ks = [16, 32, 64]

    row_labels = []
    data = []
    for model_name in MODEL_STYLES:
        if model_name not in groups:
            continue
        for k in ks:
            pts = [p for p in groups[model_name] if p["top_k"] == k]
            if not pts:
                continue
            p = pts[0]
            row_labels.append(f"{LABEL_MAP[model_name]} k={k}")
            data.append([p[ce_key] - baseline_ce for ce_key, _ in EVAL_MODES])

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(7, 8))

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd",
                   norm=plt.matplotlib.colors.LogNorm(vmin=data.min(), vmax=data.max()))

    ax.set_xticks(range(len(EVAL_MODES)))
    ax.set_xticklabels([title for _, title in EVAL_MODES], fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(EVAL_MODES)):
            val = data[i, j]
            text_color = "white" if val > np.median(data) else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    ax.set_title("CE degradation by training mode × eval mode", fontsize=12, pad=10)
    fig.colorbar(im, ax=ax, label="CE degradation (Δ)", shrink=0.8)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
                        default="experiments/exp_019_eval_e2e/output/results.json")
    args = parser.parse_args()

    baseline_ce, results, spd_results, jose_results, jose_baseline_ce = load_results(Path(args.results))
    print(f"Baseline CE: {baseline_ce:.4f}, {len(results)} results, {len(spd_results)} SPD, {len(jose_results)} Jose")
    if jose_baseline_ce is not None:
        print(f"Jose baseline CE: {jose_baseline_ce:.4f}")

    plot_three_panel(baseline_ce, results,
                     OUTPUT_DIR / "eval_e2e_three_panel.png",
                     spd_results=spd_results,
                     jose_results=jose_results,
                     jose_baseline_ce=jose_baseline_ce)
    plot_matched_vs_mismatched(baseline_ce, results,
                               OUTPUT_DIR / "eval_e2e_bars.png")
    plot_heatmap(baseline_ce, results,
                 OUTPUT_DIR / "eval_e2e_heatmap.png")


if __name__ == "__main__":
    main()
