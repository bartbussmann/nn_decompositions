"""Plot CE degradation for every layer combination of SPD.

Produces a 1x4 figure: panels for 1-layer, 2-layer, 3-layer, and 4-layer combos.
Each bar shows the CE degradation for that specific combination of replaced layers.

Usage:
    python experiments/exp_022_layer_combos/plot_layer_combos.py
"""

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_DIR = Path("experiments/exp_022_layer_combos/output")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LAYERS = [0, 1, 2, 3]

# Colors: deeper purple for CI>0.0 (more components), lighter for CI>0.5
COLOR_CI05 = "#9f7aea"   # lighter purple
COLOR_CI00 = "#6b21a8"   # deeper purple


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_spd_combos(data: dict, save_path: Path):
    baseline_ce = data["baseline_ce"]
    spd_05 = data.get("spd_ci0.5", {})
    spd_00 = data.get("spd_ci0.0", {})

    # Group combos by size
    combo_groups = {}
    for r in range(1, len(LAYERS) + 1):
        combos = [list(c) for c in combinations(LAYERS, r)]
        combo_groups[r] = combos

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5),
                              gridspec_kw={"width_ratios": [4, 6, 4, 1]})
    panel_labels = ["(a) Single layer", "(b) Two layers", "(c) Three layers", "(d) All layers"]

    for ax_idx, (n_layers, combos) in enumerate(combo_groups.items()):
        ax = axes[ax_idx]
        labels = [",".join(str(l) for l in c) for c in combos]
        combo_keys = [str(c) for c in combos]

        x = np.arange(len(combos))
        width = 0.35

        # CI>0.5 bars
        deltas_05 = [spd_05.get(k, {}).get("delta", 0) for k in combo_keys]
        # CI>0.0 bars
        deltas_00 = [spd_00.get(k, {}).get("delta", 0) for k in combo_keys]

        if len(combos) > 1:
            bars1 = ax.bar(x - width / 2, deltas_05, width, color=COLOR_CI05,
                           edgecolor="white", linewidth=0.8, label="CI > 0.5", zorder=3)
            bars2 = ax.bar(x + width / 2, deltas_00, width, color=COLOR_CI00,
                           edgecolor="white", linewidth=0.8, label="CI > 0.0", zorder=3)
        else:
            bars1 = ax.bar(x - width / 2, deltas_05, width, color=COLOR_CI05,
                           edgecolor="white", linewidth=0.8, zorder=3)
            bars2 = ax.bar(x + width / 2, deltas_00, width, color=COLOR_CI00,
                           edgecolor="white", linewidth=0.8, zorder=3)

        # Value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=8,
                        color="#444444")

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in labels], fontsize=9)
        ax.set_title(panel_labels[ax_idx], fontsize=11, fontweight="bold", pad=8)
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.3)
        ax.tick_params(direction="in", which="both", width=0.5, length=3)

    # Shared y-axis label
    axes[0].set_ylabel("CE degradation (Δ from baseline)")

    # Set consistent y-limits across all panels
    all_deltas = []
    for spd_data in [spd_05, spd_00]:
        for v in spd_data.values():
            all_deltas.append(v["delta"])
    y_max = max(all_deltas) * 1.2
    for ax in axes:
        ax.set_ylim(0, y_max)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_CI05, edgecolor="white", label="SPD (CI > 0.5)"),
        Patch(facecolor=COLOR_CI00, edgecolor="white", label="SPD (CI > 0.0)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95,
               bbox_to_anchor=(0.5, 1.04), fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    data = load_results(OUTPUT_DIR / "layer_combo_results.json")
    plot_spd_combos(data, OUTPUT_DIR / "spd_layer_combos.png")


if __name__ == "__main__":
    main()
