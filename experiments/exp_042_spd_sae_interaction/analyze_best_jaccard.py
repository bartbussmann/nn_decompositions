"""Analyze best SAE match (by Jaccard) for each SPD component.

Produces:
1. Heatmap: for each SPD component, which SAE layer has the best Jaccard match?
   Split by c_fc vs down_proj.
2. Distribution: max Jaccard of best match, split by (layer, mod_type).

Usage:
    python experiments/exp_042_spd_sae_interaction/analyze_best_jaccard.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("experiments/exp_042_spd_sae_interaction/output")
DATA_PATH = OUTPUT_DIR / "dashboard_data_v2.json"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LAYERS = [0, 1, 2, 3]


def jaccard(match: dict, spd_fire_rate: float) -> float:
    intersection = match["p_sae_given_spd"] * spd_fire_rate
    union = spd_fire_rate + match["sae_fire_rate"] - intersection
    return intersection / union if union > 0 else 0.0


def load_best_matches() -> dict[tuple[int, str], list[tuple[float, int]]]:
    """Return {(spd_layer, mod_type): [(best_jaccard, best_sae_layer), ...]}."""
    with open(DATA_PATH) as f:
        data = json.load(f)

    best_matches: dict[tuple[int, str], list[tuple[float, int]]] = defaultdict(list)
    for comp in data.values():
        spd_layer = comp["spd_layer"]
        mod_type = comp["spd_mod_type"]
        spd_fr = comp["spd_fire_rate"]

        best_jacc = 0.0
        best_sae_layer = 0
        for m in comp["sae_matches"]:
            j = jaccard(m, spd_fr)
            if j > best_jacc:
                best_jacc = j
                best_sae_layer = m["sae_layer"]

        best_matches[(spd_layer, mod_type)].append((best_jacc, best_sae_layer))

    return best_matches


def plot_layer_heatmap(best_matches: dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, mod_type, title in [(axes[0], "c_fc", "c_fc"), (axes[1], "down_proj", "down_proj")]:
        heatmap = np.zeros((len(LAYERS), len(LAYERS)))
        for spd_layer in LAYERS:
            entries = best_matches.get((spd_layer, mod_type), [])
            if not entries:
                continue
            for _, best_sae_layer in entries:
                heatmap[spd_layer][best_sae_layer] += 1
            heatmap[spd_layer] = 100 * heatmap[spd_layer] / len(entries)

        im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(LAYERS)))
        ax.set_yticks(range(len(LAYERS)))
        ax.set_xticklabels([f"SAE L{l}" for l in LAYERS])
        ax.set_yticklabels([f"SPD L{l}" for l in LAYERS])
        ax.set_xlabel("Best-match SAE layer")
        ax.set_ylabel("SPD component layer")
        ax.set_title(title)
        for i in range(len(LAYERS)):
            for j in range(len(LAYERS)):
                ax.text(j, i, f"{heatmap[i][j]:.0f}%", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if heatmap[i][j] > 50 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8, label="%")

    fig.suptitle("Best SAE match (by Jaccard) layer distribution per SPD component",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_jaccard_distribution(best_matches: dict, save_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True, sharex=True)

    for row, mod_type in enumerate(["c_fc", "down_proj"]):
        color = "#2b6cb0" if mod_type == "c_fc" else "#c05621"
        for col, spd_layer in enumerate(LAYERS):
            ax = axes[row][col]
            entries = best_matches.get((spd_layer, mod_type), [])
            jaccards = [j for j, _ in entries]

            ax.hist(jaccards, bins=20, color=color, edgecolor="white", alpha=0.8)
            ax.set_title(f"{mod_type} L{spd_layer}")
            if jaccards:
                ax.axvline(np.median(jaccards), color="red", linestyle="--", linewidth=1,
                           label=f"med={np.median(jaccards):.3f}")
                ax.legend(fontsize=8)
            if col == 0:
                ax.set_ylabel("# SPD components")
            if row == 1:
                ax.set_xlabel("Max Jaccard")

    fig.suptitle("Max Jaccard of best SAE match per SPD component",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    best_matches = load_best_matches()
    total = sum(len(v) for v in best_matches.values())
    print(f"Loaded {total} components")

    plot_layer_heatmap(best_matches, OUTPUT_DIR / "best_jaccard_layer_heatmap.png")
    plot_jaccard_distribution(best_matches, OUTPUT_DIR / "best_jaccard_distribution.png")


if __name__ == "__main__":
    main()
