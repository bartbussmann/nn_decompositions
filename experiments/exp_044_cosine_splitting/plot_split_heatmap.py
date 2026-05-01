"""Publication-quality heatmap: % components with >1 cosine match across models.

Reads precomputed data from cross_model_heatmap.py and produces a standalone
figure suitable for paper submission.

Usage:
    python experiments/exp_044_cosine_splitting/plot_split_heatmap.py
    python experiments/exp_044_cosine_splitting/plot_split_heatmap.py --threshold 0.5
    python experiments/exp_044_cosine_splitting/plot_split_heatmap.py --space input
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_DIR = Path("experiments/exp_044_cosine_splitting/output")


def plot(threshold: float = 0.3, space: str = "output", no_beta: bool = False):
    suffix = f"_{threshold}".replace(".", "p")
    space_suffix = f"_{space}" if space != "output" else ""
    data_path = OUTPUT_DIR / f"heatmap_data{suffix}{space_suffix}.json"

    with open(data_path) as f:
        data = json.load(f)

    model_names = data["models"]
    matrix = np.array(data["pct_split"])

    if no_beta:
        keep = [i for i, name in enumerate(model_names) if "b=" not in name]
        model_names = [model_names[i] for i in keep]
        matrix = matrix[np.ix_(keep, keep)]

    n = len(model_names)

    # ── Publication style ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal", vmin=0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    # Nicer model labels
    display_names = []
    for name in model_names:
        name = name.replace("SPD", "VPD").replace("TC", "PLT").replace("VPD b=", "VPD β=")
        display_names.append(name)

    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_yticklabels(display_names)

    ax.set_xlabel("Target model", labelpad=6)
    ax.set_ylabel("Source model", labelpad=6)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > 55 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=6, color=color, fontweight="medium")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("% components with >1 match", fontsize=9, labelpad=8)

    # Draw group separators
    sep_positions = [3.5] if no_beta else [3.5, 7.5]
    for pos in sep_positions:
        ax.axhline(pos, color="white", lw=1.5)
        ax.axvline(pos, color="white", lw=1.5)

    # Title
    space_str = "decoder / U" if space == "output" else "encoder / V"
    ax.set_title(
        f"Cross-model component splitting ({space_str}, cosine > {threshold})",
        fontsize=10, pad=10,
    )

    # Minor ticks off
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=3)

    fig.tight_layout()
    beta_suffix = "_no_beta" if no_beta else ""
    out_path = OUTPUT_DIR / f"split_heatmap{suffix}{space_suffix}{beta_suffix}.png"
    fig.savefig(out_path)
    out_pdf = OUTPUT_DIR / f"split_heatmap{suffix}{space_suffix}{beta_suffix}.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--space", choices=["output", "input"], default="output")
    parser.add_argument("--no-beta", action="store_true", help="Exclude SPD beta variants")
    args = parser.parse_args()
    plot(args.threshold, args.space, args.no_beta)
