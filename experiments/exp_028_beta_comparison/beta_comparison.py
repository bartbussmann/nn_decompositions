"""Compare SPD runs with different frequency penalty (beta) values.

Plots training curves and final metrics for:
  - baseline (beta=0.5)
  - beta=0
  - beta=2x (1.0)
  - beta=4x (2.0)

Usage:
    python experiments/exp_028_beta_comparison/beta_comparison.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"

RUNS = {
    "baseline": {"label": "baseline (β=0.5)", "color": "#2196F3", "beta": 0.5},
    "beta_0": {"label": "β=0", "color": "#4CAF50", "beta": 0.0},
    "beta_2x": {"label": "β=1.0 (2×)", "color": "#FF9800", "beta": 1.0},
    "beta_4x": {"label": "β=2.0 (4×)", "color": "#F44336", "beta": 2.0},
}

# Final summary metrics (from W&B summary)
SUMMARY = {
    "baseline": {
        "ce_diff_ci": 0.286034, "kl_ci": 0.343155, "l0_total": 201.453140,
        "l0_layer": [43.86, 18.60, 48.37, 90.62],
        "recon_loss": 0.857397, "stoch_recon": 0.414701, "pgd_recon": 0.652251,
        "faith_loss": 0.000002, "imp_min_loss": 1158.32,
        "ce_diff_rounded": 0.240730, "kl_rounded": 0.312946,
        "ce_diff_stoch": 0.134657, "kl_stoch": 0.213557,
    },
    "beta_0": {
        "ce_diff_ci": 0.137851, "kl_ci": 0.178413, "l0_total": 725.290527,
        "l0_layer": [123.29, 62.90, 182.49, 356.60],
        "recon_loss": 0.639822, "stoch_recon": 0.338344, "pgd_recon": 0.397144,
        "faith_loss": 0.000002, "imp_min_loss": 604.36,
        "ce_diff_rounded": 0.099619, "kl_rounded": 0.159430,
        "ce_diff_stoch": 0.037890, "kl_stoch": 0.104299,
    },
    "beta_2x": {
        "ce_diff_ci": 0.371551, "kl_ci": 0.432611, "l0_total": 138.968918,
        "l0_layer": [31.25, 13.16, 35.21, 59.34],
        "recon_loss": 1.015789, "stoch_recon": 0.468606, "pgd_recon": 0.779998,
        "faith_loss": 0.000002, "imp_min_loss": 1458.66,
        "ce_diff_rounded": 0.327148, "kl_rounded": 0.399030,
        "ce_diff_stoch": 0.188033, "kl_stoch": 0.268919,
    },
    "beta_4x": {
        "ce_diff_ci": 0.499190, "kl_ci": 0.560719, "l0_total": 93.525162,
        "l0_layer": [21.74, 9.96, 23.78, 38.04],
        "recon_loss": 1.271711, "stoch_recon": 0.567493, "pgd_recon": 0.885081,
        "faith_loss": 0.000002, "imp_min_loss": 1870.32,
        "ce_diff_rounded": 0.459691, "kl_rounded": 0.526425,
        "ce_diff_stoch": 0.279902, "kl_stoch": 0.354671,
    },
}


def load_history():
    with open(OUTPUT_DIR / "wandb_history.json") as f:
        return json.load(f)


def plot_training_curves(history):
    """Plot key metrics over training steps."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ("eval/ce_kl/ce_difference_ci_masked", "ΔCE (CI-masked)", axes[0, 0]),
        ("eval/ce_kl/kl_ci_masked", "KL (CI-masked)", axes[0, 1]),
        ("eval/l0/0.0_total", "L0 (total)", axes[0, 2]),
        ("eval/loss/ImportanceMinimalityLoss", "ImportanceMinimality Loss", axes[1, 0]),
        ("eval/loss/StochasticHiddenActsReconLoss", "Stoch Recon Loss", axes[1, 1]),
        ("train/loss/total", "Total Train Loss", axes[1, 2]),
    ]

    for metric_key, title, ax in metrics:
        for run_name, run_info in RUNS.items():
            rows = history[run_name]
            steps = [r["_step"] for r in rows if metric_key in r and r[metric_key] is not None]
            vals = [r[metric_key] for r in rows if metric_key in r and r[metric_key] is not None]
            if steps:
                ax.plot(steps, vals, label=run_info["label"], color=run_info["color"], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Beta Sweep: Training Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'training_curves.png'}")


def plot_final_bar_charts():
    """Bar charts of final metrics across beta values."""
    betas = [0.0, 0.5, 1.0, 2.0]
    run_order = ["beta_0", "baseline", "beta_2x", "beta_4x"]
    labels = [RUNS[r]["label"] for r in run_order]
    colors = [RUNS[r]["color"] for r in run_order]
    x = np.arange(len(betas))
    width = 0.6

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. ΔCE (CI-masked)
    vals = [SUMMARY[r]["ce_diff_ci"] for r in run_order]
    axes[0, 0].bar(x, vals, width, color=colors)
    axes[0, 0].set_title("ΔCE (CI-masked)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, fontsize=8, rotation=15)
    for i, v in enumerate(vals):
        axes[0, 0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # 2. KL (CI-masked)
    vals = [SUMMARY[r]["kl_ci"] for r in run_order]
    axes[0, 1].bar(x, vals, width, color=colors)
    axes[0, 1].set_title("KL (CI-masked)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, fontsize=8, rotation=15)
    for i, v in enumerate(vals):
        axes[0, 1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # 3. L0 total
    vals = [SUMMARY[r]["l0_total"] for r in run_order]
    axes[0, 2].bar(x, vals, width, color=colors)
    axes[0, 2].set_title("L0 (total)")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(labels, fontsize=8, rotation=15)
    for i, v in enumerate(vals):
        axes[0, 2].text(i, v + 10, f"{v:.0f}", ha="center", fontsize=9)

    # 4. L0 per layer (grouped bar)
    layer_colors = ["#BBDEFB", "#90CAF9", "#42A5F5", "#1565C0"]
    n_layers = 4
    bar_w = width / n_layers
    for layer_idx in range(n_layers):
        layer_vals = [SUMMARY[r]["l0_layer"][layer_idx] for r in run_order]
        offset = (layer_idx - n_layers / 2 + 0.5) * bar_w
        axes[1, 0].bar(x + offset, layer_vals, bar_w, color=layer_colors[layer_idx],
                        label=f"Layer {layer_idx}")
    axes[1, 0].set_title("L0 per Layer")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, fontsize=8, rotation=15)
    axes[1, 0].legend(fontsize=8)

    # 5. Reconstruction losses
    recon_metrics = ["stoch_recon", "pgd_recon"]
    recon_labels = ["Stoch Recon", "PGD Recon"]
    recon_colors = ["#CE93D8", "#F48FB1"]
    bar_w = width / len(recon_metrics)
    for j, (metric, mlabel, mcolor) in enumerate(zip(recon_metrics, recon_labels, recon_colors)):
        vals = [SUMMARY[r][metric] for r in run_order]
        offset = (j - len(recon_metrics) / 2 + 0.5) * bar_w
        bars = axes[1, 1].bar(x + offset, vals, bar_w, color=mcolor, label=mlabel)
    axes[1, 1].set_title("Reconstruction Losses")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, fontsize=8, rotation=15)
    axes[1, 1].legend(fontsize=8)

    # 6. CE vs L0 tradeoff scatter
    for r in run_order:
        axes[1, 2].scatter(SUMMARY[r]["l0_total"], SUMMARY[r]["ce_diff_ci"],
                           color=RUNS[r]["color"], s=120, zorder=5, edgecolors="black", linewidth=0.5)
        axes[1, 2].annotate(RUNS[r]["label"],
                            (SUMMARY[r]["l0_total"], SUMMARY[r]["ce_diff_ci"]),
                            textcoords="offset points", xytext=(8, 5), fontsize=8)
    # Connect with line
    l0s = [SUMMARY[r]["l0_total"] for r in run_order]
    ces = [SUMMARY[r]["ce_diff_ci"] for r in run_order]
    axes[1, 2].plot(l0s, ces, "k--", linewidth=0.8, alpha=0.5)
    axes[1, 2].set_xlabel("L0 (total)")
    axes[1, 2].set_ylabel("ΔCE (CI-masked)")
    axes[1, 2].set_title("ΔCE vs L0 Tradeoff")
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("Beta Sweep: Final Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'final_metrics.png'}")


def plot_per_layer_l0_curves(history):
    """L0 per layer over training for each run."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (run_name, run_info) in enumerate(RUNS.items()):
        ax = axes[idx]
        rows = history[run_name]
        for layer in range(4):
            key = f"eval/l0/0.0_layer_{layer}"
            steps = [r["_step"] for r in rows if key in r and r[key] is not None]
            vals = [r[key] for r in rows if key in r and r[key] is not None]
            if steps:
                ax.plot(steps, vals, label=f"Layer {layer}", linewidth=1.5)
        ax.set_title(run_info["label"])
        ax.set_xlabel("Step")
        ax.set_ylabel("L0")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("L0 per Layer Over Training", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "l0_per_layer_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'l0_per_layer_curves.png'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history = load_history()
    plot_training_curves(history)
    plot_final_bar_charts()
    plot_per_layer_l0_curves(history)

    # Print summary table
    run_order = ["beta_0", "baseline", "beta_2x", "beta_4x"]
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'β=0':>10} {'β=0.5':>10} {'β=1.0':>10} {'β=2.0':>10}")
    print(f"{'-'*70}")
    for metric, label in [
        ("ce_diff_ci", "ΔCE (CI-masked)"),
        ("kl_ci", "KL (CI-masked)"),
        ("l0_total", "L0 (total)"),
        ("stoch_recon", "Stoch Recon Loss"),
        ("pgd_recon", "PGD Recon Loss"),
        ("imp_min_loss", "ImpMin Loss"),
    ]:
        vals = [SUMMARY[r][metric] for r in run_order]
        if metric == "l0_total":
            print(f"{label:<25} {vals[0]:>10.1f} {vals[1]:>10.1f} {vals[2]:>10.1f} {vals[3]:>10.1f}")
        elif metric == "imp_min_loss":
            print(f"{label:<25} {vals[0]:>10.0f} {vals[1]:>10.0f} {vals[2]:>10.0f} {vals[3]:>10.0f}")
        else:
            print(f"{label:<25} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f}")


if __name__ == "__main__":
    main()
