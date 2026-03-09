"""Plot results from eval_e2e_jose.py with multiple L0 x-axis metrics.

Produces a 3x3 grid: rows = eval modes (cascading, parallel, single-MLP),
columns = L0 metrics (per component, per MLP reconstruction, total active params).

Usage:
    python experiments/exp_021_eval_e2e_jose/plot_e2e_jose_grid.py
    python experiments/exp_021_eval_e2e_jose/plot_e2e_jose_grid.py --results path/to/results.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

OUTPUT_DIR = Path("experiments/exp_021_eval_e2e_jose/output")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 13,
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
    "tc_cascading":    dict(marker="o", color="#2b6cb0", linewidth=1.8, markersize=6, linestyle="-"),
    "tc_parallel":     dict(marker="s", color="#c05621", linewidth=1.8, markersize=6, linestyle="-"),
    "tc_independent":  dict(marker="^", color="#2f855a", linewidth=1.8, markersize=6, linestyle="-"),
    "clt_cascading":   dict(marker="o", color="#2b6cb0", linewidth=1.8, markersize=6, linestyle="--"),
    "clt_parallel":    dict(marker="s", color="#c05621", linewidth=1.8, markersize=6, linestyle="--"),
}

EVAL_MODES = [
    ("ce_cascading", "All-replace cascading"),
    ("ce_parallel", "All-replace parallel"),
    ("ce_single", "Single-MLP replace (avg)"),
]

LABEL_MAP = {
    "tc_cascading": "TC cascading",
    "tc_parallel": "TC parallel",
    "tc_independent": "TC independent",
    "clt_cascading": "CLT cascading",
    "clt_parallel": "CLT parallel",
}

MATCHED_MODELS = {
    "ce_cascading": {"tc_cascading", "clt_cascading"},
    "ce_parallel":  {"tc_parallel", "clt_parallel"},
    "ce_single":    {"tc_independent"},
}

SPD_COLOR = "#6b21a8"

# Jose model dimensions
N_LAYERS = 4
D_IN = 768    # residual stream
D_OUT = 768   # residual stream
D_HIDDEN = 3072  # MLP intermediate


def load_results(path: Path) -> tuple[float, list[dict], list[dict]]:
    with open(path) as f:
        data = json.load(f)
    return data["baseline_ce"], data["results"], data.get("spd_results", [])


def group_by_model(results: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for r in results:
        key = f"{r['type']}_{r['mode']}"
        groups.setdefault(key, []).append(r)
    for v in groups.values():
        v.sort(key=lambda x: x["top_k"])
    return groups


def compute_x_metrics(model_name: str, l0: float) -> dict[str, float]:
    """Compute the three x-axis metrics from raw L0, assuming uniform L0 across layers."""
    if model_name.startswith("tc_"):
        # TC: one encoder-decoder per MLP. Feature has d_in encoder + d_out decoder params.
        per_component = l0
        per_mlp = l0
        total_params = N_LAYERS * l0 * (D_IN + D_OUT)
    elif model_name.startswith("clt_"):
        # CLT: encoder at layer i, decoder writes to layers i..n-1.
        # Per component = raw L0 per encoder layer.
        # Per MLP = avg over MLPs of total features writing to that MLP.
        #   MLP at layer j receives from encoders 0..j, each contributing l0 features.
        #   So MLP j gets (j+1)*l0. Average = l0 * (1+2+3+4)/4 = l0 * 2.5.
        # Total params = sum_i l0 * (d_in + (n-i) * d_out)
        per_component = l0
        per_mlp = l0 * sum(j + 1 for j in range(N_LAYERS)) / N_LAYERS
        total_params = l0 * sum(D_IN + (N_LAYERS - i) * D_OUT for i in range(N_LAYERS))
    else:
        assert False, f"Unknown model: {model_name}"
    return {"x_per_component": per_component, "x_per_mlp": per_mlp, "x_total_params": total_params}


def compute_spd_x_metrics(l0: float) -> dict[str, float]:
    """Compute x-axis metrics for SPD. L0 is per-module (MLP only for jose)."""
    # Jose SPD uses 8 MLP modules: c_fc and down_proj for each layer.
    # per_component = l0 (already average per module)
    # per_mlp = l0 * 2 (each MLP has c_fc + down_proj, each with ~l0 active)
    # total_params = 4 layers * (l0*(d_in+d_hidden) + l0*(d_hidden+d_out))
    per_component = l0
    per_mlp = l0 * 2
    total_params = N_LAYERS * l0 * ((D_IN + D_HIDDEN) + (D_HIDDEN + D_OUT))
    return {"x_per_component": per_component, "x_per_mlp": per_mlp, "x_total_params": total_params}


X_METRICS = [
    ("x_per_component", "Active components per module"),
    ("x_per_mlp", "Active components per MLP"),
    ("x_total_params", "Total active parameters"),
]


def plot_grid(baseline_ce: float, results: list[dict], save_path: Path,
              spd_results: list[dict] | None = None):
    groups = group_by_model(results)

    spd_ce_keys = {
        "ce_cascading": "ce_all",
        "ce_parallel": "ce_all",
        "ce_single": "ce_single",
    }

    fig, axes = plt.subplots(3, 3, figsize=(16, 13), sharey=True)

    for row, (ce_key, row_title) in enumerate(EVAL_MODES):
        matched = MATCHED_MODELS[ce_key]

        for col, (x_key, col_title) in enumerate(X_METRICS):
            ax = axes[row, col]

            # Draw mismatched first, then matched on top
            for is_matched_pass in [False, True]:
                for model_name in MODEL_STYLES:
                    if model_name not in groups:
                        continue
                    if (model_name in matched) != is_matched_pass:
                        continue

                    pts = groups[model_name]
                    xs = [compute_x_metrics(model_name, p["l0"])[x_key] for p in pts]
                    deltas = [p[ce_key] - baseline_ce for p in pts]
                    style = dict(MODEL_STYLES[model_name])

                    if model_name in matched:
                        style["linewidth"] = 2.2
                        style["markersize"] = 7
                        alpha = 1.0
                        zorder = 5
                    else:
                        style["linewidth"] = 1.2
                        style["markersize"] = 5
                        alpha = 0.45
                        zorder = 3

                    ax.plot(xs, deltas, label=LABEL_MAP[model_name],
                            markeredgecolor="white", markeredgewidth=0.8,
                            alpha=alpha, zorder=zorder, **style)

            # SPD points
            if spd_results:
                spd_ce_key = spd_ce_keys[ce_key]
                spd_markers = {"CI>0.5": "P", "CI>0.0": "D"}
                for spd_r in spd_results:
                    if "l0" not in spd_r:
                        continue
                    spd_x = compute_spd_x_metrics(spd_r["l0"])[x_key]
                    delta = spd_r[spd_ce_key] - baseline_ce
                    marker = spd_markers.get(spd_r["mode"], "P")
                    label = f"SPD ({spd_r['mode']})"
                    ax.plot(spd_x, delta, color=SPD_COLOR,
                            marker=marker, markersize=14, linestyle="none",
                            markeredgecolor="white", markeredgewidth=1.0,
                            zorder=7, label=label)

            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f"{int(x)}" if x >= 1 and x == int(x) else f"{x:g}"
            ))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.grid(True, alpha=0.15, linewidth=0.3)
            ax.tick_params(direction="in", which="both", width=0.5, length=3)

            # Column titles on top row
            if row == 0:
                ax.set_title(col_title, fontsize=12, fontweight="bold", pad=10)

            # Row labels on left column
            if col == 0:
                ax.set_ylabel("CE degradation (Δ)")

            # X-axis label on bottom row
            if row == 2:
                ax.set_xlabel(col_title)

            # Panel label
            panel_idx = row * 3 + col
            panel_label = f"({'abcdefghi'[panel_idx]})"
            ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top", ha="left")

        # Row title on right side
        axes[row, 2].text(1.08, 0.5, row_title, transform=axes[row, 2].transAxes,
                          fontsize=11, fontweight="bold", va="center", ha="left",
                          rotation=-90)

    # Shared log y-axis
    axes[0, 0].set_yscale("log")
    axes[0, 0].yaxis.set_major_locator(ticker.FixedLocator(
        [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    ))
    axes[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:g}"
    ))
    axes[0, 0].yaxis.set_minor_formatter(ticker.NullFormatter())

    # Build legend
    legend_handles = []
    seen = set()
    # Collect all labels that appear
    for row_axes in axes:
        for ax in row_axes:
            for _, l in zip(*ax.get_legend_handles_labels()):
                seen.add(l)

    for model_name in ["tc_cascading", "tc_parallel", "tc_independent",
                       "clt_cascading", "clt_parallel"]:
        if LABEL_MAP[model_name] not in seen:
            continue
        s = MODEL_STYLES[model_name]
        legend_handles.append(Line2D(
            [0], [0], color=s["color"], marker=s["marker"], markersize=6,
            markeredgecolor="white", markeredgewidth=0.5,
            linewidth=1.8, linestyle=s["linestyle"],
            label=LABEL_MAP[model_name],
        ))
    for spd_mode, spd_marker in [("CI>0.5", "P"), ("CI>0.0", "D")]:
        spd_label = f"SPD ({spd_mode})"
        if spd_label in seen:
            legend_handles.append(Line2D(
                [0], [0], color=SPD_COLOR, marker=spd_marker, markersize=10,
                markeredgecolor="white", markeredgewidth=0.6,
                linewidth=0, linestyle="none", label=spd_label,
            ))

    fig.legend(handles=legend_handles, loc="upper center",
               ncol=len(legend_handles),
               frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95,
               bbox_to_anchor=(0.5, 1.01), fontsize=9.5,
               handlelength=2.5, columnspacing=1.5)

    fig.tight_layout(rect=[0, 0, 0.95, 0.94])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
                        default="experiments/exp_021_eval_e2e_jose/output/results.json")
    args = parser.parse_args()

    baseline_ce, results, spd_results = load_results(Path(args.results))
    print(f"Baseline CE: {baseline_ce:.4f}, {len(results)} results, {len(spd_results)} SPD")

    plot_grid(baseline_ce, results,
              OUTPUT_DIR / "eval_e2e_jose_grid.png",
              spd_results=spd_results)


if __name__ == "__main__":
    main()
