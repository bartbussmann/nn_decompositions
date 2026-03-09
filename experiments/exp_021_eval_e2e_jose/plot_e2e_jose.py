"""Plot results from eval_e2e_jose.py.

Reads results.json and produces a 1x3 figure: one subplot per eval mode
(cascading / parallel / single-MLP), showing CE degradation vs L0 per module.

Usage:
    python experiments/exp_021_eval_e2e_jose/plot_e2e_jose.py
    python experiments/exp_021_eval_e2e_jose/plot_e2e_jose.py --results path/to/results.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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

# Colors: blue/orange for cascading/parallel (warm vs cool), green for independent
# Solid for TC, dashed for CLT
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

# Which training modes are "matched" to each eval panel
MATCHED_MODELS = {
    "ce_cascading": {"tc_cascading", "clt_cascading"},
    "ce_parallel":  {"tc_parallel", "clt_parallel"},
    "ce_single":    {"tc_independent"},
}

MATCHED_SUBTITLE = {
    "ce_cascading": "favors cascading-trained",
    "ce_parallel":  "favors parallel-trained",
    "ce_single":    "favors independent-trained",
}

SPD_COLOR = "#6b21a8"


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


def plot_three_panel(baseline_ce: float, results: list[dict], save_path: Path,
                     spd_results: list[dict] | None = None):
    groups = group_by_model(results)

    spd_ce_keys = {
        "ce_cascading": "ce_all",
        "ce_parallel": "ce_all",
        "ce_single": "ce_single",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    panel_labels = ["(a)", "(b)", "(c)"]

    for ax, (ce_key, title), panel_label in zip(axes, EVAL_MODES, panel_labels):
        matched = MATCHED_MODELS[ce_key]

        # Draw mismatched lines first (thinner, lower alpha), then matched on top
        for is_matched_pass in [False, True]:
            for model_name in MODEL_STYLES:
                if model_name not in groups:
                    continue
                if (model_name in matched) != is_matched_pass:
                    continue

                pts = groups[model_name]
                l0s = [p["l0"] for p in pts]
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

                ax.plot(l0s, deltas, label=LABEL_MAP[model_name],
                        markeredgecolor="white", markeredgewidth=0.8,
                        alpha=alpha, zorder=zorder, **style)

        # SPD points (separate markers, no connecting line)
        if spd_results:
            spd_ce_key = spd_ce_keys[ce_key]
            spd_markers = {"CI>0.5": "P", "CI>0.0": "D"}
            for spd_r in spd_results:
                if "l0" not in spd_r:
                    continue
                delta = spd_r[spd_ce_key] - baseline_ce
                marker = spd_markers.get(spd_r["mode"], "P")
                label = f"SPD ({spd_r['mode']})"
                ax.plot(spd_r["l0"], delta, color=SPD_COLOR,
                        marker=marker, markersize=14, linestyle="none",
                        markeredgecolor="white", markeredgewidth=1.0,
                        zorder=7, label=label)

        ax.set_xlabel("L0 (active features per module)")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        pass  # subtitle removed
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x)}" if x == int(x) else f"{x:g}"
        ))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, alpha=0.15, linewidth=0.3)
        ax.tick_params(direction="in", which="both", width=0.5, length=3)
        ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    axes[0].set_ylabel("CE degradation (Δ from baseline)")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(ticker.FixedLocator(
        [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    ))
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:g}"
    ))
    axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())

    # Build legend: deduplicate, but keep order logical
    # Collect from panel (a) which has all models visible
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)

    # Override legend handles to full opacity and consistent linewidth
    from matplotlib.lines import Line2D
    legend_handles = []
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
    # SPD
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
               bbox_to_anchor=(0.5, 1.02), fontsize=9.5,
               handlelength=2.5, columnspacing=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
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

    plot_three_panel(baseline_ce, results,
                     OUTPUT_DIR / "eval_e2e_jose_three_panel.png",
                     spd_results=spd_results)


if __name__ == "__main__":
    main()
