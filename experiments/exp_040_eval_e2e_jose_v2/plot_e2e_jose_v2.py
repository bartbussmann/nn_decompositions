"""Plot results from eval_e2e_jose_v2.py.

Produces a 1x3 three-panel figure per dict_size: one subplot per eval mode
(cascading / parallel / single-MLP), showing CE degradation vs L0.

Usage:
    python experiments/exp_040_eval_e2e_jose_v2/plot_e2e_jose_v2.py
    python experiments/exp_040_eval_e2e_jose_v2/plot_e2e_jose_v2.py --dict_sizes 4k
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

OUTPUT_DIR = Path("experiments/exp_040_eval_e2e_jose_v2/output")

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
    "tc_local_mse":    dict(marker="D", color="#9b2c2c", linewidth=1.8, markersize=6, linestyle="-"),
    "clt_cascading":   dict(marker="o", color="#2b6cb0", linewidth=1.8, markersize=6, linestyle="--"),
    "clt_parallel":    dict(marker="s", color="#c05621", linewidth=1.8, markersize=6, linestyle="--"),
    "clt_local_mse":   dict(marker="D", color="#9b2c2c", linewidth=1.8, markersize=6, linestyle="--"),
}

LABEL_MAP = {
    "tc_cascading": "TC cascading",
    "tc_parallel": "TC parallel",
    "tc_independent": "TC independent",
    "tc_local_mse": "TC local MSE",
    "clt_cascading": "CLT cascading",
    "clt_parallel": "CLT parallel",
    "clt_local_mse": "CLT local MSE",
}

EVAL_MODES = [
    ("ce_cascading", "All-replace cascading"),
    ("ce_parallel", "All-replace parallel"),
    ("ce_single", "Single-MLP replace (avg)"),
]

MATCHED_MODELS = {
    "ce_cascading": {"tc_cascading", "clt_cascading"},
    "ce_parallel":  {"tc_parallel", "clt_parallel"},
    "ce_single":    {"tc_independent"},
}

SPD_COLOR = "#6b21a8"


def plot_three_panel(baseline_ce, results, spd_results, save_path, title_suffix=""):
    groups = {}
    for r in results:
        key = f"{r['type']}_{r['mode']}"
        groups.setdefault(key, []).append(r)
    for v in groups.values():
        v.sort(key=lambda x: x["top_k"])

    spd_ce_keys = {"ce_cascading": "ce_all", "ce_parallel": "ce_all", "ce_single": "ce_single"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    panel_labels = ["(a)", "(b)", "(c)"]

    for ax, (ce_key, title), panel_label in zip(axes, EVAL_MODES, panel_labels):
        matched = MATCHED_MODELS[ce_key]

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

        if spd_results:
            spd_ce_key = spd_ce_keys[ce_key]
            spd_markers = {"CI>0.5": "P", "CI>0.0": "D"}
            for spd_r in spd_results:
                if "l0" not in spd_r:
                    continue
                delta = spd_r[spd_ce_key] - baseline_ce
                marker = spd_markers.get(spd_r["mode"], "P")
                ax.plot(spd_r["l0"], delta, color=SPD_COLOR,
                        marker=marker, markersize=14, linestyle="none",
                        markeredgecolor="white", markeredgewidth=1.0,
                        zorder=7, label=f"SPD ({spd_r['mode']})")

        ax.set_xlabel("L0 (active features per module)")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x)}" if x == int(x) else f"{x:g}"
        ))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, alpha=0.15, linewidth=0.3)
        ax.tick_params(direction="in", which="both", width=0.5, length=3)
        ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    axes[0].set_ylabel("CE degradation (\u0394 from baseline)")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(ticker.FixedLocator(
        [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    ))
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())

    # Build legend
    legend_handles = []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.add(l)

    for model_name in ["tc_cascading", "tc_parallel", "tc_independent", "tc_local_mse",
                       "clt_cascading", "clt_parallel", "clt_local_mse"]:
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
        if f"SPD ({spd_mode})" in seen:
            legend_handles.append(Line2D(
                [0], [0], color=SPD_COLOR, marker=spd_marker, markersize=10,
                markeredgecolor="white", markeredgewidth=0.6,
                linewidth=0, linestyle="none", label=f"SPD ({spd_mode})",
            ))

    fig.legend(handles=legend_handles, loc="upper center",
               ncol=min(len(legend_handles), 5),
               frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95,
               bbox_to_anchor=(0.5, 1.02), fontsize=9.5,
               handlelength=2.5, columnspacing=1.5)

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=14, fontweight="bold", y=1.08)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_sizes", nargs="+", default=["4k", "32k"], choices=["4k", "32k"])
    args = parser.parse_args()

    for dict_label in args.dict_sizes:
        results_path = OUTPUT_DIR / f"results_{dict_label}.json"
        if not results_path.exists():
            print(f"Skipping {dict_label}: {results_path} not found")
            continue

        with open(results_path) as f:
            data = json.load(f)

        baseline_ce = data["baseline_ce"]
        results = data["results"]
        spd_results = data.get("spd_results", [])
        print(f"{dict_label}: baseline={baseline_ce:.4f}, {len(results)} results, {len(spd_results)} SPD")

        plot_three_panel(
            baseline_ce, results, spd_results,
            OUTPUT_DIR / f"eval_e2e_jose_{dict_label}.png",
            title_suffix=f"Jose target model — {dict_label} dict",
        )


if __name__ == "__main__":
    main()
