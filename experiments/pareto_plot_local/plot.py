"""Plot the local Pareto figure from `pareto_plot_local.py`'s output JSON.

Produces a 1x3 three-panel figure per dict_size: one subplot per eval mode
(cascading / parallel / single-MLP), showing CE degradation vs L0.

Usage:
    python experiments/pareto_plot_local/plot.py
    python experiments/pareto_plot_local/plot.py --dict_sizes 4k
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

OUTPUT_DIR = Path("experiments/pareto_plot_local/output")

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
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_STYLES = {
    "tc_cascading":    dict(marker="o", color="#2b6cb0", linewidth=1.8, markersize=6, linestyle="-"),
    "tc_parallel":     dict(marker="s", color="#2b6cb0", linewidth=1.8, markersize=6, linestyle="--"),
    "tc_independent":  dict(marker="^", color="#2b6cb0", linewidth=1.8, markersize=10, linestyle=":"),
    "clt_cascading":   dict(marker="o", color="#dd6b20", linewidth=1.8, markersize=6, linestyle="-"),
    "clt_parallel":    dict(marker="s", color="#dd6b20", linewidth=1.8, markersize=6, linestyle="--"),
}

LABEL_MAP = {
    "tc_cascading": "PLT error-propagating",
    "tc_parallel": "PLT clean-input",
    "tc_independent": "PLT single-layer",
    "clt_cascading": "CLT error-propagating",
    "clt_parallel": "CLT clean-input",
}

EVAL_MODES = [
    ("ce_cascading", "Error-propagating eval"),
    ("ce_parallel", "Clean-input eval"),
    ("ce_single", "Single-layer eval"),
]

MATCHED_MODELS = {
    "ce_cascading": {"tc_cascading", "clt_cascading"},
    "ce_parallel":  {"tc_parallel", "clt_parallel"},
    "ce_single":    {"tc_independent"},
}

VPD_COLOR = "#6b21a8"


def plot_three_panel(baseline_ce, results, vpd_results, save_path, title_suffix=""):
    groups = {}
    for r in results:
        key = f"{r['type']}_{r['mode']}"
        if r["top_k"] > 32:
            continue
        groups.setdefault(key, []).append(r)
    for v in groups.values():
        v.sort(key=lambda x: x["top_k"])

    vpd_ce_keys = {"ce_cascading": "ce_cascading", "ce_parallel": "ce_parallel", "ce_single": "ce_single"}

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

                base_ms = style["markersize"]
                if model_name in matched:
                    style["linewidth"] = 2.2
                    style["markersize"] = max(base_ms, 7)
                    alpha = 1.0
                    zorder = 5
                else:
                    style["linewidth"] = 1.2
                    style["markersize"] = max(base_ms - 2, 5)
                    alpha = 0.45
                    zorder = 3

                ax.plot(l0s, deltas, label=LABEL_MAP[model_name],
                        markeredgecolor="white", markeredgewidth=0.8,
                        alpha=alpha, zorder=zorder, **style)

        if vpd_results:
            vpd_ce_key = vpd_ce_keys[ce_key]
            vpd_markers = {"CI>0.5": "P", "CI>0.1": "X", "CI>0.0": "D"}
            for vpd_r in vpd_results:
                if "l0" not in vpd_r:
                    continue
                # Fall back to ce_all for old results files without separate cascading/parallel
                ce_val = vpd_r.get(vpd_ce_key, vpd_r.get("ce_all"))
                delta = ce_val - baseline_ce
                marker = vpd_markers.get(vpd_r["mode"], "P")
                ax.plot(vpd_r["l0"], delta, color=VPD_COLOR,
                        marker=marker, markersize=14, linestyle="none",
                        markeredgecolor="white", markeredgewidth=1.0,
                        zorder=7, label=f"VPD ({vpd_r['mode']})")

        ax.set_xlabel("L0 (active features per module)")
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x)}" if x == int(x) else f"{x:g}"
        ))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, alpha=0.15, linewidth=0.3)
        ax.tick_params(direction="in", which="both", width=0.5, length=3)
        ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    axes[0].set_ylabel("CE degradation (\u03b4 from baseline)")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(ticker.FixedLocator(
        [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    ))
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())

    # Build legend — columns: PLT | CLT | VPD
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.add(l)

    def _model_handle(model_name):
        s = MODEL_STYLES[model_name]
        return Line2D([0], [0], color=s["color"], marker=s["marker"], markersize=6,
                      markeredgecolor="white", markeredgewidth=0.5,
                      linewidth=1.8, linestyle=s["linestyle"], label=LABEL_MAP[model_name])

    def _vpd_handle(mode, marker):
        return Line2D([0], [0], color=VPD_COLOR, marker=marker, markersize=10,
                      markeredgecolor="white", markeredgewidth=0.6,
                      linewidth=0, linestyle="none", label=f"VPD ({mode})")

    # Build column lists, then interleave for ncol=3 row-major layout
    plt_col = [_model_handle(m) for m in ["tc_cascading", "tc_parallel", "tc_independent"]
               if LABEL_MAP[m] in seen]
    clt_col = [_model_handle(m) for m in ["clt_cascading", "clt_parallel"]
               if LABEL_MAP[m] in seen]
    vpd_col = [_vpd_handle(mode, marker) for mode, marker in
               [("CI>0.5", "P"), ("CI>0.1", "X"), ("CI>0.0", "D")]
               if f"VPD ({mode})" in seen]

    # Pad columns to same length with invisible handles
    n_rows = max(len(plt_col), len(clt_col), len(vpd_col))
    for col in [plt_col, clt_col, vpd_col]:
        while len(col) < n_rows:
            col.append(Line2D([0], [0], linewidth=0, markersize=0, color="none", label=" "))

    # ncol=3 fills column-first, so just concatenate: all PLTs, all CLTs, all VPDs
    legend_handles = plt_col + clt_col + vpd_col

    fig.legend(handles=legend_handles, loc="upper center",
               ncol=3,
               frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95,
               bbox_to_anchor=(0.5, 1.02), fontsize=9.5,
               handlelength=2.5, columnspacing=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig(save_path)
    fig.savefig(str(save_path).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved {save_path}")
    print(f"Saved {str(save_path).replace('.png', '.pdf')}")


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
        vpd_results = data.get("vpd_results", [])
        print(f"{dict_label}: baseline={baseline_ce:.4f}, {len(results)} results, {len(vpd_results)} VPD")

        plot_three_panel(
            baseline_ce, results, vpd_results,
            OUTPUT_DIR / f"eval_e2e_{dict_label}.png",
        )


if __name__ == "__main__":
    main()
