"""Line plot: total components (x) vs mean max cosine similarity (y) for VPD, TCs, and CLTs.

Same x-axis and models as exp_041 alive_line_plot. Y-axis is the mean of per-alive-component
max cosine similarity (how similar each alive component is to its most similar neighbor),
averaged across modules. Higher values indicate more feature splitting.

Data is read from exp_044's precomputed cosine_capacity.json if available, otherwise
recomputed via subprocess workers (identical to exp_044).

Usage:
    python experiments/exp_047_cosine_line_plot/cosine_line_plot.py
    python experiments/exp_047_cosine_line_plot/cosine_line_plot.py --plot-only
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path("experiments/exp_047_cosine_line_plot/output")
EXP044_DATA = Path("experiments/exp_044_cosine_splitting/output/cosine_capacity.json")

LAYERS = [0, 1, 2, 3]

# Same models as exp_041
TC_RUNS = {
    4096: {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "4ziu27fn"},
    32768: {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "c4o8i98k"},
}

CLT_RUNS = {
    4096: {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "77sgz1pe"},
    32768: {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "j20m9hzr"},
}

SPD_CAPACITY_LABELS = ["0.5x", "1x", "2x", "4x"]


def _mean_max_cos_total(model_data: dict) -> float:
    """Average mean_max_cosine across all modules."""
    vals = [v["mean_max_cosine"] for v in model_data.values()]
    return sum(vals) / len(vals) if vals else 0.0


def load_data() -> dict:
    """Load precomputed data from exp_044, or recompute if missing."""
    if EXP044_DATA.exists():
        with open(EXP044_DATA) as f:
            return json.load(f)

    # Fall back to running exp_044's compute step
    import subprocess
    import sys
    print("exp_044 data not found, running exp_044 capacity computation...")
    subprocess.run(
        [sys.executable, "experiments/exp_044_cosine_splitting/cosine_splitting.py", "--step", "capacity"],
        check=True,
    )
    with open(EXP044_DATA) as f:
        return json.load(f)


def plot(data: dict):
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
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    # VPD
    spd_totals, spd_cos = [], []
    for cap in SPD_CAPACITY_LABELS:
        key = f"spd_{cap}"
        if key not in data:
            continue
        total = sum(v["total"] for v in data[key].values())
        spd_totals.append(total)
        spd_cos.append(_mean_max_cos_total(data[key]))
    ax.plot(spd_totals, spd_cos, "o-", color="#6b21a8", markersize=7, lw=2, label="VPD", zorder=3)

    # PLT
    tc_totals, tc_cos = [], []
    for ds in sorted(TC_RUNS.keys()):
        key = f"tc_{ds}"
        if key not in data:
            continue
        tc_totals.append(ds * len(LAYERS))
        tc_cos.append(_mean_max_cos_total(data[key]))
    ax.plot(tc_totals, tc_cos, "s-", color="#2b6cb0", markersize=7, lw=2, label="PLT (k=16)", zorder=3)

    # CLT
    clt_totals, clt_cos = [], []
    for ds in sorted(CLT_RUNS.keys()):
        key = f"clt_{ds}"
        if key not in data:
            continue
        clt_totals.append(ds * len(LAYERS))
        clt_cos.append(_mean_max_cos_total(data[key]))
    ax.plot(clt_totals, clt_cos, "^-", color="#dd6b20", markersize=7, lw=2, label="CLT (k=16)", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Total component capacity")
    ax.set_ylabel("Mean max cosine similarity")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, linewidth=0.5, which="both")

    fig.tight_layout()
    save_path = OUTPUT_DIR / "cosine_line_plot.png"
    fig.savefig(save_path)
    fig.savefig(OUTPUT_DIR / "cosine_line_plot.pdf")
    plt.close(fig)
    print(f"Saved {save_path}")
    print(f"Saved {OUTPUT_DIR / 'cosine_line_plot.pdf'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Only plot from saved data")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    plot(data)


if __name__ == "__main__":
    main()
