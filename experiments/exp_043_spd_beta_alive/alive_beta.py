"""Count alive SPD components (mean CI > 1e-6) for different ImpMin beta values.

All runs use the same capacity (jose's 1x) but vary beta ∈ {0.0, 0.25, 0.5, 1.0}.
Jose's original run (beta=0.5, imp_min_coeff=0.0002) is included for comparison.
Runs one model at a time via subprocess to avoid OOM accumulation.

Usage:
    python experiments/exp_043_spd_beta_alive/alive_beta.py
    python experiments/exp_043_spd_beta_alive/alive_beta.py --run "beta=0.5"
    python experiments/exp_043_spd_beta_alive/alive_beta.py --plot-only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("experiments/exp_043_spd_beta_alive/output")
OUTPUT_FILE = OUTPUT_DIR / "alive_beta.json"

SPD_RUNS = {
    "beta=0.0": "goodfire/spd/s-68b5d00f",
    "beta=0.25": "goodfire/spd/s-2d710891",
    "beta=0.5": "goodfire/spd/s-cebfd0fb",
    "beta=1.0": "goodfire/spd/s-c4895972",
    "jose (beta=0.5)": "goodfire/spd/s-55ea3f9b",
}

WORKER_SCRIPT = r"""
import sys, json, torch
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/spd')

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
DEVICE = 'cuda'
LAYERS = [0, 1, 2, 3]
BATCH_SIZE = 4
SEQ_LEN = 512
N_TOKENS = int(1e6)
N_BATCHES = N_TOKENS // (BATCH_SIZE * SEQ_LEN)
MLP_MODULE_PATTERNS = ['h.{}.mlp.c_fc', 'h.{}.mlp.down_proj']
THRESHOLD = 1e-6

label = sys.argv[1]
run_path = sys.argv[2]

from datasets import load_dataset
from tqdm import tqdm
from analysis.collect_spd_activations import load_spd_model

dataset = load_dataset('danbraunai/pile-uncopyrighted-tok', split='train', streaming=True)
dataset = dataset.shuffle(seed=0, buffer_size=10000)
data_iter = iter(dataset)
batches = []
for _ in tqdm(range(N_BATCHES), desc='Loading batches'):
    batch_ids = []
    for _ in range(BATCH_SIZE):
        sample = next(data_iter)
        ids = sample['input_ids']
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        batch_ids.append(ids[:SEQ_LEN])
    batches.append(torch.stack(batch_ids))

print(f'Loading {label} ({run_path})...')
spd_model, _ = load_spd_model(run_path)
spd_model.to(DEVICE)

mlp_modules = []
for layer in LAYERS:
    for pattern in MLP_MODULE_PATTERNS:
        mod_name = pattern.format(layer)
        if mod_name in spd_model.module_to_c:
            mlp_modules.append(mod_name)

ci_sum = {}
for mod_name in mlp_modules:
    n_c = spd_model.module_to_c[mod_name]
    ci_sum[mod_name] = torch.zeros(n_c, dtype=torch.float64, device='cpu')

torch.set_grad_enabled(False)
n_tokens_total = 0
for input_ids_cpu in tqdm(batches, desc=label):
    input_ids = input_ids_cpu.to(DEVICE)
    B, S = input_ids.shape
    n_tokens_total += B * S
    out = spd_model(input_ids, cache_type='input')
    ci = spd_model.calc_causal_importances(out.cache, sampling='continuous')
    for mod_name in mlp_modules:
        ci_vals = ci.lower_leaky[mod_name].reshape(-1, spd_model.module_to_c[mod_name])
        ci_sum[mod_name] += ci_vals.double().sum(dim=0).cpu()

results = {}
for mod_name in mlp_modules:
    mean_ci = ci_sum[mod_name] / n_tokens_total
    n_alive = (mean_ci > THRESHOLD).sum().item()
    n_total = spd_model.module_to_c[mod_name]
    results[mod_name] = {'alive': n_alive, 'total': n_total}
    print(f'  {mod_name}: {n_alive}/{n_total} alive ({100*n_alive/n_total:.1f}%)')

print('RESULT_JSON:' + json.dumps(results))
"""


def run_one(label: str, run_path: str) -> dict:
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    result = subprocess.run(
        [sys.executable, "-c", WORKER_SCRIPT, label, run_path],
        capture_output=True, text=True, timeout=1200, env=env,
    )
    print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr, end="")
    for line in result.stdout.split("\n"):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(f"No result from {label}:\n{result.stdout[-500:]}\n{result.stderr[-500:]}")


def compute():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    runs_to_do = SPD_RUNS
    if args.run:
        runs_to_do = {args.run: SPD_RUNS[args.run]}

    for label, run_path in runs_to_do.items():
        if label in all_results and not args.run:
            print(f"  Skipping {label} (already computed)")
            continue

        print(f"\n{'='*60}")
        print(f"  {label} ({run_path})")
        print(f"{'='*60}")
        results = run_one(label, run_path)
        all_results[label] = results

        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved ({len(all_results)} models so far)")

    print(f"\nDone! Results in {OUTPUT_FILE}")
    return all_results


def plot(all_results: dict | None = None):
    if all_results is None:
        with open(OUTPUT_FILE) as f:
            all_results = json.load(f)

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

    # Aggregate per-layer: sum c_fc + down_proj
    LAYERS = [0, 1, 2, 3]
    MLP_PATTERNS = ["h.{}.mlp.c_fc", "h.{}.mlp.down_proj"]

    per_layer = {}  # {label: {layer: (alive, total)}}
    for label, modules in all_results.items():
        per_layer[label] = {}
        for layer in LAYERS:
            alive = 0
            total = 0
            for pattern in MLP_PATTERNS:
                mod_name = pattern.format(layer)
                if mod_name in modules:
                    alive += modules[mod_name]["alive"]
                    total += modules[mod_name]["total"]
            per_layer[label][layer] = (alive, total)

    # ── Bar chart: alive per layer, grouped by beta ──
    labels_ordered = ["beta=0.0", "beta=0.25", "beta=0.5", "jose (beta=0.5)", "beta=1.0"]
    labels_ordered = [l for l in labels_ordered if l in per_layer]

    colors = {
        "beta=0.0": "#66c2a5",
        "beta=0.25": "#fc8d62",
        "beta=0.5": "#8da0cb",
        "jose (beta=0.5)": "#e78ac3",
        "beta=1.0": "#a6d854",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(LAYERS))
    n_groups = len(labels_ordered)
    bar_width = 0.15
    offsets = np.arange(n_groups) - (n_groups - 1) / 2

    for i, label in enumerate(labels_ordered):
        alive_vals = [per_layer[label][l][0] for l in LAYERS]
        total_vals = [per_layer[label][l][1] for l in LAYERS]
        bars = ax.bar(
            x + offsets[i] * bar_width, alive_vals, bar_width,
            label=label, color=colors.get(label, "#999"),
            edgecolor="white", linewidth=0.5,
        )
        for bar, alive in zip(bars, alive_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{alive}", ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Alive MLP components (mean CI > 1e-6)")
    ax.set_title("Alive SPD components per layer for different ImpMin beta values")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS])
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)

    fig.tight_layout()
    save_path = OUTPUT_DIR / "alive_beta_per_layer.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Per-layer plot saved to {save_path}")

    # ── Summary: total alive across all layers ──
    print("\nSummary (total across all layers):")
    for label in labels_ordered:
        total_alive = sum(per_layer[label][l][0] for l in LAYERS)
        total_cap = sum(per_layer[label][l][1] for l in LAYERS)
        print(f"  {label:>20}: {total_alive:>5}/{total_cap:>5} alive ({100*total_alive/total_cap:.1f}%)")

    # ── Line plot: beta (x) vs alive components (y), one line per layer ──
    beta_labels = ["beta=0.0", "beta=0.25", "beta=0.5", "beta=1.0"]
    beta_labels = [l for l in beta_labels if l in per_layer]
    beta_vals = [float(l.split("=")[1]) for l in beta_labels]

    layer_colors = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#984ea3"}

    fig, ax = plt.subplots(figsize=(7, 5))

    for layer in LAYERS:
        alive_vals = [per_layer[l][layer][0] for l in beta_labels]
        ax.plot(beta_vals, alive_vals, "o-", color=layer_colors[layer],
                markersize=7, lw=2, label=f"Layer {layer}", zorder=3)

    # Total line (all layers summed)
    total_vals = [sum(per_layer[l][layer][0] for layer in LAYERS) for l in beta_labels]
    ax.plot(beta_vals, total_vals, "s-", color="black", markersize=7, lw=2,
            label="Total", zorder=3)


    ax.set_xlabel("ImpMin beta")
    ax.set_ylabel("Alive MLP components (mean CI > 1e-6)")
    ax.set_title("Alive SPD components vs. frequency minimality strength")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, linewidth=0.5)

    fig.tight_layout()
    save_path = OUTPUT_DIR / "alive_beta_line.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Line plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Run only this label (e.g. 'beta=0.5')")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from saved data")
    args = parser.parse_args()

    if args.plot_only:
        plot()
    else:
        all_results = compute()
        plot(all_results)
