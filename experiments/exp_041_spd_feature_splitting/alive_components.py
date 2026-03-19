"""Count alive SPD components per layer for different C values.

A component is "alive" if it activates (CI > 0) on at least 1 in 1e6 tokens.
Produces a grouped bar plot comparing jose's original SPD with 0.5x, 2x, 4x C.

Usage:
    python experiments/exp_041_spd_feature_splitting/alive_components.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
N_TOKENS = int(1e6)
BATCH_SIZE = 8
SEQ_LEN = 512
N_BATCHES = N_TOKENS // (BATCH_SIZE * SEQ_LEN)  # ~244 batches for 1M tokens

SPD_RUNS = {
    "1x (jose)": "goodfire/spd/s-55ea3f9b",
    "0.5x": "goodfire/spd/s-b2b37c4e",
    "2x": "goodfire/spd/s-8f968bd7",
    "4x": "goodfire/spd/s-f336f91c",
}

# Focus on MLP modules
MLP_MODULE_PATTERNS = ["h.{}.mlp.c_fc", "h.{}.mlp.down_proj"]

OUTPUT_DIR = Path("experiments/exp_041_spd_feature_splitting/output")


def get_eval_batches(n_batches: int) -> list[torch.Tensor]:
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)
    batches = []
    for _ in tqdm(range(n_batches), desc="Loading batches"):
        batch_ids = []
        for _ in range(BATCH_SIZE):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:SEQ_LEN])
        batches.append(torch.stack(batch_ids).to(DEVICE))
    return batches


@torch.no_grad()
def count_alive_components(spd_model, batches, module_names) -> dict[str, tuple[int, int]]:
    """Returns {module_name: (n_alive, n_total)} where alive = fired at least once."""
    ever_active = {}
    for mod_name in module_names:
        n_c = spd_model.module_to_c[mod_name]
        ever_active[mod_name] = torch.zeros(n_c, dtype=torch.bool, device=DEVICE)

    for input_ids in tqdm(batches, desc="Counting alive components"):
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        for mod_name in module_names:
            ci_post = ci.lower_leaky[mod_name]  # (B, S, C)
            # Component is active if CI > 0 for any token in the batch
            active_in_batch = (ci_post > 0).any(dim=0).any(dim=0)  # (C,)
            ever_active[mod_name] |= active_in_batch

    results = {}
    for mod_name in module_names:
        n_alive = ever_active[mod_name].sum().item()
        n_total = spd_model.module_to_c[mod_name]
        results[mod_name] = (n_alive, n_total)
    return results


def main():
    from analysis.collect_spd_activations import load_spd_model

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {N_BATCHES} eval batches ({N_TOKENS/1e6:.0f}M tokens)...")
    batches = get_eval_batches(N_BATCHES)

    all_results = {}

    for label, run_path in SPD_RUNS.items():
        print(f"\n{'='*60}")
        print(f"  Loading SPD: {label} ({run_path})")
        print(f"{'='*60}")

        spd_model, _ = load_spd_model(run_path)
        spd_model.to(DEVICE)

        # Get MLP module names for this model
        mlp_modules = []
        for layer in LAYERS:
            for pattern in MLP_MODULE_PATTERNS:
                mod_name = pattern.format(layer)
                if mod_name in spd_model.module_to_c:
                    mlp_modules.append(mod_name)

        print(f"  Modules: {mlp_modules}")
        for m in mlp_modules:
            print(f"    {m}: C={spd_model.module_to_c[m]}")

        results = count_alive_components(spd_model, batches, mlp_modules)
        all_results[label] = results

        for mod_name, (n_alive, n_total) in results.items():
            pct = 100 * n_alive / n_total
            print(f"  {mod_name}: {n_alive}/{n_total} alive ({pct:.1f}%)")

        del spd_model
        torch.cuda.empty_cache()

    # Aggregate per layer: sum c_fc + down_proj alive/total
    per_layer = {}  # {label: {layer: (alive, total)}}
    for label, results in all_results.items():
        per_layer[label] = {}
        for layer in LAYERS:
            alive = 0
            total = 0
            for pattern in MLP_MODULE_PATTERNS:
                mod_name = pattern.format(layer)
                if mod_name in results:
                    a, t = results[mod_name]
                    alive += a
                    total += t
            per_layer[label][layer] = (alive, total)

    # Plot: grouped bar chart
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.titlesize": 12,
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

    labels_ordered = ["0.5x", "1x (jose)", "2x", "4x"]
    colors = {"0.5x": "#66c2a5", "1x (jose)": "#fc8d62", "2x": "#8da0cb", "4x": "#e78ac3"}

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(LAYERS))
    n_groups = len(labels_ordered)
    bar_width = 0.18
    offsets = np.arange(n_groups) - (n_groups - 1) / 2

    for i, label in enumerate(labels_ordered):
        alive_vals = [per_layer[label][l][0] for l in LAYERS]
        total_vals = [per_layer[label][l][1] for l in LAYERS]
        bars = ax.bar(x + offsets[i] * bar_width, alive_vals, bar_width,
                      label=f"C={label} (total={total_vals[0]})",
                      color=colors[label], edgecolor="white", linewidth=0.5)
        # Annotate with count
        for bar, alive, total in zip(bars, alive_vals, total_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{alive}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Alive MLP components")
    ax.set_title("Alive SPD components per layer (MLP c_fc + down_proj)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS])
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)

    fig.tight_layout()
    save_path = OUTPUT_DIR / "alive_components.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"\nPlot saved to {save_path}")

    # Also save raw data
    import json
    data_path = OUTPUT_DIR / "alive_components.json"
    save_data = {}
    for label, results in all_results.items():
        save_data[label] = {
            mod: {"alive": a, "total": t} for mod, (a, t) in results.items()
        }
    with open(data_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Data saved to {data_path}")


if __name__ == "__main__":
    main()
