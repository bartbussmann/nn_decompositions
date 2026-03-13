"""Activation-based feature splitting across SPD beta runs.

Asymmetric metric: for component A in run X and component B in run Y,
  coverage(A, B) = |S_A ∩ S_B| / |S_A|
i.e. what fraction of A's activations does B also activate on?

For each A, we count how many B's exceed a coverage threshold.
If A splits into B1, B2, B3 in another run, each Bi covers a portion of A.

Usage:
    python experiments/exp_030_activation_splitting/activation_splitting.py
    python experiments/exp_030_activation_splitting/activation_splitting.py --thresholds 0.5 0.25 0.1
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 2
ALIVE_THRESHOLD = 1e-5
N_EVAL_BATCHES = 20
BATCH_SIZE = 16
SEQ_LEN = 512
OUTPUT_DIR = Path(__file__).parent / "output"
CACHE_DIR = Path(__file__).parent / "output" / "cached_binary"
CHUNK_SIZE = 256

RUNS = [
    ("β=0", "goodfire/spd/s-3d327fbf"),
    ("β=0.5 (baseline)", "goodfire/spd/s-55ea3f9b"),
    ("β=1.0 (2×)", "goodfire/spd/s-c20df1cc"),
    ("β=2.0 (4×)", "goodfire/spd/s-73641aa1"),
]


def get_eval_batches() -> list[torch.Tensor]:
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)
    batches = []
    for _ in tqdm(range(N_EVAL_BATCHES), desc="Loading batches"):
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
def extract_binary_activations(run_path: str, batches: list[torch.Tensor]) -> torch.Tensor:
    """Returns binary activation matrix (n_alive, n_positions) as bool on CPU."""
    from analysis.collect_spd_activations import load_spd_model

    spd_model, _ = load_spd_model(run_path)
    spd_model.to(DEVICE)

    module_name = f"h.{LAYER}.mlp.down_proj"
    comp = spd_model.components[module_name]
    C = comp.C

    fire_counts = torch.zeros(C, device=DEVICE)
    total_positions = 0
    all_binary = []

    for input_ids in tqdm(batches, desc="Computing activations"):
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_vals = ci.lower_leaky[module_name]
        binary = (ci_vals > 0)
        fire_counts += binary.float().sum(dim=(0, 1))
        total_positions += ci_vals.shape[0] * ci_vals.shape[1]
        all_binary.append(binary.reshape(-1, C).cpu())

    density = fire_counts / total_positions
    alive_mask = density > ALIVE_THRESHOLD
    n_alive = alive_mask.sum().item()

    all_binary_cat = torch.cat(all_binary, dim=0)  # (n_positions, C)
    binary_alive = all_binary_cat[:, alive_mask.cpu()].T.contiguous()  # (n_alive, n_positions)

    print(f"  {n_alive}/{C} alive, {total_positions} positions")

    del spd_model
    gc.collect()
    torch.cuda.empty_cache()

    return binary_alive


def load_or_extract_binary(batches: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Load cached binary matrices or extract and cache them."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_binary = {}
    for name, run_path in RUNS:
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("×", "x")
        cache_path = CACHE_DIR / f"{safe_name}.pt"
        if cache_path.exists():
            print(f"Loading cached binary for {name}...")
            all_binary[name] = torch.load(cache_path)
        else:
            print(f"\nExtracting binary activations for {name} ({run_path})...")
            all_binary[name] = extract_binary_activations(run_path, batches)
            torch.save(all_binary[name], cache_path)
        print(f"  {name}: {all_binary[name].shape}")
    return all_binary


def count_coverage(
    query: torch.Tensor,   # (n_query, n_positions) bool — "A" components
    target: torch.Tensor,  # (n_target, n_positions) bool — "B" components
    threshold: float,
    exclude_self: bool = False,
) -> np.ndarray:
    """For each A in query, count how many B in target satisfy
    |S_A ∩ S_B| / |S_A| >= threshold.

    Returns (n_query,) array of counts.
    """
    n_query = query.shape[0]
    best = np.zeros(n_query, dtype=np.int32)

    target_f = target.float().to(DEVICE)
    query_sums = query.float().sum(dim=1)  # (n_query,) on CPU

    for i in tqdm(range(0, n_query, CHUNK_SIZE), desc="Coverage chunks", leave=False):
        chunk = query[i:i + CHUNK_SIZE].float().to(DEVICE)  # (chunk, n_pos)
        chunk_sums = query_sums[i:i + CHUNK_SIZE].to(DEVICE)  # (chunk,)

        # intersection
        intersection = chunk @ target_f.T  # (chunk, n_target)
        # coverage = intersection / |S_A|
        coverage = intersection / chunk_sums.unsqueeze(1).clamp(min=1)

        if exclude_self:
            for local_j in range(coverage.shape[0]):
                global_j = i + local_j
                if global_j < target.shape[0]:
                    coverage[local_j, global_j] = 0.0

        counts = (coverage >= threshold).sum(dim=1).cpu().numpy()
        best[i:i + CHUNK_SIZE] = counts

    return best


def run_analysis(all_binary: dict[str, torch.Tensor], threshold: float):
    """Run coverage counting and plot for a given threshold."""
    run_names = [name for name, _ in RUNS]
    suffix = f"_{int(threshold * 100)}pct"

    match_counts = {}
    for name_from in run_names:
        for name_to in run_names:
            print(f"\n{name_from} → {name_to} (coverage >= {threshold:.0%})...")
            exclude_self = (name_from == name_to)
            binary_from = all_binary[name_from]
            binary_to = all_binary[name_to]

            counts = count_coverage(binary_from, binary_to, threshold, exclude_self=exclude_self)

            match_counts[(name_from, name_to)] = counts
            n_zero = (counts == 0).sum()
            n_one = (counts == 1).sum()
            n_multi = (counts > 1).sum()
            n_total = len(counts)
            print(f"  0: {n_zero}/{n_total}, 1: {n_one}/{n_total}, >1 (split): {n_multi}/{n_total}")
            if n_total > 0:
                print(f"  mean={counts.mean():.2f}, max={counts.max()}")

    # 4x4 histogram grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    global_max_count = max(counts.max() for counts in match_counts.values())
    global_max_count = min(global_max_count, 29)
    bins = np.arange(-0.5, global_max_count + 1.5, 1)

    global_max_y = 0
    for counts in match_counts.values():
        hist_vals, _ = np.histogram(counts, bins=bins)
        global_max_y = max(global_max_y, hist_vals.max())

    for i, name_from in enumerate(run_names):
        for j, name_to in enumerate(run_names):
            ax = axes[i][j]
            counts = match_counts[(name_from, name_to)]

            ax.hist(counts, bins=bins, color="#42A5F5" if i != j else "#90CAF9",
                    edgecolor="white", linewidth=0.5)

            n_zero = (counts == 0).sum()
            n_one = (counts == 1).sum()
            n_multi = (counts > 1).sum()
            n_total = len(counts)

            ax.set_title(f"{name_from}\n→ {name_to}", fontsize=9)
            stats = f"0: {n_zero} ({100*n_zero/n_total:.0f}%)\n1: {n_one} ({100*n_one/n_total:.0f}%)\n>1: {n_multi} ({100*n_multi/n_total:.0f}%)"
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                    va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            if i == 3:
                ax.set_xlabel("# components covering ≥" + f"{threshold:.0%}")
            if j == 0:
                ax.set_ylabel("# features")
            ax.set_xlim(-0.5, global_max_count + 0.5)
            ax.set_ylim(0, global_max_y * 1.1)

    fig.suptitle(
        f"Activation Splitting: # components B where |S_A∩S_B|/|S_A| ≥ {threshold:.0%} (Layer {LAYER})",
        fontsize=12)
    fig.tight_layout()
    path = OUTPUT_DIR / f"activation_splitting_grid{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")

    # Summary heatmap
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (metric_name, metric_fn) in zip(axes2, [
        ("Mean # covering", lambda c: c.mean()),
        ("% with >1 (split)", lambda c: 100 * (c > 1).sum() / len(c)),
        ("% with 0 (uncovered)", lambda c: 100 * (c == 0).sum() / len(c)),
    ]):
        matrix = np.zeros((4, 4))
        for i, name_from in enumerate(run_names):
            for j, name_to in enumerate(run_names):
                counts = match_counts[(name_from, name_to)]
                matrix[i, j] = metric_fn(counts)

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(run_names, fontsize=8, rotation=30, ha="right")
        ax.set_yticklabels(run_names, fontsize=8)
        ax.set_xlabel("To run (B)")
        ax.set_ylabel("From run (A)")
        ax.set_title(metric_name)
        for ii in range(4):
            for jj in range(4):
                ax.text(jj, ii, f"{matrix[ii, jj]:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if matrix[ii, jj] > matrix.max() * 0.6 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig2.suptitle(
        f"Activation Splitting Summary: coverage(A,B) = |S_A∩S_B|/|S_A| ≥ {threshold:.0%} (Layer {LAYER})",
        fontsize=11)
    fig2.tight_layout()
    path2 = OUTPUT_DIR / f"activation_splitting_summary{suffix}.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {path2}")

    # Save JSON
    run_n_alive = {name: int(all_binary[name].shape[0]) for name in run_names}
    results = {
        "threshold": threshold,
        "layer": LAYER,
        "n_positions": int(all_binary[run_names[0]].shape[1]),
        "n_alive": run_n_alive,
        "pairs": {},
    }
    for (nf, nt), counts in match_counts.items():
        results["pairs"][f"{nf} -> {nt}"] = {
            "n_from": int(len(counts)),
            "mean": float(counts.mean()),
            "max": int(counts.max()),
            "n_zero": int((counts == 0).sum()),
            "n_one": int((counts == 1).sum()),
            "n_multi": int((counts > 1).sum()),
            "pct_split": float(100 * (counts > 1).sum() / len(counts)),
            "pct_uncovered": float(100 * (counts == 0).sum() / len(counts)),
        }
    json_path = OUTPUT_DIR / f"activation_splitting_results{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.25, 0.1])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading eval data...")
    batches = get_eval_batches()

    all_binary = load_or_extract_binary(batches)

    for threshold in args.thresholds:
        run_analysis(all_binary, threshold)


if __name__ == "__main__":
    main()
