"""Feature splitting analysis across SPD runs with different beta values.

Two features are "the same" if the cosine similarity of their down_proj U
vectors exceeds a threshold (default 0.7). For each pair of runs, we count
how many features in run B each feature in run A matches. A count > 1 in
(A→B) means the feature split in B relative to A.

Produces a 4×4 grid of histograms (run_from × run_to).

Usage:
    python experiments/exp_029_feature_splitting/feature_splitting.py
"""

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 2
ALIVE_THRESHOLD = 1e-5
N_EVAL_BATCHES = 20
BATCH_SIZE = 16
SEQ_LEN = 512
COSINE_THRESHOLD = 0.7  # default, can override via --threshold
OUTPUT_DIR = Path(__file__).parent / "output"
CACHE_DIR = Path(__file__).parent / "output" / "cached_dirs"

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
def extract_alive_directions(run_path: str, batches: list[torch.Tensor]) -> torch.Tensor:
    """Load SPD model, compute alive mask, return normalized U directions."""
    from analysis.collect_spd_activations import load_spd_model

    spd_model, _ = load_spd_model(run_path)
    spd_model.to(DEVICE)

    module_name = f"h.{LAYER}.mlp.down_proj"
    comp = spd_model.components[module_name]
    C = comp.C

    fire_counts = torch.zeros(C, device=DEVICE)
    total_positions = 0

    for input_ids in tqdm(batches, desc="Computing activations"):
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_vals = ci.lower_leaky[module_name]
        fire_counts += (ci_vals > 0).float().sum(dim=(0, 1))
        total_positions += ci_vals.shape[0] * ci_vals.shape[1]

    density = fire_counts / total_positions
    alive_mask = density > ALIVE_THRESHOLD
    n_alive = alive_mask.sum().item()

    U = comp.U.float()
    U_alive = F.normalize(U[alive_mask], dim=1).cpu()

    print(f"  {n_alive}/{C} alive")

    # Free GPU memory
    del spd_model
    gc.collect()
    torch.cuda.empty_cache()

    return U_alive


def count_matches(dirs_from: torch.Tensor, dirs_to: torch.Tensor, threshold: float) -> np.ndarray:
    """For each direction in dirs_from, count how many in dirs_to exceed |cosine| > threshold."""
    # Compute on GPU in chunks to handle large matrices
    dirs_to_gpu = dirs_to.to(DEVICE)
    chunk_size = 512
    counts = []

    for i in range(0, dirs_from.shape[0], chunk_size):
        chunk = dirs_from[i:i + chunk_size].to(DEVICE)
        cosine = (chunk @ dirs_to_gpu.T).abs()  # (chunk, n_to)
        counts.append((cosine > threshold).sum(dim=1).cpu().numpy())

    return np.concatenate(counts)


def load_or_extract_directions(batches: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Load cached directions or extract and cache them."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_dirs = {}
    for name, run_path in RUNS:
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("×", "x")
        cache_path = CACHE_DIR / f"{safe_name}.pt"
        if cache_path.exists():
            print(f"Loading cached directions for {name}...")
            all_dirs[name] = torch.load(cache_path)
        else:
            print(f"\nExtracting directions for {name} ({run_path})...")
            all_dirs[name] = extract_alive_directions(run_path, batches)
            torch.save(all_dirs[name], cache_path)
        print(f"  {name}: {all_dirs[name].shape}")
    return all_dirs


def run_analysis(all_dirs: dict[str, torch.Tensor], threshold: float):
    """Run match counting and plot for a given threshold."""
    run_names = [name for name, _ in RUNS]
    suffix = f"_{threshold:.1f}".replace(".", "p")

    match_counts = {}
    for name_from in run_names:
        for name_to in run_names:
            print(f"\n{name_from} → {name_to} (threshold={threshold})...")
            exclude_self = (name_from == name_to)
            dirs_from = all_dirs[name_from]
            dirs_to = all_dirs[name_to]

            if exclude_self:
                raw_counts = count_matches(dirs_from, dirs_to, threshold)
                counts = np.maximum(raw_counts - 1, 0)
            else:
                counts = count_matches(dirs_from, dirs_to, threshold)

            match_counts[(name_from, name_to)] = counts
            n_zero = (counts == 0).sum()
            n_one = (counts == 1).sum()
            n_multi = (counts > 1).sum()
            print(f"  0 matches: {n_zero}/{len(counts)}, 1 match: {n_one}/{len(counts)}, >1 (split): {n_multi}/{len(counts)}")
            if len(counts) > 0:
                print(f"  mean={counts.mean():.2f}, max={counts.max()}")

    # Plot 4x4 histogram grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i, name_from in enumerate(run_names):
        for j, name_to in enumerate(run_names):
            ax = axes[i][j]
            counts = match_counts[(name_from, name_to)]
            max_count = max(counts.max(), 1) if len(counts) > 0 else 1
            bins = np.arange(-0.5, min(max_count + 1.5, 20.5), 1)

            ax.hist(counts, bins=bins, color="#42A5F5" if i != j else "#90CAF9",
                    edgecolor="white", linewidth=0.5)

            n_zero = (counts == 0).sum()
            n_one = (counts == 1).sum()
            n_multi = (counts > 1).sum()
            n_total = len(counts)

            ax.set_title(f"{name_from}\n→ {name_to}", fontsize=9)
            stats_text = f"0: {n_zero} ({100*n_zero/n_total:.0f}%)\n1: {n_one} ({100*n_one/n_total:.0f}%)\n>1: {n_multi} ({100*n_multi/n_total:.0f}%)"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=7,
                    va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            if i == 3:
                ax.set_xlabel("# matches")
            if j == 0:
                ax.set_ylabel("# features")
            ax.set_xlim(-0.5, min(max_count + 0.5, 19.5))

    fig.suptitle(f"Feature Splitting: # matches per feature (|cos| > {threshold}, Layer {LAYER}, alive only)",
                 fontsize=13)
    fig.tight_layout()
    path = OUTPUT_DIR / f"feature_splitting_grid{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")

    # Summary heatmap
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (metric_name, metric_fn) in zip(axes2, [
        ("Mean # matches", lambda c: c.mean()),
        ("% with >1 match (split)", lambda c: 100 * (c > 1).sum() / len(c)),
        ("% with 0 matches (novel)", lambda c: 100 * (c == 0).sum() / len(c)),
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
        ax.set_xlabel("To run")
        ax.set_ylabel("From run")
        ax.set_title(metric_name)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig2.suptitle(f"Feature Splitting Summary (|cos| > {threshold}, Layer {LAYER})", fontsize=13)
    fig2.tight_layout()
    path2 = OUTPUT_DIR / f"feature_splitting_summary{suffix}.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {path2}")

    # Save numeric results
    results = {
        "threshold": threshold,
        "layer": LAYER,
        "n_alive": {name: int(all_dirs[name].shape[0]) for name in run_names},
        "pairs": {},
    }
    for (name_from, name_to), counts in match_counts.items():
        key = f"{name_from} -> {name_to}"
        results["pairs"][key] = {
            "n_from": int(len(counts)),
            "mean_matches": float(counts.mean()),
            "max_matches": int(counts.max()),
            "n_zero": int((counts == 0).sum()),
            "n_one": int((counts == 1).sum()),
            "n_multi": int((counts > 1).sum()),
            "pct_split": float(100 * (counts > 1).sum() / len(counts)),
            "pct_novel": float(100 * (counts == 0).sum() / len(counts)),
        }
    json_path = OUTPUT_DIR / f"feature_splitting_results{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {json_path}")

    return match_counts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=float, nargs="+", default=[COSINE_THRESHOLD])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading eval data...")
    batches = get_eval_batches()

    all_dirs = load_or_extract_directions(batches)

    for threshold in args.thresholds:
        run_analysis(all_dirs, threshold)


if __name__ == "__main__":
    main()
