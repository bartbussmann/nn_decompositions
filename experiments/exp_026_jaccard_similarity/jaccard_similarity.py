"""Jaccard similarity between SPD and TC component activation sets.

For each alive component, computes:
  1. Max Jaccard with another component from the SAME method
  2. Max Jaccard with a component from the OTHER method

Jaccard(A, B) = |A ∩ B| / |A ∪ B|  where A, B are sets of token positions
where each component fires (activation > 0).

Usage:
    python experiments/exp_026_jaccard_similarity/jaccard_similarity.py
"""

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
CHUNK_SIZE = 256  # rows per chunk for pairwise Jaccard computation


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


def load_transcoder(checkpoint_dir: str) -> BatchTopKTranscoder:
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["dtype"] = getattr(torch, cfg_dict.get("dtype", "torch.float32").replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    encoder = BatchTopKTranscoder(cfg)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    encoder.eval()
    return encoder


@torch.no_grad()
def collect_spd_binary(spd_model, batches) -> torch.Tensor:
    """Returns binary activation matrix (n_alive, n_positions) on CPU as bool."""
    module_name = f"h.{LAYER}.mlp.down_proj"
    comp = spd_model.components[module_name]
    C = comp.C

    fire_counts = torch.zeros(C, device=DEVICE)
    total_positions = 0
    all_binary = []

    for input_ids in tqdm(batches, desc="SPD activations"):
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_vals = ci.lower_leaky[module_name]  # (batch, seq, C)
        binary = (ci_vals > 0)  # (batch, seq, C)
        fire_counts += binary.float().sum(dim=(0, 1))
        total_positions += ci_vals.shape[0] * ci_vals.shape[1]
        # Flatten to (batch*seq, C)
        all_binary.append(binary.reshape(-1, C).cpu())

    density = fire_counts / total_positions
    alive_mask = density > ALIVE_THRESHOLD
    n_alive = alive_mask.sum().item()

    # (n_positions, C) -> filter to alive -> transpose to (n_alive, n_positions)
    all_binary_cat = torch.cat(all_binary, dim=0)  # (n_positions, C)
    binary_alive = all_binary_cat[:, alive_mask.cpu()].T.contiguous()  # (n_alive, n_positions)

    print(f"SPD: {n_alive}/{C} alive, {total_positions} positions")
    return binary_alive, density[alive_mask].cpu().numpy()


@torch.no_grad()
def collect_tc_binary(tc, base_model, batches) -> torch.Tensor:
    """Returns binary activation matrix (n_alive, n_positions) on CPU as bool."""
    dict_size = tc.cfg.dict_size

    fire_counts = torch.zeros(dict_size, device=DEVICE)
    total_positions = 0
    all_binary = []

    for input_ids in tqdm(batches, desc="TC activations"):
        captured = {}
        def _hook(_mod, _inp, out):
            captured["mlp_in"] = out.detach()
        hook = base_model.h[LAYER].rms_2.register_forward_hook(_hook)
        base_model(input_ids)
        hook.remove()

        mlp_in = captured["mlp_in"]
        flat = mlp_in.reshape(-1, mlp_in.shape[-1])

        use_pre_enc_bias = tc.cfg.pre_enc_bias and tc.cfg.input_size == tc.cfg.output_size
        x_enc = flat - tc.b_dec if use_pre_enc_bias else flat
        acts = F.relu(x_enc @ tc.W_enc + tc.b_enc)  # (n_tokens, dict_size)

        binary = (acts > 0)
        fire_counts += binary.float().sum(dim=0)
        total_positions += flat.shape[0]
        all_binary.append(binary.cpu())

    density = fire_counts / total_positions
    alive_mask = density > ALIVE_THRESHOLD

    all_binary_cat = torch.cat(all_binary, dim=0)  # (n_positions, dict_size)
    binary_alive = all_binary_cat[:, alive_mask.cpu()].T.contiguous()  # (n_alive, n_positions)
    n_alive = alive_mask.sum().item()

    print(f"TC: {n_alive}/{dict_size} alive, {total_positions} positions")
    return binary_alive, density[alive_mask].cpu().numpy()


def max_jaccard_chunked(
    query: torch.Tensor,   # (n_query, n_positions) bool
    target: torch.Tensor,  # (n_target, n_positions) bool
    exclude_self: bool = False,
) -> np.ndarray:
    """For each row in query, find max Jaccard with any row in target.

    Uses float32 on GPU in chunks to stay within memory.
    Returns (n_query,) array of max Jaccard values.
    """
    n_query = query.shape[0]
    n_target = target.shape[0]
    best = np.zeros(n_query, dtype=np.float32)

    # Precompute target firing counts
    target_f = target.float().to(DEVICE)
    target_sums = target_f.sum(dim=1)  # (n_target,)

    for i in tqdm(range(0, n_query, CHUNK_SIZE), desc="Jaccard chunks"):
        chunk = query[i:i + CHUNK_SIZE].float().to(DEVICE)  # (chunk, n_pos)
        chunk_sums = chunk.sum(dim=1)  # (chunk,)

        # intersection: chunk @ target^T
        intersection = chunk @ target_f.T  # (chunk, n_target)
        # union = |A| + |B| - |A ∩ B|
        union = chunk_sums.unsqueeze(1) + target_sums.unsqueeze(0) - intersection
        jaccard = intersection / union.clamp(min=1)  # avoid div by zero

        if exclude_self:
            # Zero out the diagonal block (self-comparisons)
            for local_j in range(jaccard.shape[0]):
                global_j = i + local_j
                if global_j < n_target:
                    jaccard[local_j, global_j] = 0.0

        best[i:i + CHUNK_SIZE] = jaccard.max(dim=1).values.cpu().numpy()

    return best


def plot_histograms(
    spd_within: np.ndarray,
    spd_cross: np.ndarray,
    tc_within: np.ndarray,
    tc_cross: np.ndarray,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    bins = np.linspace(0, 1, 51)

    def _hist(ax, data, title, color):
        ax.hist(data, bins=bins, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.set_title(title)
        ax.set_xlabel("Max Jaccard similarity")
        ax.set_ylabel("Count")
        median = np.median(data)
        mean = np.mean(data)
        ax.axvline(median, color="black", linestyle="--", linewidth=1, label=f"median={median:.3f}")
        ax.axvline(mean, color="red", linestyle="--", linewidth=1, label=f"mean={mean:.3f}")
        ax.legend()

    _hist(axes[0, 0], spd_within, "SPD: max Jaccard with other SPD component", "#2196F3")
    _hist(axes[0, 1], spd_cross, "SPD: max Jaccard with any TC component", "#9C27B0")
    _hist(axes[1, 0], tc_within, "TC: max Jaccard with other TC component", "#FF9800")
    _hist(axes[1, 1], tc_cross, "TC: max Jaccard with any SPD component", "#9C27B0")

    fig.suptitle(f"Jaccard Similarity of Activation Sets (Layer {LAYER})", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "jaccard_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'jaccard_histograms.png'}")


def plot_joint_scatter(
    spd_within: np.ndarray,
    spd_cross: np.ndarray,
    tc_within: np.ndarray,
    tc_cross: np.ndarray,
    spd_density: np.ndarray,
    tc_density: np.ndarray,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc0 = axes[0].scatter(spd_within, spd_cross, c=np.log10(spd_density), s=4, alpha=0.5, cmap="viridis")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
    axes[0].set_xlabel("Max Jaccard within SPD")
    axes[0].set_ylabel("Max Jaccard with TC")
    axes[0].set_title("SPD components")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    plt.colorbar(sc0, ax=axes[0], label="log10(density)")

    sc1 = axes[1].scatter(tc_within, tc_cross, c=np.log10(tc_density), s=4, alpha=0.5, cmap="viridis")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
    axes[1].set_xlabel("Max Jaccard within TC")
    axes[1].set_ylabel("Max Jaccard with SPD")
    axes[1].set_title("TC components")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    plt.colorbar(sc1, ax=axes[1], label="log10(density)")

    fig.suptitle(f"Within-method vs Cross-method Max Jaccard (Layer {LAYER})", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "jaccard_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'jaccard_scatter.png'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading eval data...")
    batches = get_eval_batches()

    # Load SPD
    from analysis.collect_spd_activations import load_spd_model
    print("\nLoading SPD model...")
    spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    spd_binary, spd_density = collect_spd_binary(spd_model, batches)
    tc_checkpoint = (Path(__file__).resolve().parent.parent.parent
                     / "checkpoints/jose/tc_independent_k32_tc_independent_k32_checkpoint_layer2_final")
    tc = load_transcoder(str(tc_checkpoint))
    tc_binary, tc_density = collect_tc_binary(tc, base_model, batches)

    print(f"\nBinary matrices: SPD {spd_binary.shape}, TC {tc_binary.shape}")

    # Compute max Jaccard similarities
    print("\n--- SPD within SPD ---")
    spd_within = max_jaccard_chunked(spd_binary, spd_binary, exclude_self=True)

    print("\n--- SPD vs TC ---")
    spd_cross = max_jaccard_chunked(spd_binary, tc_binary)

    print("\n--- TC within TC ---")
    tc_within = max_jaccard_chunked(tc_binary, tc_binary, exclude_self=True)

    print("\n--- TC vs SPD ---")
    tc_cross = max_jaccard_chunked(tc_binary, spd_binary)

    # Stats
    def _stats(name, vals):
        print(f"  {name}: mean={vals.mean():.4f}, median={np.median(vals):.4f}, "
              f"max={vals.max():.4f}, >0.5: {(vals > 0.5).sum()}/{len(vals)}, "
              f">0.1: {(vals > 0.1).sum()}/{len(vals)}")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    _stats("SPD↔SPD (within)", spd_within)
    _stats("SPD→TC  (cross) ", spd_cross)
    _stats("TC↔TC   (within)", tc_within)
    _stats("TC→SPD  (cross) ", tc_cross)

    # Save results
    results = {
        "layer": LAYER,
        "n_positions": int(spd_binary.shape[1]),
        "spd_n_alive": int(spd_binary.shape[0]),
        "tc_n_alive": int(tc_binary.shape[0]),
        "spd_within": {
            "mean": float(spd_within.mean()), "median": float(np.median(spd_within)),
            "max": float(spd_within.max()), "gt_0.5": int((spd_within > 0.5).sum()),
            "gt_0.1": int((spd_within > 0.1).sum()),
        },
        "spd_cross": {
            "mean": float(spd_cross.mean()), "median": float(np.median(spd_cross)),
            "max": float(spd_cross.max()), "gt_0.5": int((spd_cross > 0.5).sum()),
            "gt_0.1": int((spd_cross > 0.1).sum()),
        },
        "tc_within": {
            "mean": float(tc_within.mean()), "median": float(np.median(tc_within)),
            "max": float(tc_within.max()), "gt_0.5": int((tc_within > 0.5).sum()),
            "gt_0.1": int((tc_within > 0.1).sum()),
        },
        "tc_cross": {
            "mean": float(tc_cross.mean()), "median": float(np.median(tc_cross)),
            "max": float(tc_cross.max()), "gt_0.5": int((tc_cross > 0.5).sum()),
            "gt_0.1": int((tc_cross > 0.1).sum()),
        },
    }
    with open(OUTPUT_DIR / "jaccard_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT_DIR / 'jaccard_results.json'}")

    # Plots
    plot_histograms(spd_within, spd_cross, tc_within, tc_cross)
    plot_joint_scatter(spd_within, spd_cross, tc_within, tc_cross, spd_density, tc_density)


if __name__ == "__main__":
    main()
