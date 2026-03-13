"""UMAP co-embedding of SPD component directions and transcoder decoder directions.

Compares SPD down_proj U vectors with transcoder W_dec rows in residual stream space.
Only includes alive components (activation density > threshold).
Focuses on layer 2.

Usage:
    python experiments/exp_024_umap_spd_vs_tc/umap_spd_vs_tc.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import umap
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig
from spd.models.components import make_mask_infos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 2
ALIVE_THRESHOLD = 1e-5
N_EVAL_BATCHES = 20
BATCH_SIZE = 16
SEQ_LEN = 512
OUTPUT_DIR = Path(__file__).parent / "output"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


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
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    encoder = BatchTopKTranscoder(cfg)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    encoder.eval()
    return encoder


@torch.no_grad()
def get_spd_directions_alive(spd_model, batches: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """Extract unit-normalized U vectors from alive SPD down_proj components (layer 2).

    Returns (directions, densities) for alive components only.
    """
    module_name = f"h.{LAYER}.mlp.down_proj"
    comp = spd_model.components[module_name]
    C = comp.C

    # Compute activation density: fraction of positions where CI > 0
    fire_counts = torch.zeros(C, device=DEVICE)
    total_positions = 0

    for input_ids in tqdm(batches, desc="SPD activation density"):
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_vals = ci.lower_leaky[module_name]  # (batch, seq, C)
        fire_counts += (ci_vals > 0).float().sum(dim=(0, 1))
        total_positions += ci_vals.shape[0] * ci_vals.shape[1]

    density = fire_counts / total_positions
    alive_mask = density > ALIVE_THRESHOLD
    n_alive = alive_mask.sum().item()

    U = comp.U.float()
    U_alive = U[alive_mask]
    U_norm = F.normalize(U_alive, dim=1)
    density_alive = density[alive_mask]

    print(f"SPD: {n_alive}/{C} alive (density > {ALIVE_THRESHOLD}), d={U_norm.shape[1]}")
    print(f"  Density stats: min={density_alive.min():.6f}, "
          f"median={density_alive.median():.4f}, "
          f"max={density_alive.max():.4f}")
    return U_norm.cpu().numpy(), density_alive.cpu().numpy()


@torch.no_grad()
def get_tc_directions_alive(
    checkpoint_dir: str,
    base_model: torch.nn.Module,
    batches: list[torch.Tensor],
) -> np.ndarray:
    """Extract unit-normalized W_dec rows from alive transcoder features."""
    tc = load_transcoder(checkpoint_dir)
    dict_size = tc.cfg.dict_size

    fire_counts = torch.zeros(dict_size, device=DEVICE)
    total_positions = 0

    for input_ids in tqdm(batches, desc="TC activation density"):
        # Collect MLP input (post-RMSNorm) for the target layer
        captured = {}
        def _hook(_mod, _inp, out):
            captured["mlp_in"] = out.detach()
        hook = base_model.h[LAYER].rms_2.register_forward_hook(_hook)
        base_model(input_ids)
        hook.remove()

        mlp_in = captured["mlp_in"]  # (batch, seq, d)
        flat = mlp_in.reshape(-1, mlp_in.shape[-1])

        # Get pre-topk activations
        use_pre_enc_bias = tc.cfg.pre_enc_bias and tc.cfg.input_size == tc.cfg.output_size
        x_enc = flat - tc.b_dec if use_pre_enc_bias else flat
        acts = F.relu(x_enc @ tc.W_enc)  # (n_tokens, dict_size)

        fire_counts += (acts > 0).float().sum(dim=0)
        total_positions += flat.shape[0]

    density = fire_counts / total_positions
    alive_mask = density > ALIVE_THRESHOLD
    n_alive = alive_mask.sum().item()

    W_dec = tc.W_dec.float()
    W_dec_alive = W_dec[alive_mask]
    W_dec_norm = F.normalize(W_dec_alive, dim=1)

    density_alive = density[alive_mask]
    print(f"TC:  {n_alive}/{dict_size} alive (density > {ALIVE_THRESHOLD}), d={W_dec_norm.shape[1]}")
    print(f"  Density stats: min={density_alive.min():.6f}, "
          f"median={density_alive.median():.4f}, "
          f"max={density_alive.max():.4f}")
    return W_dec_norm.cpu().numpy(), density_alive.cpu().numpy()


def plot_umap_single(
    dirs: np.ndarray,
    densities: np.ndarray,
    label: str,
    color: str,
    save_path: Path,
):
    """UMAP of a single method's directions using Anthropic's Towards Monosemanticity params.

    - 2D UMAP with n_neighbors=15, metric=cosine, min_dist=0.05
    - HDBSCAN clustering on a 10D UMAP (n_neighbors=15, cosine, min_dist=0.1)
    """
    import hdbscan

    n = dirs.shape[0]
    print(f"Running UMAPs on {n} {label} vectors...")

    # 10D UMAP for clustering
    reducer_10d = umap.UMAP(
        n_components=10, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=42,
    )
    emb_10d = reducer_10d.fit_transform(dirs)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
    cluster_labels = clusterer.fit_predict(emb_10d)
    n_clusters = cluster_labels.max() + 1
    n_noise = (cluster_labels == -1).sum()
    print(f"  HDBSCAN: {n_clusters} clusters, {n_noise}/{n} noise points")

    # 2D UMAP for visualization
    reducer_2d = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.05,
        metric="cosine", random_state=42,
    )
    emb_2d = reducer_2d.fit_transform(dirs)

    log_density = np.log10(densities)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel (a): colored by cluster
    ax = axes[0]
    noise_mask = cluster_labels == -1
    ax.scatter(emb_2d[noise_mask, 0], emb_2d[noise_mask, 1],
               s=3, alpha=0.15, c="#cccccc", rasterized=True)
    if n_clusters > 0:
        cmap = plt.cm.get_cmap("tab20", min(n_clusters, 20))
        for ci in range(n_clusters):
            mask = cluster_labels == ci
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       s=5, alpha=0.5, c=[cmap(ci % 20)], rasterized=True)
    ax.set_title(f"{label} UMAP ({n} alive, {n_clusters} clusters)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    # Panel (b): colored by activation density
    ax = axes[1]
    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5, c=log_density,
                    cmap="viridis", alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label="log10(activation density)", shrink=0.8)
    ax.set_title(f"{label} colored by activation density")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.03, 0.97, "(b)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    # Panel (c): cluster size distribution
    ax = axes[2]
    if n_clusters > 0:
        cluster_sizes = np.bincount(cluster_labels[~noise_mask])
        ax.hist(cluster_sizes[cluster_sizes > 0], bins=50, color=color, alpha=0.7)
    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Count")
    ax.set_title(f"Cluster size distribution")
    ax.text(0.03, 0.97, "(c)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_umap(
    spd_dirs: np.ndarray,
    tc_dirs: np.ndarray,
    save_path: Path,
):
    n_spd = spd_dirs.shape[0]
    n_tc = tc_dirs.shape[0]

    # Co-embed with Towards Monosemanticity params
    all_dirs = np.concatenate([spd_dirs, tc_dirs], axis=0)
    print(f"Running UMAP on {all_dirs.shape[0]} vectors (d={all_dirs.shape[1]})...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.05,
        metric="cosine", random_state=42,
    )
    embedding = reducer.fit_transform(all_dirs)

    spd_emb = embedding[:n_spd]
    tc_emb = embedding[n_spd:]

    # Compute nearest-neighbor stats
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(spd_dirs, tc_dirs)
    max_sims_spd_to_tc = sims.max(axis=1)  # for each SPD, best TC match
    max_sims_tc_to_spd = sims.max(axis=0)  # for each TC, best SPD match

    print(f"\nCosine similarity stats (original space):")
    print(f"  SPD->TC: mean best match = {max_sims_spd_to_tc.mean():.3f}, "
          f"median = {np.median(max_sims_spd_to_tc):.3f}, "
          f">0.9: {(max_sims_spd_to_tc > 0.9).sum()}/{n_spd}")
    print(f"  TC->SPD: mean best match = {max_sims_tc_to_spd.mean():.3f}, "
          f"median = {np.median(max_sims_tc_to_spd):.3f}, "
          f">0.9: {(max_sims_tc_to_spd > 0.9).sum()}/{n_tc}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel (a): Co-embedding
    ax = axes[0]
    ax.scatter(tc_emb[:, 0], tc_emb[:, 1], s=6, alpha=0.4, c="#1f77b4", label=f"TC alive ({n_tc})", rasterized=True)
    ax.scatter(spd_emb[:, 0], spd_emb[:, 1], s=6, alpha=0.4, c="#9467bd", label=f"SPD alive ({n_spd})", rasterized=True)
    ax.set_title(f"UMAP co-embedding (layer {LAYER} down_proj, alive only)")
    ax.legend(markerscale=3, frameon=True, fancybox=False, edgecolor="#ccc")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    # Panel (b): Color SPD points by their best TC match
    ax = axes[1]
    sc = ax.scatter(
        spd_emb[:, 0], spd_emb[:, 1], s=8, c=max_sims_spd_to_tc,
        cmap="RdYlGn", vmin=0, vmax=1, alpha=0.7, rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="Best cosine sim to TC", shrink=0.8)
    ax.set_title("SPD components colored by\nbest TC match")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.03, 0.97, "(b)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    # Panel (c): Histograms of best-match similarities
    ax = axes[2]
    ax.hist(max_sims_spd_to_tc, bins=60, alpha=0.6, color="#9467bd", label="SPD→TC", density=True)
    ax.hist(max_sims_tc_to_spd, bins=60, alpha=0.6, color="#1f77b4", label="TC→SPD", density=True)
    ax.set_xlabel("Best cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of best-match\ncosine similarities")
    ax.legend(frameon=True, fancybox=False, edgecolor="#ccc")
    ax.text(0.03, 0.97, "(c)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"\nPlot saved to {save_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spd_run = "goodfire/spd/s-55ea3f9b"
    tc_checkpoint = (
        "checkpoints/jose/tc_independent_k32_tc_independent_k32_checkpoint_layer2_final"
    )

    print("Loading eval data...")
    batches = get_eval_batches()

    # Load SPD (also gives us the base model for TC eval)
    from analysis.collect_spd_activations import load_spd_model
    print(f"\nLoading SPD model from {spd_run}...")
    spd_model, _ = load_spd_model(spd_run)
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    print("\nComputing SPD alive directions...")
    spd_dirs, spd_densities = get_spd_directions_alive(spd_model, batches)

    print(f"\nComputing TC alive directions from {tc_checkpoint}...")
    tc_dirs, tc_densities = get_tc_directions_alive(tc_checkpoint, base_model, batches)

    plot_umap(
        spd_dirs,
        tc_dirs,
        save_path=OUTPUT_DIR / "umap_spd_vs_tc_layer2.png",
    )

    print("\n--- Separate UMAPs ---")
    plot_umap_single(
        spd_dirs, spd_densities, label="SPD", color="#9467bd",
        save_path=OUTPUT_DIR / "umap_spd_only_layer2.png",
    )
    plot_umap_single(
        tc_dirs, tc_densities, label="TC", color="#1f77b4",
        save_path=OUTPUT_DIR / "umap_tc_only_layer2.png",
    )


if __name__ == "__main__":
    main()
