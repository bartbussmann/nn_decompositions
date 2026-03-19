"""Investigate how SPD components interact with residual stream SAE features.

For each SPD MLP component (c_fc + down_proj per layer), we compute:
1. Correlation between SPD component CI and SAE feature activations in the
   residual stream before and after the MLP (i.e., layer input and output)
2. "Mapping sparsity": how many SAE features does each SPD component map
   between? Ideally a component maps a small set of input features to a
   small set of output features.

Uses res_stream_local SAEs (k=32) from sae_sweep_jose3.

Usage:
    python experiments/exp_042_spd_sae_interaction/spd_sae_interaction.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
SAE_WANDB_PROJECT = "mats-sprint/sae_sweep_jose3"
SAE_K = 32  # Use k=32 SAEs
OUTPUT_DIR = Path("experiments/exp_042_spd_sae_interaction/output")

BATCH_SIZE = 8
SEQ_LEN = 512
N_BATCHES = 50  # ~200k tokens


# =============================================================================
# Data & model loading
# =============================================================================


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


def load_sae(checkpoint_dir: Path) -> BatchTopKTranscoder:
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    sae = BatchTopKTranscoder(cfg)
    sae.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    sae.eval()
    return sae


def download_resid_stream_saes() -> dict[int, Path]:
    """Download res_stream_local SAEs. Returns {layer: path}."""
    api = wandb.Api()
    runs = api.runs(SAE_WANDB_PROJECT)

    sae_paths = {}
    for run in runs:
        if run.state != "finished":
            continue
        if run.name != f"sae_resid_stream_local_k{SAE_K}":
            continue

        arts = [a for a in run.logged_artifacts() if a.type == "model"]
        for art in arts:
            art_base = art.name.split(":")[0]
            for layer in LAYERS:
                if f"layer{layer}_final" in art_base:
                    dest = Path(f"checkpoints/jose_sae_resid_local_k{SAE_K}_layer{layer}")
                    if not (dest / "encoder.pt").exists():
                        art.download(root=str(dest))
                        print(f"  Downloaded {art.name} -> {dest}")
                    else:
                        print(f"  Cached {dest}")
                    sae_paths[layer] = dest
                    break

    assert set(sae_paths.keys()) == set(LAYERS), f"Missing SAE layers: got {set(sae_paths.keys())}"
    return sae_paths


# =============================================================================
# Collect co-activations
# =============================================================================


@torch.no_grad()
def collect_spd_sae_coactivations(
    spd_model, saes: dict[int, BatchTopKTranscoder], batches,
) -> dict:
    """For each MLP layer, collect co-activation statistics between SPD components
    and SAE features in the residual stream before and after the MLP.

    Returns per-layer stats:
      {layer: {
          "corr_input": (n_spd_components, n_sae_features) correlation matrix,
          "corr_output": (n_spd_components, n_sae_features) correlation matrix,
          "spd_c": int,
          "sae_dict_size": int,
      }}
    """
    base_model = spd_model.target_model
    results = {}

    for layer_idx in LAYERS:
        cfc_name = f"h.{layer_idx}.mlp.c_fc"
        down_name = f"h.{layer_idx}.mlp.down_proj"
        n_cfc = spd_model.module_to_c[cfc_name]
        n_down = spd_model.module_to_c[down_name]
        sae_dict = saes[layer_idx].cfg.dict_size

        # Accumulators for Pearson correlation
        # We'll track sum(x), sum(y), sum(xy), sum(x^2), sum(y^2), n
        # for SPD CI (c_fc) vs SAE features at resid_stream[layer] (input to MLP)
        # and SPD CI (c_fc) vs SAE features at resid_stream[layer+1] (after MLP, but
        # that's actually resid_stream[layer] since the block output includes the MLP)

        # Actually: resid_stream SAE at layer L reconstructs model.h[L] output,
        # which is the residual stream AFTER layer L's MLP + residual connection.
        # So: SAE[layer-1] = input to layer's MLP, SAE[layer] = output of layer's MLP
        # For layer 0, there's no SAE[-1], so we skip input correlation for layer 0.

        # For simplicity, let's correlate SPD CI with SAE features at the SAME layer
        # (residual stream after the MLP), which captures what SPD writes to.
        # And with SAE features at layer-1 (residual stream before the MLP),
        # which captures what SPD reads from.

        print(f"\nLayer {layer_idx}: c_fc C={n_cfc}, down_proj C={n_down}, SAE dict={sae_dict}")

        # Collect all CI values and SAE activations across batches
        all_ci_cfc = []     # (n_tokens, n_cfc)
        all_ci_down = []    # (n_tokens, n_down)
        all_sae_output = [] # (n_tokens, sae_dict) - SAE at layer_idx
        all_sae_input = []  # (n_tokens, sae_dict) - SAE at layer_idx-1 (if exists)

        for input_ids in tqdm(batches, desc=f"Layer {layer_idx}"):
            # SPD forward to get CI
            out = spd_model(input_ids, cache_type="input")
            ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

            ci_cfc = ci.lower_leaky[cfc_name]  # (B, S, n_cfc)
            ci_down = ci.lower_leaky[down_name]  # (B, S, n_down)

            # Clean forward through base model to get residual stream activations
            captured = {}
            hooks = []
            for l in LAYERS:
                def _make_hook(li):
                    def _hook(_mod, _inp, out):
                        captured[li] = out.detach()
                    return _hook
                hooks.append(base_model.h[l].register_forward_hook(_make_hook(l)))
            base_model(input_ids)
            for h in hooks:
                h.remove()

            # SAE activations at layer output
            resid_out = captured[layer_idx].reshape(-1, 768)
            sae_acts_out = saes[layer_idx].encode(resid_out)

            all_ci_cfc.append(ci_cfc.reshape(-1, n_cfc))
            all_ci_down.append(ci_down.reshape(-1, n_down))
            all_sae_output.append(sae_acts_out)

            # SAE activations at layer input (previous layer's residual stream)
            if layer_idx > 0:
                resid_in = captured[layer_idx - 1].reshape(-1, 768)
                sae_acts_in = saes[layer_idx - 1].encode(resid_in)
                all_sae_input.append(sae_acts_in)

        all_ci_cfc = torch.cat(all_ci_cfc, dim=0)  # (N, n_cfc)
        all_ci_down = torch.cat(all_ci_down, dim=0)  # (N, n_down)
        all_sae_output = torch.cat(all_sae_output, dim=0)  # (N, sae_dict)

        # Compute correlation: SPD c_fc CI vs output SAE features
        # Use just the top-100 most active SPD components to keep it manageable
        cfc_activity = (all_ci_cfc > 0).float().mean(dim=0)  # firing rate per component
        top_cfc = cfc_activity.topk(min(100, n_cfc)).indices

        down_activity = (all_ci_down > 0).float().mean(dim=0)
        top_down = down_activity.topk(min(100, n_down)).indices

        sae_activity = (all_sae_output > 0).float().mean(dim=0)
        top_sae_out = sae_activity.topk(min(200, sae_dict)).indices

        # Correlation between top SPD c_fc components and top SAE output features
        ci_subset = all_ci_cfc[:, top_cfc].float()  # (N, 100)
        sae_subset = all_sae_output[:, top_sae_out].float()  # (N, 200)

        # Pearson correlation
        ci_centered = ci_subset - ci_subset.mean(dim=0, keepdim=True)
        sae_centered = sae_subset - sae_subset.mean(dim=0, keepdim=True)
        ci_std = ci_centered.pow(2).sum(dim=0).sqrt().clamp(min=1e-8)
        sae_std = sae_centered.pow(2).sum(dim=0).sqrt().clamp(min=1e-8)

        corr_output = (ci_centered.T @ sae_centered) / (ci_std.unsqueeze(1) * sae_std.unsqueeze(0))
        # corr_output: (100, 200) - correlation between each SPD component and SAE feature

        # Input SAE correlation
        corr_input = None
        if layer_idx > 0:
            all_sae_input = torch.cat(all_sae_input, dim=0)
            sae_in_activity = (all_sae_input > 0).float().mean(dim=0)
            top_sae_in = sae_in_activity.topk(min(200, sae_dict)).indices

            sae_in_subset = all_sae_input[:, top_sae_in].float()
            sae_in_centered = sae_in_subset - sae_in_subset.mean(dim=0, keepdim=True)
            sae_in_std = sae_in_centered.pow(2).sum(dim=0).sqrt().clamp(min=1e-8)
            corr_input = (ci_centered.T @ sae_in_centered) / (ci_std.unsqueeze(1) * sae_in_std.unsqueeze(0))

        # Per-component stats: how many SAE features does each SPD component correlate with?
        n_corr_gt_03_output = (corr_output.abs() > 0.3).sum(dim=1).float()  # per SPD component
        n_corr_gt_05_output = (corr_output.abs() > 0.5).sum(dim=1).float()

        results[layer_idx] = {
            "corr_output": corr_output.cpu(),
            "corr_input": corr_input.cpu() if corr_input is not None else None,
            "n_spd_cfc": n_cfc,
            "n_spd_down": n_down,
            "sae_dict_size": sae_dict,
            "top_cfc_indices": top_cfc.cpu(),
            "top_sae_out_indices": top_sae_out.cpu(),
            "n_corr_gt_03_output": n_corr_gt_03_output.cpu(),
            "n_corr_gt_05_output": n_corr_gt_05_output.cpu(),
            "max_corr_per_spd": corr_output.abs().max(dim=1).values.cpu(),
        }

        print(f"  Correlation stats (SPD c_fc vs output SAE):")
        print(f"    Max abs corr: {corr_output.abs().max():.3f}")
        print(f"    Mean max-per-component: {corr_output.abs().max(dim=1).values.mean():.3f}")
        print(f"    Median SAE features with |corr|>0.3 per component: {n_corr_gt_03_output.median():.0f}")
        print(f"    Median SAE features with |corr|>0.5 per component: {n_corr_gt_05_output.median():.0f}")

        if corr_input is not None:
            n_corr_gt_03_input = (corr_input.abs() > 0.3).sum(dim=1).float()
            print(f"  Correlation stats (SPD c_fc vs input SAE):")
            print(f"    Max abs corr: {corr_input.abs().max():.3f}")
            print(f"    Mean max-per-component: {corr_input.abs().max(dim=1).values.mean():.3f}")
            print(f"    Median SAE features with |corr|>0.3 per component: {n_corr_gt_03_input.median():.0f}")

        del all_ci_cfc, all_ci_down, all_sae_output
        if layer_idx > 0:
            del all_sae_input
        torch.cuda.empty_cache()

    return results


# =============================================================================
# Plotting
# =============================================================================


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_correlation_histograms(results: dict, save_path: Path):
    """Histogram of max |correlation| per SPD component across layers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for layer_idx, ax in zip(LAYERS, axes.flat):
        r = results[layer_idx]
        max_corrs = r["max_corr_per_spd"].numpy()
        ax.hist(max_corrs, bins=50, color="#2b6cb0", edgecolor="white", linewidth=0.5, alpha=0.8)
        ax.axvline(0.3, color="#d62728", linestyle="--", linewidth=1, alpha=0.7, label="|corr|=0.3")
        ax.axvline(0.5, color="#2f855a", linestyle="--", linewidth=1, alpha=0.7, label="|corr|=0.5")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Max |correlation| with any SAE feature")
        ax.set_ylabel("SPD components")
        if layer_idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Max SPD-SAE correlation per SPD c_fc component", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_mapping_sparsity(results: dict, save_path: Path):
    """Histogram of # SAE features with |corr|>0.3 per SPD component."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for layer_idx, ax in zip(LAYERS, axes.flat):
        r = results[layer_idx]
        n_corr = r["n_corr_gt_03_output"].numpy()
        ax.hist(n_corr, bins=range(0, int(n_corr.max()) + 2), color="#c05621",
                edgecolor="white", linewidth=0.5, alpha=0.8)
        ax.set_title(f"Layer {layer_idx} (median={np.median(n_corr):.0f})")
        ax.set_xlabel("# SAE features with |corr| > 0.3")
        ax.set_ylabel("SPD components")

    fig.suptitle("Mapping sparsity: SAE features per SPD component", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_top_correlation_heatmap(results: dict, save_path: Path):
    """Heatmap of top-20 SPD components vs top-50 SAE features for each layer."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for layer_idx, ax in zip(LAYERS, axes.flat):
        r = results[layer_idx]
        corr = r["corr_output"].numpy()

        # Top 20 SPD components by max correlation
        max_per_spd = np.abs(corr).max(axis=1)
        top_spd = np.argsort(max_per_spd)[-20:][::-1]

        # Top 50 SAE features by max correlation with any of those SPD components
        sub_corr = corr[top_spd, :]
        max_per_sae = np.abs(sub_corr).max(axis=0)
        top_sae = np.argsort(max_per_sae)[-50:][::-1]

        heatmap = corr[np.ix_(top_spd, top_sae)]
        im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("SAE features (top 50)")
        ax.set_ylabel("SPD c_fc components (top 20)")
        ax.set_xticks([])
        ax.set_yticks(range(len(top_spd)))
        ax.set_yticklabels([str(r["top_cfc_indices"][i].item()) for i in top_spd], fontsize=7)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Pearson correlation")
    fig.suptitle("SPD component — SAE feature correlation (top components)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    from analysis.collect_spd_activations import load_spd_model

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading jose SPD model...")
    spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
    spd_model.to(DEVICE)

    print("\nDownloading residual stream SAEs...")
    sae_paths = download_resid_stream_saes()
    saes = {layer: load_sae(sae_paths[layer]) for layer in LAYERS}
    print(f"Loaded SAEs for layers {list(saes.keys())}")

    print(f"\nLoading {N_BATCHES} eval batches...")
    batches = get_eval_batches(N_BATCHES)

    print("\nCollecting SPD-SAE co-activations...")
    results = collect_spd_sae_coactivations(spd_model, saes, batches)

    # Plots
    plot_correlation_histograms(results, OUTPUT_DIR / "spd_sae_max_correlation.png")
    plot_mapping_sparsity(results, OUTPUT_DIR / "spd_sae_mapping_sparsity.png")
    plot_top_correlation_heatmap(results, OUTPUT_DIR / "spd_sae_correlation_heatmap.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
