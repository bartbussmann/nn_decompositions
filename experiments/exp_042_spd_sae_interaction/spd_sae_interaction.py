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

        # Streaming correlation: accumulate sufficient statistics on CPU
        # For Pearson corr we need: sum_x, sum_y, sum_xy, sum_x2, sum_y2, n
        # We compute for ALL c_fc components vs ALL SAE output features,
        # then select top ones after.
        n_spd = n_cfc
        n_sae = sae_dict

        sum_ci = torch.zeros(n_spd, device="cpu", dtype=torch.float64)
        sum_sae_out = torch.zeros(n_sae, device="cpu", dtype=torch.float64)
        sum_ci2 = torch.zeros(n_spd, device="cpu", dtype=torch.float64)
        sum_sae_out2 = torch.zeros(n_sae, device="cpu", dtype=torch.float64)
        sum_ci_sae_out = torch.zeros(n_spd, n_sae, device="cpu", dtype=torch.float64)

        has_input = layer_idx > 0
        if has_input:
            sum_sae_in = torch.zeros(n_sae, device="cpu", dtype=torch.float64)
            sum_sae_in2 = torch.zeros(n_sae, device="cpu", dtype=torch.float64)
            sum_ci_sae_in = torch.zeros(n_spd, n_sae, device="cpu", dtype=torch.float64)

        n_total = 0

        for input_ids in tqdm(batches, desc=f"Layer {layer_idx}"):
            # SPD forward to get CI
            out = spd_model(input_ids, cache_type="input")
            ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
            ci_cfc = ci.lower_leaky[cfc_name].reshape(-1, n_cfc).float()  # (B*S, n_spd)

            # Clean forward for residual stream
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

            # SAE output activations
            resid_out = captured[layer_idx].reshape(-1, 768)
            sae_out = saes[layer_idx].encode(resid_out).float()  # (B*S, n_sae)

            # Move to CPU and accumulate
            ci_cpu = ci_cfc.cpu().double()
            sae_out_cpu = sae_out.cpu().double()
            bs = ci_cpu.shape[0]
            n_total += bs

            sum_ci += ci_cpu.sum(dim=0)
            sum_sae_out += sae_out_cpu.sum(dim=0)
            sum_ci2 += ci_cpu.pow(2).sum(dim=0)
            sum_sae_out2 += sae_out_cpu.pow(2).sum(dim=0)
            sum_ci_sae_out += ci_cpu.T @ sae_out_cpu

            if has_input:
                resid_in = captured[layer_idx - 1].reshape(-1, 768)
                sae_in = saes[layer_idx - 1].encode(resid_in).float()
                sae_in_cpu = sae_in.cpu().double()
                sum_sae_in += sae_in_cpu.sum(dim=0)
                sum_sae_in2 += sae_in_cpu.pow(2).sum(dim=0)
                sum_ci_sae_in += ci_cpu.T @ sae_in_cpu

        # Compute Pearson correlation from sufficient statistics
        mean_ci = sum_ci / n_total
        mean_sae_out = sum_sae_out / n_total
        std_ci = ((sum_ci2 / n_total - mean_ci.pow(2)).clamp(min=1e-16)).sqrt()
        std_sae_out = ((sum_sae_out2 / n_total - mean_sae_out.pow(2)).clamp(min=1e-16)).sqrt()
        cov_out = sum_ci_sae_out / n_total - mean_ci.unsqueeze(1) * mean_sae_out.unsqueeze(0)
        corr_output = (cov_out / (std_ci.unsqueeze(1) * std_sae_out.unsqueeze(0)).clamp(min=1e-16)).float()

        corr_input = None
        if has_input:
            mean_sae_in = sum_sae_in / n_total
            std_sae_in = ((sum_sae_in2 / n_total - mean_sae_in.pow(2)).clamp(min=1e-16)).sqrt()
            cov_in = sum_ci_sae_in / n_total - mean_ci.unsqueeze(1) * mean_sae_in.unsqueeze(0)
            corr_input = (cov_in / (std_ci.unsqueeze(1) * std_sae_in.unsqueeze(0)).clamp(min=1e-16)).float()

        # Select top components/features for detailed analysis
        cfc_activity = std_ci.float()  # proxy for activity
        top_cfc = cfc_activity.topk(min(100, n_spd)).indices
        sae_out_activity = std_sae_out.float()
        top_sae_out = sae_out_activity.topk(min(200, n_sae)).indices

        corr_output_sub = corr_output[top_cfc][:, top_sae_out]

        n_corr_gt_03_output = (corr_output_sub.abs() > 0.3).sum(dim=1).float()
        n_corr_gt_05_output = (corr_output_sub.abs() > 0.5).sum(dim=1).float()

        results[layer_idx] = {
            "corr_output": corr_output_sub,
            "corr_input": corr_input[top_cfc][:, top_sae_out] if corr_input is not None else None,
            "n_spd_cfc": n_cfc,
            "n_spd_down": n_down,
            "sae_dict_size": sae_dict,
            "top_cfc_indices": top_cfc,
            "top_sae_out_indices": top_sae_out,
            "n_corr_gt_03_output": n_corr_gt_03_output,
            "n_corr_gt_05_output": n_corr_gt_05_output,
            "max_corr_per_spd": corr_output_sub.abs().max(dim=1).values,
        }

        print(f"  Correlation stats (SPD c_fc vs output SAE, top 100 x 200):")
        print(f"    Max abs corr: {corr_output_sub.abs().max():.3f}")
        print(f"    Mean max-per-component: {corr_output_sub.abs().max(dim=1).values.mean():.3f}")
        print(f"    Median SAE features with |corr|>0.3 per component: {n_corr_gt_03_output.median():.0f}")
        print(f"    Median SAE features with |corr|>0.5 per component: {n_corr_gt_05_output.median():.0f}")

        if corr_input is not None:
            corr_input_sub = corr_input[top_cfc][:, top_sae_out]
            n_corr_gt_03_input = (corr_input_sub.abs() > 0.3).sum(dim=1).float()
            print(f"  Correlation stats (SPD c_fc vs input SAE, top 100 x 200):")
            print(f"    Max abs corr: {corr_input_sub.abs().max():.3f}")
            print(f"    Mean max-per-component: {corr_input_sub.abs().max(dim=1).values.mean():.3f}")
            print(f"    Median SAE features with |corr|>0.3 per component: {n_corr_gt_03_input.median():.0f}")

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
