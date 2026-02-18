"""Mechanistic Faithfulness (Jacobian Correlation) for SPD and Transcoder.

Compares Jacobians of the original MLP, transcoder, and SPD replacement layers
at layer 3 of LlamaSimpleMLP. High cosine similarity = the replacement responds
to input perturbations the same way as the original.

Usage:
    python experiments/faithfulness_analysis.py
    python experiments/faithfulness_analysis.py --n_samples 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from base import BatchTopK, TopK

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 3


# =============================================================================
# Model loading
# =============================================================================


def load_transcoder(checkpoint_dir: str):
    from config import EncoderConfig

    ENCODER_CLASSES = {
        "vanilla": __import__("base", fromlist=["Vanilla"]).Vanilla,
        "topk": TopK,
        "batchtopk": BatchTopK,
        "jumprelu": __import__("base", fromlist=["JumpReLUEncoder"]).JumpReLUEncoder,
    }

    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg = EncoderConfig(**cfg_dict)
    encoder = ENCODER_CLASSES[cfg.encoder_type](cfg)
    encoder.load_state_dict(
        torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE)
    )
    encoder.eval()
    return encoder


def load_spd_model(wandb_path: str):
    from analysis.collect_spd_activations import load_spd_model as _load

    return _load(wandb_path)


# =============================================================================
# Data loading
# =============================================================================


def load_eval_tokens(n_sequences: int, seq_len: int) -> torch.Tensor:
    dataset = load_dataset(
        "danbraunai/pile-uncopyrighted-tok", split="train", streaming=True
    )
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)

    rows = []
    for _ in tqdm(range(n_sequences), desc="Loading tokenized data"):
        sample = next(data_iter)
        ids = sample["input_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        rows.append(ids[:seq_len])
    return torch.stack(rows)


# =============================================================================
# MLP input collection
# =============================================================================


@torch.no_grad()
def collect_mlp_inputs(
    base_model,
    input_ids: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Collect post-rms_2 activations (MLP inputs) for all tokens."""
    all_inputs = []

    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i : i + batch_size].to(DEVICE)
        captured = {}

        def _hook(mod, inp, out):
            captured["mlp_input"] = out.detach()

        hook = base_model.h[LAYER].rms_2.register_forward_hook(_hook)
        base_model(batch)
        hook.remove()

        mlp_in = captured["mlp_input"].reshape(-1, captured["mlp_input"].shape[-1])
        all_inputs.append(mlp_in.cpu())

    return torch.cat(all_inputs, dim=0)


# =============================================================================
# Layer functions for Jacobian computation
# =============================================================================


def make_original_mlp_fn(base_model):
    """Return fn(x) -> mlp(x) for the original MLP."""
    mlp = base_model.h[LAYER].mlp

    def fn(x):
        return mlp(x)

    return fn


def make_transcoder_fn(transcoder):
    """Return fn(x) -> transcoder reconstruction with top-k sparsity."""
    k = transcoder.cfg.top_k

    def fn(x):
        use_pre_enc_bias = (
            transcoder.cfg.pre_enc_bias
            and transcoder.input_size == transcoder.output_size
        )
        x_enc = x - transcoder.b_dec if use_pre_enc_bias else x

        if isinstance(transcoder, (TopK, BatchTopK)):
            acts = F.relu(x_enc @ transcoder.W_enc)
        else:
            acts = F.relu(x_enc @ transcoder.W_enc + transcoder.b_enc)

        if k < acts.shape[-1]:
            topk = torch.topk(acts, k, dim=-1)
            acts = torch.zeros_like(acts).scatter(-1, topk.indices, topk.values)

        return acts @ transcoder.W_dec + transcoder.b_dec

    return fn


def make_spd_mlp_fn(spd_model, base_model):
    """Return fn(x) -> SPD replacement MLP output (all components active)."""
    cfc_name = f"h.{LAYER}.mlp.c_fc"
    down_name = f"h.{LAYER}.mlp.down_proj"
    gelu = base_model.h[LAYER].mlp.gelu

    cfc_comp = spd_model.components[cfc_name]
    down_comp = spd_model.components[down_name]

    def fn(x):
        # c_fc replacement: x @ V @ U + bias
        h = x @ cfc_comp.V
        h = h @ cfc_comp.U
        if cfc_comp.bias is not None:
            h = h + cfc_comp.bias

        h = gelu(h)

        # down_proj replacement: h @ V @ U + bias
        out = h @ down_comp.V
        out = out @ down_comp.U
        if down_comp.bias is not None:
            out = out + down_comp.bias

        return out

    return fn


# =============================================================================
# Jacobian correlation
# =============================================================================


def compute_jacobian_correlations(
    fn_original,
    fn_replacement,
    mlp_inputs: torch.Tensor,
    n_samples: int,
    label: str = "",
) -> list[float]:
    """Compute per-sample Jacobian cosine similarities between two functions."""
    n_total = mlp_inputs.shape[0]
    assert n_samples <= n_total, f"n_samples={n_samples} > available tokens={n_total}"

    indices = torch.linspace(0, n_total - 1, n_samples).long()
    scores = []

    for idx in tqdm(indices, desc=f"Jacobians ({label})"):
        x = mlp_inputs[idx].to(DEVICE).float()

        J_orig = torch.autograd.functional.jacobian(fn_original, x, vectorize=True)
        J_repl = torch.autograd.functional.jacobian(fn_replacement, x, vectorize=True)

        sim = F.cosine_similarity(
            J_orig.reshape(1, -1), J_repl.reshape(1, -1)
        ).item()
        scores.append(sim)

    return scores


# =============================================================================
# Plotting
# =============================================================================


def plot_faithfulness(tc_scores: list[float], spd_scores: list[float], save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart with SE error bars
    ax = axes[0]
    methods = ["Transcoder", "SPD"]
    means = [np.mean(tc_scores), np.mean(spd_scores)]
    ses = [
        np.std(tc_scores) / np.sqrt(len(tc_scores)),
        np.std(spd_scores) / np.sqrt(len(spd_scores)),
    ]
    colors = ["tab:blue", "tab:orange"]

    bars = ax.bar(
        methods, means, yerr=ses, color=colors,
        edgecolor="black", linewidth=0.8, capsize=5,
        error_kw={"linewidth": 1.5, "capthick": 1.5},
    )
    for bar, mean, se in zip(bars, means, ses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + se + 0.01,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Jacobian Cosine Similarity")
    ax.set_title("Mechanistic Faithfulness\n(LlamaSimpleMLP Layer 3 MLP)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Histogram of per-token scores
    ax = axes[1]
    ax.hist(tc_scores, bins=30, alpha=0.6, color="tab:blue", label="Transcoder", density=True)
    ax.hist(spd_scores, bins=30, alpha=0.6, color="tab:orange", label="SPD", density=True)
    ax.axvline(np.mean(tc_scores), color="tab:blue", linestyle="--", linewidth=2)
    ax.axvline(np.mean(spd_scores), color="tab:orange", linestyle="--", linewidth=2)
    ax.set_xlabel("Jacobian Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Per-Token Scores")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Jacobian Correlation: SPD vs Transcoder on Layer 3 MLP"
    )
    parser.add_argument(
        "--transcoder_path", type=str,
        default="checkpoints/4096_batchtopk_k24_0.0003_final",
    )
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument(
        "--n_samples", type=int, default=200,
        help="Number of tokens to compute Jacobians for",
    )
    parser.add_argument("--n_sequences", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--save_path", type=str,
        default="experiments/faithfulness_results.json",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    start_time = time.time()

    # Load SPD model (includes the base LlamaSimpleMLP model)
    print("Loading SPD model...")
    spd_model, raw_config = load_spd_model(args.spd_run)
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    # Load transcoder
    print("Loading transcoder...")
    transcoder = load_transcoder(args.transcoder_path)
    transcoder.to(DEVICE)

    cfc_name = f"h.{LAYER}.mlp.c_fc"
    down_name = f"h.{LAYER}.mlp.down_proj"
    print(f"  Transcoder: dict_size={transcoder.cfg.dict_size}, top_k={transcoder.cfg.top_k}")
    print(f"  SPD c_fc: {spd_model.components[cfc_name].C} components")
    print(f"  SPD down_proj: {spd_model.components[down_name].C} components")

    # Load eval data
    print(f"Loading {args.n_sequences} sequences (seq_len={args.seq_len})...")
    tokenized_data = load_eval_tokens(args.n_sequences, args.seq_len)

    # Collect MLP inputs
    print("Collecting MLP inputs (post-rms_2)...")
    mlp_inputs = collect_mlp_inputs(base_model, tokenized_data, args.batch_size)
    n_tokens = mlp_inputs.shape[0]
    d_model = mlp_inputs.shape[1]
    print(f"  {n_tokens} tokens, d_model={d_model}")

    n_samples = min(args.n_samples, n_tokens)

    # Build layer functions
    fn_orig = make_original_mlp_fn(base_model)
    fn_tc = make_transcoder_fn(transcoder)
    fn_spd = make_spd_mlp_fn(spd_model, base_model)

    # Compute Jacobian correlations
    print(f"\nComputing Jacobian correlations ({n_samples} samples)...")
    print(f"  Jacobian size: ({d_model}, {d_model}) = {d_model**2:,} elements per sample")

    tc_scores = compute_jacobian_correlations(
        fn_orig, fn_tc, mlp_inputs, n_samples, label="Transcoder"
    )

    spd_scores = compute_jacobian_correlations(
        fn_orig, fn_spd, mlp_inputs, n_samples, label="SPD"
    )

    # Summarize
    tc_mean, tc_std = np.mean(tc_scores), np.std(tc_scores)
    tc_se = tc_std / np.sqrt(len(tc_scores))

    spd_mean, spd_std = np.mean(spd_scores), np.std(spd_scores)
    spd_se = spd_std / np.sqrt(len(spd_scores))

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("RESULTS: Jacobian Correlation (Mechanistic Faithfulness)")
    print(f"{'='*60}")
    print(f"  Transcoder:  {tc_mean:.4f} +/- {tc_se:.4f}  (std={tc_std:.4f})")
    print(f"  SPD:         {spd_mean:.4f} +/- {spd_se:.4f}  (std={spd_std:.4f})")
    print(f"  N samples:   {n_samples}")
    print(f"  Time:        {elapsed:.1f}s")

    # Save
    results = {
        "config": {
            "transcoder_path": args.transcoder_path,
            "spd_run": args.spd_run,
            "n_samples": n_samples,
            "n_sequences": args.n_sequences,
            "seq_len": args.seq_len,
            "layer": LAYER,
        },
        "transcoder": {
            "mean": float(tc_mean),
            "std": float(tc_std),
            "se": float(tc_se),
            "scores": tc_scores,
        },
        "spd": {
            "mean": float(spd_mean),
            "std": float(spd_std),
            "se": float(spd_se),
            "scores": spd_scores,
        },
    }

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Plot
    plot_path = str(save_path).replace(".json", ".png")
    plot_faithfulness(tc_scores, spd_scores, plot_path)


if __name__ == "__main__":
    main()
