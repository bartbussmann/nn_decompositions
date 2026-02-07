"""Compare L0 vs CE-loss Pareto curves for SPD and Transcoder decompositions.

For a given L0 = k:
- Transcoder: keep the k features with highest activations per token
- SPD: keep the k components with highest pre-sigmoid CI score per token,
  evaluating each weight matrix (c_fc, c_proj) separately

Usage:
    python experiments/pareto_comparison.py \
        --transcoder_path checkpoints/my_transcoder/ \
        --spd_path /path/to/spd/checkpoint/ \
        --l0_values 1 2 3 5 10 20 50 100 200 500
"""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import BatchTopK, JumpReLUEncoder, TopK, Vanilla
from config import EncoderConfig

from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 8


# =============================================================================
# Data loading
# =============================================================================


def get_eval_batches(
    tokenizer, n_batches: int, batch_size: int, seq_len: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load evaluation batches from OpenWebText."""
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)

    batches = []
    for _ in range(n_batches):
        texts = []
        for _ in range(batch_size):
            sample = next(data_iter)
            texts.append(sample["text"])

        tokens = tokenizer(
            texts,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(DEVICE)
        attention_mask = tokens["attention_mask"].to(DEVICE)
        batches.append((input_ids, attention_mask))

    return batches


# =============================================================================
# Helpers
# =============================================================================


@contextmanager
def patched_forward(module: nn.Module, patched_fn):
    """Temporarily replace a module's forward method."""
    original = module.forward
    module.forward = patched_fn
    try:
        yield
    finally:
        module.forward = original


def compute_ce_loss(model, tokenizer, input_ids, attention_mask):
    """Compute CE loss for a causal LM."""
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100
    return model(input_ids, attention_mask=attention_mask, labels=labels).loss.item()


# =============================================================================
# Transcoder
# =============================================================================

ENCODER_CLASSES = {
    "vanilla": Vanilla,
    "topk": TopK,
    "batchtopk": BatchTopK,
    "jumprelu": JumpReLUEncoder,
}


def load_transcoder(checkpoint_dir: str) -> TopK | BatchTopK | Vanilla | JumpReLUEncoder:
    """Load a transcoder from a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)

    # Parse dtype back from string
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))

    cfg = EncoderConfig(**cfg_dict)
    encoder = ENCODER_CLASSES[cfg.encoder_type](cfg)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    encoder.eval()
    return encoder


def transcoder_topk_reconstruction(transcoder, x_in: torch.Tensor, k: int) -> torch.Tensor:
    """Reconstruct using only top-k features by activation magnitude."""
    use_pre_enc_bias = transcoder.cfg.pre_enc_bias and transcoder.input_size == transcoder.output_size
    x_enc = x_in - transcoder.b_dec if use_pre_enc_bias else x_in

    # TopK/BatchTopK don't use b_enc; Vanilla/JumpReLU do
    if isinstance(transcoder, (TopK, BatchTopK)):
        acts = F.relu(x_enc @ transcoder.W_enc)
    else:
        acts = F.relu(x_enc @ transcoder.W_enc + transcoder.b_enc)

    # Select top-k per token
    if k < acts.shape[-1]:
        topk = torch.topk(acts, k, dim=-1)
        acts = torch.zeros_like(acts).scatter(-1, topk.indices, topk.values)

    return acts @ transcoder.W_dec + transcoder.b_dec


@torch.no_grad()
def eval_transcoder_ce(
    model: GPT2LMHeadModel,
    tokenizer,
    transcoder,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    k: int,
) -> float:
    """Evaluate CE loss with transcoder using top-k features."""
    mlp = model.transformer.h[LAYER].mlp
    input_size = transcoder.cfg.input_size

    total_loss = 0.0
    for input_ids, attention_mask in batches:

        def _patched(hidden_states):
            flat = hidden_states.reshape(-1, input_size)
            recon = transcoder_topk_reconstruction(transcoder, flat, k)
            return recon.reshape(hidden_states.shape)

        with patched_forward(mlp, _patched):
            total_loss += compute_ce_loss(model, tokenizer, input_ids, attention_mask)

    return total_loss / len(batches)


# =============================================================================
# SPD
# =============================================================================


def load_spd(checkpoint_path: str) -> ComponentModel:
    """Load a trained SPD model from a run directory or .pth file."""
    path = Path(checkpoint_path)
    if path.is_dir():
        # Resolve the latest model checkpoint in the directory
        pth_files = sorted(path.glob("model_*.pth"))
        assert pth_files, f"No model_*.pth files found in {path}"
        path = pth_files[-1]
    return ComponentModel.from_pretrained(str(path))


@torch.no_grad()
def eval_spd_ce(
    spd_model: ComponentModel,
    tokenizer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    module_name: str,
    k: int,
) -> float:
    """Evaluate CE loss with SPD using top-k components for one module.

    Only the specified module is decomposed; the other uses original weights.
    """
    total_loss = 0.0
    for input_ids, attention_mask in batches:
        # Get input activations via caching forward pass
        out = spd_model(input_ids, attention_mask=attention_mask, cache_type="input")
        pre_weight_acts = out.cache

        # Compute CI scores
        ci = spd_model.calc_causal_importances(pre_weight_acts, sampling="continuous")
        ci_scores = ci.pre_sigmoid[module_name]  # (batch, seq_len, C)

        # Top-k mask
        topk_indices = torch.topk(ci_scores, k, dim=-1).indices
        mask = torch.zeros_like(ci_scores)
        mask.scatter_(-1, topk_indices, 1.0)

        # Forward with mask on only the target module
        mask_infos = make_mask_infos({module_name: mask})
        logits = spd_model(input_ids, attention_mask=attention_mask, mask_infos=mask_infos)

        # CE loss (next-token prediction)
        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total_loss += loss.item()

    return total_loss / len(batches)


# =============================================================================
# Baselines
# =============================================================================


@torch.no_grad()
def eval_baselines(
    model: GPT2LMHeadModel,
    tokenizer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Compute baseline CE losses: original and zero-ablation."""
    mlp = model.transformer.h[LAYER].mlp
    original_forward = mlp.forward

    original_loss = 0.0
    zero_loss = 0.0
    for input_ids, attention_mask in batches:
        original_loss += compute_ce_loss(model, tokenizer, input_ids, attention_mask)

        def _zero(hidden_states):
            return torch.zeros_like(original_forward(hidden_states))

        with patched_forward(mlp, _zero):
            zero_loss += compute_ce_loss(model, tokenizer, input_ids, attention_mask)

    n = len(batches)
    return {
        "original_ce": original_loss / n,
        "zero_ablation_ce": zero_loss / n,
    }


# =============================================================================
# Plotting
# =============================================================================


def plot_pareto(
    l0_values: list[int],
    transcoder_ces: list[float],
    spd_cfc_ces: list[float],
    spd_cproj_ces: list[float],
    baselines: dict[str, float],
    save_path: str,
):
    """Plot L0 vs CE-loss Pareto curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(l0_values, transcoder_ces, "o-", color="tab:blue", label="Transcoder")
    ax.plot(l0_values, spd_cfc_ces, "s-", color="tab:orange", label="SPD (c_fc)")
    ax.plot(l0_values, spd_cproj_ces, "^-", color="tab:green", label="SPD (c_proj)")

    ax.axhline(baselines["original_ce"], color="gray", linestyle="--", alpha=0.7, label="Original CE")
    ax.axhline(
        baselines["zero_ablation_ce"],
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Zero-ablation CE",
    )

    ax.set_xlabel("L0 (number of active components)")
    ax.set_ylabel("CE Loss")
    ax.set_title("L0 vs CE Loss: SPD vs Transcoder (GPT-2 Layer 8 MLP)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="L0 vs CE Pareto comparison")
    parser.add_argument("--transcoder_path", type=str, required=True)
    parser.add_argument("--spd_path", type=str, required=True)
    parser.add_argument(
        "--l0_values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 10, 20, 50, 100, 200, 500],
    )
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--save_path", type=str, default="experiments/pareto_comparison.png")
    args = parser.parse_args()

    # Load GPT-2
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load decompositions
    print("Loading transcoder...")
    transcoder = load_transcoder(args.transcoder_path)

    print("Loading SPD model...")
    spd_model = load_spd(args.spd_path)

    # SPD module names
    cfc_name = f"transformer.h.{LAYER}.mlp.c_fc"
    cproj_name = f"transformer.h.{LAYER}.mlp.c_proj"
    assert cfc_name in spd_model.target_module_paths, (
        f"{cfc_name} not in SPD modules: {spd_model.target_module_paths}"
    )
    assert cproj_name in spd_model.target_module_paths, (
        f"{cproj_name} not in SPD modules: {spd_model.target_module_paths}"
    )

    # Load eval data
    print(f"Loading {args.n_eval_batches} eval batches...")
    batches = get_eval_batches(tokenizer, args.n_eval_batches, args.batch_size, args.seq_len)

    # SPD needs input_ids without attention_mask handled separately
    # (ComponentModel's target_model is its own GPT-2 copy)

    # Baselines
    print("Computing baselines...")
    baselines = eval_baselines(model, tokenizer, batches)
    print(f"  Original CE: {baselines['original_ce']:.4f}")
    print(f"  Zero-ablation CE: {baselines['zero_ablation_ce']:.4f}")

    # Evaluate at each L0
    transcoder_ces = []
    spd_cfc_ces = []
    spd_cproj_ces = []

    print(f"\n{'L0':>6} | {'Transcoder':>12} | {'SPD c_fc':>12} | {'SPD c_proj':>12}")
    print("-" * 55)

    for k in args.l0_values:
        tc_ce = eval_transcoder_ce(model, tokenizer, transcoder, batches, k)
        cfc_ce = eval_spd_ce(spd_model, tokenizer, batches, cfc_name, k)
        cproj_ce = eval_spd_ce(spd_model, tokenizer, batches, cproj_name, k)

        transcoder_ces.append(tc_ce)
        spd_cfc_ces.append(cfc_ce)
        spd_cproj_ces.append(cproj_ce)

        print(f"{k:>6} | {tc_ce:>12.4f} | {cfc_ce:>12.4f} | {cproj_ce:>12.4f}")

    # Plot
    plot_pareto(
        args.l0_values, transcoder_ces, spd_cfc_ces, spd_cproj_ces, baselines, args.save_path
    )


if __name__ == "__main__":
    main()
