"""Compare L0 vs CE-loss Pareto curves for SPD and Transcoder on LlamaSimpleMLP.

For a given L0 = k:
- Transcoder: keep the k features with highest activations per token
- SPD: keep the k components with highest pre-sigmoid CI score per token,
  evaluating each weight matrix (c_fc, down_proj) separately

Uses the same base model (wandb:goodfire/spd/t-32d1bb3b), dataset
(danbraunai/pile-uncopyrighted-tok), and tokenizer (EleutherAI/gpt-neox-20b)
that both models were trained on.

Usage:
    python experiments/pareto_pile.py
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from encoders import BatchTopK, JumpReLU, TopK, Vanilla
from config import EncoderConfig
from spd.models.components import make_mask_infos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 3


# =============================================================================
# Data loading (pre-tokenized Pile)
# =============================================================================


def get_eval_batches(
    n_batches: int, batch_size: int, seq_len: int
) -> list[torch.Tensor]:
    """Load pre-tokenized Pile batches."""
    dataset = load_dataset(
        "danbraunai/pile-uncopyrighted-tok", split="train", streaming=True
    )
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)

    batches = []
    for _ in tqdm(range(n_batches), desc="Loading batches"):
        batch_ids = []
        for _ in range(batch_size):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:seq_len])
        batches.append(torch.stack(batch_ids).to(DEVICE))
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


def compute_ce_loss(model, input_ids: torch.Tensor) -> float:
    """Compute CE loss for LlamaSimpleMLP (next-token prediction)."""
    logits, _ = model(input_ids)
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)
    ).item()


def compute_ce_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Compute CE loss from pre-computed logits."""
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)
    ).item()


# =============================================================================
# MLP activation collection (for MSE)
# =============================================================================


@torch.no_grad()
def get_mlp_activations(
    model,
    batches: list[torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Collect (mlp_input, mlp_output) pairs from the base model."""
    rms2 = model.h[LAYER].rms_2
    mlp = model.h[LAYER].mlp
    pairs = []

    for input_ids in batches:
        captured = {}

        def _capture_rms2(_mod, _inp, out):
            captured["mlp_in"] = out.detach()

        def _capture_mlp(_mod, _inp, out):
            captured["mlp_out"] = out.detach()

        h1 = rms2.register_forward_hook(_capture_rms2)
        h2 = mlp.register_forward_hook(_capture_mlp)
        model(input_ids)
        h1.remove()
        h2.remove()

        mlp_in = captured["mlp_in"].reshape(-1, captured["mlp_in"].shape[-1])
        mlp_out = captured["mlp_out"].reshape(-1, captured["mlp_out"].shape[-1])
        pairs.append((mlp_in, mlp_out))

    return pairs


# =============================================================================
# Transcoder
# =============================================================================

ENCODER_CLASSES = {
    "vanilla": Vanilla,
    "topk": TopK,
    "batchtopk": BatchTopK,
    "jumprelu": JumpReLU,
}


def load_transcoder(checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
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

    if isinstance(transcoder, (TopK, BatchTopK)):
        acts = F.relu(x_enc @ transcoder.W_enc)
    else:
        acts = F.relu(x_enc @ transcoder.W_enc + transcoder.b_enc)

    if k < acts.shape[-1]:
        topk = torch.topk(acts, k, dim=-1)
        acts = torch.zeros_like(acts).scatter(-1, topk.indices, topk.values)

    return acts @ transcoder.W_dec + transcoder.b_dec


def transcoder_batchtopk_reconstruction(transcoder, x_in: torch.Tensor, k: int) -> torch.Tensor:
    """Reconstruct using batch top-k."""
    use_pre_enc_bias = transcoder.cfg.pre_enc_bias and transcoder.input_size == transcoder.output_size
    x_enc = x_in - transcoder.b_dec if use_pre_enc_bias else x_in

    if isinstance(transcoder, (TopK, BatchTopK)):
        acts = F.relu(x_enc @ transcoder.W_enc)
    else:
        acts = F.relu(x_enc @ transcoder.W_enc + transcoder.b_enc)

    n_tokens = acts.shape[0]
    n_keep = k * n_tokens
    if n_keep < acts.numel():
        topk = torch.topk(acts.flatten(), n_keep, dim=-1)
        acts = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, topk.indices, topk.values)
            .reshape(acts.shape)
        )

    return acts @ transcoder.W_dec + transcoder.b_dec


@torch.no_grad()
def eval_transcoder_ce(
    base_model,
    transcoder,
    batches: list[torch.Tensor],
    k: int,
) -> float:
    """Evaluate CE loss with transcoder using top-k features."""
    mlp = base_model.h[LAYER].mlp
    input_size = transcoder.cfg.input_size

    total_loss = 0.0
    for input_ids in batches:

        def _patched(hidden_states):
            flat = hidden_states.reshape(-1, input_size)
            recon = transcoder_topk_reconstruction(transcoder, flat, k)
            return recon.reshape(hidden_states.shape)

        with patched_forward(mlp, _patched):
            total_loss += compute_ce_loss(base_model, input_ids)

    return total_loss / len(batches)


@torch.no_grad()
def eval_transcoder_mse(
    transcoder,
    mlp_activations: list[tuple[torch.Tensor, torch.Tensor]],
    k: int,
) -> float:
    total_mse = 0.0
    for mlp_in, mlp_out in mlp_activations:
        recon = transcoder_topk_reconstruction(transcoder, mlp_in, k)
        total_mse += F.mse_loss(recon, mlp_out).item()
    return total_mse / len(mlp_activations)


@torch.no_grad()
def eval_transcoder_batchtopk_ce(
    base_model,
    transcoder,
    batches: list[torch.Tensor],
    k: int,
) -> tuple[float, float]:
    """Returns (mean_l0, mean_ce)."""
    mlp = base_model.h[LAYER].mlp
    rms2 = base_model.h[LAYER].rms_2
    input_size = transcoder.cfg.input_size

    total_loss = 0.0
    total_l0 = 0.0
    for input_ids in batches:

        def _patched(hidden_states):
            flat = hidden_states.reshape(-1, input_size)
            recon = transcoder_batchtopk_reconstruction(transcoder, flat, k)
            return recon.reshape(hidden_states.shape)

        with patched_forward(mlp, _patched):
            total_loss += compute_ce_loss(base_model, input_ids)

        # Compute actual L0
        captured = {}

        def _hook(_mod, _inp, out):
            captured["out"] = out.detach()

        h = rms2.register_forward_hook(_hook)
        base_model(input_ids)
        h.remove()
        flat = captured["out"].reshape(-1, input_size)
        acts = _get_batchtopk_acts(transcoder, flat, k)
        total_l0 += (acts > 0).float().sum(-1).mean().item()

    n = len(batches)
    return total_l0 / n, total_loss / n


@torch.no_grad()
def eval_transcoder_batchtopk_mse(
    transcoder,
    mlp_activations: list[tuple[torch.Tensor, torch.Tensor]],
    k: int,
) -> tuple[float, float]:
    total_mse = 0.0
    total_l0 = 0.0
    for mlp_in, mlp_out in mlp_activations:
        recon = transcoder_batchtopk_reconstruction(transcoder, mlp_in, k)
        total_mse += F.mse_loss(recon, mlp_out).item()
        acts = _get_batchtopk_acts(transcoder, mlp_in, k)
        total_l0 += (acts > 0).float().sum(-1).mean().item()
    n = len(mlp_activations)
    return total_l0 / n, total_mse / n


def _get_batchtopk_acts(transcoder, x_in, k):
    use_pre_enc_bias = transcoder.cfg.pre_enc_bias and transcoder.input_size == transcoder.output_size
    x_enc = x_in - transcoder.b_dec if use_pre_enc_bias else x_in

    if isinstance(transcoder, (TopK, BatchTopK)):
        acts = F.relu(x_enc @ transcoder.W_enc)
    else:
        acts = F.relu(x_enc @ transcoder.W_enc + transcoder.b_enc)

    n_keep = k * acts.shape[0]
    if n_keep < acts.numel():
        topk = torch.topk(acts.flatten(), n_keep, dim=-1)
        acts = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, topk.indices, topk.values)
            .reshape(acts.shape)
        )
    return acts


# =============================================================================
# Neurons (top-k hidden activations of the original MLP)
# =============================================================================


def neuron_topk_reconstruction(mlp: nn.Module, x_in: torch.Tensor, k: int) -> torch.Tensor:
    """Reconstruct MLP output keeping only top-k neurons by activation magnitude.

    LlamaSimpleMLP MLP: c_fc -> gelu -> down_proj
    """
    h = mlp.gelu(mlp.c_fc(x_in))
    if k < h.shape[-1]:
        topk = torch.topk(h.abs(), k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk.indices, 1.0)
        h = h * mask
    return mlp.down_proj(h)


@torch.no_grad()
def eval_neuron_ce(
    base_model,
    batches: list[torch.Tensor],
    k: int,
) -> float:
    mlp = base_model.h[LAYER].mlp

    total_loss = 0.0
    for input_ids in batches:

        def _patched(hidden_states):
            return neuron_topk_reconstruction(mlp, hidden_states, k)

        with patched_forward(mlp, _patched):
            total_loss += compute_ce_loss(base_model, input_ids)

    return total_loss / len(batches)


@torch.no_grad()
def eval_neuron_mse(
    base_model,
    mlp_activations: list[tuple[torch.Tensor, torch.Tensor]],
    k: int,
) -> float:
    mlp = base_model.h[LAYER].mlp
    total_mse = 0.0
    for mlp_in, mlp_out in mlp_activations:
        recon = neuron_topk_reconstruction(mlp, mlp_in, k)
        total_mse += F.mse_loss(recon, mlp_out).item()
    return total_mse / len(mlp_activations)


# =============================================================================
# SPD
# =============================================================================


@torch.no_grad()
def eval_spd_ce(
    spd_model,
    batches: list[torch.Tensor],
    module_name: str,
    k: int,
) -> float:
    """Evaluate CE loss with SPD using top-k components for one module."""
    total_loss = 0.0
    for input_ids in batches:
        out = spd_model(input_ids, cache_type="input")
        pre_weight_acts = out.cache

        ci = spd_model.calc_causal_importances(pre_weight_acts, sampling="continuous")
        ci_scores = ci.lower_leaky[module_name].clamp(0, 1)
        n_components = ci_scores.shape[-1]
        actual_k = min(k, n_components)

        topk_indices = torch.topk(ci_scores, actual_k, dim=-1).indices
        mask = torch.zeros_like(ci_scores)
        mask.scatter_(-1, topk_indices, 1.0)

        mask_infos = make_mask_infos({module_name: mask})
        logits = spd_model(input_ids, mask_infos=mask_infos)

        total_loss += compute_ce_from_logits(logits, input_ids)

    return total_loss / len(batches)


@torch.no_grad()
def eval_spd_both_ce(
    spd_model,
    batches: list[torch.Tensor],
    module_names: list[str],
    k: int,
) -> tuple[float, float]:
    """Evaluate CE loss with SPD masking both modules simultaneously.

    Returns (avg_l0, ce_loss) where avg_l0 is the mean of the actual k used per module.
    """
    total_loss = 0.0
    total_l0 = 0.0
    for input_ids in batches:
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        masks = {}
        batch_l0 = 0.0
        for mod_name in module_names:
            ci_scores = ci.lower_leaky[mod_name].clamp(0, 1)
            n_components = ci_scores.shape[-1]
            actual_k = min(k, n_components)

            topk_indices = torch.topk(ci_scores, actual_k, dim=-1).indices
            mask = torch.zeros_like(ci_scores)
            mask.scatter_(-1, topk_indices, 1.0)
            masks[mod_name] = mask
            batch_l0 += actual_k

        avg_l0 = batch_l0 / len(module_names)
        total_l0 += avg_l0

        mask_infos = make_mask_infos(masks)
        logits = spd_model(input_ids, mask_infos=mask_infos)
        total_loss += compute_ce_from_logits(logits, input_ids)

    n = len(batches)
    return total_l0 / n, total_loss / n


@torch.no_grad()
def eval_spd_both_mse(
    spd_model,
    batches: list[torch.Tensor],
    module_names: list[str],
    k: int,
) -> tuple[float, float]:
    """Evaluate MSE with SPD masking both modules simultaneously.

    Returns (avg_l0, mse).
    """
    target_mlp = spd_model.target_model.h[LAYER].mlp
    total_mse = 0.0
    total_l0 = 0.0

    for input_ids in batches:
        captured_orig = {}

        def _capture_orig(_mod, _inp, out):
            captured_orig["out"] = out.detach()

        hook = target_mlp.register_forward_hook(_capture_orig)
        spd_model(input_ids)
        hook.remove()

        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        masks = {}
        batch_l0 = 0.0
        for mod_name in module_names:
            ci_scores = ci.lower_leaky[mod_name].clamp(0, 1)
            n_components = ci_scores.shape[-1]
            actual_k = min(k, n_components)

            topk_indices = torch.topk(ci_scores, actual_k, dim=-1).indices
            mask = torch.zeros_like(ci_scores)
            mask.scatter_(-1, topk_indices, 1.0)
            masks[mod_name] = mask
            batch_l0 += actual_k

        avg_l0 = batch_l0 / len(module_names)
        total_l0 += avg_l0

        captured_masked = {}

        def _capture_masked(_mod, _inp, out):
            captured_masked["out"] = out.detach()

        hook = target_mlp.register_forward_hook(_capture_masked)
        mask_infos = make_mask_infos(masks)
        spd_model(input_ids, mask_infos=mask_infos)
        hook.remove()

        total_mse += F.mse_loss(captured_masked["out"], captured_orig["out"]).item()

    n = len(batches)
    return total_l0 / n, total_mse / n


def _global_topk_masks(
    ci: object, module_names: list[str], k: int,
) -> dict[str, torch.Tensor]:
    """Take global top-k across concatenated CI scores from all modules.

    Returns per-module binary masks. k is the total number of active components
    across all modules combined.
    """
    all_scores = []
    sizes = []
    for mod_name in module_names:
        scores = ci.lower_leaky[mod_name].clamp(0, 1)  # (B, S, C_i)
        all_scores.append(scores)
        sizes.append(scores.shape[-1])

    # Concatenate along component dim: (B, S, sum(C_i))
    concat = torch.cat(all_scores, dim=-1)
    total_c = concat.shape[-1]
    actual_k = min(k, total_c)

    topk_indices = torch.topk(concat, actual_k, dim=-1).indices
    concat_mask = torch.zeros_like(concat)
    concat_mask.scatter_(-1, topk_indices, 1.0)

    # Split back into per-module masks
    masks = {}
    offset = 0
    for mod_name, size in zip(module_names, sizes):
        masks[mod_name] = concat_mask[..., offset:offset + size]
        offset += size

    return masks


@torch.no_grad()
def eval_spd_global_ce(
    spd_model,
    batches: list[torch.Tensor],
    module_names: list[str],
    k: int,
) -> float:
    """Evaluate CE with global top-k across both modules. k = total active components."""
    total_loss = 0.0
    for input_ids in batches:
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        masks = _global_topk_masks(ci, module_names, k)
        mask_infos = make_mask_infos(masks)
        logits = spd_model(input_ids, mask_infos=mask_infos)
        total_loss += compute_ce_from_logits(logits, input_ids)

    return total_loss / len(batches)


@torch.no_grad()
def eval_spd_global_mse(
    spd_model,
    batches: list[torch.Tensor],
    module_names: list[str],
    k: int,
) -> float:
    """Evaluate MSE with global top-k across both modules. k = total active components."""
    target_mlp = spd_model.target_model.h[LAYER].mlp
    total_mse = 0.0

    for input_ids in batches:
        captured_orig = {}

        def _capture_orig(_mod, _inp, out):
            captured_orig["out"] = out.detach()

        hook = target_mlp.register_forward_hook(_capture_orig)
        spd_model(input_ids)
        hook.remove()

        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        masks = _global_topk_masks(ci, module_names, k)

        captured_masked = {}

        def _capture_masked(_mod, _inp, out):
            captured_masked["out"] = out.detach()

        hook = target_mlp.register_forward_hook(_capture_masked)
        mask_infos = make_mask_infos(masks)
        spd_model(input_ids, mask_infos=mask_infos)
        hook.remove()

        total_mse += F.mse_loss(captured_masked["out"], captured_orig["out"]).item()

    return total_mse / len(batches)


@torch.no_grad()
def eval_spd_ce_thresholded(
    spd_model,
    batches: list[torch.Tensor],
    module_name: str,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Returns (mean_l0, mean_ce_loss)."""
    total_loss = 0.0
    total_l0 = 0.0
    for input_ids in batches:
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_post = ci.lower_leaky[module_name]

        mask = (ci_post > threshold).float()
        total_l0 += mask.sum(-1).mean().item()

        mask_infos = make_mask_infos({module_name: mask})
        logits = spd_model(input_ids, mask_infos=mask_infos)
        total_loss += compute_ce_from_logits(logits, input_ids)

    n = len(batches)
    return total_l0 / n, total_loss / n


@torch.no_grad()
def eval_spd_both_ce_thresholded(
    spd_model,
    batches: list[torch.Tensor],
    module_names: list[str],
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Threshold both modules simultaneously. Returns (avg_l0, ce_loss)."""
    total_loss = 0.0
    total_l0 = 0.0
    for input_ids in batches:
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        masks = {}
        batch_l0 = 0.0
        for mod_name in module_names:
            ci_post = ci.lower_leaky[mod_name]
            mask = (ci_post > threshold).float()
            masks[mod_name] = mask
            batch_l0 += mask.sum(-1).mean().item()

        total_l0 += batch_l0 / len(module_names)
        mask_infos = make_mask_infos(masks)
        logits = spd_model(input_ids, mask_infos=mask_infos)
        total_loss += compute_ce_from_logits(logits, input_ids)

    n = len(batches)
    return total_l0 / n, total_loss / n


@torch.no_grad()
def eval_spd_both_mse_thresholded(
    spd_model,
    batches: list[torch.Tensor],
    module_names: list[str],
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Threshold both modules simultaneously. Returns (avg_l0, mse)."""
    target_mlp = spd_model.target_model.h[LAYER].mlp
    total_mse = 0.0
    total_l0 = 0.0

    for input_ids in batches:
        captured_orig = {}

        def _capture_orig(_mod, _inp, out):
            captured_orig["out"] = out.detach()

        hook = target_mlp.register_forward_hook(_capture_orig)
        spd_model(input_ids)
        hook.remove()

        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        masks = {}
        batch_l0 = 0.0
        for mod_name in module_names:
            ci_post = ci.lower_leaky[mod_name]
            mask = (ci_post > threshold).float()
            masks[mod_name] = mask
            batch_l0 += mask.sum(-1).mean().item()

        total_l0 += batch_l0 / len(module_names)

        captured_masked = {}

        def _capture_masked(_mod, _inp, out):
            captured_masked["out"] = out.detach()

        hook = target_mlp.register_forward_hook(_capture_masked)
        mask_infos = make_mask_infos(masks)
        spd_model(input_ids, mask_infos=mask_infos)
        hook.remove()

        total_mse += F.mse_loss(captured_masked["out"], captured_orig["out"]).item()

    n = len(batches)
    return total_l0 / n, total_mse / n


@torch.no_grad()
def eval_spd_mse(
    spd_model,
    batches: list[torch.Tensor],
    module_name: str,
    k: int,
) -> float:
    """Evaluate MSE at the MLP output level with SPD top-k components."""
    target_mlp = spd_model.target_model.h[LAYER].mlp
    total_mse = 0.0

    for input_ids in batches:
        # Original MLP output
        captured_orig = {}

        def _capture_orig(_mod, _inp, out):
            captured_orig["out"] = out.detach()

        hook = target_mlp.register_forward_hook(_capture_orig)
        spd_model(input_ids)
        hook.remove()

        # Masked MLP output
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_scores = ci.lower_leaky[module_name].clamp(0, 1)
        n_components = ci_scores.shape[-1]
        actual_k = min(k, n_components)

        topk_indices = torch.topk(ci_scores, actual_k, dim=-1).indices
        mask = torch.zeros_like(ci_scores)
        mask.scatter_(-1, topk_indices, 1.0)

        captured_masked = {}

        def _capture_masked(_mod, _inp, out):
            captured_masked["out"] = out.detach()

        hook = target_mlp.register_forward_hook(_capture_masked)
        mask_infos = make_mask_infos({module_name: mask})
        spd_model(input_ids, mask_infos=mask_infos)
        hook.remove()

        total_mse += F.mse_loss(captured_masked["out"], captured_orig["out"]).item()

    return total_mse / len(batches)


@torch.no_grad()
def eval_spd_mse_thresholded(
    spd_model,
    batches: list[torch.Tensor],
    module_name: str,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Returns (mean_l0, mean_mse)."""
    target_mlp = spd_model.target_model.h[LAYER].mlp
    total_mse = 0.0
    total_l0 = 0.0

    for input_ids in batches:
        captured_orig = {}

        def _capture_orig(_mod, _inp, out):
            captured_orig["out"] = out.detach()

        hook = target_mlp.register_forward_hook(_capture_orig)
        spd_model(input_ids)
        hook.remove()

        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_post = ci.lower_leaky[module_name]

        mask = (ci_post > threshold).float()
        total_l0 += mask.sum(-1).mean().item()

        captured_masked = {}

        def _capture_masked(_mod, _inp, out):
            captured_masked["out"] = out.detach()

        hook = target_mlp.register_forward_hook(_capture_masked)
        mask_infos = make_mask_infos({module_name: mask})
        spd_model(input_ids, mask_infos=mask_infos)
        hook.remove()

        total_mse += F.mse_loss(captured_masked["out"], captured_orig["out"]).item()

    n = len(batches)
    return total_l0 / n, total_mse / n


# =============================================================================
# Baselines
# =============================================================================


@torch.no_grad()
def eval_baselines(
    base_model,
    batches: list[torch.Tensor],
) -> dict[str, float]:
    """Compute baseline CE losses: original and zero-ablation."""
    mlp = base_model.h[LAYER].mlp
    original_forward = mlp.forward

    original_loss = 0.0
    zero_loss = 0.0
    for input_ids in batches:
        original_loss += compute_ce_loss(base_model, input_ids)

        def _zero(hidden_states):
            return torch.zeros_like(original_forward(hidden_states))

        with patched_forward(mlp, _zero):
            zero_loss += compute_ce_loss(base_model, input_ids)

    n = len(batches)
    return {
        "original_ce": original_loss / n,
        "zero_ablation_ce": zero_loss / n,
    }


# =============================================================================
# Plotting
# =============================================================================


def plot_pareto(
    l0_values, transcoder_ces, spd_cfc_ces, spd_down_ces,
    both_indep_l0s, both_indep_ces,
    neuron_ces, baselines, special_points, save_path,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(l0_values, transcoder_ces, "o-", color="tab:blue", label="Transcoder")
    ax.plot(l0_values, spd_cfc_ces, "s-", color="tab:orange", label="SPD (c_fc)")
    ax.plot(l0_values, spd_down_ces, "^-", color="tab:green", label="SPD (down_proj)")
    ax.plot(both_indep_l0s, both_indep_ces, "P-", color="tab:purple", label="SPD (both)")
    ax.plot(l0_values, neuron_ces, "d-", color="tab:red", label="Neurons")

    sp_colors = {
        "BatchTopK (train k)": "tab:blue",
        "SPD c_fc (CI>0.5)": "tab:orange",
        "SPD down_proj (CI>0.5)": "tab:green",
        "SPD c_fc (CI>0)": "tab:orange",
        "SPD down_proj (CI>0)": "tab:green",
        "SPD both (CI>0.5)": "tab:purple",
        "SPD both (CI>0)": "tab:purple",
    }
    sp_markers = {
        "SPD c_fc (CI>0)": "D", "SPD down_proj (CI>0)": "D",
        "SPD both (CI>0)": "D",
    }
    for label, (l0, ce) in special_points.items():
        color = sp_colors.get(label, "black")
        marker = sp_markers.get(label, "*")
        size = 12 if marker == "D" else 18
        ax.plot(l0, ce, marker, color=color, markersize=size, markeredgecolor="black",
                markeredgewidth=0.8, label=label, zorder=5)

    ax.axhline(baselines["original_ce"], color="gray", linestyle="--", alpha=0.7, label="Original CE")
    ax.axhline(baselines["zero_ablation_ce"], color="red", linestyle="--", alpha=0.5, label="Zero-ablation CE")

    ax.set_xlabel("L0 (number of active components)")
    ax.set_ylabel("CE Loss")
    ax.set_title("L0 vs CE Loss: SPD vs Transcoder (LlamaSimpleMLP Layer 3 MLP)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


def plot_pareto_mse(
    l0_values, transcoder_mses, spd_cfc_mses, spd_down_mses,
    both_indep_l0s, both_indep_mses,
    neuron_mses, special_points, save_path,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(l0_values, transcoder_mses, "o-", color="tab:blue", label="Transcoder")
    ax.plot(l0_values, spd_cfc_mses, "s-", color="tab:orange", label="SPD (c_fc)")
    ax.plot(l0_values, spd_down_mses, "^-", color="tab:green", label="SPD (down_proj)")
    ax.plot(both_indep_l0s, both_indep_mses, "P-", color="tab:purple", label="SPD (both)")
    ax.plot(l0_values, neuron_mses, "d-", color="tab:red", label="Neurons")

    sp_colors = {
        "BatchTopK (train k)": "tab:blue",
        "SPD c_fc (CI>0.5)": "tab:orange",
        "SPD down_proj (CI>0.5)": "tab:green",
        "SPD c_fc (CI>0)": "tab:orange",
        "SPD down_proj (CI>0)": "tab:green",
        "SPD both (CI>0.5)": "tab:purple",
        "SPD both (CI>0)": "tab:purple",
    }
    sp_markers = {
        "SPD c_fc (CI>0)": "D", "SPD down_proj (CI>0)": "D",
        "SPD both (CI>0)": "D",
    }
    for label, (l0, mse) in special_points.items():
        color = sp_colors.get(label, "black")
        marker = sp_markers.get(label, "*")
        size = 12 if marker == "D" else 18
        ax.plot(l0, mse, marker, color=color, markersize=size, markeredgecolor="black",
                markeredgewidth=0.8, label=label, zorder=5)

    ax.set_xlabel("L0 (number of active components)")
    ax.set_ylabel("MLP Output MSE")
    ax.set_title("L0 vs MLP Reconstruction MSE: SPD vs Transcoder (LlamaSimpleMLP Layer 3)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="L0 vs CE Pareto: SPD vs Transcoder (Pile/LlamaSimpleMLP)")
    parser.add_argument("--transcoder_path", type=str, default="checkpoints/4096_batchtopk_k24_0.0003_final")
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument(
        "--l0_values", type=int, nargs="+",
        default=[1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3072, 4096],
    )
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--save_path", type=str, default="experiments/pareto_pile.png")
    args = parser.parse_args()

    # Load SPD model (includes the base LlamaSimpleMLP)
    from analysis.collect_spd_activations import load_spd_model

    print("Loading SPD model...")
    spd_model, raw_config = load_spd_model(args.spd_run)
    spd_model.to(DEVICE)

    base_model = spd_model.target_model
    base_model.eval()

    # Module names for layer 3
    cfc_name = f"h.{LAYER}.mlp.c_fc"
    down_name = f"h.{LAYER}.mlp.down_proj"
    assert cfc_name in spd_model.target_module_paths
    assert down_name in spd_model.target_module_paths

    # Load transcoder
    print("Loading transcoder...")
    transcoder = load_transcoder(args.transcoder_path)
    transcoder.to(DEVICE)

    # Load eval data (pre-tokenized Pile)
    print(f"Loading {args.n_eval_batches} eval batches (seq_len={args.seq_len})...")
    batches = get_eval_batches(args.n_eval_batches, args.batch_size, args.seq_len)

    # Baselines
    print("Computing baselines...")
    baselines = eval_baselines(base_model, batches)
    print(f"  Original CE: {baselines['original_ce']:.4f}")
    print(f"  Zero-ablation CE: {baselines['zero_ablation_ce']:.4f}")

    # MLP activations for MSE
    print("Collecting MLP activations for MSE...")
    mlp_activations = get_mlp_activations(base_model, batches)

    # Sweep L0
    both_module_names = [cfc_name, down_name]
    tc_ces, cfc_ces, down_ces, n_ces = [], [], [], []
    both_indep_ces, both_indep_l0s = [], []
    tc_mses, cfc_mses, down_mses, n_mses = [], [], [], []
    both_indep_mses, both_indep_l0s_m = [], []

    header = f"{'L0':>6} | {'TC CE':>10} | {'cfc CE':>10} | {'down CE':>10} | {'both CE':>10} | {'both L0':>8} | {'Neur CE':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for k in args.l0_values:
        tc_ce = eval_transcoder_ce(base_model, transcoder, batches, k)
        cfc_ce = eval_spd_ce(spd_model, batches, cfc_name, k)
        down_ce = eval_spd_ce(spd_model, batches, down_name, k)
        indep_avg_l0, indep_ce = eval_spd_both_ce(spd_model, batches, both_module_names, k)
        n_ce = eval_neuron_ce(base_model, batches, k)

        tc_mse = eval_transcoder_mse(transcoder, mlp_activations, k)
        cfc_mse = eval_spd_mse(spd_model, batches, cfc_name, k)
        down_mse = eval_spd_mse(spd_model, batches, down_name, k)
        indep_avg_l0_m, indep_mse = eval_spd_both_mse(spd_model, batches, both_module_names, k)
        n_mse = eval_neuron_mse(base_model, mlp_activations, k)

        tc_ces.append(tc_ce)
        cfc_ces.append(cfc_ce)
        down_ces.append(down_ce)
        both_indep_ces.append(indep_ce)
        both_indep_l0s.append(indep_avg_l0)
        n_ces.append(n_ce)
        tc_mses.append(tc_mse)
        cfc_mses.append(cfc_mse)
        down_mses.append(down_mse)
        both_indep_mses.append(indep_mse)
        both_indep_l0s_m.append(indep_avg_l0_m)
        n_mses.append(n_mse)

        print(f"{k:>6} | {tc_ce:>10.4f} | {cfc_ce:>10.4f} | {down_ce:>10.4f} | {indep_ce:>10.4f} | {indep_avg_l0:>8.1f} | {n_ce:>10.4f}")

    # Special points (CE)
    print("\nSpecial points (CE):")
    sp_ce = {}

    train_k = transcoder.cfg.top_k
    tc_l0, tc_train_ce = eval_transcoder_batchtopk_ce(base_model, transcoder, batches, train_k)
    sp_ce["BatchTopK (train k)"] = (tc_l0, tc_train_ce)
    print(f"  BatchTopK (train k={train_k}): L0={tc_l0:.1f}, CE={tc_train_ce:.4f}")

    cfc_l0, cfc_t_ce = eval_spd_ce_thresholded(spd_model, batches, cfc_name)
    sp_ce["SPD c_fc (CI>0.5)"] = (cfc_l0, cfc_t_ce)
    print(f"  SPD c_fc (CI>0.5): L0={cfc_l0:.1f}, CE={cfc_t_ce:.4f}")

    down_l0, down_t_ce = eval_spd_ce_thresholded(spd_model, batches, down_name)
    sp_ce["SPD down_proj (CI>0.5)"] = (down_l0, down_t_ce)
    print(f"  SPD down_proj (CI>0.5): L0={down_l0:.1f}, CE={down_t_ce:.4f}")

    cfc_l0_0, cfc_t_ce_0 = eval_spd_ce_thresholded(spd_model, batches, cfc_name, threshold=0.0)
    sp_ce["SPD c_fc (CI>0)"] = (cfc_l0_0, cfc_t_ce_0)
    print(f"  SPD c_fc (CI>0): L0={cfc_l0_0:.1f}, CE={cfc_t_ce_0:.4f}")

    down_l0_0, down_t_ce_0 = eval_spd_ce_thresholded(spd_model, batches, down_name, threshold=0.0)
    sp_ce["SPD down_proj (CI>0)"] = (down_l0_0, down_t_ce_0)
    print(f"  SPD down_proj (CI>0): L0={down_l0_0:.1f}, CE={down_t_ce_0:.4f}")

    both_l0_05, both_ce_05 = eval_spd_both_ce_thresholded(spd_model, batches, both_module_names)
    sp_ce["SPD both (CI>0.5)"] = (both_l0_05, both_ce_05)
    print(f"  SPD both (CI>0.5): L0={both_l0_05:.1f}, CE={both_ce_05:.4f}")

    both_l0_0, both_ce_0 = eval_spd_both_ce_thresholded(spd_model, batches, both_module_names, threshold=0.0)
    sp_ce["SPD both (CI>0)"] = (both_l0_0, both_ce_0)
    print(f"  SPD both (CI>0): L0={both_l0_0:.1f}, CE={both_ce_0:.4f}")

    # Special points (MSE)
    print("\nSpecial points (MSE):")
    sp_mse = {}

    tc_l0_m, tc_train_mse = eval_transcoder_batchtopk_mse(transcoder, mlp_activations, train_k)
    sp_mse["BatchTopK (train k)"] = (tc_l0_m, tc_train_mse)
    print(f"  BatchTopK (train k={train_k}): L0={tc_l0_m:.1f}, MSE={tc_train_mse:.6f}")

    cfc_l0_m, cfc_t_mse = eval_spd_mse_thresholded(spd_model, batches, cfc_name)
    sp_mse["SPD c_fc (CI>0.5)"] = (cfc_l0_m, cfc_t_mse)
    print(f"  SPD c_fc (CI>0.5): L0={cfc_l0_m:.1f}, MSE={cfc_t_mse:.6f}")

    down_l0_m, down_t_mse = eval_spd_mse_thresholded(spd_model, batches, down_name)
    sp_mse["SPD down_proj (CI>0.5)"] = (down_l0_m, down_t_mse)
    print(f"  SPD down_proj (CI>0.5): L0={down_l0_m:.1f}, MSE={down_t_mse:.6f}")

    cfc_l0_m0, cfc_t_mse_0 = eval_spd_mse_thresholded(spd_model, batches, cfc_name, threshold=0.0)
    sp_mse["SPD c_fc (CI>0)"] = (cfc_l0_m0, cfc_t_mse_0)
    print(f"  SPD c_fc (CI>0): L0={cfc_l0_m0:.1f}, MSE={cfc_t_mse_0:.6f}")

    down_l0_m0, down_t_mse_0 = eval_spd_mse_thresholded(spd_model, batches, down_name, threshold=0.0)
    sp_mse["SPD down_proj (CI>0)"] = (down_l0_m0, down_t_mse_0)
    print(f"  SPD down_proj (CI>0): L0={down_l0_m0:.1f}, MSE={down_t_mse_0:.6f}")

    both_l0_m05, both_mse_05 = eval_spd_both_mse_thresholded(spd_model, batches, both_module_names)
    sp_mse["SPD both (CI>0.5)"] = (both_l0_m05, both_mse_05)
    print(f"  SPD both (CI>0.5): L0={both_l0_m05:.1f}, MSE={both_mse_05:.6f}")

    both_l0_m0, both_mse_0 = eval_spd_both_mse_thresholded(spd_model, batches, both_module_names, threshold=0.0)
    sp_mse["SPD both (CI>0)"] = (both_l0_m0, both_mse_0)
    print(f"  SPD both (CI>0): L0={both_l0_m0:.1f}, MSE={both_mse_0:.6f}")

    # Plots
    plot_pareto(
        args.l0_values, tc_ces, cfc_ces, down_ces,
        both_indep_l0s, both_indep_ces,
        n_ces, baselines, sp_ce, args.save_path,
    )

    mse_path = args.save_path.replace(".png", "_mse.png")
    plot_pareto_mse(
        args.l0_values, tc_mses, cfc_mses, down_mses,
        both_indep_l0s_m, both_indep_mses,
        n_mses, sp_mse, mse_path,
    )


if __name__ == "__main__":
    main()
