"""Compare L0 vs CE-loss Pareto curves for SPD and Transcoder on LlamaSimpleMLP.

All-layers variant: replaces ALL MLP layers simultaneously with their
respective transcoders/decompositions.

For a given L0 = k:
- Transcoder: keep the k features with highest activations per token, all layers
- SPD: keep the k components with highest pre-sigmoid CI score per token,
  evaluating each weight matrix (c_fc, down_proj) separately, all layers

Uses the same base model (wandb:goodfire/spd/t-32d1bb3b), dataset
(danbraunai/pile-uncopyrighted-tok), and tokenizer (EleutherAI/gpt-neox-20b)
that both models were trained on.

Usage:
    python "experiments/pareto_pile all_layers.py"
"""

import argparse
import json
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from transcoder import BatchTopKTranscoder, JumpReLUTranscoder, TopKTranscoder, VanillaTranscoder
from config import EncoderConfig, CLTConfig
from clt import CrossLayerTranscoder
from spd.models.components import make_mask_infos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
LAYER_CHECKPOINT_MAP = {
    0: "checkpoints/4096_batchtopk_k21_0.0003_final",
    1: "checkpoints/4096_batchtopk_k8_0.0003_final",
    2: "checkpoints/4096_batchtopk_k13_0.0003_final",
    3: "checkpoints/4096_batchtopk_k27_0.0003_final",
}


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
# MLP activation collection (for MSE) — all layers
# =============================================================================


@torch.no_grad()
def get_mlp_activations(
    model,
    batches: list[torch.Tensor],
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Collect (mlp_input, mlp_output) pairs from all layers of the base model."""
    all_pairs = {layer_idx: [] for layer_idx in LAYERS}

    for input_ids in batches:
        captured = {}
        hooks = []

        for layer_idx in LAYERS:
            rms2 = model.h[layer_idx].rms_2
            mlp = model.h[layer_idx].mlp

            def _make_hooks(li):
                def _capture_rms2(_mod, _inp, out):
                    captured[f"mlp_in_{li}"] = out.detach()

                def _capture_mlp(_mod, _inp, out):
                    captured[f"mlp_out_{li}"] = out.detach()

                return _capture_rms2, _capture_mlp

            h_rms, h_mlp = _make_hooks(layer_idx)
            hooks.append(rms2.register_forward_hook(h_rms))
            hooks.append(mlp.register_forward_hook(h_mlp))

        model(input_ids)
        for h in hooks:
            h.remove()

        for layer_idx in LAYERS:
            mlp_in = captured[f"mlp_in_{layer_idx}"]
            mlp_in = mlp_in.reshape(-1, mlp_in.shape[-1])
            mlp_out = captured[f"mlp_out_{layer_idx}"]
            mlp_out = mlp_out.reshape(-1, mlp_out.shape[-1])
            all_pairs[layer_idx].append((mlp_in, mlp_out))

    return all_pairs


# =============================================================================
# Transcoder
# =============================================================================

ENCODER_CLASSES = {
    "vanilla": VanillaTranscoder,
    "topk": TopKTranscoder,
    "batchtopk": BatchTopKTranscoder,
    "jumprelu": JumpReLUTranscoder,
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

    if isinstance(transcoder, (TopKTranscoder, BatchTopKTranscoder)):
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

    if isinstance(transcoder, (TopKTranscoder, BatchTopKTranscoder)):
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
    transcoders: dict[int, nn.Module],
    batches: list[torch.Tensor],
    k: int,
) -> float:
    """Evaluate CE loss with transcoders replacing all layers using top-k features."""
    total_loss = 0.0
    for input_ids in batches:
        with ExitStack() as stack:
            for layer_idx in LAYERS:
                mlp = base_model.h[layer_idx].mlp
                tc = transcoders[layer_idx]
                input_size = tc.cfg.input_size

                def _make_patched(tc_, input_size_):
                    def _patched(hidden_states):
                        flat = hidden_states.reshape(-1, input_size_)
                        recon = transcoder_topk_reconstruction(tc_, flat, k)
                        return recon.reshape(hidden_states.shape)
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(tc, input_size)))
            total_loss += compute_ce_loss(base_model, input_ids)

    return total_loss / len(batches)


@torch.no_grad()
def eval_transcoder_mse(
    transcoders: dict[int, nn.Module],
    all_mlp_activations: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    k: int,
) -> float:
    """Evaluate average MSE across all layers."""
    total_mse = 0.0
    for layer_idx in LAYERS:
        layer_mse = 0.0
        for mlp_in, mlp_out in all_mlp_activations[layer_idx]:
            recon = transcoder_topk_reconstruction(transcoders[layer_idx], mlp_in, k)
            layer_mse += F.mse_loss(recon, mlp_out).item()
        layer_mse /= len(all_mlp_activations[layer_idx])
        total_mse += layer_mse
    return total_mse / len(LAYERS)


@torch.no_grad()
def eval_transcoder_batchtopk_ce(
    base_model,
    transcoders: dict[int, nn.Module],
    batches: list[torch.Tensor],
) -> tuple[float, float]:
    """Returns (mean_l0, mean_ce) using each layer's native batch-topk k."""
    total_loss = 0.0
    total_l0 = 0.0

    for input_ids in batches:
        with ExitStack() as stack:
            for layer_idx in LAYERS:
                mlp = base_model.h[layer_idx].mlp
                tc = transcoders[layer_idx]
                input_size = tc.cfg.input_size
                layer_k = tc.cfg.top_k

                def _make_patched(tc_, input_size_, k_):
                    def _patched(hidden_states):
                        flat = hidden_states.reshape(-1, input_size_)
                        recon = transcoder_batchtopk_reconstruction(tc_, flat, k_)
                        return recon.reshape(hidden_states.shape)
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(tc, input_size, layer_k)))
            total_loss += compute_ce_loss(base_model, input_ids)

        # Compute actual L0 per layer and average
        batch_l0 = 0.0
        captured = {}
        hooks = []
        for layer_idx in LAYERS:
            rms2 = base_model.h[layer_idx].rms_2

            def _make_hook(li):
                def _hook(_mod, _inp, out):
                    captured[li] = out.detach()
                return _hook

            hooks.append(rms2.register_forward_hook(_make_hook(layer_idx)))

        base_model(input_ids)
        for h in hooks:
            h.remove()

        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            input_size = tc.cfg.input_size
            layer_k = tc.cfg.top_k
            flat = captured[layer_idx].reshape(-1, input_size)
            acts = _get_batchtopk_acts(tc, flat, layer_k)
            batch_l0 += (acts > 0).float().sum(-1).mean().item()

        total_l0 += batch_l0 / len(LAYERS)

    n = len(batches)
    return total_l0 / n, total_loss / n


@torch.no_grad()
def eval_transcoder_batchtopk_mse(
    transcoders: dict[int, nn.Module],
    all_mlp_activations: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
) -> tuple[float, float]:
    """Returns (mean_l0, mean_mse) using each layer's native batch-topk k."""
    total_mse = 0.0
    total_l0 = 0.0
    n_batches = len(all_mlp_activations[LAYERS[0]])

    for batch_idx in range(n_batches):
        batch_mse = 0.0
        batch_l0 = 0.0
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            layer_k = tc.cfg.top_k
            mlp_in, mlp_out = all_mlp_activations[layer_idx][batch_idx]
            recon = transcoder_batchtopk_reconstruction(tc, mlp_in, layer_k)
            batch_mse += F.mse_loss(recon, mlp_out).item()
            acts = _get_batchtopk_acts(tc, mlp_in, layer_k)
            batch_l0 += (acts > 0).float().sum(-1).mean().item()
        total_mse += batch_mse / len(LAYERS)
        total_l0 += batch_l0 / len(LAYERS)

    return total_l0 / n_batches, total_mse / n_batches


def _get_batchtopk_acts(transcoder, x_in, k):
    use_pre_enc_bias = transcoder.cfg.pre_enc_bias and transcoder.input_size == transcoder.output_size
    x_enc = x_in - transcoder.b_dec if use_pre_enc_bias else x_in

    if isinstance(transcoder, (TopKTranscoder, BatchTopKTranscoder)):
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
# Cross-Layer Transcoder (CLT)
# =============================================================================


def load_clt(checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["layers"] = json.loads(cfg_dict["layers"])
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg = CLTConfig(**cfg_dict)
    clt = CrossLayerTranscoder(cfg)
    clt.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    clt.eval()
    return clt


def clt_topk_reconstruction(
    clt: CrossLayerTranscoder, inputs: list[torch.Tensor], k: int,
) -> list[torch.Tensor]:
    """Reconstruct all layers using per-token top-k features at each encoder layer."""
    all_acts = []
    for i in range(clt.cfg.n_layers):
        pre_acts = F.relu(inputs[i] @ clt.W_enc[i] + clt.b_enc[i])
        if k < pre_acts.shape[-1]:
            topk = torch.topk(pre_acts, k, dim=-1)
            acts = torch.zeros_like(pre_acts).scatter(-1, topk.indices, topk.values)
        else:
            acts = pre_acts
        all_acts.append(acts)
    return clt.decode(all_acts)


def clt_batchtopk_acts(
    clt: CrossLayerTranscoder, inputs: list[torch.Tensor], k: int,
) -> list[torch.Tensor]:
    """Compute batch-topk activations per encoder layer. Returns list of activations."""
    all_acts = []
    for i in range(clt.cfg.n_layers):
        pre_acts = F.relu(inputs[i] @ clt.W_enc[i] + clt.b_enc[i])
        n_keep = k * pre_acts.shape[0]
        if n_keep < pre_acts.numel():
            topk = torch.topk(pre_acts.flatten(), n_keep, dim=-1)
            acts = (
                torch.zeros_like(pre_acts.flatten())
                .scatter(-1, topk.indices, topk.values)
                .reshape(pre_acts.shape)
            )
        else:
            acts = pre_acts
        all_acts.append(acts)
    return all_acts


def _collect_rms2_outputs(base_model, input_ids):
    """Collect rms_2 outputs (residual stream inputs to MLPs) from all layers."""
    captured = {}
    hooks = []
    for layer_idx in LAYERS:
        rms2 = base_model.h[layer_idx].rms_2

        def _make_hook(li):
            def _hook(_mod, _inp, out):
                captured[li] = out.detach()
            return _hook

        hooks.append(rms2.register_forward_hook(_make_hook(layer_idx)))
    base_model(input_ids)
    for h in hooks:
        h.remove()
    return captured


@torch.no_grad()
def eval_clt_ce(
    base_model,
    clt: CrossLayerTranscoder,
    batches: list[torch.Tensor],
    k: int,
) -> float:
    """Evaluate CE loss with CLT replacing all layers using per-token top-k."""
    total_loss = 0.0
    for input_ids in batches:
        # Collect clean residual stream inputs
        captured = _collect_rms2_outputs(base_model, input_ids)
        seq_shape = captured[LAYERS[0]].shape

        # Run CLT with custom top-k
        clt_inputs = [captured[l].reshape(-1, clt.cfg.input_size) for l in LAYERS]
        recons = clt_topk_reconstruction(clt, clt_inputs, k)
        recons = [r.reshape(seq_shape) for r in recons]

        # Patch all MLPs with pre-computed CLT reconstructions
        with ExitStack() as stack:
            for i, layer_idx in enumerate(LAYERS):
                mlp = base_model.h[layer_idx].mlp

                def _make_patched(recon):
                    def _patched(hidden_states):
                        return recon
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(recons[i])))
            total_loss += compute_ce_loss(base_model, input_ids)

    return total_loss / len(batches)


@torch.no_grad()
def eval_clt_mse(
    clt: CrossLayerTranscoder,
    all_mlp_activations: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    k: int,
) -> float:
    """Evaluate average MSE across all layers with CLT top-k."""
    total_mse = 0.0
    n_batches = len(all_mlp_activations[LAYERS[0]])

    for batch_idx in range(n_batches):
        clt_inputs = [all_mlp_activations[l][batch_idx][0] for l in LAYERS]
        targets = [all_mlp_activations[l][batch_idx][1] for l in LAYERS]

        recons = clt_topk_reconstruction(clt, clt_inputs, k)

        batch_mse = 0.0
        for i in range(len(LAYERS)):
            batch_mse += F.mse_loss(recons[i], targets[i]).item()
        total_mse += batch_mse / len(LAYERS)

    return total_mse / n_batches


@torch.no_grad()
def eval_clt_batchtopk_ce(
    base_model,
    clt: CrossLayerTranscoder,
    batches: list[torch.Tensor],
) -> tuple[float, float]:
    """Returns (mean_l0_per_layer, mean_ce) using CLT's native batch-topk k."""
    total_loss = 0.0
    total_l0 = 0.0

    for input_ids in batches:
        captured = _collect_rms2_outputs(base_model, input_ids)
        seq_shape = captured[LAYERS[0]].shape

        clt_inputs = [captured[l].reshape(-1, clt.cfg.input_size) for l in LAYERS]
        all_acts = clt_batchtopk_acts(clt, clt_inputs, clt.cfg.top_k)
        recons = clt.decode(all_acts)
        recons = [r.reshape(seq_shape) for r in recons]

        # Compute L0 (average per layer)
        batch_l0 = 0.0
        for acts in all_acts:
            batch_l0 += (acts > 0).float().sum(-1).mean().item()
        total_l0 += batch_l0 / len(LAYERS)

        with ExitStack() as stack:
            for i, layer_idx in enumerate(LAYERS):
                mlp = base_model.h[layer_idx].mlp

                def _make_patched(recon):
                    def _patched(hidden_states):
                        return recon
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(recons[i])))
            total_loss += compute_ce_loss(base_model, input_ids)

    n = len(batches)
    return total_l0 / n, total_loss / n


@torch.no_grad()
def eval_clt_batchtopk_mse(
    clt: CrossLayerTranscoder,
    all_mlp_activations: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
) -> tuple[float, float]:
    """Returns (mean_l0_per_layer, mean_mse) using CLT's native batch-topk k."""
    total_mse = 0.0
    total_l0 = 0.0
    n_batches = len(all_mlp_activations[LAYERS[0]])

    for batch_idx in range(n_batches):
        clt_inputs = [all_mlp_activations[l][batch_idx][0] for l in LAYERS]
        targets = [all_mlp_activations[l][batch_idx][1] for l in LAYERS]

        all_acts = clt_batchtopk_acts(clt, clt_inputs, clt.cfg.top_k)
        recons = clt.decode(all_acts)

        batch_l0 = 0.0
        for acts in all_acts:
            batch_l0 += (acts > 0).float().sum(-1).mean().item()
        total_l0 += batch_l0 / len(LAYERS)

        batch_mse = 0.0
        for i in range(len(LAYERS)):
            batch_mse += F.mse_loss(recons[i], targets[i]).item()
        total_mse += batch_mse / len(LAYERS)

    return total_l0 / n_batches, total_mse / n_batches


# =============================================================================
# Neurons (top-k hidden activations of the original MLP) — all layers
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
    """Evaluate CE loss with neuron top-k on all layers."""
    total_loss = 0.0
    for input_ids in batches:
        with ExitStack() as stack:
            for layer_idx in LAYERS:
                mlp = base_model.h[layer_idx].mlp

                def _make_patched(mlp_):
                    def _patched(hidden_states):
                        return neuron_topk_reconstruction(mlp_, hidden_states, k)
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(mlp)))
            total_loss += compute_ce_loss(base_model, input_ids)

    return total_loss / len(batches)


@torch.no_grad()
def eval_neuron_mse(
    base_model,
    all_mlp_activations: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    k: int,
) -> float:
    """Evaluate average neuron MSE across all layers."""
    total_mse = 0.0
    for layer_idx in LAYERS:
        mlp = base_model.h[layer_idx].mlp
        layer_mse = 0.0
        for mlp_in, mlp_out in all_mlp_activations[layer_idx]:
            recon = neuron_topk_reconstruction(mlp, mlp_in, k)
            layer_mse += F.mse_loss(recon, mlp_out).item()
        layer_mse /= len(all_mlp_activations[layer_idx])
        total_mse += layer_mse
    return total_mse / len(LAYERS)


# =============================================================================
# SPD — all layers
# =============================================================================


@torch.no_grad()
def eval_spd_ce(
    spd_model,
    batches: list[torch.Tensor],
    module_names: list[str],
    k: int,
) -> float:
    """Evaluate CE loss with SPD using top-k components for given modules."""
    total_loss = 0.0
    for input_ids in batches:
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        masks = {}
        for mod_name in module_names:
            ci_scores = ci.lower_leaky[mod_name].clamp(0, 1)
            n_components = ci_scores.shape[-1]
            actual_k = min(k, n_components)

            topk_indices = torch.topk(ci_scores, actual_k, dim=-1).indices
            mask = torch.zeros_like(ci_scores)
            mask.scatter_(-1, topk_indices, 1.0)
            masks[mod_name] = mask

        mask_infos = make_mask_infos(masks)
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
    """Evaluate CE loss with SPD masking all given modules simultaneously.

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


def _capture_all_layer_mlp_outputs(spd_model, input_ids, **model_kwargs):
    """Run model and capture MLP outputs from all layers. Returns dict[int, Tensor]."""
    captured = {}
    hooks = []
    for layer_idx in LAYERS:
        target_mlp = spd_model.target_model.h[layer_idx].mlp

        def _make_hook(li):
            def _capture(_mod, _inp, out):
                captured[li] = out.detach()
            return _capture

        hooks.append(target_mlp.register_forward_hook(_make_hook(layer_idx)))

    spd_model(input_ids, **model_kwargs)
    for h in hooks:
        h.remove()

    return captured


def _avg_mse_across_layers(captured_orig, captured_masked):
    """Compute average MSE across all layers."""
    total = 0.0
    for layer_idx in LAYERS:
        total += F.mse_loss(captured_masked[layer_idx], captured_orig[layer_idx]).item()
    return total / len(LAYERS)


@torch.no_grad()
def eval_spd_both_mse(
    spd_model,
    batches: list[torch.Tensor],
    module_names: list[str],
    k: int,
) -> tuple[float, float]:
    """Evaluate MSE with SPD masking all given modules simultaneously.

    Returns (avg_l0, mse) where mse is averaged across all layers.
    """
    total_mse = 0.0
    total_l0 = 0.0

    for input_ids in batches:
        captured_orig = _capture_all_layer_mlp_outputs(spd_model, input_ids)

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
        captured_masked = _capture_all_layer_mlp_outputs(
            spd_model, input_ids, mask_infos=mask_infos
        )

        total_mse += _avg_mse_across_layers(captured_orig, captured_masked)

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
    """Evaluate CE with global top-k across all modules."""
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
    """Evaluate MSE with global top-k across all modules, averaged across layers."""
    total_mse = 0.0

    for input_ids in batches:
        captured_orig = _capture_all_layer_mlp_outputs(spd_model, input_ids)

        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        masks = _global_topk_masks(ci, module_names, k)
        mask_infos = make_mask_infos(masks)
        captured_masked = _capture_all_layer_mlp_outputs(
            spd_model, input_ids, mask_infos=mask_infos
        )

        total_mse += _avg_mse_across_layers(captured_orig, captured_masked)

    return total_mse / len(batches)


@torch.no_grad()
def eval_spd_ce_thresholded(
    spd_model,
    batches: list[torch.Tensor],
    module_names: list[str],
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Threshold given modules. Returns (avg_l0, ce_loss)."""
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
def eval_spd_mse_thresholded(
    spd_model,
    batches: list[torch.Tensor],
    module_names: list[str],
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Threshold given modules. Returns (avg_l0, mse) averaged across layers."""
    total_mse = 0.0
    total_l0 = 0.0

    for input_ids in batches:
        captured_orig = _capture_all_layer_mlp_outputs(spd_model, input_ids)

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
        captured_masked = _capture_all_layer_mlp_outputs(
            spd_model, input_ids, mask_infos=mask_infos
        )

        total_mse += _avg_mse_across_layers(captured_orig, captured_masked)

    n = len(batches)
    return total_l0 / n, total_mse / n


@torch.no_grad()
def eval_spd_mse(
    spd_model,
    batches: list[torch.Tensor],
    module_names: list[str],
    k: int,
) -> float:
    """Evaluate MSE with SPD top-k components, averaged across all layers."""
    total_mse = 0.0

    for input_ids in batches:
        captured_orig = _capture_all_layer_mlp_outputs(spd_model, input_ids)

        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        masks = {}
        for mod_name in module_names:
            ci_scores = ci.lower_leaky[mod_name].clamp(0, 1)
            n_components = ci_scores.shape[-1]
            actual_k = min(k, n_components)

            topk_indices = torch.topk(ci_scores, actual_k, dim=-1).indices
            mask = torch.zeros_like(ci_scores)
            mask.scatter_(-1, topk_indices, 1.0)
            masks[mod_name] = mask

        mask_infos = make_mask_infos(masks)
        captured_masked = _capture_all_layer_mlp_outputs(
            spd_model, input_ids, mask_infos=mask_infos
        )

        total_mse += _avg_mse_across_layers(captured_orig, captured_masked)

    return total_mse / len(batches)


# =============================================================================
# Baselines — all layers
# =============================================================================


@torch.no_grad()
def eval_baselines(
    base_model,
    batches: list[torch.Tensor],
) -> dict[str, float]:
    """Compute baseline CE losses: original and zero-ablation of all layers."""
    original_forwards = {l: base_model.h[l].mlp.forward for l in LAYERS}

    original_loss = 0.0
    zero_loss = 0.0
    for input_ids in batches:
        original_loss += compute_ce_loss(base_model, input_ids)

        with ExitStack() as stack:
            for layer_idx in LAYERS:
                mlp = base_model.h[layer_idx].mlp

                def _make_zero(orig_fwd):
                    def _zero(hidden_states):
                        return torch.zeros_like(orig_fwd(hidden_states))
                    return _zero

                stack.enter_context(patched_forward(mlp, _make_zero(original_forwards[layer_idx])))
            zero_loss += compute_ce_loss(base_model, input_ids)

    n = len(batches)
    return {
        "original_ce": original_loss / n,
        "zero_ablation_ce": zero_loss / n,
    }


# =============================================================================
# Plotting
# =============================================================================


def plot_pareto_general(
    lines, special_points, baselines, xlabel, ylabel, title, save_path,
):
    """General Pareto plot.

    Args:
        lines: List of (x_values, y_values, marker_style, color, label) tuples.
        special_points: Dict of label -> (x, y, marker, color, size).
        baselines: Dict with "original_ce" and "zero_ablation_ce" keys, or None.
        xlabel, ylabel, title, save_path: Plot configuration.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for x_vals, y_vals, style, color, label in lines:
        ax.plot(x_vals, y_vals, style, color=color, label=label)

    for label, (x, y, marker, color, size) in special_points.items():
        ax.plot(x, y, marker, color=color, markersize=size, markeredgecolor="black",
                markeredgewidth=0.8, label=label, zorder=5)

    if baselines is not None:
        ax.axhline(baselines["original_ce"], color="gray", linestyle="--", alpha=0.7, label="Original CE")
        ax.axhline(baselines["zero_ablation_ce"], color="red", linestyle="--", alpha=0.5, label="Zero-ablation CE")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


# Line and special-point style definitions
LINE_DEFS = [
    ("transcoder", "o-", "tab:blue", "Transcoder"),
    ("clt", "X-", "tab:cyan", "CLT"),
    ("spd_both", "P-", "tab:purple", "SPD"),
    ("neuron", "d-", "tab:red", "Neurons"),
]

SP_STYLES = {
    "BatchTopKTranscoder (train k)": ("*", "tab:blue", 18),
    "CLT (train k)": ("*", "tab:cyan", 18),
    "SPD (CI>0.5)": ("*", "tab:purple", 18),
    "SPD (CI>0)": ("D", "tab:purple", 12),
}

SP_METHOD_MAP = {
    "BatchTopKTranscoder (train k)": "transcoder",
    "CLT (train k)": "clt",
    "SPD (CI>0.5)": "spd_both",
    "SPD (CI>0)": "spd_both",
}


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="L0 vs CE Pareto: SPD vs Transcoder (Pile/LlamaSimpleMLP, all layers)")
    parser.add_argument("--clt_path", type=str, default="checkpoints/clt_L0-3_4096_batchtopk_k16_0.0003_final")
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument(
        "--l0_values", type=int, nargs="+",
        default=[4, 8, 16, 24, 32, 50, 100, 200, 1000, 4096],
    )
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--save_path", type=str, default="experiments/exp_007_pareto_pile_all_layers/output/pareto_pile_all_layers.png")
    args = parser.parse_args()

    # Load SPD model (includes the base LlamaSimpleMLP)
    from analysis.collect_spd_activations import load_spd_model

    print("Loading SPD model...")
    spd_model, raw_config = load_spd_model(args.spd_run)
    spd_model.to(DEVICE)

    base_model = spd_model.target_model
    base_model.eval()

    # Module names for all layers
    all_cfc_names = [f"h.{l}.mlp.c_fc" for l in LAYERS]
    all_down_names = [f"h.{l}.mlp.down_proj" for l in LAYERS]
    all_module_names = all_cfc_names + all_down_names
    for name in all_module_names:
        assert name in spd_model.target_module_paths, f"{name} not in SPD model"

    # Load transcoders for all layers
    print("Loading transcoders for all layers...")
    transcoders = {}
    for layer_idx in LAYERS:
        path = LAYER_CHECKPOINT_MAP[layer_idx]
        print(f"  Layer {layer_idx}: {path}")
        tc = load_transcoder(path)
        tc.to(DEVICE)
        transcoders[layer_idx] = tc

    # Load CLT
    print(f"Loading CLT from {args.clt_path}...")
    clt = load_clt(args.clt_path)
    clt.to(DEVICE)

    # Load eval data (pre-tokenized Pile)
    print(f"Loading {args.n_eval_batches} eval batches (seq_len={args.seq_len})...")
    batches = get_eval_batches(args.n_eval_batches, args.batch_size, args.seq_len)

    # Baselines
    print("Computing baselines (zero-ablation of all layers)...")
    baselines = eval_baselines(base_model, batches)
    print(f"  Original CE: {baselines['original_ce']:.4f}")
    print(f"  Zero-ablation CE: {baselines['zero_ablation_ce']:.4f}")

    # MLP activations for MSE (all layers)
    print("Collecting MLP activations for MSE (all layers)...")
    mlp_activations = get_mlp_activations(base_model, batches)

    # Sweep L0
    tc_ces, clt_ces, n_ces = [], [], []
    both_indep_ces, both_indep_l0s = [], []
    tc_mses, clt_mses, n_mses = [], [], []
    both_indep_mses, both_indep_l0s_m = [], []

    header = f"{'L0':>6} | {'TC CE':>10} | {'CLT CE':>10} | {'SPD CE':>10} | {'SPD L0':>8} | {'Neur CE':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for k in args.l0_values:
        tc_ce = eval_transcoder_ce(base_model, transcoders, batches, k)
        clt_ce = eval_clt_ce(base_model, clt, batches, k)
        indep_avg_l0, indep_ce = eval_spd_both_ce(spd_model, batches, all_module_names, k)
        n_ce = eval_neuron_ce(base_model, batches, k)

        tc_mse = eval_transcoder_mse(transcoders, mlp_activations, k)
        clt_mse = eval_clt_mse(clt, mlp_activations, k)
        indep_avg_l0_m, indep_mse = eval_spd_both_mse(spd_model, batches, all_module_names, k)
        n_mse = eval_neuron_mse(base_model, mlp_activations, k)

        tc_ces.append(tc_ce)
        clt_ces.append(clt_ce)
        both_indep_ces.append(indep_ce)
        both_indep_l0s.append(indep_avg_l0)
        n_ces.append(n_ce)
        tc_mses.append(tc_mse)
        clt_mses.append(clt_mse)
        both_indep_mses.append(indep_mse)
        both_indep_l0s_m.append(indep_avg_l0_m)
        n_mses.append(n_mse)

        print(f"{k:>6} | {tc_ce:>10.4f} | {clt_ce:>10.4f} | {indep_ce:>10.4f} | {indep_avg_l0:>8.1f} | {n_ce:>10.4f}")

    # Special points (CE)
    print("\nSpecial points (CE):")
    sp_ce = {}

    train_ks = {l: transcoders[l].cfg.top_k for l in LAYERS}
    tc_l0, tc_train_ce = eval_transcoder_batchtopk_ce(base_model, transcoders, batches)
    sp_ce["BatchTopKTranscoder (train k)"] = (tc_l0, tc_train_ce)
    print(f"  BatchTopKTranscoder (train k={train_ks}): L0={tc_l0:.1f}, CE={tc_train_ce:.4f}")

    clt_l0, clt_train_ce = eval_clt_batchtopk_ce(base_model, clt, batches)
    sp_ce["CLT (train k)"] = (clt_l0, clt_train_ce)
    print(f"  CLT (train k={clt.cfg.top_k}): L0={clt_l0:.1f}, CE={clt_train_ce:.4f}")

    both_l0_05, both_ce_05 = eval_spd_ce_thresholded(spd_model, batches, all_module_names)
    sp_ce["SPD (CI>0.5)"] = (both_l0_05, both_ce_05)
    print(f"  SPD (CI>0.5): L0={both_l0_05:.1f}, CE={both_ce_05:.4f}")

    both_l0_0, both_ce_0 = eval_spd_ce_thresholded(spd_model, batches, all_module_names, threshold=0.0)
    sp_ce["SPD (CI>0)"] = (both_l0_0, both_ce_0)
    print(f"  SPD (CI>0): L0={both_l0_0:.1f}, CE={both_ce_0:.4f}")

    # Special points (MSE)
    print("\nSpecial points (MSE):")
    sp_mse = {}

    tc_l0_m, tc_train_mse = eval_transcoder_batchtopk_mse(transcoders, mlp_activations)
    sp_mse["BatchTopKTranscoder (train k)"] = (tc_l0_m, tc_train_mse)
    print(f"  BatchTopKTranscoder (train k={train_ks}): L0={tc_l0_m:.1f}, MSE={tc_train_mse:.6f}")

    clt_l0_m, clt_train_mse = eval_clt_batchtopk_mse(clt, mlp_activations)
    sp_mse["CLT (train k)"] = (clt_l0_m, clt_train_mse)
    print(f"  CLT (train k={clt.cfg.top_k}): L0={clt_l0_m:.1f}, MSE={clt_train_mse:.6f}")

    both_l0_m05, both_mse_05 = eval_spd_mse_thresholded(spd_model, batches, all_module_names)
    sp_mse["SPD (CI>0.5)"] = (both_l0_m05, both_mse_05)
    print(f"  SPD (CI>0.5): L0={both_l0_m05:.1f}, MSE={both_mse_05:.6f}")

    both_l0_m0, both_mse_0 = eval_spd_mse_thresholded(spd_model, batches, all_module_names, threshold=0.0)
    sp_mse["SPD (CI>0)"] = (both_l0_m0, both_mse_0)
    print(f"  SPD (CI>0): L0={both_l0_m0:.1f}, MSE={both_mse_0:.6f}")

    # Organize sweep data by method key
    sweep_ce = {
        "transcoder": ([float(x) for x in args.l0_values], tc_ces),
        "clt": ([float(x) for x in args.l0_values], clt_ces),
        "spd_both": (both_indep_l0s, both_indep_ces),
        "neuron": ([float(x) for x in args.l0_values], n_ces),
    }
    sweep_mse = {
        "transcoder": ([float(x) for x in args.l0_values], tc_mses),
        "clt": ([float(x) for x in args.l0_values], clt_mses),
        "spd_both": (both_indep_l0s_m, both_indep_mses),
        "neuron": ([float(x) for x in args.l0_values], n_mses),
    }

    # Compute x-axis scaling factors
    d_in = clt.cfg.input_size
    d_out = clt.cfg.output_size
    d_hidden = base_model.h[0].mlp.c_fc.weight.shape[0]
    n = len(LAYERS)

    x_factors = {
        "per_component": {
            "transcoder": 1, "clt": 1, "spd_both": 1, "neuron": 1,
        },
        "per_mlp": {
            "transcoder": 1, "clt": (n + 1) / 2, "spd_both": 2, "neuron": 1,
        },
        "total_params": {
            "transcoder": n * (d_in + d_out),
            "clt": sum(d_in + (n - i) * d_out for i in range(n)),
            "spd_both": n * (d_in + 2 * d_hidden + d_out),
            "neuron": n * (d_in + d_out),
        },
    }

    axis_configs = [
        ("per_component", "L0 (active components per module)", ""),
        ("per_mlp", "L0 (active components per MLP)", "_per_mlp"),
        ("total_params", "Total active parameters", "_total_params"),
    ]

    for axis_type, xlabel, suffix in axis_configs:
        factors = x_factors[axis_type]

        for metric, sweep_data, sp_raw, ylabel, title_metric in [
            ("ce", sweep_ce, sp_ce, "CE Loss",
             "CE Loss: SPD vs Transcoder (All Layers)"),
            ("mse", sweep_mse, sp_mse, "MLP Output MSE",
             "MLP Reconstruction MSE: SPD vs Transcoder (All Layers)"),
        ]:
            lines = []
            for method_key, style, color, label in LINE_DEFS:
                raw_x, y = sweep_data[method_key]
                f = factors[method_key]
                scaled_x = [x * f for x in raw_x]
                lines.append((scaled_x, y, style, color, label))

            scaled_sp = {}
            for sp_label, (l0, val) in sp_raw.items():
                method = SP_METHOD_MAP[sp_label]
                f = factors[method]
                marker, color, size = SP_STYLES[sp_label]
                scaled_sp[sp_label] = (l0 * f, val, marker, color, size)

            bl = baselines if metric == "ce" else None
            save = args.save_path.replace(".png", f"{suffix}_{metric}.png") if suffix else (
                args.save_path if metric == "ce" else args.save_path.replace(".png", "_mse.png")
            )

            plot_pareto_general(
                lines, scaled_sp, bl,
                xlabel=xlabel, ylabel=ylabel,
                title=f"{xlabel} vs {title_metric}",
                save_path=save,
            )


if __name__ == "__main__":
    main()
