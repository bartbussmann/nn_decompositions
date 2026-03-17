"""Evaluate transcoders using JumpReLU threshold inference instead of BatchTopK.

Estimates threshold θ = E[min positive activation] over calibration batches,
then uses JumpReLU(z, θ) = z * (z > θ) at inference time. This removes the
batch dependency introduced by BatchTopK.

Usage:
    python experiments/exp_037_pareto_jose_per_layer/eval_with_threshold.py
"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
ENCODER_CLASSES = {"batchtopk": BatchTopKTranscoder}


def load_transcoder(checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    encoder = ENCODER_CLASSES[cfg.encoder_type](cfg)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    encoder.eval()
    return encoder


def _pre_activations(tc, x_in):
    """Compute pre-topk activations (ReLU output before sparsification)."""
    use_pre_enc_bias = tc.cfg.pre_enc_bias and tc.input_size == tc.output_size
    x_enc = x_in - tc.b_dec if use_pre_enc_bias else x_in
    return F.relu(x_enc @ tc.W_enc)


def _batchtopk_recon(tc, x_in, k):
    """Standard BatchTopK reconstruction."""
    acts = _pre_activations(tc, x_in)
    n_keep = k * acts.shape[0]
    if n_keep < acts.numel():
        topk = torch.topk(acts.flatten(), n_keep, dim=-1)
        acts = torch.zeros_like(acts.flatten()).scatter(-1, topk.indices, topk.values).reshape(acts.shape)
    return acts, acts @ tc.W_dec + tc.b_dec


def _jumprelu_recon(tc, x_in, threshold):
    """JumpReLU reconstruction: zero out activations below threshold."""
    acts = _pre_activations(tc, x_in)
    acts = acts * (acts > threshold).float()
    return acts, acts @ tc.W_dec + tc.b_dec


@torch.no_grad()
def estimate_threshold(tc, mlp_inputs, n_calibration_batches: int = 10, batch_size: int = 4096):
    """Estimate θ = E[min positive activation per example] over calibration data."""
    min_positives = []
    k = tc.cfg.top_k

    for i in range(min(n_calibration_batches, len(mlp_inputs))):
        x_in = mlp_inputs[i]
        acts = _pre_activations(tc, x_in)

        # BatchTopK to get the activated values
        n_keep = k * acts.shape[0]
        if n_keep < acts.numel():
            topk = torch.topk(acts.flatten(), n_keep, dim=-1)
            sparse_acts = torch.zeros_like(acts.flatten()).scatter(
                -1, topk.indices, topk.values
            ).reshape(acts.shape)
        else:
            sparse_acts = acts

        # For each example, find the minimum positive activation
        for j in range(sparse_acts.shape[0]):
            row = sparse_acts[j]
            positives = row[row > 0]
            if len(positives) > 0:
                min_positives.append(positives.min().item())

    threshold = sum(min_positives) / len(min_positives)
    return threshold


@contextmanager
def patched_forward(module: nn.Module, patched_fn):
    original = module.forward
    module.forward = patched_fn
    try:
        yield
    finally:
        module.forward = original


def compute_ce_loss(model, input_ids: torch.Tensor) -> float:
    logits, _ = model(input_ids)
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)).item()


def get_eval_batches(n_batches: int, batch_size: int, seq_len: int):
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
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


@torch.no_grad()
def get_mlp_activations(model, batches):
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
            mlp_in = captured[f"mlp_in_{layer_idx}"].reshape(-1, captured[f"mlp_in_{layer_idx}"].shape[-1])
            mlp_out = captured[f"mlp_out_{layer_idx}"].reshape(-1, captured[f"mlp_out_{layer_idx}"].shape[-1])
            all_pairs[layer_idx].append((mlp_in, mlp_out))
    return all_pairs


@torch.no_grad()
def eval_single_layer(base_model, tc, layer_idx, batches, mlp_activations, mode, threshold=None):
    """Evaluate a transcoder on one layer with either batchtopk or jumprelu."""
    total_ce, total_mse, total_l0 = 0.0, 0.0, 0.0
    k = tc.cfg.top_k

    for batch_idx, input_ids in enumerate(batches):
        mlp_in, mlp_out = mlp_activations[layer_idx][batch_idx]

        if mode == "batchtopk":
            acts, recon_flat = _batchtopk_recon(tc, mlp_in, k)
        else:
            acts, recon_flat = _jumprelu_recon(tc, mlp_in, threshold)

        total_l0 += (acts > 0).float().sum(-1).mean().item()
        total_mse += F.mse_loss(recon_flat, mlp_out).item()

        # CE: patch the MLP
        mlp = base_model.h[layer_idx].mlp
        recon_shaped = recon_flat.reshape(input_ids.shape[0], input_ids.shape[1], -1)

        def _make_patched(r):
            def _patched(hidden_states):
                return r
            return _patched

        with patched_forward(mlp, _make_patched(recon_shaped)):
            total_ce += compute_ce_loss(base_model, input_ids)

    n = len(batches)
    recon_norm = recon_flat.norm(dim=-1).mean().item()
    target_norm = mlp_out.norm(dim=-1).mean().item()
    return {
        "l0": total_l0 / n,
        "ce": total_ce / n,
        "mse": total_mse / n,
        "recon_norm": recon_norm,
        "target_norm": target_norm,
    }


def main():
    from analysis.collect_spd_activations import load_spd_model

    print("Loading base model...")
    spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    print("Loading eval batches...")
    batches = get_eval_batches(10, 8, 512)

    print("Collecting MLP activations...")
    mlp_activations = get_mlp_activations(base_model, batches)

    top_ks = [8, 16, 32, 64]
    baseline_ce = 0.0
    for input_ids in batches:
        baseline_ce += compute_ce_loss(base_model, input_ids)
    baseline_ce /= len(batches)
    print(f"Baseline CE: {baseline_ce:.4f}")

    print("\n" + "=" * 90)
    print(f"{'Layer':>5} {'k':>4} {'Mode':>10} {'L0':>8} {'CE':>8} {'CE_deg':>8} "
          f"{'MSE':>10} {'R_norm':>8} {'T_norm':>8} {'Ratio':>7}")
    print("-" * 90)

    for layer_idx in LAYERS:
        mlp_inputs = [mlp_activations[layer_idx][i][0] for i in range(len(batches))]

        for k in top_ks:
            tc = load_transcoder(f"checkpoints/jose_tc_tc_k{k}_layer{layer_idx}")
            tc.to(DEVICE)

            # Estimate threshold
            threshold = estimate_threshold(tc, mlp_inputs)

            # Eval with BatchTopK
            r_btk = eval_single_layer(base_model, tc, layer_idx, batches, mlp_activations, "batchtopk")
            print(f"{layer_idx:>5} {k:>4} {'batchtopk':>10} {r_btk['l0']:>8.1f} {r_btk['ce']:>8.4f} "
                  f"{r_btk['ce']-baseline_ce:>8.4f} {r_btk['mse']:>10.6f} "
                  f"{r_btk['recon_norm']:>8.2f} {r_btk['target_norm']:>8.2f} "
                  f"{r_btk['recon_norm']/r_btk['target_norm']:>7.3f}")

            # Eval with JumpReLU threshold
            r_jr = eval_single_layer(base_model, tc, layer_idx, batches, mlp_activations, "jumprelu", threshold)
            print(f"{layer_idx:>5} {k:>4} {'jumprelu':>10} {r_jr['l0']:>8.1f} {r_jr['ce']:>8.4f} "
                  f"{r_jr['ce']-baseline_ce:>8.4f} {r_jr['mse']:>10.6f} "
                  f"{r_jr['recon_norm']:>8.2f} {r_jr['target_norm']:>8.2f} "
                  f"{r_jr['recon_norm']/r_jr['target_norm']:>7.3f}  θ={threshold:.4f}")

            del tc
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
