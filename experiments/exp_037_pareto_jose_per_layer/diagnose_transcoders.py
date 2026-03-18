"""Diagnose why transcoders get worse MSE/CE at higher k.

Checks per-layer: dead features, feature activation stats, decoder cosine
similarities, and per-example reconstruction quality.

Usage:
    python experiments/exp_037_pareto_jose_per_layer/diagnose_transcoders.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
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


def _transcoder_batchtopk_recon(tc, x_in, k):
    use_pre_enc_bias = tc.cfg.pre_enc_bias and tc.input_size == tc.output_size
    x_enc = x_in - tc.b_dec if use_pre_enc_bias else x_in
    acts = F.relu(x_enc @ tc.W_enc + tc.b_enc)
    n_keep = k * acts.shape[0]
    if n_keep < acts.numel():
        topk = torch.topk(acts.flatten(), n_keep, dim=-1)
        acts = torch.zeros_like(acts.flatten()).scatter(-1, topk.indices, topk.values).reshape(acts.shape)
    return acts, acts @ tc.W_dec + tc.b_dec


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
def diagnose_transcoder(tc, layer_idx: int, mlp_activations, top_ks: list[int]):
    """Run diagnostics on a single transcoder at multiple k values."""
    dict_size = tc.cfg.dict_size

    # Collect all activations for this layer across batches
    all_mlp_in = torch.cat([pair[0] for pair in mlp_activations[layer_idx]], dim=0)
    all_mlp_out = torch.cat([pair[1] for pair in mlp_activations[layer_idx]], dim=0)
    n_tokens = all_mlp_in.shape[0]

    # Pre-encode (before top-k): get all pre-activation values
    use_pre_enc_bias = tc.cfg.pre_enc_bias and tc.input_size == tc.output_size
    x_enc = all_mlp_in - tc.b_dec if use_pre_enc_bias else all_mlp_in
    pre_acts = F.relu(x_enc @ tc.W_enc + tc.b_enc)  # (n_tokens, dict_size)

    # Feature activation stats (before any top-k)
    ever_active = (pre_acts > 0).any(dim=0)  # (dict_size,)
    n_ever_active = ever_active.sum().item()
    n_dead = dict_size - n_ever_active
    activation_freq = (pre_acts > 0).float().mean(dim=0)  # per-feature firing rate
    mean_act_when_active = []
    for f in range(dict_size):
        mask = pre_acts[:, f] > 0
        if mask.any():
            mean_act_when_active.append(pre_acts[:, f][mask].mean().item())
        else:
            mean_act_when_active.append(0.0)
    mean_act_when_active = torch.tensor(mean_act_when_active)

    print(f"\n  Layer {layer_idx}: dict_size={dict_size}, tokens={n_tokens}")
    print(f"    Dead features (never fire on eval set): {n_dead}/{dict_size} ({100*n_dead/dict_size:.1f}%)")
    print(f"    Alive features: {n_ever_active}")
    print(f"    Mean firing rate (alive only): {activation_freq[ever_active].mean():.4f}")
    print(f"    Median firing rate (alive only): {activation_freq[ever_active].median():.4f}")

    # Decoder cosine similarity analysis
    W_dec = tc.W_dec.data  # (dict_size, output_size)
    W_dec_normed = W_dec / (W_dec.norm(dim=-1, keepdim=True) + 1e-8)
    # Sample alive features for cosine sim (full matrix is too big)
    alive_idxs = ever_active.nonzero().squeeze(-1)
    if len(alive_idxs) > 500:
        perm = torch.randperm(len(alive_idxs))[:500]
        sample_idxs = alive_idxs[perm]
    else:
        sample_idxs = alive_idxs
    cos_sim = W_dec_normed[sample_idxs] @ W_dec_normed[sample_idxs].T
    # Zero out diagonal
    cos_sim.fill_diagonal_(0)
    print(f"    Decoder cos sim (alive, off-diag): mean={cos_sim.mean():.4f}, "
          f"max={cos_sim.max():.4f}, >0.5: {(cos_sim.abs() > 0.5).sum().item()//2} pairs")

    # Now evaluate at different k values
    for k in top_ks:
        acts, recon = _transcoder_batchtopk_recon(tc, all_mlp_in, k)
        mse = F.mse_loss(recon, all_mlp_out).item()
        l0 = (acts > 0).float().sum(-1).mean().item()

        # Which features actually get selected?
        selected = (acts > 0).any(dim=0)
        n_selected = selected.sum().item()

        # Per-example L0 distribution
        per_example_l0 = (acts > 0).float().sum(-1)
        l0_std = per_example_l0.std().item()
        l0_min = per_example_l0.min().item()
        l0_max = per_example_l0.max().item()

        # Per-example MSE distribution
        per_example_mse = (recon - all_mlp_out).pow(2).mean(-1)
        mse_median = per_example_mse.median().item()
        mse_p95 = per_example_mse.quantile(0.95).item()
        mse_p99 = per_example_mse.quantile(0.99).item()

        # Reconstruction norm vs target norm
        recon_norm = recon.norm(dim=-1).mean().item()
        target_norm = all_mlp_out.norm(dim=-1).mean().item()

        # b_dec contribution
        b_dec_norm = tc.b_dec.norm().item()
        b_dec_mse = F.mse_loss(tc.b_dec.unsqueeze(0).expand_as(all_mlp_out), all_mlp_out).item()

        print(f"\n    k={k}: MSE={mse:.6f}, L0={l0:.1f}")
        print(f"      Features used (ever selected): {n_selected}/{n_ever_active} alive")
        print(f"      Per-example L0: mean={l0:.1f}, std={l0_std:.1f}, min={l0_min:.0f}, max={l0_max:.0f}")
        print(f"      Per-example MSE: median={mse_median:.6f}, p95={mse_p95:.6f}, p99={mse_p99:.6f}")
        print(f"      Recon norm={recon_norm:.2f}, target norm={target_norm:.2f}, ratio={recon_norm/target_norm:.3f}")
        print(f"      b_dec norm={b_dec_norm:.4f}, b_dec-only MSE={b_dec_mse:.6f}")


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

    # Free base model memory
    del spd_model, base_model
    torch.cuda.empty_cache()

    top_ks_to_test = [8, 16, 32, 64]

    # Diagnose 4k transcoders
    print("\n" + "=" * 80)
    print("4k dict_size transcoders (pile_local_sweep_jose)")
    print("=" * 80)
    for layer_idx in LAYERS:
        # Load the k=8 transcoder (all k values share the same weights,
        # only differing in top_k config — actually no, they're different models!)
        for k in top_ks_to_test:
            tc = load_transcoder(f"checkpoints/jose_tc_tc_k{k}_layer{layer_idx}")
            tc.to(DEVICE)
            print(f"\n  --- TC trained with k={k} ---")
            diagnose_transcoder(tc, layer_idx, mlp_activations, [k])
            del tc
            torch.cuda.empty_cache()

    # Also test: what if we evaluate the k=8 model at higher k?
    print("\n" + "=" * 80)
    print("Cross-k evaluation: k=8 model evaluated at all k values")
    print("=" * 80)
    for layer_idx in [2, 3]:  # Focus on problematic layers
        tc = load_transcoder(f"checkpoints/jose_tc_tc_k8_layer{layer_idx}")
        tc.to(DEVICE)
        diagnose_transcoder(tc, layer_idx, mlp_activations, top_ks_to_test)
        del tc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
