"""Pareto plot with only naturally-trained operating points (exp_011).

Each decomposition method is evaluated at its trained operating point — no
post-hoc L0 sweeps.

- SPD: 2 points (CI > 0.5 and CI > 0 thresholds across all decomposed modules)
- Transcoders: All runs from mats-sprint/pile_transcoder_sweep3, each evaluated
  at training-time batch-top-k on its specific layer
- CLTs: All runs from mats-sprint/pile_clt, each evaluated at training-time
  top-k across all layers

Usage:
    python experiments/pareto_pile_natural.py
"""

import argparse
import json
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import fields as dataclass_fields
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from clt import CrossLayerTranscoder
from config import CLTConfig, EncoderConfig
from transcoder import (
    BatchTopKTranscoder,
    JumpReLUTranscoder,
    TopKTranscoder,
    VanillaTranscoder,
)
from spd.models.components import make_mask_infos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]

ENCODER_CLASSES = {
    "vanilla": VanillaTranscoder,
    "topk": TopKTranscoder,
    "batchtopk": BatchTopKTranscoder,
    "jumprelu": JumpReLUTranscoder,
}


# ====================================================================
# Data loading
# ====================================================================


def get_eval_batches(
    n_batches: int, batch_size: int, seq_len: int
) -> list[torch.Tensor]:
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


# ====================================================================
# Helpers
# ====================================================================


@contextmanager
def patched_forward(module, patched_fn):
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
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)
    ).item()


def compute_ce_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)
    ).item()


# ====================================================================
# Wandb download helpers
# ====================================================================


def _filter_config(cfg_dict: dict, config_class) -> dict:
    known = {f.name for f in dataclass_fields(config_class)}
    return {k: v for k, v in cfg_dict.items() if k in known}


def _parse_dtype(cfg_dict: dict) -> dict:
    cfg_dict = dict(cfg_dict)
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    if isinstance(dtype_str, str):
        cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    return cfg_dict


def download_all_transcoders(
    project: str,
) -> list[tuple[int, object, str]]:
    """Download all transcoder checkpoints from a wandb project.

    Returns list of (layer, transcoder, run_name).
    """
    import wandb

    api = wandb.Api()
    runs = api.runs(project)

    results = []
    for run in tqdm(list(runs), desc="Downloading transcoders"):
        artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
        if not artifacts:
            print(f"  Skip {run.name}: no model artifacts")
            continue

        artifact = artifacts[-1]
        artifact_dir = Path(artifact.download())

        with open(artifact_dir / "config.json") as f:
            cfg_dict = json.load(f)

        layer = run.config.get("layer") or cfg_dict.get("layer")
        assert layer is not None, f"No layer info for run {run.name}"
        layer = int(layer)

        cfg_dict = _parse_dtype(cfg_dict)
        cfg_dict = _filter_config(cfg_dict, EncoderConfig)
        cfg = EncoderConfig(**cfg_dict)

        transcoder = ENCODER_CLASSES[cfg.encoder_type](cfg)
        state_dict = torch.load(
            artifact_dir / "encoder.pt", map_location=DEVICE, weights_only=True
        )
        transcoder.load_state_dict(state_dict)
        transcoder.eval().to(DEVICE)

        results.append((layer, transcoder, run.name))
        print(
            f"  Layer {layer}: {run.name} "
            f"(type={cfg.encoder_type}, top_k={cfg.top_k}, dict={cfg.dict_size})"
        )

    return results


def download_all_clts(project: str) -> list[tuple[object, str]]:
    """Download all CLT checkpoints from a wandb project.

    Returns list of (clt, run_name).
    """
    import wandb
    import ast

    api = wandb.Api()
    runs = api.runs(project)

    results = []
    for run in tqdm(list(runs), desc="Downloading CLTs"):
        artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
        if not artifacts:
            print(f"  Skip {run.name}: no model artifacts")
            continue

        artifact = artifacts[-1]
        artifact_dir = Path(artifact.download())

        with open(artifact_dir / "config.json") as f:
            cfg_dict = json.load(f)

        cfg_dict = _parse_dtype(cfg_dict)

        # layers may be stored as string repr of list
        if isinstance(cfg_dict.get("layers"), str):
            cfg_dict["layers"] = ast.literal_eval(cfg_dict["layers"])

        cfg_dict = _filter_config(cfg_dict, CLTConfig)
        cfg = CLTConfig(**cfg_dict)

        clt = CrossLayerTranscoder(cfg)
        state_dict = torch.load(
            artifact_dir / "encoder.pt", map_location=DEVICE, weights_only=True
        )
        clt.load_state_dict(state_dict)
        clt.eval().to(DEVICE)

        results.append((clt, run.name))
        print(
            f"  {run.name} "
            f"(layers={cfg.layers}, type={cfg.encoder_type}, top_k={cfg.top_k})"
        )

    return results


# ====================================================================
# Evaluation: Transcoders
# ====================================================================


@torch.no_grad()
def eval_transcoder_natural_ce(
    base_model,
    transcoder,
    batches: list[torch.Tensor],
    layer: int,
) -> tuple[float, float]:
    """Evaluate transcoder at training-time sparsity, patching one layer.

    Returns (actual_l0, ce_loss).
    """
    mlp = base_model.h[layer].mlp
    input_size = transcoder.cfg.input_size

    total_loss = 0.0
    total_l0 = 0.0

    for input_ids in batches:
        batch_l0 = {}

        def _patched(hidden_states, tc=transcoder, bl0=batch_l0):
            flat = hidden_states.reshape(-1, input_size)
            acts = tc.encode(flat)
            bl0["l0"] = (acts > 0).float().sum(-1).mean().item()
            recon = tc.decode(acts)
            return recon.reshape(hidden_states.shape)

        with patched_forward(mlp, _patched):
            total_loss += compute_ce_loss(base_model, input_ids)

        total_l0 += batch_l0["l0"]

    n = len(batches)
    return total_l0 / n, total_loss / n


# ====================================================================
# Evaluation: CLTs
# ====================================================================


@torch.no_grad()
def eval_clt_natural_ce(
    base_model,
    clt,
    batches: list[torch.Tensor],
) -> tuple[float, float]:
    """Evaluate CLT at training-time sparsity, patching all CLT layers.

    Returns (actual_l0_per_layer, ce_loss).
    L0 is normalized per layer for comparability with single-layer methods.
    """
    layers = clt.cfg.layers

    total_loss = 0.0
    total_l0 = 0.0

    for input_ids in batches:
        # Capture pre-MLP residual streams at all layers
        layer_inputs = {}
        hooks = []
        for layer in layers:

            def _hook(_mod, _inp, out, l=layer):
                layer_inputs[l] = out.detach()

            h = base_model.h[layer].rms_2.register_forward_hook(_hook)
            hooks.append(h)

        base_model(input_ids)
        for h in hooks:
            h.remove()

        # Encode all layers
        flat_inputs = [
            layer_inputs[l].reshape(-1, clt.cfg.input_size) for l in layers
        ]
        all_acts = [clt.encode_layer(x, i) for i, x in enumerate(flat_inputs)]

        # L0 per layer (averaged)
        acts_cat = torch.cat(all_acts, dim=-1)
        total_l0 += (acts_cat != 0).float().sum(-1).mean().item() / len(layers)

        # Decode
        reconstructions = clt.decode(all_acts)
        seq_shape = layer_inputs[layers[0]].shape
        reconstructions = [r.reshape(seq_shape) for r in reconstructions]

        # Patch all MLPs simultaneously
        def _make_const_fn(tensor):
            return lambda *a, **kw: tensor

        with ExitStack() as stack:
            for layer, recon in zip(layers, reconstructions):
                stack.enter_context(
                    patched_forward(base_model.h[layer].mlp, _make_const_fn(recon))
                )
            total_loss += compute_ce_loss(base_model, input_ids)

    n = len(batches)
    return total_l0 / n, total_loss / n


# ====================================================================
# Evaluation: SPD
# ====================================================================


@torch.no_grad()
def eval_spd_natural_ce(
    spd_model,
    batches: list[torch.Tensor],
    threshold: float,
) -> tuple[float, float]:
    """Evaluate SPD with CI thresholding across all decomposed modules.

    Returns (avg_l0_per_module, ce_loss).
    """
    module_names = list(spd_model.module_to_c.keys())

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


# ====================================================================
# Baselines
# ====================================================================


@torch.no_grad()
def eval_baselines(
    base_model, batches: list[torch.Tensor]
) -> dict[str, float]:
    total_orig = 0.0
    total_zero = 0.0

    for input_ids in batches:
        total_orig += compute_ce_loss(base_model, input_ids)

        with ExitStack() as stack:
            for layer in LAYERS:
                mlp = base_model.h[layer].mlp
                orig_fwd = mlp.forward

                def _zero(hs, fwd=orig_fwd):
                    return torch.zeros_like(fwd(hs))

                stack.enter_context(patched_forward(mlp, _zero))
            total_zero += compute_ce_loss(base_model, input_ids)

    n = len(batches)
    return {
        "original_ce": total_orig / n,
        "zero_ablation_ce": total_zero / n,
    }


# ====================================================================
# Plotting
# ====================================================================


def plot_natural_pareto(
    spd_points: dict[str, tuple[float, float]],
    tc_points: list[tuple[float, float, str]],
    clt_points: list[tuple[float, float, str]],
    baselines: dict[str, float],
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(
        baselines["original_ce"],
        color="gray", linestyle="--", alpha=0.7, label="Original CE",
    )
    ax.axhline(
        baselines["zero_ablation_ce"],
        color="red", linestyle="--", alpha=0.5, label="Zero-ablation CE",
    )

    for label, (l0, ce) in spd_points.items():
        ax.plot(
            l0, ce, "*", color="tab:orange", markersize=18,
            markeredgecolor="black", markeredgewidth=0.8,
            label=f"SPD ({label})", zorder=5,
        )

    if tc_points:
        tc_l0s = [p[0] for p in tc_points]
        tc_ces = [p[1] for p in tc_points]
        ax.scatter(
            tc_l0s, tc_ces, marker="o", color="tab:blue", s=100,
            edgecolors="black", linewidths=0.8, label="Transcoder", zorder=5,
        )

    if clt_points:
        clt_l0s = [p[0] for p in clt_points]
        clt_ces = [p[1] for p in clt_points]
        ax.scatter(
            clt_l0s, clt_ces, marker="s", color="tab:green", s=100,
            edgecolors="black", linewidths=0.8, label="CLT", zorder=5,
        )

    ax.set_xlabel("L0 (active features per token)")
    ax.set_ylabel("CE Loss")
    ax.set_title("Naturally Trained Operating Points: L0 vs CE Loss")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


# ====================================================================
# Main
# ====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Pareto plot with naturally-trained operating points"
    )
    parser.add_argument(
        "--spd_run", type=str, default="goodfire/spd/s-275c8f21",
    )
    parser.add_argument(
        "--tc_project", type=str, default="mats-sprint/pile_transcoder_sweep3",
    )
    parser.add_argument(
        "--clt_project", type=str, default="mats-sprint/pile_clt",
    )
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument(
        "--save_path", type=str, default="experiments/pareto_pile_natural.png",
    )
    args = parser.parse_args()

    from analysis.collect_spd_activations import load_spd_model

    # Load SPD model (includes the base LlamaSimpleMLP)
    print("Loading SPD model...")
    spd_model, _ = load_spd_model(args.spd_run)
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()
    print(f"  SPD modules: {list(spd_model.module_to_c.keys())}")

    # Load eval data
    print(f"\nLoading {args.n_eval_batches} eval batches (seq_len={args.seq_len})...")
    batches = get_eval_batches(args.n_eval_batches, args.batch_size, args.seq_len)

    # Baselines
    print("\nComputing baselines...")
    baselines = eval_baselines(base_model, batches)
    print(f"  Original CE: {baselines['original_ce']:.4f}")
    print(f"  Zero-ablation CE: {baselines['zero_ablation_ce']:.4f}")

    # SPD (2 naturally-trained points)
    print("\nEvaluating SPD...")
    spd_points = {}
    for label, threshold in [("CI > 0.5", 0.5), ("CI > 0", 0.0)]:
        l0, ce = eval_spd_natural_ce(spd_model, batches, threshold)
        spd_points[label] = (l0, ce)
        print(f"  {label}: L0={l0:.1f}, CE={ce:.4f}")

    # Transcoders (one point per run, at training-time k)
    print(f"\nDownloading transcoders from {args.tc_project}...")
    tc_data = download_all_transcoders(args.tc_project)

    print("\nEvaluating transcoders...")
    tc_points = []
    for layer, transcoder, name in tc_data:
        l0, ce = eval_transcoder_natural_ce(base_model, transcoder, batches, layer)
        tc_points.append((l0, ce, name))
        print(f"  {name} (layer {layer}): L0={l0:.1f}, CE={ce:.4f}")

    # CLTs (one point per run, at training-time k)
    print(f"\nDownloading CLTs from {args.clt_project}...")
    clt_data = download_all_clts(args.clt_project)

    print("\nEvaluating CLTs...")
    clt_points = []
    for clt, name in clt_data:
        l0, ce = eval_clt_natural_ce(base_model, clt, batches)
        clt_points.append((l0, ce, name))
        print(f"  {name}: L0={l0:.1f}, CE={ce:.4f}")

    # Plot
    plot_natural_pareto(spd_points, tc_points, clt_points, baselines, args.save_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary of naturally-trained points:")
    print("=" * 60)
    print(f"\nSPD ({len(spd_points)} points):")
    for label, (l0, ce) in spd_points.items():
        print(f"  {label}: L0={l0:.1f}, CE={ce:.4f}")
    print(f"\nTranscoders ({len(tc_points)} points):")
    for l0, ce, name in tc_points:
        print(f"  {name}: L0={l0:.1f}, CE={ce:.4f}")
    print(f"\nCLTs ({len(clt_points)} points):")
    for l0, ce, name in clt_points:
        print(f"  {name}: L0={l0:.1f}, CE={ce:.4f}")


if __name__ == "__main__":
    main()
