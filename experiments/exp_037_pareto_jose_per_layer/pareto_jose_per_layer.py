"""Per-layer Pareto plots for jose's target model (t-9d2b8f02).

Like exp_035 but replacing only one MLP at a time, producing a 2x2 grid
(one subplot per layer). This isolates each layer's decomposition quality.

For CLT, all encoders run and full triangular decode is computed, but only
the target layer's MLP is replaced — capturing cross-layer contributions.

Usage:
    python experiments/exp_037_pareto_jose_per_layer/pareto_jose_per_layer.py
"""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig, CLTConfig
from nn_decompositions.clt import CrossLayerTranscoder
from spd.models.components import make_mask_infos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]


# =============================================================================
# Artifact downloading (reused from exp_035)
# =============================================================================


def download_wandb_artifact(project: str, artifact_name: str, dest: Path) -> Path:
    if dest.exists() and (dest / "encoder.pt").exists():
        print(f"  Using cached {dest}")
        return dest
    api = wandb.Api()
    artifact = api.artifact(f"{project}/{artifact_name}")
    artifact.download(root=str(dest))
    print(f"  Downloaded {artifact_name} -> {dest}")
    return dest


def download_transcoders(project: str) -> dict[int, dict[int, Path]]:
    """Returns {top_k: {layer: path}}."""
    api = wandb.Api()
    runs = api.runs(project)
    tc_paths: dict[int, dict[int, Path]] = {}
    for run in runs:
        if run.state != "finished":
            continue
        name = run.name
        if not name.startswith("tc_k"):
            continue
        top_k = run.config.get("top_k")
        assert top_k is not None
        arts = [a for a in run.logged_artifacts() if a.type == "model"]
        layer_arts = {}
        for a in arts:
            aname = a.name.split(":")[0]
            for layer_idx in LAYERS:
                if f"layer{layer_idx}_final" in aname:
                    layer_arts[layer_idx] = a
                    break
        assert set(layer_arts.keys()) == set(LAYERS), (
            f"Expected all layers in run {name}, got {set(layer_arts.keys())}"
        )
        layer_paths = {}
        for layer_idx in LAYERS:
            dest = Path(f"checkpoints/jose_tc_{name}_layer{layer_idx}")
            download_wandb_artifact(project, layer_arts[layer_idx].name, dest)
            layer_paths[layer_idx] = dest
        tc_paths[top_k] = layer_paths
    return tc_paths


def download_clts(project: str) -> list[tuple[int, Path]]:
    """Returns [(top_k, path), ...]."""
    api = wandb.Api()
    runs = api.runs(project)
    clt_paths = []
    for run in runs:
        if run.state != "finished":
            continue
        name = run.name
        if not name.startswith("clt_k"):
            continue
        top_k = run.config.get("top_k")
        assert top_k is not None
        arts = [a for a in run.logged_artifacts() if a.type == "model"]
        final_arts = [a for a in arts if "final" in a.name]
        assert len(final_arts) == 1
        dest = Path(f"checkpoints/jose_clt_{name}")
        download_wandb_artifact(project, final_arts[0].name, dest)
        clt_paths.append((top_k, dest))
    clt_paths.sort(key=lambda x: x[0])
    return clt_paths


# =============================================================================
# Data loading
# =============================================================================


def get_eval_batches(n_batches: int, batch_size: int, seq_len: int) -> list[torch.Tensor]:
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


# =============================================================================
# Helpers
# =============================================================================


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


def compute_ce_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)).item()


# =============================================================================
# MLP activation collection
# =============================================================================


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


# =============================================================================
# Model loading
# =============================================================================

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


def load_clt(checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["layers"] = json.loads(cfg_dict["layers"])
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = CLTConfig(**cfg_dict)
    clt = CrossLayerTranscoder(cfg)
    clt.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    clt.eval()
    return clt


# =============================================================================
# Single-layer transcoder eval
# =============================================================================


def _transcoder_batchtopk_recon(tc, x_in, k):
    use_pre_enc_bias = tc.cfg.pre_enc_bias and tc.input_size == tc.output_size
    x_enc = x_in - tc.b_dec if use_pre_enc_bias else x_in
    acts = F.relu(x_enc @ tc.W_enc)
    n_keep = k * acts.shape[0]
    if n_keep < acts.numel():
        topk = torch.topk(acts.flatten(), n_keep, dim=-1)
        acts = torch.zeros_like(acts.flatten()).scatter(-1, topk.indices, topk.values).reshape(acts.shape)
    return acts, acts @ tc.W_dec + tc.b_dec


@torch.no_grad()
def eval_transcoder_single_layer(
    base_model, tc, layer_idx: int, batches, mlp_activations,
) -> dict:
    """Evaluate a single transcoder on one layer. Returns {l0, ce, mse}."""
    total_ce, total_mse, total_l0 = 0.0, 0.0, 0.0
    k = tc.cfg.top_k

    for batch_idx, input_ids in enumerate(batches):
        mlp = base_model.h[layer_idx].mlp

        def _make_patched(tc_, input_size_, k_):
            def _patched(hidden_states):
                flat = hidden_states.reshape(-1, input_size_)
                _, recon = _transcoder_batchtopk_recon(tc_, flat, k_)
                return recon.reshape(hidden_states.shape)
            return _patched

        with patched_forward(mlp, _make_patched(tc, tc.cfg.input_size, k)):
            total_ce += compute_ce_loss(base_model, input_ids)

        mlp_in, mlp_out = mlp_activations[layer_idx][batch_idx]
        acts, recon = _transcoder_batchtopk_recon(tc, mlp_in, k)
        total_l0 += (acts > 0).float().sum(-1).mean().item()
        total_mse += F.mse_loss(recon, mlp_out).item()

    n = len(batches)
    return {"l0": total_l0 / n, "ce": total_ce / n, "mse": total_mse / n}


# =============================================================================
# Single-layer CLT eval
# =============================================================================


def _collect_rms2_outputs(base_model, input_ids):
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


def _clt_batchtopk_acts(clt, inputs, k):
    all_acts = []
    for i in range(clt.cfg.n_layers):
        pre_acts = F.relu(inputs[i] @ clt.W_enc[i] + clt.b_enc[i])
        n_keep = k * pre_acts.shape[0]
        if n_keep < pre_acts.numel():
            topk = torch.topk(pre_acts.flatten(), n_keep, dim=-1)
            acts = torch.zeros_like(pre_acts.flatten()).scatter(-1, topk.indices, topk.values).reshape(pre_acts.shape)
        else:
            acts = pre_acts
        all_acts.append(acts)
    return all_acts


@torch.no_grad()
def eval_clt_single_layer(
    base_model, clt: CrossLayerTranscoder, target_layer: int, batches, mlp_activations,
) -> dict:
    """Evaluate CLT replacing only one layer's MLP. Returns {l0, ce, mse}."""
    total_ce, total_mse, total_l0 = 0.0, 0.0, 0.0
    k = clt.cfg.top_k

    for batch_idx, input_ids in enumerate(batches):
        captured = _collect_rms2_outputs(base_model, input_ids)
        seq_shape = captured[LAYERS[0]].shape

        clt_inputs = [captured[l].reshape(-1, clt.cfg.input_size) for l in LAYERS]
        all_acts = _clt_batchtopk_acts(clt, clt_inputs, k)
        recons = clt.decode(all_acts)
        target_recon = recons[target_layer].reshape(seq_shape)

        # L0: encoder L0 at the target layer
        total_l0 += (all_acts[target_layer] > 0).float().sum(-1).mean().item()

        # CE: only patch target layer
        mlp = base_model.h[target_layer].mlp

        def _make_patched(recon):
            def _patched(hidden_states):
                return recon
            return _patched

        with patched_forward(mlp, _make_patched(target_recon)):
            total_ce += compute_ce_loss(base_model, input_ids)

        # MSE at target layer
        mlp_out = mlp_activations[target_layer][batch_idx][1]
        total_mse += F.mse_loss(recons[target_layer], mlp_out).item()

    n = len(batches)
    return {"l0": total_l0 / n, "ce": total_ce / n, "mse": total_mse / n}


# =============================================================================
# Single-layer SPD eval
# =============================================================================


@torch.no_grad()
def eval_spd_single_layer(
    spd_model, layer_idx: int, batches, mlp_activations, threshold: float,
) -> dict:
    """Evaluate SPD at a CI threshold, masking only one layer. Returns {l0, ce, mse}."""
    total_ce, total_mse, total_l0 = 0.0, 0.0, 0.0
    cfc_name = f"h.{layer_idx}.mlp.c_fc"
    down_name = f"h.{layer_idx}.mlp.down_proj"
    module_names = [cfc_name, down_name]
    # We also need the full set of module names for make_mask_infos —
    # non-target layers get identity masks (all ones).
    all_cfc = [f"h.{l}.mlp.c_fc" for l in LAYERS]
    all_down = [f"h.{l}.mlp.down_proj" for l in LAYERS]
    all_module_names = all_cfc + all_down

    for batch_idx, input_ids in enumerate(batches):
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        masks = {}
        for mod_name in all_module_names:
            ci_post = ci.lower_leaky[mod_name]
            if mod_name in module_names:
                mask = (ci_post > threshold).float()
                total_l0 += mask.sum(-1).mean().item()
            else:
                # Identity: keep all components for non-target layers
                mask = torch.ones_like(ci_post)
            masks[mod_name] = mask

        mask_infos = make_mask_infos(masks)
        logits = spd_model(input_ids, mask_infos=mask_infos)
        total_ce += compute_ce_from_logits(logits, input_ids)

        # MSE at target layer
        target_mlp = spd_model.target_model.h[layer_idx].mlp

        def _capture_mlp_output(model, ids, **kwargs):
            captured = {}
            def _hook(_mod, _inp, out_):
                captured["out"] = out_.detach()
            h = target_mlp.register_forward_hook(_hook)
            model(ids, **kwargs)
            h.remove()
            return captured["out"]

        orig_out = _capture_mlp_output(spd_model, input_ids)
        masked_out = _capture_mlp_output(spd_model, input_ids, mask_infos=mask_infos)
        total_mse += F.mse_loss(masked_out, orig_out).item()

    n = len(batches)
    avg_l0 = total_l0 / n / len(module_names)
    return {"l0": avg_l0, "ce": total_ce / n, "mse": total_mse / n}


# =============================================================================
# Single-layer neuron eval
# =============================================================================


def neuron_topk_reconstruction(mlp: nn.Module, x_in: torch.Tensor, k: int) -> torch.Tensor:
    h = mlp.gelu(mlp.c_fc(x_in))
    if k < h.shape[-1]:
        topk = torch.topk(h.abs(), k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk.indices, 1.0)
        h = h * mask
    return mlp.down_proj(h)


@torch.no_grad()
def eval_neuron_single_layer(
    base_model, layer_idx: int, batches, mlp_activations, k: int,
) -> dict:
    """Evaluate top-k neuron baseline on a single layer. Returns {l0, ce, mse}."""
    total_ce, total_mse = 0.0, 0.0

    for batch_idx, input_ids in enumerate(batches):
        mlp = base_model.h[layer_idx].mlp

        def _make_patched(mlp_):
            def _patched(hidden_states):
                return neuron_topk_reconstruction(mlp_, hidden_states, k)
            return _patched

        with patched_forward(mlp, _make_patched(mlp)):
            total_ce += compute_ce_loss(base_model, input_ids)

        mlp_in, mlp_out = mlp_activations[layer_idx][batch_idx]
        recon = neuron_topk_reconstruction(mlp, mlp_in, k)
        total_mse += F.mse_loss(recon, mlp_out).item()

    n = len(batches)
    return {"l0": float(k), "ce": total_ce / n, "mse": total_mse / n}


# =============================================================================
# Baselines (per-layer)
# =============================================================================


@torch.no_grad()
def eval_baselines_per_layer(base_model, batches) -> dict:
    """Returns {original_ce, zero_ablation_ce_per_layer: {layer: ce}}."""
    original_loss = 0.0
    zero_losses = {l: 0.0 for l in LAYERS}

    for input_ids in batches:
        original_loss += compute_ce_loss(base_model, input_ids)
        for layer_idx in LAYERS:
            mlp = base_model.h[layer_idx].mlp
            orig_fwd = mlp.forward

            def _make_zero(fwd):
                def _zero(hidden_states):
                    return torch.zeros_like(fwd(hidden_states))
                return _zero

            with patched_forward(mlp, _make_zero(orig_fwd)):
                zero_losses[layer_idx] += compute_ce_loss(base_model, input_ids)

    n = len(batches)
    return {
        "original_ce": original_loss / n,
        "zero_ablation_ce": {l: zero_losses[l] / n for l in LAYERS},
    }


# =============================================================================
# Plotting
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_STYLES = {
    "Transcoders": dict(marker="o", color="#1f77b4", linestyle="-", linewidth=1.8, markersize=8, zorder=5),
    "CLT":         dict(marker="X", color="#17becf", linestyle="-", linewidth=1.8, markersize=9, zorder=5),
    "SPD (CI>0.5)": dict(marker="P", color="#9467bd", linestyle="none", markersize=11, zorder=6),
    "SPD (CI>0)":   dict(marker="D", color="#9467bd", linestyle="none", markersize=9, zorder=6),
    "Neurons":     dict(marker="d", color="#d62728", linestyle="-", linewidth=1.8, markersize=8, zorder=4),
}

PLOT_ORDER = ["Neurons", "Transcoders", "CLT", "SPD (CI>0.5)", "SPD (CI>0)"]


def plot_per_layer(
    per_layer_points: dict[int, dict[str, list[dict]]],
    baselines: dict,
    y_key: str,
    ylabel: str,
    save_path: str,
):
    """2x2 grid of Pareto plots, one per layer."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)
    baseline_ce = baselines["original_ce"]
    ce_degradation = y_key == "ce"

    for layer_idx, ax in zip(LAYERS, axes.flat):
        points = per_layer_points[layer_idx]

        for label in PLOT_ORDER:
            pts = points.get(label, [])
            if not pts:
                continue
            pts_sorted = sorted(pts, key=lambda p: p["l0"])
            xs = [p["l0"] for p in pts_sorted]
            if ce_degradation:
                ys = [p["ce"] - baseline_ce for p in pts_sorted]
            else:
                ys = [p["mse"] for p in pts_sorted]
            style = METHOD_STYLES[label]
            ax.plot(xs, ys, label=label, markeredgecolor="white", markeredgewidth=0.8, **style)

        if ce_degradation:
            zero_deg = baselines["zero_ablation_ce"][layer_idx] - baseline_ce
            ax.axhline(zero_deg, color="#d62728", linestyle=":",
                       linewidth=1.0, alpha=0.5, label="Zero ablation", zorder=1)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(ticker.FixedLocator(
                [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
            ))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, _: f"{y:g}"
            ))

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x)}" if x == int(x) else f"{x:g}"
        ))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.tick_params(direction="in", which="both")
        ax.set_title(f"Layer {layer_idx}", fontsize=12)

    # Shared labels
    for ax in axes[1]:
        ax.set_xlabel("L0")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Per-layer Pareto plot for jose's target model")
    parser.add_argument("--project", type=str, default="mats-sprint/pile_local_sweep_jose")
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-55ea3f9b")
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--neuron_ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--save_path", type=str,
                        default="experiments/exp_037_pareto_jose_per_layer/output/pareto_jose_per_layer.png")
    args = parser.parse_args()

    from analysis.collect_spd_activations import load_spd_model

    print("Loading SPD model (jose)...")
    spd_model, raw_config = load_spd_model(args.spd_run)
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    print(f"\nDownloading transcoders from {args.project}...")
    tc_paths = download_transcoders(args.project)
    print(f"  Found top_k values: {sorted(tc_paths.keys())}")

    print(f"\nDownloading CLTs from {args.project}...")
    clt_paths = download_clts(args.project)
    print(f"  Found CLTs: {[(k, str(p)) for k, p in clt_paths]}")

    print(f"\nLoading {args.n_eval_batches} eval batches (seq_len={args.seq_len})...")
    batches = get_eval_batches(args.n_eval_batches, args.batch_size, args.seq_len)

    print("Computing per-layer baselines...")
    baselines = eval_baselines_per_layer(base_model, batches)
    print(f"  Original CE: {baselines['original_ce']:.4f}")
    for l in LAYERS:
        print(f"  Zero-ablation CE (layer {l}): {baselines['zero_ablation_ce'][l]:.4f}")

    print("Collecting MLP activations for MSE...")
    mlp_activations = get_mlp_activations(base_model, batches)

    # Build per-layer points: {layer: {method: [points]}}
    per_layer_points: dict[int, dict[str, list[dict]]] = {l: {} for l in LAYERS}

    # Transcoders
    for top_k in sorted(tc_paths.keys()):
        layer_paths = tc_paths[top_k]
        for layer_idx in LAYERS:
            tc = load_transcoder(str(layer_paths[layer_idx]))
            tc.to(DEVICE)
            print(f"  Evaluating TC k={top_k} layer={layer_idx}...")
            result = eval_transcoder_single_layer(base_model, tc, layer_idx, batches, mlp_activations)
            result["top_k"] = top_k
            per_layer_points[layer_idx].setdefault("Transcoders", []).append(result)
            print(f"    L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # CLTs
    for top_k, path in clt_paths:
        clt = load_clt(str(path))
        clt.to(DEVICE)
        for layer_idx in LAYERS:
            print(f"  Evaluating CLT k={top_k} layer={layer_idx}...")
            result = eval_clt_single_layer(base_model, clt, layer_idx, batches, mlp_activations)
            result["top_k"] = top_k
            per_layer_points[layer_idx].setdefault("CLT", []).append(result)
            print(f"    L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # SPD
    for threshold, label in [(0.5, "SPD (CI>0.5)"), (0.0, "SPD (CI>0)")]:
        for layer_idx in LAYERS:
            print(f"  Evaluating {label} layer={layer_idx}...")
            result = eval_spd_single_layer(spd_model, layer_idx, batches, mlp_activations, threshold)
            per_layer_points[layer_idx].setdefault(label, []).append(result)
            print(f"    L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # Neurons
    for k in args.neuron_ks:
        for layer_idx in LAYERS:
            print(f"  Evaluating Neuron k={k} layer={layer_idx}...")
            result = eval_neuron_single_layer(base_model, layer_idx, batches, mlp_activations, k)
            result["top_k"] = k
            per_layer_points[layer_idx].setdefault("Neurons", []).append(result)
            print(f"    L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # Plots
    plot_per_layer(
        per_layer_points, baselines,
        y_key="ce", ylabel="CE degradation (\u0394 from baseline)",
        save_path=args.save_path,
    )
    plot_per_layer(
        per_layer_points, baselines,
        y_key="mse", ylabel="MLP reconstruction MSE",
        save_path=args.save_path.replace(".png", "_mse.png"),
    )

    # Summary table
    print("\n" + "=" * 80)
    print("Per-layer summary")
    print(f"{'Method':<20} {'k':>6} {'Layer':>6} {'L0':>8} {'CE':>10} {'MSE':>12}")
    print("-" * 66)
    for layer_idx in LAYERS:
        for method in ["Transcoders", "CLT", "SPD (CI>0.5)", "SPD (CI>0)", "Neurons"]:
            for p in per_layer_points[layer_idx].get(method, []):
                k_str = str(p.get("top_k", ""))
                print(f"{method:<20} {k_str:>6} {layer_idx:>6} {p['l0']:>8.1f} {p['ce']:>10.4f} {p['mse']:>12.6f}")


if __name__ == "__main__":
    main()
