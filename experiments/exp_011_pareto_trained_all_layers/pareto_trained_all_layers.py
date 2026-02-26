"""Pareto plot using only naturally-trained checkpoints (no synthetic L0 sweep).

Each point on the plot is a model evaluated at its native (training) top_k:
- Transcoders: downloaded from wandb mats-sprint/pile_transcoder_sweep3
  (4 top_k values × 4 layers, all layers replaced simultaneously)
- CLTs: downloaded from wandb mats-sprint/pile_clt
  (each CLT covers all 4 layers)
- SPD: 2 threshold points (CI>0.5 and CI>0)

Usage:
    python experiments/exp_011_pareto_trained_all_layers/pareto_trained_all_layers.py
"""

import argparse
import json
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from transcoder import BatchTopKTranscoder
from config import EncoderConfig, CLTConfig
from clt import CrossLayerTranscoder
from spd.models.components import make_mask_infos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]


# =============================================================================
# Artifact downloading
# =============================================================================


def download_wandb_artifact(project: str, artifact_name: str, dest: Path) -> Path:
    """Download a wandb artifact and return the local path."""
    if dest.exists() and (dest / "encoder.pt").exists():
        print(f"  Using cached {dest}")
        return dest
    api = wandb.Api()
    artifact = api.artifact(f"{project}/{artifact_name}")
    artifact.download(root=str(dest))
    print(f"  Downloaded {artifact_name} -> {dest}")
    return dest


def download_transcoders(project: str) -> dict[int, dict[int, Path]]:
    """Download all transcoder artifacts. Returns {top_k: {layer: path}}."""
    api = wandb.Api()
    runs = api.runs(project)

    tc_paths: dict[int, dict[int, Path]] = {}
    for run in runs:
        if run.state != "finished":
            continue
        cfg = run.config
        top_k = cfg.get("top_k")
        run_name = run.name
        # Parse layer from run name: ...L{layer}_...
        layer = None
        for part in run_name.split("_"):
            if part.startswith("L") and part[1:].isdigit():
                layer = int(part[1:])
                break
        assert layer is not None, f"Could not parse layer from run name: {run_name}"
        assert top_k is not None, f"No top_k in config for run: {run_name}"

        arts = [a for a in run.logged_artifacts() if a.type == "model"]
        assert len(arts) == 1, f"Expected 1 model artifact for {run_name}, got {len(arts)}"

        dest = Path(f"checkpoints/sweep3_{run_name}_final")
        download_wandb_artifact(project, f"{arts[0].name}", dest)

        tc_paths.setdefault(top_k, {})[layer] = dest

    return tc_paths


def download_clts(project: str) -> list[tuple[int, Path]]:
    """Download all finished CLT artifacts. Returns [(top_k, path), ...]."""
    api = wandb.Api()
    runs = api.runs(project)

    clt_paths = []
    for run in runs:
        if run.state != "finished":
            continue
        cfg = run.config
        top_k = cfg.get("top_k")
        assert top_k is not None, f"No top_k in config for run: {run.name}"

        arts = [a for a in run.logged_artifacts() if a.type == "model"]
        assert len(arts) == 1, f"Expected 1 model artifact for {run.name}, got {len(arts)}"

        dest = Path(f"checkpoints/clt_{run.name}_final")
        download_wandb_artifact(project, f"{arts[0].name}", dest)
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
# MLP activation collection (for MSE)
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
# Transcoder loading & eval (batchtopk at native k)
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
def eval_transcoder_batchtopk(
    base_model, transcoders: dict[int, nn.Module], batches, mlp_activations,
) -> dict:
    """Evaluate transcoders at their native batchtopk k. Returns {l0, ce, mse, layer_l0s}."""
    total_ce, total_mse = 0.0, 0.0
    layer_l0_totals = {l: 0.0 for l in LAYERS}

    for batch_idx, input_ids in enumerate(batches):
        # CE: patch all MLPs
        with ExitStack() as stack:
            for layer_idx in LAYERS:
                mlp = base_model.h[layer_idx].mlp
                tc = transcoders[layer_idx]
                input_size = tc.cfg.input_size
                layer_k = tc.cfg.top_k

                def _make_patched(tc_, input_size_, k_):
                    def _patched(hidden_states):
                        flat = hidden_states.reshape(-1, input_size_)
                        _, recon = _transcoder_batchtopk_recon(tc_, flat, k_)
                        return recon.reshape(hidden_states.shape)
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(tc, input_size, layer_k)))
            total_ce += compute_ce_loss(base_model, input_ids)

        # L0 and MSE
        batch_mse = 0.0
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            mlp_in, mlp_out = mlp_activations[layer_idx][batch_idx]
            acts, recon = _transcoder_batchtopk_recon(tc, mlp_in, tc.cfg.top_k)
            layer_l0_totals[layer_idx] += (acts > 0).float().sum(-1).mean().item()
            batch_mse += F.mse_loss(recon, mlp_out).item()
        total_mse += batch_mse / len(LAYERS)

    n = len(batches)
    layer_l0s = {l: layer_l0_totals[l] / n for l in LAYERS}
    avg_l0 = sum(layer_l0s.values()) / len(LAYERS)
    return {"l0": avg_l0, "ce": total_ce / n, "mse": total_mse / n, "layer_l0s": layer_l0s}


# =============================================================================
# CLT loading & eval (batchtopk at native k)
# =============================================================================


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
def eval_clt_batchtopk(
    base_model, clt: CrossLayerTranscoder, batches, mlp_activations,
) -> dict:
    """Evaluate CLT at its native batchtopk k. Returns {l0, ce, mse, layer_l0s}."""
    total_ce, total_mse = 0.0, 0.0
    layer_l0_totals = {i: 0.0 for i in range(len(LAYERS))}
    k = clt.cfg.top_k

    for batch_idx, input_ids in enumerate(batches):
        captured = _collect_rms2_outputs(base_model, input_ids)
        seq_shape = captured[LAYERS[0]].shape

        clt_inputs = [captured[l].reshape(-1, clt.cfg.input_size) for l in LAYERS]
        all_acts = _clt_batchtopk_acts(clt, clt_inputs, k)
        recons = clt.decode(all_acts)
        recons_shaped = [r.reshape(seq_shape) for r in recons]

        # L0
        for i, a in enumerate(all_acts):
            layer_l0_totals[i] += (a > 0).float().sum(-1).mean().item()

        # CE
        with ExitStack() as stack:
            for i, layer_idx in enumerate(LAYERS):
                mlp = base_model.h[layer_idx].mlp

                def _make_patched(recon):
                    def _patched(hidden_states):
                        return recon
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(recons_shaped[i])))
            total_ce += compute_ce_loss(base_model, input_ids)

        # MSE
        targets = [mlp_activations[l][batch_idx][1] for l in LAYERS]
        batch_mse = sum(F.mse_loss(recons[i], targets[i]).item() for i in range(len(LAYERS))) / len(LAYERS)
        total_mse += batch_mse

    n_batches = len(batches)
    layer_l0s = {i: layer_l0_totals[i] / n_batches for i in range(len(LAYERS))}
    avg_l0 = sum(layer_l0s.values()) / len(LAYERS)
    return {"l0": avg_l0, "ce": total_ce / n_batches, "mse": total_mse / n_batches, "layer_l0s": layer_l0s}


# =============================================================================
# SPD (thresholded)
# =============================================================================


def _capture_all_layer_mlp_outputs(spd_model, input_ids, **model_kwargs):
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


@torch.no_grad()
def eval_spd_thresholded(
    spd_model, batches, module_names, mlp_activations, threshold: float,
) -> dict:
    """Evaluate SPD at a CI threshold. Returns {l0, ce, mse, module_l0s}."""
    total_ce, total_mse = 0.0, 0.0
    module_l0_totals = {name: 0.0 for name in module_names}

    for batch_idx, input_ids in enumerate(batches):
        # CE
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        masks = {}
        for mod_name in module_names:
            ci_post = ci.lower_leaky[mod_name]
            mask = (ci_post > threshold).float()
            masks[mod_name] = mask
            module_l0_totals[mod_name] += mask.sum(-1).mean().item()

        mask_infos = make_mask_infos(masks)
        logits = spd_model(input_ids, mask_infos=mask_infos)
        total_ce += compute_ce_from_logits(logits, input_ids)

        # MSE
        captured_orig = _capture_all_layer_mlp_outputs(spd_model, input_ids)
        captured_masked = _capture_all_layer_mlp_outputs(spd_model, input_ids, mask_infos=mask_infos)
        batch_mse = 0.0
        for layer_idx in LAYERS:
            batch_mse += F.mse_loss(captured_masked[layer_idx], captured_orig[layer_idx]).item()
        total_mse += batch_mse / len(LAYERS)

    n = len(batches)
    module_l0s = {name: module_l0_totals[name] / n for name in module_names}
    avg_l0 = sum(module_l0s.values()) / len(module_names)
    return {"l0": avg_l0, "ce": total_ce / n, "mse": total_mse / n, "module_l0s": module_l0s}


# =============================================================================
# Baselines
# =============================================================================


@torch.no_grad()
def eval_baselines(base_model, batches) -> dict[str, float]:
    original_loss, zero_loss = 0.0, 0.0
    for input_ids in batches:
        original_loss += compute_ce_loss(base_model, input_ids)
        with ExitStack() as stack:
            for layer_idx in LAYERS:
                mlp = base_model.h[layer_idx].mlp
                orig_fwd = mlp.forward

                def _make_zero(fwd):
                    def _zero(hidden_states):
                        return torch.zeros_like(fwd(hidden_states))
                    return _zero

                stack.enter_context(patched_forward(mlp, _make_zero(orig_fwd)))
            zero_loss += compute_ce_loss(base_model, input_ids)
    n = len(batches)
    return {"original_ce": original_loss / n, "zero_ablation_ce": zero_loss / n}


# =============================================================================
# Neuron baseline (top-k neurons)
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
def eval_neuron_topk(base_model, batches, mlp_activations, k: int) -> dict:
    """Evaluate top-k neuron baseline on all layers. Returns {l0, ce, mse, layer_l0s}."""
    total_ce, total_mse = 0.0, 0.0

    for batch_idx, input_ids in enumerate(batches):
        with ExitStack() as stack:
            for layer_idx in LAYERS:
                mlp = base_model.h[layer_idx].mlp

                def _make_patched(mlp_):
                    def _patched(hidden_states):
                        return neuron_topk_reconstruction(mlp_, hidden_states, k)
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(mlp)))
            total_ce += compute_ce_loss(base_model, input_ids)

        batch_mse = 0.0
        for layer_idx in LAYERS:
            mlp = base_model.h[layer_idx].mlp
            mlp_in, mlp_out = mlp_activations[layer_idx][batch_idx]
            recon = neuron_topk_reconstruction(mlp, mlp_in, k)
            batch_mse += F.mse_loss(recon, mlp_out).item()
        total_mse += batch_mse / len(LAYERS)

    n = len(batches)
    layer_l0s = {l: float(k) for l in LAYERS}
    return {"l0": float(k), "ce": total_ce / n, "mse": total_mse / n, "layer_l0s": layer_l0s}


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


def _plot_on_ax(ax, points, baselines, x_key, y_key):
    ce_degradation = y_key == "ce" and baselines
    baseline_ce = baselines["original_ce"] if ce_degradation else 0.0

    for label in PLOT_ORDER:
        pts = points.get(label, [])
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[x_key])
        xs = [p[x_key] for p in pts_sorted]
        ys = [p[y_key] - baseline_ce for p in pts_sorted] if ce_degradation else [p[y_key] for p in pts_sorted]
        style = METHOD_STYLES[label]
        ax.plot(xs, ys, label=label, markeredgecolor="white", markeredgewidth=0.8, **style)

    if ce_degradation:
        zero_abl_deg = baselines["zero_ablation_ce"] - baseline_ce
        ax.axhline(zero_abl_deg, color="#d62728", linestyle=":",
                    linewidth=1.0, alpha=0.5, label="Zero ablation", zorder=1)
        ax.set_yscale("log")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x)}" if x == int(x) else f"{x:g}"
    ))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(direction="in", which="both")


def plot_pareto(
    points: dict[str, list[dict]],
    baselines: dict[str, float],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
    x_key: str = "l0",
    y_key: str = "ce",
):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_on_ax(ax, points, baselines, x_key, y_key)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_pareto_combined(
    points: dict[str, list[dict]],
    baselines: dict[str, float],
    axis_configs: list[tuple[str, str, str]],
    y_key: str,
    ylabel: str,
    save_path: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)

    subplot_labels = ["(a)", "(b)", "(c)"]
    for ax, (x_key, xlabel, _suffix), panel_label in zip(axes, axis_configs, subplot_labels):
        _plot_on_ax(ax, points, baselines, x_key, y_key)
        ax.set_xlabel(xlabel)
        ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Combined plot saved to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Pareto plot from naturally-trained checkpoints (all layers)")
    parser.add_argument("--tc_project", type=str, default="mats-sprint/pile_transcoder_sweep3")
    parser.add_argument("--clt_project", type=str, default="mats-sprint/pile_clt")
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--neuron_ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--save_path", type=str,
                        default="experiments/exp_011_pareto_trained_all_layers/output/pareto_trained_all_layers.png")
    args = parser.parse_args()

    # Load SPD model (includes the base LlamaSimpleMLP)
    from analysis.collect_spd_activations import load_spd_model

    print("Loading SPD model...")
    spd_model, raw_config = load_spd_model(args.spd_run)
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    all_cfc_names = [f"h.{l}.mlp.c_fc" for l in LAYERS]
    all_down_names = [f"h.{l}.mlp.down_proj" for l in LAYERS]
    all_module_names = all_cfc_names + all_down_names

    # Download transcoders
    print(f"\nDownloading transcoders from {args.tc_project}...")
    tc_paths = download_transcoders(args.tc_project)
    print(f"  Found top_k values: {sorted(tc_paths.keys())}")

    # Download CLTs
    print(f"\nDownloading CLTs from {args.clt_project}...")
    clt_paths = download_clts(args.clt_project)
    print(f"  Found CLTs: {[(k, str(p)) for k, p in clt_paths]}")

    # Load eval data
    print(f"\nLoading {args.n_eval_batches} eval batches (seq_len={args.seq_len})...")
    batches = get_eval_batches(args.n_eval_batches, args.batch_size, args.seq_len)

    # Baselines
    print("Computing baselines...")
    baselines = eval_baselines(base_model, batches)
    print(f"  Original CE: {baselines['original_ce']:.4f}")
    print(f"  Zero-ablation CE: {baselines['zero_ablation_ce']:.4f}")

    # MLP activations for MSE
    print("Collecting MLP activations for MSE...")
    mlp_activations = get_mlp_activations(base_model, batches)

    # Evaluate transcoders
    tc_points = []
    for top_k in sorted(tc_paths.keys()):
        layer_paths = tc_paths[top_k]
        assert set(layer_paths.keys()) == set(LAYERS), f"Missing layers for k={top_k}: got {set(layer_paths.keys())}"
        transcoders = {}
        for layer_idx in LAYERS:
            tc = load_transcoder(str(layer_paths[layer_idx]))
            tc.to(DEVICE)
            transcoders[layer_idx] = tc

        print(f"\nEvaluating Transcoder k={top_k}...")
        result = eval_transcoder_batchtopk(base_model, transcoders, batches, mlp_activations)
        result["top_k"] = top_k
        tc_points.append(result)
        print(f"  L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # Evaluate CLTs
    clt_points = []
    for top_k, path in clt_paths:
        clt = load_clt(str(path))
        clt.to(DEVICE)

        print(f"\nEvaluating CLT k={top_k}...")
        result = eval_clt_batchtopk(base_model, clt, batches, mlp_activations)
        result["top_k"] = top_k
        clt_points.append(result)
        print(f"  L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # Evaluate SPD
    spd_points = []
    for threshold, label in [(0.5, "CI>0.5"), (0.0, "CI>0")]:
        print(f"\nEvaluating SPD ({label})...")
        result = eval_spd_thresholded(spd_model, batches, all_module_names, mlp_activations, threshold)
        result["threshold"] = threshold
        spd_points.append(result)
        print(f"  L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # Evaluate neurons
    neuron_points = []
    for k in args.neuron_ks:
        print(f"\nEvaluating Neuron top-k={k}...")
        result = eval_neuron_topk(base_model, batches, mlp_activations, k)
        result["top_k"] = k
        neuron_points.append(result)
        print(f"  L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # Compute x-axis values from actual per-layer L0s
    d_in = base_model.config.n_embd
    d_out = base_model.config.n_embd
    d_hidden = base_model.h[0].mlp.c_fc.weight.shape[0]
    n = len(LAYERS)

    def compute_x_values(method: str, result: dict) -> dict:
        if method in ("Transcoders", "Neurons"):
            ll = result["layer_l0s"]
            per_component = sum(ll.values()) / n
            per_mlp = per_component
            total_params = sum(ll[l] * (d_in + d_out) for l in LAYERS)
        elif method == "CLT":
            ll = result["layer_l0s"]
            per_component = sum(ll.values()) / n
            per_mlp = sum(ll[i] * (n - i) for i in range(n)) / n
            total_params = sum(ll[i] * (d_in + (n - i) * d_out) for i in range(n))
        elif method == "SPD":
            ml = result["module_l0s"]
            per_component = sum(ml.values()) / len(ml)
            per_mlp = sum(
                ml[f"h.{l}.mlp.c_fc"] + ml[f"h.{l}.mlp.down_proj"] for l in LAYERS
            ) / n
            total_params = sum(
                ml[f"h.{l}.mlp.c_fc"] * (d_in + d_hidden)
                + ml[f"h.{l}.mlp.down_proj"] * (d_hidden + d_out)
                for l in LAYERS
            )
        else:
            assert False, f"Unknown method: {method}"
        return {"x_per_component": per_component, "x_per_mlp": per_mlp, "x_total_params": total_params}

    for method, pts in [
        ("Transcoders", tc_points), ("CLT", clt_points),
        ("SPD", spd_points), ("Neurons", neuron_points),
    ]:
        for p in pts:
            p.update(compute_x_values(method, p))

    all_points = {
        "Transcoders": tc_points,
        "CLT": clt_points,
        "SPD (CI>0.5)": [spd_points[0]],
        "SPD (CI>0)": [spd_points[1]],
        "Neurons": neuron_points,
    }

    axis_configs = [
        ("x_per_component", "Active components per module", ""),
        ("x_per_mlp", "Active components per MLP reconstruction", "_per_mlp"),
        ("x_total_params", "Total active parameters", "_total_params"),
    ]

    for x_key, xlabel, suffix in axis_configs:
        for y_key, ylabel, metric_suffix in [
            ("ce", "Cross-entropy degradation", ""),
            ("mse", "MLP reconstruction MSE", "_mse"),
        ]:
            save = args.save_path.replace(".png", f"{suffix}{metric_suffix}.png")
            plot_pareto(
                all_points, baselines,
                xlabel=xlabel, ylabel=ylabel,
                title=f"{ylabel} vs. {xlabel.lower()}",
                save_path=save,
                x_key=x_key,
                y_key=y_key,
            )

    # Combined 3-subplot figures (one for CE, one for MSE)
    plot_pareto_combined(
        all_points, baselines, axis_configs,
        y_key="ce", ylabel="Cross-entropy degradation",
        save_path=args.save_path.replace(".png", "_combined_ce.png"),
    )
    plot_pareto_combined(
        all_points, baselines, axis_configs,
        y_key="mse", ylabel="MLP reconstruction MSE",
        save_path=args.save_path.replace(".png", "_combined_mse.png"),
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary (per-component L0)")
    print(f"{'Method':<20} {'k':>6} {'L0':>8} {'CE':>10} {'MSE':>12}")
    print("-" * 60)
    for p in tc_points:
        print(f"{'Transcoders':<20} {p['top_k']:>6} {p['l0']:>8.1f} {p['ce']:>10.4f} {p['mse']:>12.6f}")
    for p in clt_points:
        print(f"{'CLT':<20} {p['top_k']:>6} {p['l0']:>8.1f} {p['ce']:>10.4f} {p['mse']:>12.6f}")
    for p in spd_points:
        label = f"SPD (CI>{p['threshold']})"
        print(f"{label:<20} {'':>6} {p['l0']:>8.1f} {p['ce']:>10.4f} {p['mse']:>12.6f}")
    for p in neuron_points:
        print(f"{'Neurons':<20} {p['top_k']:>6} {p['l0']:>8.1f} {p['ce']:>10.4f} {p['mse']:>12.6f}")


if __name__ == "__main__":
    main()
