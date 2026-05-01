"""Combined Pareto plot: 4k + 32k dict sizes on the base LLM.

Single figure showing CE / MSE vs three capacity definitions for
PLT (BatchTopK Transcoders) and CLTs at both 4k and 32k dict sizes,
overlaid with VPD baselines (three CI thresholds) and the neuron baseline.

Usage:
    python experiments/pareto_plot_e2e/pareto_plot_e2e.py
    python experiments/pareto_plot_e2e/pareto_plot_e2e.py --plot-only
"""

import argparse
import json
import re
from contextlib import ExitStack
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from nn_decompositions.eval_utils import (
    collect_mlp_inputs,
    compute_ce_from_logits,
    compute_ce_loss,
    get_pile_batches,
    load_clt,
    load_vpd_model,
    load_transcoder,
    patched_forward,
)
from experiments.paper_runs import VPD_BASELINE_RUN
from nn_decompositions.clt import CrossLayerTranscoder  # for type hints
from spd.models.components import make_mask_infos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]

OUTPUT_DIR = Path("experiments/pareto_plot_e2e/output")


# =============================================================================
# Artifact downloading
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


def download_transcoders(project: str, prefix: str) -> dict[int, dict[int, Path]]:
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
        if top_k is None:
            m = re.search(r"tc_k(\d+)", name)
            assert m, f"Cannot parse top_k from run name: {name}"
            top_k = int(m.group(1))

        arts = [a for a in run.logged_artifacts() if a.type == "model"]
        layer_arts = {}
        for a in arts:
            aname = a.name.split(":")[0]
            for layer_idx in LAYERS:
                if f"layer{layer_idx}_final" in aname:
                    layer_arts[layer_idx] = a
                    break

        if set(layer_arts.keys()) != set(LAYERS):
            print(f"  Skipping {name}: only got layers {set(layer_arts.keys())}")
            continue

        layer_paths = {}
        for layer_idx in LAYERS:
            dest = Path(f"checkpoints/{prefix}_tc_{name}_layer{layer_idx}")
            download_wandb_artifact(project, layer_arts[layer_idx].name, dest)
            layer_paths[layer_idx] = dest

        tc_paths[top_k] = layer_paths

    return tc_paths


def download_clts(project: str, prefix: str) -> list[tuple[int, Path]]:
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
        if top_k is None:
            m = re.search(r"clt_k(\d+)", name)
            assert m, f"Cannot parse top_k from run name: {name}"
            top_k = int(m.group(1))

        arts = [a for a in run.logged_artifacts() if a.type == "model"]
        final_arts = [a for a in arts if "final" in a.name]
        assert len(final_arts) == 1, (
            f"Expected 1 final artifact for CLT run {name}, got {len(final_arts)}"
        )

        dest = Path(f"checkpoints/{prefix}_clt_{name}")
        download_wandb_artifact(project, final_arts[0].name, dest)
        clt_paths.append((top_k, dest))

    clt_paths.sort(key=lambda x: x[0])
    return clt_paths


# =============================================================================
# Data loading
# =============================================================================


def get_eval_batches(n_batches: int, batch_size: int, seq_len: int) -> list[torch.Tensor]:
    return get_pile_batches(n_batches, batch_size, seq_len, device=DEVICE)


# =============================================================================
# Helpers
# =============================================================================


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
# Transcoder loading & eval
# =============================================================================


@torch.no_grad()
def eval_transcoder_batchtopk(
    base_model, transcoders: dict[int, nn.Module], batches, mlp_activations,
) -> dict:
    total_ce, total_mse = 0.0, 0.0
    layer_l0_totals = {l: 0.0 for l in LAYERS}

    for batch_idx, input_ids in enumerate(batches):
        with ExitStack() as stack:
            for layer_idx in LAYERS:
                mlp = base_model.h[layer_idx].mlp
                tc = transcoders[layer_idx]

                def _make_patched(tc_):
                    def _patched(hidden_states):
                        flat = hidden_states.reshape(-1, tc_.cfg.input_size)
                        acts = tc_.encode(flat)
                        recon = tc_.decode(acts)
                        return recon.reshape(hidden_states.shape)
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(tc)))
            total_ce += compute_ce_loss(base_model, input_ids)

        batch_mse = 0.0
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            mlp_in, mlp_out = mlp_activations[layer_idx][batch_idx]
            acts = tc.encode(mlp_in)
            recon = tc.decode(acts)
            layer_l0_totals[layer_idx] += (acts > 0).float().sum(-1).mean().item()
            batch_mse += F.mse_loss(recon, mlp_out).item()
        total_mse += batch_mse / len(LAYERS)

    n = len(batches)
    layer_l0s = {l: layer_l0_totals[l] / n for l in LAYERS}
    avg_l0 = sum(layer_l0s.values()) / len(LAYERS)
    return {"l0": avg_l0, "ce": total_ce / n, "mse": total_mse / n, "layer_l0s": layer_l0s}


# =============================================================================
# CLT loading & eval
# =============================================================================


def _collect_rms2_outputs(base_model, input_ids):
    return collect_mlp_inputs(base_model, input_ids, LAYERS)


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

        for i, a in enumerate(all_acts):
            layer_l0_totals[i] += (a > 0).float().sum(-1).mean().item()

        with ExitStack() as stack:
            for i, layer_idx in enumerate(LAYERS):
                mlp = base_model.h[layer_idx].mlp

                def _make_patched(recon):
                    def _patched(hidden_states):
                        return recon
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(recons_shaped[i])))
            total_ce += compute_ce_loss(base_model, input_ids)

        targets = [mlp_activations[l][batch_idx][1] for l in LAYERS]
        batch_mse = sum(F.mse_loss(recons[i], targets[i]).item() for i in range(len(LAYERS))) / len(LAYERS)
        total_mse += batch_mse

    n_batches = len(batches)
    layer_l0s = {i: layer_l0_totals[i] / n_batches for i in range(len(LAYERS))}
    avg_l0 = sum(layer_l0s.values()) / len(LAYERS)
    return {"l0": avg_l0, "ce": total_ce / n_batches, "mse": total_mse / n_batches, "layer_l0s": layer_l0s}


# =============================================================================
# VPD (thresholded)
# =============================================================================


def _capture_all_layer_mlp_outputs(vpd_model, input_ids, **model_kwargs):
    captured = {}
    hooks = []
    for layer_idx in LAYERS:
        target_mlp = vpd_model.target_model.h[layer_idx].mlp

        def _make_hook(li):
            def _capture(_mod, _inp, out):
                captured[li] = out.detach()
            return _capture

        hooks.append(target_mlp.register_forward_hook(_make_hook(layer_idx)))
    vpd_model(input_ids, **model_kwargs)
    for h in hooks:
        h.remove()
    return captured


@torch.no_grad()
def eval_vpd_thresholded(
    vpd_model, batches, module_names, mlp_activations, threshold: float,
) -> dict:
    total_ce, total_mse = 0.0, 0.0
    module_l0_totals = {name: 0.0 for name in module_names}

    for batch_idx, input_ids in enumerate(batches):
        out = vpd_model(input_ids, cache_type="input")
        ci = vpd_model.calc_causal_importances(out.cache, sampling="continuous")
        masks = {}
        for mod_name in module_names:
            ci_post = ci.lower_leaky[mod_name]
            mask = (ci_post > threshold).float()
            masks[mod_name] = mask
            module_l0_totals[mod_name] += mask.sum(-1).mean().item()

        mask_infos = make_mask_infos(masks)
        logits = vpd_model(input_ids, mask_infos=mask_infos)
        total_ce += compute_ce_from_logits(logits, input_ids)

        captured_orig = _capture_all_layer_mlp_outputs(vpd_model, input_ids)
        captured_masked = _capture_all_layer_mlp_outputs(vpd_model, input_ids, mask_infos=mask_infos)
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
# X-axis value computation
# =============================================================================


def compute_x_values(method: str, result: dict, d_in: int, d_out: int, d_hidden: int) -> dict:
    n = len(LAYERS)
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
    elif method == "VPD":
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
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_STYLES = {
    "PLT (4k)":      dict(marker="o", color="#2b6cb0", linestyle="-", linewidth=1.8, markersize=8,  zorder=5),
    "PLT (32k)":     dict(marker="o", color="#63b3ed", linestyle="-", linewidth=1.8, markersize=8,  zorder=5),
    "CLT (4k)":     dict(marker="X", color="#dd6b20", linestyle="-", linewidth=1.8, markersize=9,  zorder=5),
    "CLT (32k)":    dict(marker="X", color="#f6ad55", linestyle="-", linewidth=1.8, markersize=9,  zorder=5),
    "VPD (CI>0.5)": dict(marker="P", color="#6b21a8", linestyle="none", markersize=11, zorder=6),
    "VPD (CI>0.1)": dict(marker="X", color="#6b21a8", linestyle="none", markersize=10, zorder=6),
    "VPD (CI>0)":   dict(marker="D", color="#6b21a8", linestyle="none", markersize=9,  zorder=6),
    "Neurons":      dict(marker="d", color="#d62728", linestyle="-",    linewidth=1.8, markersize=8,  zorder=4),
}

PLOT_ORDER = ["Neurons", "PLT (4k)", "PLT (32k)", "CLT (4k)", "CLT (32k)", "VPD (CI>0.5)", "VPD (CI>0.1)", "VPD (CI>0)"]


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
        ax.yaxis.set_major_locator(ticker.FixedLocator(
            [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
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


def plot_pareto(points, baselines, xlabel, ylabel, save_path, x_key="l0", y_key="ce"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_on_ax(ax, points, baselines, x_key, y_key)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(save_path)
    fig.savefig(str(save_path).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_pareto_combined(points, baselines, axis_configs, y_key, ylabel, save_path):
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

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(save_path)
    fig.savefig(str(save_path).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Combined plot saved to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Combined Pareto plot (4k + 32k)")
    parser.add_argument("--project_4k", type=str, default="mats-sprint/pile_local_sweep_jose")
    parser.add_argument("--project_32k", type=str, default="mats-sprint/pile_local_sweep_jose_32k")
    parser.add_argument("--vpd_run", type=str, default=VPD_BASELINE_RUN)
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--neuron_ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--plot-only", action="store_true", help="Replot from cached data")
    args = parser.parse_args()

    if args.plot_only:
        data_path = OUTPUT_DIR / "pareto_data.json"
        assert data_path.exists(), f"No cached data at {data_path}"
        with open(data_path) as f:
            saved = json.load(f)
        all_points = saved["all_points"]
        baselines = saved["baselines"]

        axis_configs = [
            ("x_per_component", "Active subcomponents per module", ""),
            ("x_per_mlp", "Active subcomponents per MLP reconstruction", "_per_mlp"),
            ("x_total_params", "Total active parameters", "_total_params"),
        ]
        base_path = str(OUTPUT_DIR / "pareto_combined.png")
        for x_key, xlabel, suffix in axis_configs:
            for y_key, ylabel, metric_suffix in [
                ("ce", "CE degradation (\u03b4 from baseline)", ""),
                ("mse", "MLP reconstruction MSE", "_mse"),
            ]:
                save = base_path.replace(".png", f"{suffix}{metric_suffix}.png")
                plot_pareto(all_points, baselines, xlabel=xlabel, ylabel=ylabel,
                            save_path=save, x_key=x_key, y_key=y_key)
        plot_pareto_combined(all_points, baselines, axis_configs,
                             y_key="ce", ylabel="CE degradation (\u03b4 from baseline)",
                             save_path=base_path.replace(".png", "_combined_ce.png"))
        plot_pareto_combined(all_points, baselines, axis_configs,
                             y_key="mse", ylabel="MLP reconstruction MSE",
                             save_path=base_path.replace(".png", "_combined_mse.png"))
        return

    print("Loading VPD model...")
    vpd_model, raw_config = load_vpd_model(args.vpd_run)
    vpd_model.to(DEVICE)
    base_model = vpd_model.target_model
    base_model.eval()

    all_cfc_names = [f"h.{l}.mlp.c_fc" for l in LAYERS]
    all_down_names = [f"h.{l}.mlp.down_proj" for l in LAYERS]
    all_module_names = all_cfc_names + all_down_names

    d_in = base_model.config.n_embd
    d_out = base_model.config.n_embd
    d_hidden = base_model.h[0].mlp.c_fc.weight.shape[0]

    # Download transcoders and CLTs from both projects
    print(f"\nDownloading 4k transcoders from {args.project_4k}...")
    tc_paths_4k = download_transcoders(args.project_4k, "plt")
    print(f"  Found top_k values: {sorted(tc_paths_4k.keys())}")

    print(f"\nDownloading 32k transcoders from {args.project_32k}...")
    tc_paths_32k = download_transcoders(args.project_32k, "plt32k")
    print(f"  Found top_k values: {sorted(tc_paths_32k.keys())}")

    print(f"\nDownloading 4k CLTs from {args.project_4k}...")
    clt_paths_4k = download_clts(args.project_4k, "plt")
    print(f"  Found CLTs: {[(k, str(p)) for k, p in clt_paths_4k]}")

    print(f"\nDownloading 32k CLTs from {args.project_32k}...")
    clt_paths_32k = download_clts(args.project_32k, "plt32k")
    print(f"  Found CLTs: {[(k, str(p)) for k, p in clt_paths_32k]}")

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

    # Evaluate 4k transcoders
    tc_points_4k = []
    for top_k in sorted(tc_paths_4k.keys()):
        layer_paths = tc_paths_4k[top_k]
        transcoders = {l: load_transcoder(layer_paths[l], DEVICE) for l in LAYERS}
        for tc in transcoders.values():
            tc.to(DEVICE)

        print(f"\nEvaluating TC 4k k={top_k}...")
        result = eval_transcoder_batchtopk(base_model, transcoders, batches, mlp_activations)
        result["top_k"] = top_k
        result.update(compute_x_values("Transcoders", result, d_in, d_out, d_hidden))
        tc_points_4k.append(result)
        print(f"  L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # Evaluate 32k transcoders
    tc_points_32k = []
    for top_k in sorted(tc_paths_32k.keys()):
        layer_paths = tc_paths_32k[top_k]
        transcoders = {l: load_transcoder(layer_paths[l], DEVICE) for l in LAYERS}
        for tc in transcoders.values():
            tc.to(DEVICE)

        print(f"\nEvaluating TC 32k k={top_k}...")
        result = eval_transcoder_batchtopk(base_model, transcoders, batches, mlp_activations)
        result["top_k"] = top_k
        result.update(compute_x_values("Transcoders", result, d_in, d_out, d_hidden))
        tc_points_32k.append(result)
        print(f"  L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # Evaluate 4k CLTs
    clt_points_4k = []
    for top_k, path in clt_paths_4k:
        clt = load_clt(path, DEVICE)
        clt.to(DEVICE)

        print(f"\nEvaluating CLT 4k k={top_k}...")
        result = eval_clt_batchtopk(base_model, clt, batches, mlp_activations)
        result["top_k"] = top_k
        result.update(compute_x_values("CLT", result, d_in, d_out, d_hidden))
        clt_points_4k.append(result)
        print(f"  L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # Evaluate 32k CLTs
    clt_points_32k = []
    for top_k, path in clt_paths_32k:
        clt = load_clt(path, DEVICE)
        clt.to(DEVICE)

        print(f"\nEvaluating CLT 32k k={top_k}...")
        result = eval_clt_batchtopk(base_model, clt, batches, mlp_activations)
        result["top_k"] = top_k
        result.update(compute_x_values("CLT", result, d_in, d_out, d_hidden))
        clt_points_32k.append(result)
        print(f"  L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # Evaluate VPD
    vpd_points = []
    for threshold, label in [(0.5, "CI>0.5"), (0.1, "CI>0.1"), (0.0, "CI>0")]:
        print(f"\nEvaluating VPD ({label})...")
        result = eval_vpd_thresholded(vpd_model, batches, all_module_names, mlp_activations, threshold)
        result["threshold"] = threshold
        result.update(compute_x_values("VPD", result, d_in, d_out, d_hidden))
        vpd_points.append(result)
        print(f"  L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    # Evaluate neurons
    neuron_points = []
    for k in args.neuron_ks:
        print(f"\nEvaluating Neuron top-k={k}...")
        result = eval_neuron_topk(base_model, batches, mlp_activations, k)
        result["top_k"] = k
        result.update(compute_x_values("Neurons", result, d_in, d_out, d_hidden))
        neuron_points.append(result)
        print(f"  L0={result['l0']:.1f}, CE={result['ce']:.4f}, MSE={result['mse']:.6f}")

    all_points = {
        "PLT (4k)": tc_points_4k,
        "PLT (32k)": tc_points_32k,
        "CLT (4k)": clt_points_4k,
        "CLT (32k)": clt_points_32k,
        "VPD (CI>0.5)": [vpd_points[0]],
        "VPD (CI>0.1)": [vpd_points[1]],
        "VPD (CI>0)": [vpd_points[2]],
        "Neurons": neuron_points,
    }

    # Save data for --plot-only reruns
    with open(OUTPUT_DIR / "pareto_data.json", "w") as f:
        json.dump({"all_points": all_points, "baselines": baselines}, f, indent=2, default=str)

    axis_configs = [
        ("x_per_component", "Active subcomponents per module", ""),
        ("x_per_mlp", "Active subcomponents per MLP reconstruction", "_per_mlp"),
        ("x_total_params", "Total active parameters", "_total_params"),
    ]

    base_path = str(OUTPUT_DIR / "pareto_combined.png")
    for x_key, xlabel, suffix in axis_configs:
        for y_key, ylabel, metric_suffix in [
            ("ce", "CE degradation (\u03b4 from baseline)", ""),
            ("mse", "MLP reconstruction MSE", "_mse"),
        ]:
            save = base_path.replace(".png", f"{suffix}{metric_suffix}.png")
            plot_pareto(
                all_points, baselines,
                xlabel=xlabel, ylabel=ylabel,
                save_path=save,
                x_key=x_key,
                y_key=y_key,
            )

    # Combined 3-subplot figures
    plot_pareto_combined(
        all_points, baselines, axis_configs,
        y_key="ce", ylabel="CE degradation (\u03b4 from baseline)",
        save_path=base_path.replace(".png", "_combined_ce.png"),
    )
    plot_pareto_combined(
        all_points, baselines, axis_configs,
        y_key="mse", ylabel="MLP reconstruction MSE",
        save_path=base_path.replace(".png", "_combined_mse.png"),
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary (per-component L0)")
    print(f"{'Method':<20} {'k':>6} {'L0':>8} {'CE':>10} {'MSE':>12}")
    print("-" * 60)
    for label, pts in [("PLT (4k)", tc_points_4k), ("PLT (32k)", tc_points_32k),
                       ("CLT (4k)", clt_points_4k), ("CLT (32k)", clt_points_32k)]:
        for p in pts:
            print(f"{label:<20} {p['top_k']:>6} {p['l0']:>8.1f} {p['ce']:>10.4f} {p['mse']:>12.6f}")
    for p in vpd_points:
        label = f"VPD (CI>{p['threshold']})"
        print(f"{label:<20} {'':>6} {p['l0']:>8.1f} {p['ce']:>10.4f} {p['mse']:>12.6f}")
    for p in neuron_points:
        print(f"{'Neurons':<20} {p['top_k']:>6} {p['l0']:>8.1f} {p['ce']:>10.4f} {p['mse']:>12.6f}")


if __name__ == "__main__":
    main()
