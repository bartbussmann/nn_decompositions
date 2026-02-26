"""Comprehensive comparison table of all decomposition methods.

Reports per-layer statistics (dict size, alive features, dead %, L0, mean activation)
and aggregate metrics (CE, MSE, total parameters) for Transcoders, CLTs, and SPD.

Usage:
    python experiments/exp_012_model_table/model_table.py
"""

import argparse
import csv
import json
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from tabulate import tabulate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from transcoder import BatchTopKTranscoder
from config import EncoderConfig, CLTConfig
from clt import CrossLayerTranscoder
from spd.models.components import make_mask_infos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
OUTPUT_DIR = Path("experiments/exp_012_model_table/output")


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class LayerStats:
    layer: int
    dict_size: int
    alive: int
    dead_pct: float
    l0: float
    mean_act: float


@dataclass
class ModelStats:
    method: str
    label: str
    ce: float
    ce_degradation: float
    mse: float
    mean_l0: float
    total_params: int
    training_tokens: int | None
    layer_stats: list[LayerStats] = field(default_factory=list)


# =============================================================================
# Artifact downloading (from exp_011)
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
    api = wandb.Api()
    runs = api.runs(project)
    tc_paths: dict[int, dict[int, Path]] = {}
    for run in runs:
        if run.state != "finished":
            continue
        cfg = run.config
        top_k = cfg.get("top_k")
        run_name = run.name
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
# Transcoder stats
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
def compute_transcoder_stats(
    transcoders: dict[int, nn.Module],
    base_model,
    batches,
    mlp_activations,
    baseline_ce: float,
    top_k: int,
) -> ModelStats:
    dict_size = transcoders[LAYERS[0]].cfg.dict_size
    training_tokens = transcoders[LAYERS[0]].cfg.num_tokens

    # Per-layer accumulators
    ever_active = {l: torch.zeros(dict_size, dtype=torch.bool, device=DEVICE) for l in LAYERS}
    l0_totals = {l: 0.0 for l in LAYERS}
    act_sum = {l: 0.0 for l in LAYERS}
    act_count = {l: 0 for l in LAYERS}
    total_ce, total_mse = 0.0, 0.0

    for batch_idx, input_ids in enumerate(batches):
        # CE
        with ExitStack() as stack:
            for layer_idx in LAYERS:
                mlp = base_model.h[layer_idx].mlp
                tc = transcoders[layer_idx]

                def _make_patched(tc_, k_):
                    def _patched(hidden_states):
                        flat = hidden_states.reshape(-1, tc_.cfg.input_size)
                        _, recon = _transcoder_batchtopk_recon(tc_, flat, k_)
                        return recon.reshape(hidden_states.shape)
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(tc, tc.cfg.top_k)))
            total_ce += compute_ce_loss(base_model, input_ids)

        # Per-layer stats
        batch_mse = 0.0
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            mlp_in, mlp_out = mlp_activations[layer_idx][batch_idx]
            acts, recon = _transcoder_batchtopk_recon(tc, mlp_in, tc.cfg.top_k)

            active_mask = acts > 0
            ever_active[layer_idx] |= active_mask.any(dim=0)
            l0_totals[layer_idx] += active_mask.float().sum(-1).mean().item()
            act_sum[layer_idx] += acts[active_mask].sum().item()
            act_count[layer_idx] += active_mask.sum().item()
            batch_mse += F.mse_loss(recon, mlp_out).item()
        total_mse += batch_mse / len(LAYERS)

    n = len(batches)
    ce = total_ce / n
    mse = total_mse / n
    total_params = sum(p.numel() for tc in transcoders.values() for p in tc.parameters())

    layer_stats_list = []
    for l in LAYERS:
        alive = ever_active[l].sum().item()
        l0 = l0_totals[l] / n
        mean_act = act_sum[l] / act_count[l] if act_count[l] > 0 else 0.0
        layer_stats_list.append(LayerStats(
            layer=l, dict_size=dict_size, alive=int(alive),
            dead_pct=1.0 - alive / dict_size, l0=l0, mean_act=mean_act,
        ))

    mean_l0 = sum(ls.l0 for ls in layer_stats_list) / len(LAYERS)
    return ModelStats(
        method="Transcoder", label=f"TC k={top_k}",
        ce=ce, ce_degradation=ce - baseline_ce, mse=mse,
        mean_l0=mean_l0, total_params=total_params,
        training_tokens=training_tokens, layer_stats=layer_stats_list,
    )


# =============================================================================
# CLT stats
# =============================================================================


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


@torch.no_grad()
def compute_clt_stats(
    clt: CrossLayerTranscoder,
    base_model,
    batches,
    mlp_activations,
    baseline_ce: float,
    top_k: int,
) -> ModelStats:
    dict_size = clt.cfg.dict_size
    n_layers = clt.cfg.n_layers
    training_tokens = clt.cfg.num_tokens
    k = clt.cfg.top_k

    ever_active = {i: torch.zeros(dict_size, dtype=torch.bool, device=DEVICE) for i in range(n_layers)}
    l0_totals = {i: 0.0 for i in range(n_layers)}
    act_sum = {i: 0.0 for i in range(n_layers)}
    act_count = {i: 0 for i in range(n_layers)}
    total_ce, total_mse = 0.0, 0.0

    for batch_idx, input_ids in enumerate(batches):
        captured = _collect_rms2_outputs(base_model, input_ids)
        seq_shape = captured[LAYERS[0]].shape

        clt_inputs = [captured[l].reshape(-1, clt.cfg.input_size) for l in LAYERS]
        all_acts = _clt_batchtopk_acts(clt, clt_inputs, k)
        recons = clt.decode(all_acts)
        recons_shaped = [r.reshape(seq_shape) for r in recons]

        for i, a in enumerate(all_acts):
            active_mask = a > 0
            ever_active[i] |= active_mask.any(dim=0)
            l0_totals[i] += active_mask.float().sum(-1).mean().item()
            act_sum[i] += a[active_mask].sum().item()
            act_count[i] += active_mask.sum().item()

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
        batch_mse = sum(F.mse_loss(recons[i], targets[i]).item() for i in range(n_layers)) / n_layers
        total_mse += batch_mse

    n = len(batches)
    ce = total_ce / n
    mse = total_mse / n
    total_params = sum(p.numel() for p in clt.parameters())

    layer_stats_list = []
    for i in range(n_layers):
        alive = ever_active[i].sum().item()
        l0 = l0_totals[i] / n
        mean_act = act_sum[i] / act_count[i] if act_count[i] > 0 else 0.0
        layer_stats_list.append(LayerStats(
            layer=i, dict_size=dict_size, alive=int(alive),
            dead_pct=1.0 - alive / dict_size, l0=l0, mean_act=mean_act,
        ))

    mean_l0 = sum(ls.l0 for ls in layer_stats_list) / n_layers
    return ModelStats(
        method="CLT", label=f"CLT k={top_k}",
        ce=ce, ce_degradation=ce - baseline_ce, mse=mse,
        mean_l0=mean_l0, total_params=total_params,
        training_tokens=training_tokens, layer_stats=layer_stats_list,
    )


# =============================================================================
# SPD stats
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
def compute_spd_stats(
    spd_model,
    base_model,
    batches,
    mlp_activations,
    module_names: list[str],
    baseline_ce: float,
    threshold: float,
) -> list[ModelStats]:
    """Returns 3 ModelStats: c_fc only, down_proj only, and combined total."""
    components_per_module = {name: spd_model.module_to_c[name] for name in module_names}

    ever_active = {
        name: torch.zeros(components_per_module[name], dtype=torch.bool, device=DEVICE)
        for name in module_names
    }
    l0_totals = {name: 0.0 for name in module_names}
    total_ce, total_mse = 0.0, 0.0

    for batch_idx, input_ids in enumerate(batches):
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        masks = {}
        for mod_name in module_names:
            ci_post = ci.lower_leaky[mod_name]
            mask = (ci_post > threshold).float()
            masks[mod_name] = mask
            l0_totals[mod_name] += mask.sum(-1).mean().item()
            ever_active[mod_name] |= (ci_post > threshold).any(dim=0).any(dim=0)

        mask_infos = make_mask_infos(masks)
        logits = spd_model(input_ids, mask_infos=mask_infos)
        total_ce += compute_ce_from_logits(logits, input_ids)

        captured_orig = _capture_all_layer_mlp_outputs(spd_model, input_ids)
        captured_masked = _capture_all_layer_mlp_outputs(spd_model, input_ids, mask_infos=mask_infos)
        batch_mse = 0.0
        for layer_idx in LAYERS:
            batch_mse += F.mse_loss(captured_masked[layer_idx], captured_orig[layer_idx]).item()
        total_mse += batch_mse / len(LAYERS)

    n = len(batches)
    ce = total_ce / n
    mse = total_mse / n
    threshold_label = f"CI>{threshold}" if threshold > 0 else "CI>0"

    # Build per-layer stats for each view: c_fc, down_proj, total
    def _make_layer_stats(get_names_fn) -> list[LayerStats]:
        stats = []
        for l in LAYERS:
            names = get_names_fn(l)
            dict_size = sum(components_per_module[nm] for nm in names)
            alive = sum(int(ever_active[nm].sum().item()) for nm in names)
            l0 = sum(l0_totals[nm] / n for nm in names) / len(names)
            stats.append(LayerStats(
                layer=l, dict_size=dict_size, alive=alive,
                dead_pct=1.0 - alive / dict_size, l0=l0, mean_act=0.0,
            ))
        return stats

    def _make_model_stats(suffix: str, names_for_layer, all_names: list[str]) -> ModelStats:
        layer_stats = _make_layer_stats(names_for_layer)
        params = sum(
            spd_model.components[nm].V.numel() + spd_model.components[nm].U.numel()
            for nm in all_names
        )
        mean_l0 = sum(l0_totals[nm] / n for nm in all_names) / len(all_names)
        return ModelStats(
            method="SPD", label=f"SPD {suffix} ({threshold_label})",
            ce=ce, ce_degradation=ce - baseline_ce, mse=mse,
            mean_l0=mean_l0, total_params=params,
            training_tokens=None, layer_stats=layer_stats,
        )

    cfc_names = [f"h.{l}.mlp.c_fc" for l in LAYERS]
    dp_names = [f"h.{l}.mlp.down_proj" for l in LAYERS]

    return [
        _make_model_stats("c_fc", lambda l: [f"h.{l}.mlp.c_fc"], cfc_names),
        _make_model_stats("down_proj", lambda l: [f"h.{l}.mlp.down_proj"], dp_names),
        _make_model_stats("total", lambda l: [f"h.{l}.mlp.c_fc", f"h.{l}.mlp.down_proj"], module_names),
    ]


# =============================================================================
# Output formatting
# =============================================================================


def print_summary_table(all_stats: list[ModelStats]):
    headers = [
        "Model", "CE", "CE degr.", "MSE", "Mean L0", "Params",
        *[f"L{l} dict" for l in LAYERS],
        *[f"L{l} alive" for l in LAYERS],
        *[f"L{l} dead%" for l in LAYERS],
        *[f"L{l} L0" for l in LAYERS],
        *[f"L{l} act" for l in LAYERS],
    ]
    rows = []
    for s in all_stats:
        row = [
            s.label,
            f"{s.ce:.4f}",
            f"{s.ce_degradation:.4f}",
            f"{s.mse:.6f}",
            f"{s.mean_l0:.1f}",
            f"{s.total_params:,}",
        ]
        for l in LAYERS:
            ls = s.layer_stats[l]
            row.append(str(ls.dict_size))
        for l in LAYERS:
            ls = s.layer_stats[l]
            row.append(str(ls.alive))
        for l in LAYERS:
            ls = s.layer_stats[l]
            row.append(f"{ls.dead_pct:.1%}")
        for l in LAYERS:
            ls = s.layer_stats[l]
            row.append(f"{ls.l0:.1f}")
        for l in LAYERS:
            ls = s.layer_stats[l]
            row.append(f"{ls.mean_act:.4f}" if ls.mean_act > 0 else "-")
        rows.append(row)

    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))


def print_per_layer_table(all_stats: list[ModelStats]):
    print("\n" + "=" * 90)
    print("Per-layer detail")
    print("=" * 90)

    headers = ["Model", "Layer", "Dict size", "Alive", "Dead %", "L0", "Mean act"]
    rows = []
    for s in all_stats:
        for ls in s.layer_stats:
            rows.append([
                s.label, ls.layer, ls.dict_size, ls.alive,
                f"{ls.dead_pct:.1%}", f"{ls.l0:.1f}",
                f"{ls.mean_act:.4f}" if ls.mean_act > 0 else "-",
            ])
        rows.append([""] * len(headers))

    print(tabulate(rows, headers=headers, tablefmt="simple"))


def save_csv(all_stats: list[ModelStats], path: Path):
    fieldnames = [
        "method", "label", "ce", "ce_degradation", "mse", "mean_l0",
        "total_params", "training_tokens",
    ]
    for l in LAYERS:
        fieldnames.extend([
            f"L{l}_dict_size", f"L{l}_alive", f"L{l}_dead_pct",
            f"L{l}_l0", f"L{l}_mean_act",
        ])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_stats:
            row = {
                "method": s.method, "label": s.label,
                "ce": f"{s.ce:.6f}", "ce_degradation": f"{s.ce_degradation:.6f}",
                "mse": f"{s.mse:.8f}", "mean_l0": f"{s.mean_l0:.2f}",
                "total_params": s.total_params,
                "training_tokens": s.training_tokens or "",
            }
            for ls in s.layer_stats:
                l = ls.layer
                row[f"L{l}_dict_size"] = ls.dict_size
                row[f"L{l}_alive"] = ls.alive
                row[f"L{l}_dead_pct"] = f"{ls.dead_pct:.4f}"
                row[f"L{l}_l0"] = f"{ls.l0:.2f}"
                row[f"L{l}_mean_act"] = f"{ls.mean_act:.6f}" if ls.mean_act > 0 else ""
            writer.writerow(row)
    print(f"\nCSV saved to {path}")


# =============================================================================
# Plotting
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LABEL_COLORS = {
    "Transcoder": "#1f77b4",
    "CLT": "#17becf",
    "SPD c_fc": "#7b4ea3",
    "SPD down_proj": "#b07cd8",
    "SPD total": "#9467bd",
}


def _get_color(stats: ModelStats) -> str:
    for prefix, color in LABEL_COLORS.items():
        if stats.label.startswith(prefix):
            return color
    return "#888888"


def _bar_plot(ax, all_stats: list[ModelStats], value_fn, ylabel: str, title: str,
              log_scale: bool = False):
    labels = [s.label for s in all_stats]
    values = [value_fn(s) for s in all_stats]
    colors = [_get_color(s) for s in all_stats]
    x = np.arange(len(labels))

    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=11)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)


def _grouped_bar_plot(ax, all_stats: list[ModelStats], value_fn, ylabel: str, title: str):
    n_models = len(all_stats)
    n_layers = len(LAYERS)
    bar_width = 0.8 / n_models
    x = np.arange(n_layers)

    for i, s in enumerate(all_stats):
        values = [value_fn(s.layer_stats[l]) for l in range(n_layers)]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=s.label,
               color=_get_color(s), edgecolor="white", linewidth=0.3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)


def plot_comparison(all_stats: list[ModelStats], baselines: dict[str, float], save_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (a) CE degradation
    _bar_plot(axes[0, 0], all_stats,
              lambda s: s.ce_degradation,
              "CE degradation", "(a) Cross-entropy degradation",
              log_scale=True)

    # (b) MSE
    _bar_plot(axes[0, 1], all_stats,
              lambda s: s.mse,
              "MSE", "(b) MLP reconstruction MSE",
              log_scale=True)

    # (c) Total parameters
    _bar_plot(axes[0, 2], all_stats,
              lambda s: s.total_params,
              "Parameters", "(c) Total parameters",
              log_scale=True)
    axes[0, 2].yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"
    ))

    # (d) Per-layer L0
    _grouped_bar_plot(axes[1, 0], all_stats,
                      lambda ls: ls.l0,
                      "L0", "(d) Per-layer L0")

    # (e) Per-layer alive features
    _grouped_bar_plot(axes[1, 1], all_stats,
                      lambda ls: ls.alive,
                      "Alive features", "(e) Per-layer alive features")

    # (f) Per-layer dead %
    _grouped_bar_plot(axes[1, 2], all_stats,
                      lambda ls: ls.dead_pct * 100,
                      "Dead features (%)", "(f) Per-layer dead features")

    # Shared legend
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 8),
               frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95,
               bbox_to_anchor=(0.5, 1.01), fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Model comparison table")
    parser.add_argument("--tc_project", type=str, default="mats-sprint/pile_transcoder_sweep3")
    parser.add_argument("--clt_project", type=str, default="mats-sprint/pile_clt")
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # Download models
    print(f"\nDownloading transcoders from {args.tc_project}...")
    tc_paths = download_transcoders(args.tc_project)
    print(f"  Found top_k values: {sorted(tc_paths.keys())}")

    print(f"\nDownloading CLTs from {args.clt_project}...")
    clt_paths = download_clts(args.clt_project)
    print(f"  Found CLTs: {[(k, str(p)) for k, p in clt_paths]}")

    # Load eval data
    print(f"\nLoading {args.n_eval_batches} eval batches...")
    batches = get_eval_batches(args.n_eval_batches, args.batch_size, args.seq_len)

    # Baselines
    print("Computing baselines...")
    baselines = eval_baselines(base_model, batches)
    baseline_ce = baselines["original_ce"]
    print(f"  Original CE: {baseline_ce:.4f}")
    print(f"  Zero-ablation CE: {baselines['zero_ablation_ce']:.4f}")

    # MLP activations
    print("Collecting MLP activations...")
    mlp_activations = get_mlp_activations(base_model, batches)

    all_stats: list[ModelStats] = []

    # Transcoders
    for top_k in sorted(tc_paths.keys()):
        layer_paths = tc_paths[top_k]
        assert set(layer_paths.keys()) == set(LAYERS)
        transcoders = {}
        for layer_idx in LAYERS:
            tc = load_transcoder(str(layer_paths[layer_idx]))
            tc.to(DEVICE)
            transcoders[layer_idx] = tc

        print(f"\nComputing stats for Transcoder k={top_k}...")
        stats = compute_transcoder_stats(
            transcoders, base_model, batches, mlp_activations, baseline_ce, top_k,
        )
        all_stats.append(stats)
        print(f"  CE={stats.ce:.4f}, MSE={stats.mse:.6f}, Mean L0={stats.mean_l0:.1f}")

    # CLTs
    for top_k, path in clt_paths:
        clt = load_clt(str(path))
        clt.to(DEVICE)

        print(f"\nComputing stats for CLT k={top_k}...")
        stats = compute_clt_stats(clt, base_model, batches, mlp_activations, baseline_ce, top_k)
        all_stats.append(stats)
        print(f"  CE={stats.ce:.4f}, MSE={stats.mse:.6f}, Mean L0={stats.mean_l0:.1f}")

    # SPD
    for threshold in [0.5, 0.0]:
        label = f"CI>{threshold}" if threshold > 0 else "CI>0"
        print(f"\nComputing stats for SPD ({label})...")
        spd_stats_list = compute_spd_stats(
            spd_model, base_model, batches, mlp_activations,
            all_module_names, baseline_ce, threshold,
        )
        for stats in spd_stats_list:
            all_stats.append(stats)
            print(f"  {stats.label}: CE={stats.ce:.4f}, MSE={stats.mse:.6f}, Mean L0={stats.mean_l0:.1f}")

    # Output
    print_summary_table(all_stats)
    print_per_layer_table(all_stats)
    save_csv(all_stats, OUTPUT_DIR / "model_comparison.csv")
    plot_comparison(all_stats, baselines, OUTPUT_DIR / "model_comparison.png")


if __name__ == "__main__":
    main()
