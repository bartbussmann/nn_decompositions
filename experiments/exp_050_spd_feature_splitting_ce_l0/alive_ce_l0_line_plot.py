"""
exp_050: same model lineup and alive-count methodology as exp_049
(SPD 0.5x/1x/2x/4x, TC 4k/32k k=16, CLT 4k/32k k=16), but additionally
records CE degradation and L0 for every model.

Usage:
  python experiments/exp_050_spd_feature_splitting_ce_l0/alive_ce_l0_line_plot.py
  python experiments/exp_050_spd_feature_splitting_ce_l0/alive_ce_l0_line_plot.py --plot-only
  python experiments/exp_050_spd_feature_splitting_ce_l0/alive_ce_l0_line_plot.py --reuse-cache
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
N_TOKENS = int(1e6)
BATCH_SIZE = 8
SEQ_LEN = 512
N_BATCHES = N_TOKENS // (BATCH_SIZE * SEQ_LEN)
ALIVE_THRESHOLD = 1e-6
SPD_PER_TOKEN_THRESHOLD = 0.1

OUTPUT_DIR = Path("experiments/exp_050_spd_feature_splitting_ce_l0/output")
SPD_RAW_FILE = OUTPUT_DIR / "spd_raw.json"
LINE_DATA_FILE = OUTPUT_DIR / "alive_ce_l0_line_data.json"
JOSE_MODEL_CACHE = Path("experiments/exp_019_eval_e2e/jose_model_cache")
CHECKPOINT_DIR = Path("checkpoints/exp_050")

SPD_RUNS = {
    "0.5x": "goodfire/spd/s-b2b37c4e",
    "1x (jose)": "goodfire/spd/s-55ea3f9b",
    "2x": "goodfire/spd/s-266cb440",
    "4x": "goodfire/spd/s-d3834f54",
}

TC_RUNS = {
    4096: {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "4ziu27fn"},
    32768: {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "c4o8i98k"},
}

CLT_RUNS = {
    4096: {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "77sgz1pe"},
    32768: {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "j20m9hzr"},
}

MLP_MODULE_PATTERNS = ["h.{}.mlp.c_fc", "h.{}.mlp.down_proj"]


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


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_ce_loss(model, input_ids: torch.Tensor) -> float:
    logits, _ = model(input_ids)
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)).item()


def get_eval_batches(n_batches: int) -> list[torch.Tensor]:
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)
    batches: list[torch.Tensor] = []
    for _ in tqdm(range(n_batches), desc="Loading eval batches"):
        batch_ids = []
        for _ in range(BATCH_SIZE):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:SEQ_LEN])
        batches.append(torch.stack(batch_ids).to(DEVICE))
    return batches


def collect_mlp_inputs(base_model, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for layer_idx in LAYERS:
        def _make_hook(li: int):
            def _hook(_mod, _inp, out):
                captured[li] = out.detach()
            return _hook
        hooks.append(base_model.h[layer_idx].rms_2.register_forward_hook(_make_hook(layer_idx)))
    base_model(input_ids)
    for hook in hooks:
        hook.remove()
    return captured


# =============================================================================
# Loaders (full classes — encode + decode)
# =============================================================================


def load_transcoder(checkpoint_dir: Path) -> BatchTopKTranscoder:
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    tc = BatchTopKTranscoder(cfg)
    tc.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    tc.eval()
    return tc


def load_clt(checkpoint_dir: Path) -> CrossLayerTranscoder:
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
# WandB downloads
# =============================================================================


def download_wandb_artifact(project: str, artifact_name: str, dest: Path) -> Path:
    if dest.exists() and (dest / "encoder.pt").exists():
        print(f"  Using cached artifact at {dest}")
        return dest
    api = wandb.Api()
    artifact = api.artifact(f"{project}/{artifact_name}")
    artifact.download(root=str(dest))
    print(f"  Downloaded {artifact_name} -> {dest}")
    return dest


def download_tc_artifacts(run_info: dict) -> dict[int, Path]:
    api = wandb.Api()
    run = api.run(f"{run_info['project']}/runs/{run_info['run_id']}")
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    layer_paths: dict[int, Path] = {}
    for artifact in artifacts:
        artifact_name = artifact.name.split(":")[0]
        for layer_idx in LAYERS:
            if f"layer{layer_idx}_final" in artifact_name:
                dest = CHECKPOINT_DIR / f"tc_{run_info['run_id']}_layer{layer_idx}"
                download_wandb_artifact(run_info["project"], artifact.name, dest)
                layer_paths[layer_idx] = dest
                break
    if set(layer_paths.keys()) != set(LAYERS):
        raise RuntimeError(f"Missing TC layers; got {sorted(layer_paths.keys())}")
    return layer_paths


def download_clt_artifact(run_info: dict) -> Path:
    api = wandb.Api()
    run = api.run(f"{run_info['project']}/runs/{run_info['run_id']}")
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    final_artifacts = [a for a in artifacts if "final" in a.name]
    if len(final_artifacts) != 1:
        raise RuntimeError(f"Expected 1 CLT final artifact, got {len(final_artifacts)}")
    dest = CHECKPOINT_DIR / f"clt_{run_info['run_id']}"
    download_wandb_artifact(run_info["project"], final_artifacts[0].name, dest)
    return dest


# =============================================================================
# Alive counts (matches exp_049 methodology)
# =============================================================================


@torch.no_grad()
def count_spd_alive_mean_ci(spd_model, batches, threshold: float) -> dict[str, dict[str, int]]:
    mlp_modules = []
    for layer in LAYERS:
        for pattern in MLP_MODULE_PATTERNS:
            mod_name = pattern.format(layer)
            if mod_name in spd_model.module_to_c:
                mlp_modules.append(mod_name)

    ci_sum = {
        mod_name: torch.zeros(spd_model.module_to_c[mod_name], dtype=torch.float64, device="cpu")
        for mod_name in mlp_modules
    }
    n_tokens_total = 0
    for input_ids in tqdm(batches, desc="SPD mean-CI alive count"):
        bsz, seq = input_ids.shape
        n_tokens_total += bsz * seq
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        for mod_name in mlp_modules:
            n_c = spd_model.module_to_c[mod_name]
            ci_vals = ci.lower_leaky[mod_name].reshape(-1, n_c)
            ci_sum[mod_name] += ci_vals.double().sum(dim=0).cpu()

    results: dict[str, dict[str, int]] = {}
    for mod_name in mlp_modules:
        mean_ci = ci_sum[mod_name] / n_tokens_total
        alive = int((mean_ci > threshold).sum().item())
        total = int(spd_model.module_to_c[mod_name])
        results[mod_name] = {"alive": alive, "total": total}
    return results


@torch.no_grad()
def count_tc_alive(transcoders: dict[int, BatchTopKTranscoder], base_model, batches) -> tuple[int, int]:
    dict_size = next(iter(transcoders.values())).cfg.dict_size
    fire_count = {l: torch.zeros(dict_size, dtype=torch.int64, device=DEVICE) for l in LAYERS}
    n_tokens = 0
    for input_ids in tqdm(batches, desc="TC alive count"):
        mlp_inputs = collect_mlp_inputs(base_model, input_ids)
        bsz, seq = input_ids.shape
        n_tokens += bsz * seq
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            flat = mlp_inputs[layer_idx].reshape(-1, tc.cfg.input_size)
            acts = tc.encode(flat)
            fire_count[layer_idx] += (acts > 0).sum(dim=0).to(torch.int64)
    total = dict_size * len(LAYERS)
    alive = 0
    for layer_idx in LAYERS:
        sparsity = fire_count[layer_idx].float() / n_tokens
        alive += int((sparsity > ALIVE_THRESHOLD).sum().item())
    return total, alive


@torch.no_grad()
def count_clt_alive(clt: CrossLayerTranscoder, base_model, batches) -> tuple[int, int]:
    dict_size = clt.cfg.dict_size
    n_layers = clt.cfg.n_layers
    fire_count = {l: torch.zeros(dict_size, dtype=torch.int64, device=DEVICE) for l in range(n_layers)}
    n_tokens = 0
    for input_ids in tqdm(batches, desc="CLT alive count"):
        mlp_inputs = collect_mlp_inputs(base_model, input_ids)
        bsz, seq = input_ids.shape
        n_tokens += bsz * seq
        for layer_idx in range(n_layers):
            flat = mlp_inputs[LAYERS[layer_idx]].reshape(-1, clt.cfg.input_size)
            acts = clt.encode_layer(flat, layer_idx)
            fire_count[layer_idx] += (acts > 0).sum(dim=0).to(torch.int64)
    total = dict_size * n_layers
    alive = 0
    for layer_idx in range(n_layers):
        sparsity = fire_count[layer_idx].float() / n_tokens
        alive += int((sparsity > ALIVE_THRESHOLD).sum().item())
    return total, alive


# =============================================================================
# CE + L0
# =============================================================================


@torch.no_grad()
def eval_tc_ce_l0(base_model, transcoders, batches) -> tuple[float, float]:
    """Replace all 4 MLPs in parallel with TC reconstructions; report CE and mean L0 per layer."""
    layer_l0_totals = {l: 0.0 for l in LAYERS}
    total_ce = 0.0
    for input_ids in batches:
        captured = collect_mlp_inputs(base_model, input_ids)
        recons_shaped = {}
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            flat = captured[layer_idx].reshape(-1, tc.cfg.input_size)
            acts = tc.encode(flat)
            layer_l0_totals[layer_idx] += (acts > 0).float().sum(-1).mean().item()
            recon = tc.decode(acts)
            recons_shaped[layer_idx] = recon.reshape(captured[layer_idx].shape)

        def _make_const(tensor):
            return lambda *a, **kw: tensor

        with ExitStack() as stack:
            for layer_idx in LAYERS:
                stack.enter_context(patched_forward(base_model.h[layer_idx].mlp, _make_const(recons_shaped[layer_idx])))
            total_ce += compute_ce_loss(base_model, input_ids)

    n = len(batches)
    avg_l0 = sum(v / n for v in layer_l0_totals.values()) / len(LAYERS)
    return total_ce / n, avg_l0


@torch.no_grad()
def eval_clt_ce_l0(base_model, clt: CrossLayerTranscoder, batches) -> tuple[float, float]:
    """Replace all 4 MLPs in parallel with CLT reconstructions; report CE and mean L0 per layer."""
    layer_l0_totals = [0.0] * clt.cfg.n_layers
    total_ce = 0.0
    for input_ids in batches:
        captured = collect_mlp_inputs(base_model, input_ids)
        seq_shape = captured[LAYERS[0]].shape
        flat_inputs = [captured[LAYERS[i]].reshape(-1, clt.cfg.input_size) for i in range(clt.cfg.n_layers)]
        all_acts = [clt.encode_layer(flat_inputs[i], i) for i in range(clt.cfg.n_layers)]
        for i, acts in enumerate(all_acts):
            layer_l0_totals[i] += (acts > 0).float().sum(-1).mean().item()
        recons = clt.decode(all_acts)
        recons_shaped = [r.reshape(seq_shape) for r in recons]

        def _make_const(tensor):
            return lambda *a, **kw: tensor

        with ExitStack() as stack:
            for i, layer_idx in enumerate(LAYERS):
                stack.enter_context(patched_forward(base_model.h[layer_idx].mlp, _make_const(recons_shaped[i])))
            total_ce += compute_ce_loss(base_model, input_ids)

    n = len(batches)
    avg_l0 = sum(v / n for v in layer_l0_totals) / clt.cfg.n_layers
    return total_ce / n, avg_l0


@torch.no_grad()
def eval_spd_ce_l0(spd_model, batches, module_names, threshold: float) -> tuple[float, float]:
    """SPD CE eval with per-token CI mask; report CE (clean-input parallel patching) and mean L0 per module."""
    from spd.models.components import make_mask_infos

    base_model = spd_model.target_model
    module_l0_totals = {name: 0.0 for name in module_names}
    total_ce = 0.0
    for input_ids in batches:
        clean_inputs = collect_mlp_inputs(base_model, input_ids)
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        masks = {m: (ci.lower_leaky[m] > threshold).float() for m in module_names}
        for m in module_names:
            module_l0_totals[m] += masks[m].sum(-1).mean().item()
        mask_infos = make_mask_infos(masks)

        recons = {}
        for layer_idx in LAYERS:
            x = clean_inputs[layer_idx]
            cfc_name = f"h.{layer_idx}.mlp.c_fc"
            down_name = f"h.{layer_idx}.mlp.down_proj"
            cfc_components = spd_model.components[cfc_name]
            down_components = spd_model.components[down_name]
            hidden = cfc_components(
                x, mask=mask_infos[cfc_name].component_mask,
                weight_delta_and_mask=mask_infos[cfc_name].weight_delta_and_mask,
            )
            hidden = base_model.h[layer_idx].mlp.gelu(hidden)
            mlp_out = down_components(
                hidden, mask=mask_infos[down_name].component_mask,
                weight_delta_and_mask=mask_infos[down_name].weight_delta_and_mask,
            )
            recons[layer_idx] = mlp_out

        def _make_const(tensor):
            return lambda *a, **kw: tensor

        with ExitStack() as stack:
            for layer_idx in LAYERS:
                stack.enter_context(patched_forward(base_model.h[layer_idx].mlp, _make_const(recons[layer_idx])))
            total_ce += compute_ce_loss(base_model, input_ids)

    n = len(batches)
    avg_l0 = sum(v / n for v in module_l0_totals.values()) / len(module_names)
    return total_ce / n, avg_l0


# =============================================================================
# Main
# =============================================================================


def load_jose_base_model():
    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig
    with open(JOSE_MODEL_CACHE / "config.json") as f:
        model_cfg = LlamaSimpleMLPConfig(**json.load(f))
    base_model = LlamaSimpleMLP(model_cfg)
    sd = torch.load(JOSE_MODEL_CACHE / "state_dict.pt", map_location="cpu", weights_only=True)
    base_model.load_state_dict(sd)
    base_model.to(DEVICE)
    base_model.eval()
    del sd
    return base_model


def plot(spd, tc, clt, baseline_ce: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.titlesize": 13,
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
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    spd_color, tc_color, clt_color = "#6b21a8", "#2b6cb0", "#dd6b20"

    spd_totals = [d["total"] for d in spd]
    spd_alive = [d["alive"] for d in spd]
    spd_ce_deg = [d["ce"] - baseline_ce for d in spd]
    spd_l0 = [d["l0"] for d in spd]

    tc_totals = [d["total"] for d in tc]
    tc_alive = [d["alive"] for d in tc]
    tc_ce_deg = [d["ce"] - baseline_ce for d in tc]
    tc_l0 = [d["l0"] for d in tc]

    clt_totals = [d["total"] for d in clt]
    clt_alive = [d["alive"] for d in clt]
    clt_ce_deg = [d["ce"] - baseline_ce for d in clt]
    clt_l0 = [d["l0"] for d in clt]

    # Panel 1: alive vs total
    ax = axes[0]
    all_totals = spd_totals + tc_totals + clt_totals
    lo, hi = min(all_totals) * 0.7, max(all_totals) * 1.5
    ax.plot([lo, hi], [lo, hi], ls="--", color="#cccccc", lw=1, zorder=0, label="y = x (all alive)")
    ax.plot(spd_totals, spd_alive, "o-", color=spd_color, markersize=7, lw=2, label="VPD", zorder=3)
    ax.plot(tc_totals, tc_alive, "s-", color=tc_color, markersize=7, lw=2, label="PLT (k=16)", zorder=3)
    ax.plot(clt_totals, clt_alive, "^-", color=clt_color, markersize=7, lw=2, label="CLT (k=16)", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total component capacity")
    ax.set_ylabel("Alive components")
    ax.set_title("Alive features")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, linewidth=0.5, which="both")

    # Panel 2: CE degradation vs total
    ax = axes[1]
    ax.plot(spd_totals, spd_ce_deg, "o-", color=spd_color, markersize=7, lw=2, label=f"VPD (CI>{SPD_PER_TOKEN_THRESHOLD})", zorder=3)
    ax.plot(tc_totals, tc_ce_deg, "s-", color=tc_color, markersize=7, lw=2, label="PLT (k=16)", zorder=3)
    ax.plot(clt_totals, clt_ce_deg, "^-", color=clt_color, markersize=7, lw=2, label="CLT (k=16)", zorder=3)
    ax.axhline(0, ls="--", color="#cccccc", lw=1, zorder=0, label="baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Total component capacity")
    ax.set_ylabel(rf"$\Delta$ CE (vs baseline {baseline_ce:.3f})")
    ax.set_title("CE degradation (parallel patch)")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, linewidth=0.5, which="both")

    # Panel 3: L0 vs total
    ax = axes[2]
    ax.plot(spd_totals, spd_l0, "o-", color=spd_color, markersize=7, lw=2, label=f"VPD (CI>{SPD_PER_TOKEN_THRESHOLD})", zorder=3)
    ax.plot(tc_totals, tc_l0, "s-", color=tc_color, markersize=7, lw=2, label="PLT (k=16)", zorder=3)
    ax.plot(clt_totals, clt_l0, "^-", color=clt_color, markersize=7, lw=2, label="CLT (k=16)", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Total component capacity")
    ax.set_ylabel("L0 (mean active features per token, per layer)")
    ax.set_title("Sparsity (L0)")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, linewidth=0.5, which="both")

    fig.tight_layout()
    png_path = OUTPUT_DIR / "alive_ce_l0_line_plot.png"
    pdf_path = OUTPUT_DIR / "alive_ce_l0_line_plot.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved plot to {png_path} and {pdf_path}")


def main(plot_only: bool, reuse_cache: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if plot_only:
        with open(LINE_DATA_FILE) as f:
            data = json.load(f)
        plot(data["spd"], data["tc"], data["clt"], data["baseline_ce"])
        return

    print(f"Preparing {N_BATCHES} batches ({N_TOKENS / 1e6:.0f}M tokens)")
    batches = get_eval_batches(N_BATCHES)

    base_model = load_jose_base_model()
    print("Computing baseline CE...")
    baseline_ce = sum(compute_ce_loss(base_model, b) for b in batches) / len(batches)
    print(f"  Baseline CE: {baseline_ce:.4f}")

    # ----- SPD -----
    print("\n=== Stage 1/3: SPD models ===")
    if reuse_cache and SPD_RAW_FILE.exists():
        with open(SPD_RAW_FILE) as f:
            spd_raw = json.load(f)
    else:
        spd_raw = {}

    spd_records: list[dict] = []
    from analysis.collect_spd_activations import load_spd_model

    for label, run_path in SPD_RUNS.items():
        cached = spd_raw.get(label) if reuse_cache else None
        if cached and "alive_per_module" in cached and "ce" in cached and "l0" in cached:
            print(f"\nUsing cached SPD results for {label}")
            spd_records.append({
                "label": label,
                "total": cached["total"],
                "alive": cached["alive"],
                "ce": cached["ce"],
                "l0": cached["l0"],
                "alive_per_module": cached["alive_per_module"],
            })
            continue

        print(f"\n{'=' * 64}\nSPD {label}: {run_path}\n{'=' * 64}")
        spd_model, _ = load_spd_model(run_path)
        spd_model.to(DEVICE)
        spd_model.eval()

        alive_per_module = count_spd_alive_mean_ci(spd_model, batches, threshold=ALIVE_THRESHOLD)
        total = sum(v["total"] for v in alive_per_module.values())
        alive = sum(v["alive"] for v in alive_per_module.values())

        module_names = [f"h.{l}.mlp.c_fc" for l in LAYERS] + [f"h.{l}.mlp.down_proj" for l in LAYERS]
        module_names = [m for m in module_names if m in spd_model.module_to_c]
        ce, l0 = eval_spd_ce_l0(spd_model, batches, module_names, threshold=SPD_PER_TOKEN_THRESHOLD)

        rec = {
            "label": label,
            "total": total,
            "alive": alive,
            "ce": ce,
            "l0": l0,
            "alive_per_module": alive_per_module,
        }
        spd_records.append(rec)
        spd_raw[label] = rec
        with open(SPD_RAW_FILE, "w") as f:
            json.dump(spd_raw, f, indent=2)
        print(f"  alive={alive}/{total} ({100 * alive / total:.1f}%)  CE={ce:.4f} (Δ {ce - baseline_ce:+.4f})  L0={l0:.2f}")

        del spd_model
        cleanup_cuda()

    # Reload base model after SPD evals (each SPD load swaps target_model on its own copy)
    base_model = load_jose_base_model()

    # ----- TC -----
    print("\n=== Stage 2/3: TC models ===")
    tc_records: list[dict] = []
    for dict_size, run_info in TC_RUNS.items():
        print(f"\nTC dict_size={dict_size}, run={run_info['run_id']}")
        layer_paths = download_tc_artifacts(run_info)
        transcoders = {l: load_transcoder(p) for l, p in layer_paths.items()}

        total, alive = count_tc_alive(transcoders, base_model, batches)
        ce, l0 = eval_tc_ce_l0(base_model, transcoders, batches)

        tc_records.append({
            "dict_size": dict_size,
            "total": total,
            "alive": alive,
            "ce": ce,
            "l0": l0,
        })
        print(f"  alive={alive}/{total} ({100 * alive / total:.1f}%)  CE={ce:.4f} (Δ {ce - baseline_ce:+.4f})  L0={l0:.2f}")
        del transcoders
        cleanup_cuda()

    # ----- CLT -----
    print("\n=== Stage 3/3: CLT models ===")
    clt_records: list[dict] = []
    for dict_size, run_info in CLT_RUNS.items():
        print(f"\nCLT dict_size={dict_size}, run={run_info['run_id']}")
        clt_path = download_clt_artifact(run_info)
        clt = load_clt(clt_path)

        total, alive = count_clt_alive(clt, base_model, batches)
        ce, l0 = eval_clt_ce_l0(base_model, clt, batches)

        clt_records.append({
            "dict_size": dict_size,
            "total": total,
            "alive": alive,
            "ce": ce,
            "l0": l0,
        })
        print(f"  alive={alive}/{total} ({100 * alive / total:.1f}%)  CE={ce:.4f} (Δ {ce - baseline_ce:+.4f})  L0={l0:.2f}")
        del clt
        cleanup_cuda()

    output_data = {
        "baseline_ce": baseline_ce,
        "spd_per_token_threshold": SPD_PER_TOKEN_THRESHOLD,
        "alive_threshold": ALIVE_THRESHOLD,
        "spd": spd_records,
        "tc": tc_records,
        "clt": clt_records,
    }
    with open(LINE_DATA_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved line data to {LINE_DATA_FILE}")

    plot(spd_records, tc_records, clt_records, baseline_ce)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Generate the figure from saved line data")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse cached SPD per-model results if present")
    args = parser.parse_args()
    main(plot_only=args.plot_only, reuse_cache=args.reuse_cache)
