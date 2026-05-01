"""Evaluate e2e-trained transcoders/CLTs on jose's target model (v2).

Same as exp_021 but:
  - Evaluates both 4k and 32k dict_size models
  - Also includes local_mse models from pile_local_sweep_jose[_32k]
  - Uses tc.encode()/tc.decode() directly (no hand-rolled forward pass)
  - Produces separate results files per dict_size

Usage:
    python experiments/exp_040_eval_e2e_jose_v2/eval_e2e_jose_v2.py
    python experiments/exp_040_eval_e2e_jose_v2/eval_e2e_jose_v2.py --dict_sizes 4k
    python experiments/exp_040_eval_e2e_jose_v2/eval_e2e_jose_v2.py --skip_spd
"""

import json
import re
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

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
JOSE_MODEL_CACHE = Path("experiments/exp_019_eval_e2e/jose_model_cache")
CHECKPOINT_DIR = Path("checkpoints/jose_v2")
OUTPUT_DIR = Path("experiments/exp_040_eval_e2e_jose_v2/output")

PROJECTS = {
    "4k": {
        "e2e": "mats-sprint/pile_e2e_sweep_jose",
        "local": "mats-sprint/pile_local_sweep_jose",
    },
    "32k": {
        "e2e": "mats-sprint/pile_e2e_sweep_jose_32k",
        "local": "mats-sprint/pile_local_sweep_jose_32k",
    },
}


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


def _collect_rms2_outputs(base_model, input_ids):
    captured = {}
    hooks = []
    for layer_idx in LAYERS:
        def _make_hook(li):
            def _hook(_mod, _inp, out):
                captured[li] = out.detach()
            return _hook
        hooks.append(base_model.h[layer_idx].rms_2.register_forward_hook(_make_hook(layer_idx)))
    base_model(input_ids)
    for h in hooks:
        h.remove()
    return captured


# =============================================================================
# Model loading
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
# TC eval modes (using encode/decode)
# =============================================================================


@torch.no_grad()
def compute_tc_l0(transcoders: dict[int, BatchTopKTranscoder], base_model, batches) -> float:
    layer_l0_totals = {l: 0.0 for l in LAYERS}
    for input_ids in batches:
        captured = _collect_rms2_outputs(base_model, input_ids)
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            flat = captured[layer_idx].reshape(-1, tc.cfg.input_size)
            acts = tc.encode(flat)
            layer_l0_totals[layer_idx] += (acts > 0).float().sum(-1).mean().item()
    return sum(v / len(batches) for v in layer_l0_totals.values()) / len(LAYERS)


@torch.no_grad()
def eval_tc_all_parallel(base_model, transcoders, batches) -> float:
    total_ce = 0.0
    for input_ids in batches:
        captured = _collect_rms2_outputs(base_model, input_ids)
        recons_shaped = {}
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            flat = captured[layer_idx].reshape(-1, tc.cfg.input_size)
            recon = tc.decode(tc.encode(flat))
            recons_shaped[layer_idx] = recon.reshape(captured[layer_idx].shape)

        def _make_const(tensor):
            return lambda *a, **kw: tensor

        with ExitStack() as stack:
            for layer_idx in LAYERS:
                stack.enter_context(patched_forward(base_model.h[layer_idx].mlp, _make_const(recons_shaped[layer_idx])))
            total_ce += compute_ce_loss(base_model, input_ids)
    return total_ce / len(batches)


@torch.no_grad()
def eval_tc_all_cascading(base_model, transcoders, batches) -> float:
    total_ce = 0.0
    for input_ids in batches:
        hooks = []
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]

            def _make_hook(tc_):
                def _hook(_module, inp, _output):
                    flat = inp[0].reshape(-1, tc_.cfg.input_size)
                    recon = tc_.decode(tc_.encode(flat))
                    return recon.reshape(inp[0].shape)
                return _hook

            hooks.append(base_model.h[layer_idx].mlp.register_forward_hook(_make_hook(tc)))
        total_ce += compute_ce_loss(base_model, input_ids)
        for h in hooks:
            h.remove()
    return total_ce / len(batches)


@torch.no_grad()
def eval_tc_single_mlp(base_model, transcoders, batches) -> float:
    total_ce = 0.0
    for layer_idx in LAYERS:
        tc = transcoders[layer_idx]
        layer_ce = 0.0
        for input_ids in batches:
            def _make_patched(tc_):
                def _patched(hidden_states):
                    flat = hidden_states.reshape(-1, tc_.cfg.input_size)
                    recon = tc_.decode(tc_.encode(flat))
                    return recon.reshape(hidden_states.shape)
                return _patched

            with patched_forward(base_model.h[layer_idx].mlp, _make_patched(tc)):
                layer_ce += compute_ce_loss(base_model, input_ids)
        total_ce += layer_ce / len(batches)
    return total_ce / len(LAYERS)


# =============================================================================
# CLT eval modes
# =============================================================================


@torch.no_grad()
def compute_clt_l0(clt, base_model, batches) -> float:
    layer_l0_totals = [0.0] * clt.cfg.n_layers
    for input_ids in batches:
        captured = _collect_rms2_outputs(base_model, input_ids)
        for i in range(clt.cfg.n_layers):
            flat = captured[LAYERS[i]].reshape(-1, clt.cfg.input_size)
            acts = clt.encode_layer(flat, i)
            layer_l0_totals[i] += (acts > 0).float().sum(-1).mean().item()
    return sum(v / len(batches) for v in layer_l0_totals) / clt.cfg.n_layers


@torch.no_grad()
def eval_clt_all_parallel(base_model, clt, batches) -> float:
    total_ce = 0.0
    for input_ids in batches:
        captured = _collect_rms2_outputs(base_model, input_ids)
        seq_shape = captured[LAYERS[0]].shape
        flat_inputs = [captured[l].reshape(-1, clt.cfg.input_size) for l in LAYERS]
        all_acts = [clt.encode_layer(flat_inputs[i], i) for i in range(clt.cfg.n_layers)]
        recons = clt.decode(all_acts)
        recons_shaped = [r.reshape(seq_shape) for r in recons]

        def _make_const(tensor):
            return lambda *a, **kw: tensor

        with ExitStack() as stack:
            for i, layer_idx in enumerate(LAYERS):
                stack.enter_context(patched_forward(base_model.h[layer_idx].mlp, _make_const(recons_shaped[i])))
            total_ce += compute_ce_loss(base_model, input_ids)
    return total_ce / len(batches)


@torch.no_grad()
def eval_clt_all_cascading(base_model, clt, batches) -> float:
    total_ce = 0.0
    for input_ids in batches:
        all_acts: list[torch.Tensor] = []

        def _make_cascading_hook(layer_idx: int):
            def _hook(_module, inp, _output):
                flat = inp[0].reshape(-1, clt.cfg.input_size)
                acts = clt.encode_layer(flat, layer_idx)
                all_acts.append(acts)
                recon = clt.b_dec[layer_idx].unsqueeze(0).expand(flat.shape[0], -1).clone()
                for i in range(layer_idx + 1):
                    recon = recon + all_acts[i] @ clt.W_dec[i][layer_idx - i]
                return recon.reshape(inp[0].shape)
            return _hook

        hooks = []
        for layer_idx in range(clt.cfg.n_layers):
            h = base_model.h[LAYERS[layer_idx]].mlp.register_forward_hook(_make_cascading_hook(layer_idx))
            hooks.append(h)
        total_ce += compute_ce_loss(base_model, input_ids)
        for h in hooks:
            h.remove()
    return total_ce / len(batches)


@torch.no_grad()
def eval_clt_single_mlp(base_model, clt, batches) -> float:
    total_ce = 0.0
    for target in range(clt.cfg.n_layers):
        layer_ce = 0.0
        for input_ids in batches:
            captured = _collect_rms2_outputs(base_model, input_ids)
            seq_shape = captured[LAYERS[0]].shape
            flat_inputs = [captured[LAYERS[i]].reshape(-1, clt.cfg.input_size) for i in range(target + 1)]
            source_acts = [clt.encode_layer(flat_inputs[i], i) for i in range(target + 1)]
            recon = clt.b_dec[target].unsqueeze(0).expand(source_acts[0].shape[0], -1).clone()
            for i in range(target + 1):
                recon = recon + source_acts[i] @ clt.W_dec[i][target - i]
            recon_shaped = recon.reshape(seq_shape)

            def _make_const(tensor):
                return lambda *a, **kw: tensor

            with patched_forward(base_model.h[LAYERS[target]].mlp, _make_const(recon_shaped)):
                layer_ce += compute_ce_loss(base_model, input_ids)
        total_ce += layer_ce / len(batches)
    return total_ce / clt.cfg.n_layers


# =============================================================================
# SPD eval
# =============================================================================


@torch.no_grad()
def eval_spd_all(spd_model, batches, module_names, threshold) -> tuple[float, float]:
    from spd.models.components import make_mask_infos
    total_ce = 0.0
    module_l0_totals = {name: 0.0 for name in module_names}
    for input_ids in batches:
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        masks = {}
        for mod_name in module_names:
            mask = (ci.lower_leaky[mod_name] > threshold).float()
            masks[mod_name] = mask
            module_l0_totals[mod_name] += mask.sum(-1).mean().item()
        mask_infos = make_mask_infos(masks)
        logits = spd_model(input_ids, mask_infos=mask_infos)
        total_ce += compute_ce_from_logits(logits, input_ids)
    n = len(batches)
    avg_l0 = sum(v / n for v in module_l0_totals.values()) / len(module_names)
    return total_ce / n, avg_l0


@torch.no_grad()
def eval_spd_all_parallel(spd_model, batches, module_names, threshold) -> float:
    """Clean-input SPD: compute masked MLP outputs from clean residual streams."""
    from spd.models.components import make_mask_infos
    base_model = spd_model.target_model
    total_ce = 0.0
    for input_ids in batches:
        # Get clean MLP inputs from original model
        clean_inputs = _collect_rms2_outputs(base_model, input_ids)

        # Compute CI from clean forward pass
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        masks = {m: (ci.lower_leaky[m] > threshold).float() for m in module_names}
        mask_infos = make_mask_infos(masks)

        # For each layer, compute masked MLP output from clean input
        recons = {}
        for layer_idx in LAYERS:
            x = clean_inputs[layer_idx]  # (batch, seq, d_model)
            cfc_name = f"h.{layer_idx}.mlp.c_fc"
            down_name = f"h.{layer_idx}.mlp.down_proj"

            cfc_components = spd_model.components[cfc_name]
            down_components = spd_model.components[down_name]

            # c_fc: x -> hidden (with mask)
            hidden = cfc_components(
                x, mask=mask_infos[cfc_name].component_mask,
                weight_delta_and_mask=mask_infos[cfc_name].weight_delta_and_mask,
            )
            # GELU activation
            hidden = base_model.h[layer_idx].mlp.gelu(hidden)
            # down_proj: hidden -> output (with mask)
            mlp_out = down_components(
                hidden, mask=mask_infos[down_name].component_mask,
                weight_delta_and_mask=mask_infos[down_name].weight_delta_and_mask,
            )
            recons[layer_idx] = mlp_out

        # Patch MLPs and compute CE
        def _make_const(tensor):
            return lambda *a, **kw: tensor

        with ExitStack() as stack:
            for layer_idx in LAYERS:
                stack.enter_context(patched_forward(
                    base_model.h[layer_idx].mlp, _make_const(recons[layer_idx])
                ))
            total_ce += compute_ce_loss(base_model, input_ids)
    return total_ce / len(batches)


@torch.no_grad()
def eval_spd_single_mlp(spd_model, batches, threshold) -> float:
    from spd.models.components import make_mask_infos
    total_ce = 0.0
    for layer_idx in LAYERS:
        layer_mods = [f"h.{layer_idx}.mlp.c_fc", f"h.{layer_idx}.mlp.down_proj"]
        layer_ce = 0.0
        for input_ids in batches:
            out = spd_model(input_ids, cache_type="input")
            ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
            masks = {m: (ci.lower_leaky[m] > threshold).float() for m in layer_mods}
            mask_infos = make_mask_infos(masks)
            logits = spd_model(input_ids, mask_infos=mask_infos)
            layer_ce += compute_ce_from_logits(logits, input_ids)
        total_ce += layer_ce / len(batches)
    return total_ce / len(LAYERS)


# =============================================================================
# Download from wandb
# =============================================================================


def download_artifacts(dict_size_label: str) -> dict:
    """Download all finished runs for a dict size. Returns {(type, mode, k): path_or_layer_paths}."""
    api = wandb.Api()
    models = {}

    for project_type in ("e2e", "local"):
        project = PROJECTS[dict_size_label][project_type]
        runs = api.runs(project)

        for run in runs:
            if run.state != "finished":
                continue
            arts = [a for a in run.logged_artifacts() if a.type == "model"]
            if not arts:
                continue

            name = run.name
            top_k = run.config.get("top_k")
            # Fallback: extract k from run name (e.g. tc_cascading_k16 -> 16)
            if top_k is None:
                m = re.search(r"_k(\d+)", name)
                if m:
                    top_k = int(m.group(1))

            if name.startswith("tc_"):
                # e2e: tc_cascading_k16, tc_parallel_k8, tc_independent_k32
                # local: tc_k8, tc_k16, etc.
                if project_type == "local":
                    mode = "local_mse"
                else:
                    parts = name.split("_")
                    mode = parts[1]  # cascading, parallel, independent

                layer_paths = {}
                for art in arts:
                    dest = CHECKPOINT_DIR / dict_size_label / f"{name}_{art.name.split(':')[0]}"
                    if not (dest / "encoder.pt").exists():
                        art.download(root=str(dest))
                        print(f"  Downloaded {art.name} -> {dest}")
                    else:
                        print(f"  Cached {dest}")
                    art_base = art.name.split(":")[0]
                    for layer in LAYERS:
                        if f"layer{layer}_final" in art_base:
                            layer_paths[layer] = dest
                            break

                if len(layer_paths) == len(LAYERS):
                    models[("tc", mode, top_k)] = layer_paths

            elif name.startswith("clt_"):
                if project_type == "local":
                    mode = "local_mse"
                else:
                    parts = name.split("_")
                    mode = parts[1]

                final_arts = [a for a in arts if "final" in a.name]
                if not final_arts:
                    continue
                dest = CHECKPOINT_DIR / dict_size_label / f"{name}_final"
                if not (dest / "encoder.pt").exists():
                    final_arts[0].download(root=str(dest))
                    print(f"  Downloaded {final_arts[0].name} -> {dest}")
                else:
                    print(f"  Cached {dest}")
                models[("clt", mode, top_k)] = dest

    return models


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate jose e2e + local models (v2)")
    parser.add_argument("--dict_sizes", nargs="+", default=["4k", "32k"], choices=["4k", "32k"])
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--spd_thresholds", type=float, nargs="+", default=[0.5, 0.1, 0.0])
    parser.add_argument("--skip_spd", action="store_true")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load jose base model
    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig

    print("Loading jose base model...")
    with open(JOSE_MODEL_CACHE / "config.json") as f:
        model_cfg = LlamaSimpleMLPConfig(**json.load(f))
    base_model = LlamaSimpleMLP(model_cfg)
    sd = torch.load(JOSE_MODEL_CACHE / "state_dict.pt", map_location="cpu", weights_only=True)
    base_model.load_state_dict(sd)
    base_model.to(DEVICE)
    base_model.eval()
    del sd

    print(f"Loading {args.n_eval_batches} eval batches...")
    batches = get_eval_batches(args.n_eval_batches, args.batch_size, args.seq_len)

    print("Computing baseline CE...")
    baseline_ce = sum(compute_ce_loss(base_model, b) for b in batches) / len(batches)
    print(f"  Baseline CE: {baseline_ce:.4f}")

    for dict_label in args.dict_sizes:
        print(f"\n{'='*80}")
        print(f"  Evaluating {dict_label} models")
        print(f"{'='*80}")

        if not args.skip_download:
            print(f"\nDownloading {dict_label} artifacts...")
            models = download_artifacts(dict_label)
        else:
            # Discover from cache
            models = {}
            print("Skipping download, discovering from cache...")

        results = []

        # Evaluate TC models
        tc_models = sorted([(m, k, p) for (t, m, k), p in models.items() if t == "tc"], key=lambda x: (x[0], x[1]))
        print(f"\nFound {len(tc_models)} TC model sets")
        for mode, k, layer_paths in tc_models:
            print(f"\n--- TC {mode} k={k} ---")
            transcoders = {layer: load_transcoder(layer_paths[layer]) for layer in LAYERS}

            l0 = compute_tc_l0(transcoders, base_model, batches)
            ce_cascading = eval_tc_all_cascading(base_model, transcoders, batches)
            ce_parallel = eval_tc_all_parallel(base_model, transcoders, batches)
            ce_single = eval_tc_single_mlp(base_model, transcoders, batches)

            print(f"  L0={l0:.1f}  casc={ce_cascading:.4f} ({ce_cascading-baseline_ce:+.4f})  "
                  f"para={ce_parallel:.4f} ({ce_parallel-baseline_ce:+.4f})  "
                  f"single={ce_single:.4f} ({ce_single-baseline_ce:+.4f})")

            results.append({
                "type": "tc", "mode": mode, "top_k": k, "l0": l0,
                "ce_cascading": ce_cascading, "ce_parallel": ce_parallel, "ce_single": ce_single,
            })
            del transcoders; torch.cuda.empty_cache()

        # Evaluate CLT models
        clt_models = sorted([(m, k, p) for (t, m, k), p in models.items() if t == "clt"], key=lambda x: (x[0], x[1]))
        print(f"\nFound {len(clt_models)} CLT models")
        for mode, k, path in clt_models:
            print(f"\n--- CLT {mode} k={k} ---")
            clt = load_clt(path)

            l0 = compute_clt_l0(clt, base_model, batches)
            ce_cascading = eval_clt_all_cascading(base_model, clt, batches)
            ce_parallel = eval_clt_all_parallel(base_model, clt, batches)
            ce_single = eval_clt_single_mlp(base_model, clt, batches)

            print(f"  L0={l0:.1f}  casc={ce_cascading:.4f} ({ce_cascading-baseline_ce:+.4f})  "
                  f"para={ce_parallel:.4f} ({ce_parallel-baseline_ce:+.4f})  "
                  f"single={ce_single:.4f} ({ce_single-baseline_ce:+.4f})")

            results.append({
                "type": "clt", "mode": mode, "top_k": k, "l0": l0,
                "ce_cascading": ce_cascading, "ce_parallel": ce_parallel, "ce_single": ce_single,
            })
            del clt; torch.cuda.empty_cache()

        # SPD
        spd_results = []
        if not args.skip_spd:
            print("\nLoading jose SPD model...")
            from analysis.collect_spd_activations import load_spd_model
            spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
            spd_model.to(DEVICE)
            all_module_names = [f"h.{l}.mlp.c_fc" for l in LAYERS] + [f"h.{l}.mlp.down_proj" for l in LAYERS]

            for threshold in args.spd_thresholds:
                label = f"CI>{threshold}"
                print(f"\n--- SPD ({label}) ---")
                ce_cascading, l0 = eval_spd_all(spd_model, batches, all_module_names, threshold)
                ce_parallel = eval_spd_all_parallel(spd_model, batches, all_module_names, threshold)
                ce_single = eval_spd_single_mlp(spd_model, batches, threshold)
                print(f"  L0={l0:.1f}  casc={ce_cascading:.4f} ({ce_cascading-baseline_ce:+.4f})  "
                      f"para={ce_parallel:.4f} ({ce_parallel-baseline_ce:+.4f})  "
                      f"single={ce_single:.4f} ({ce_single-baseline_ce:+.4f})")
                spd_results.append({
                    "type": "jose", "mode": label, "threshold": threshold, "l0": l0,
                    "ce_all": ce_cascading, "ce_cascading": ce_cascading,
                    "ce_parallel": ce_parallel, "ce_single": ce_single,
                })
            del spd_model; torch.cuda.empty_cache()

        # Save results
        output_path = OUTPUT_DIR / f"results_{dict_label}.json"
        with open(output_path, "w") as f:
            json.dump({"baseline_ce": baseline_ce, "results": results, "spd_results": spd_results}, f, indent=2)
        print(f"\nResults saved to {output_path}")

        # Summary table
        print(f"\n{'Model':<25} {'k':>4} {'L0':>6}  {'Casc':>10}  {'Para':>10}  {'Single':>10}  | {'Δ Casc':>8}  {'Δ Para':>8}  {'Δ Sing':>8}")
        print("-" * 100)
        for r in results:
            label = f"{r['type']}_{r['mode']}"
            k_str = str(r['top_k']) if r['top_k'] is not None else "?"
            print(f"{label:<25} {k_str:>4} {r['l0']:>6.1f}"
                  f"  {r['ce_cascading']:>10.4f}  {r['ce_parallel']:>10.4f}  {r['ce_single']:>10.4f}"
                  f"  | {r['ce_cascading']-baseline_ce:>8.4f}  {r['ce_parallel']-baseline_ce:>8.4f}  {r['ce_single']-baseline_ce:>8.4f}")
        for r in spd_results:
            label = f"spd_{r['mode']}"
            print(f"{label:<25} {'':>4} {r['l0']:>6.1f}"
                  f"  {r['ce_cascading']:>10.4f}  {r['ce_parallel']:>10.4f}  {r['ce_single']:>10.4f}"
                  f"  | {r['ce_cascading']-baseline_ce:>8.4f}  {r['ce_parallel']-baseline_ce:>8.4f}  {r['ce_single']-baseline_ce:>8.4f}")
        print(f"\nBaseline CE: {baseline_ce:.4f}")


if __name__ == "__main__":
    main()
