"""Evaluate e2e-trained transcoders and CLTs in three CE settings:

1. All-replace cascading: replace all MLPs, each encoder sees modified residual stream
2. All-replace parallel: replace all MLPs, each encoder sees clean inputs
3. Single-MLP replace: replace one MLP at a time, average CE over all layers

Loads models from exp_018 checkpoints (tc_cascading, tc_parallel, tc_independent,
clt_cascading, clt_parallel) at k=16/32/64.

Usage:
    python experiments/exp_019_eval_e2e/eval_e2e.py
    python experiments/exp_019_eval_e2e/eval_e2e.py --n_eval_batches 20
"""

import json
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig, CLTConfig
from nn_decompositions.clt import CrossLayerTranscoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
CHECKPOINT_DIR = Path("checkpoints")


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


def patched_forward(module: nn.Module, patched_fn):
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        original = module.forward
        module.forward = patched_fn
        try:
            yield
        finally:
            module.forward = original

    return _ctx()


def compute_ce_loss(model, input_ids: torch.Tensor) -> float:
    logits, _ = model(input_ids)
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)).item()


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
# Transcoder eval modes
# =============================================================================


@torch.no_grad()
def compute_tc_l0(transcoders: dict[int, BatchTopKTranscoder], base_model, batches) -> float:
    """Compute average L0 per MLP for transcoders (from clean inputs)."""
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
def eval_tc_all_parallel(base_model, transcoders: dict[int, BatchTopKTranscoder], batches) -> float:
    """Replace all MLPs with transcoders, each encoding from clean (unmodified) inputs."""
    total_ce = 0.0
    for input_ids in batches:
        captured = _collect_rms2_outputs(base_model, input_ids)

        recons_shaped = {}
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            clean_input = captured[layer_idx]
            flat = clean_input.reshape(-1, tc.cfg.input_size)
            acts = tc.encode(flat)
            recon = tc.decode(acts)
            recons_shaped[layer_idx] = recon.reshape(clean_input.shape)

        def _make_const(tensor):
            return lambda *a, **kw: tensor

        with ExitStack() as stack:
            for layer_idx in LAYERS:
                stack.enter_context(patched_forward(base_model.h[layer_idx].mlp, _make_const(recons_shaped[layer_idx])))
            total_ce += compute_ce_loss(base_model, input_ids)
    return total_ce / len(batches)


@torch.no_grad()
def eval_tc_all_cascading(base_model, transcoders: dict[int, BatchTopKTranscoder], batches) -> float:
    """Replace all MLPs with transcoders, cascading: each encoder sees modified residual stream."""
    total_ce = 0.0
    for input_ids in batches:
        hooks = []
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]

            def _make_hook(tc_):
                def _hook(_module, inp, _output):
                    encoder_input = inp[0]
                    flat = encoder_input.reshape(-1, tc_.cfg.input_size)
                    acts = tc_.encode(flat)
                    recon = tc_.decode(acts)
                    return recon.reshape(encoder_input.shape)
                return _hook

            h = base_model.h[layer_idx].mlp.register_forward_hook(_make_hook(tc))
            hooks.append(h)

        total_ce += compute_ce_loss(base_model, input_ids)

        for h in hooks:
            h.remove()

    return total_ce / len(batches)


@torch.no_grad()
def eval_tc_single_mlp(base_model, transcoders: dict[int, BatchTopKTranscoder], batches) -> float:
    """Replace one MLP at a time, return average CE over all layers."""
    total_ce = 0.0
    for layer_idx in LAYERS:
        tc = transcoders[layer_idx]
        layer_ce = 0.0
        for input_ids in batches:
            def _make_patched(tc_):
                def _patched(hidden_states):
                    flat = hidden_states.reshape(-1, tc_.cfg.input_size)
                    acts = tc_.encode(flat)
                    recon = tc_.decode(acts)
                    return recon.reshape(hidden_states.shape)
                return _patched

            with patched_forward(base_model.h[layer_idx].mlp, _make_patched(tc)):
                layer_ce += compute_ce_loss(base_model, input_ids)
        total_ce += layer_ce / len(batches)
    return total_ce / len(LAYERS)


# =============================================================================
# CLT eval modes
# =============================================================================


def _collect_rms2_outputs(base_model, input_ids):
    """Run clean forward pass and capture rms_2 outputs for all layers."""
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


@torch.no_grad()
def compute_clt_l0(clt: CrossLayerTranscoder, base_model, batches) -> float:
    """Compute average L0 per encoder layer for CLT (from clean inputs)."""
    layer_l0_totals = [0.0] * clt.cfg.n_layers
    for input_ids in batches:
        captured = _collect_rms2_outputs(base_model, input_ids)
        flat_inputs = [captured[LAYERS[i]].reshape(-1, clt.cfg.input_size) for i in range(clt.cfg.n_layers)]
        for i in range(clt.cfg.n_layers):
            acts = clt.encode_layer(flat_inputs[i], i)
            layer_l0_totals[i] += (acts > 0).float().sum(-1).mean().item()
    return sum(v / len(batches) for v in layer_l0_totals) / clt.cfg.n_layers


@torch.no_grad()
def eval_clt_all_parallel(base_model, clt: CrossLayerTranscoder, batches) -> float:
    """Replace all MLPs with CLT, encoding from clean inputs (non-cascading)."""
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
def eval_clt_all_cascading(base_model, clt: CrossLayerTranscoder, batches) -> float:
    """Replace all MLPs with CLT, cascading: each layer encodes from modified residual stream."""
    total_ce = 0.0
    for input_ids in batches:
        all_acts: list[torch.Tensor] = []

        def _make_cascading_hook(layer_idx: int):
            def _hook(_module, inp, _output):
                encoder_input = inp[0]
                flat = encoder_input.reshape(-1, clt.cfg.input_size)
                acts = clt.encode_layer(flat, layer_idx)
                all_acts.append(acts)

                # Triangular decode: recon[j] = b_dec[j] + sum_{i<=j} acts[i] @ W_dec[i][j-i]
                recon = clt.b_dec[layer_idx].unsqueeze(0).expand(flat.shape[0], -1).clone()
                for i in range(layer_idx + 1):
                    recon = recon + all_acts[i] @ clt.W_dec[i][layer_idx - i]

                return recon.reshape(encoder_input.shape)
            return _hook

        hooks = []
        for layer_idx in range(clt.cfg.n_layers):
            h = base_model.h[LAYERS[layer_idx]].mlp.register_forward_hook(
                _make_cascading_hook(layer_idx)
            )
            hooks.append(h)

        total_ce += compute_ce_loss(base_model, input_ids)

        for h in hooks:
            h.remove()

    return total_ce / len(batches)


@torch.no_grad()
def eval_clt_single_mlp(base_model, clt: CrossLayerTranscoder, batches) -> float:
    """Replace one MLP at a time with CLT reconstruction, average CE over all layers.

    For each target layer j, we run a clean forward pass to get rms_2 outputs for
    all source layers i <= j, encode them, then decode only target j.
    """
    total_ce = 0.0
    for target_layer_idx in range(clt.cfg.n_layers):
        layer_ce = 0.0
        for input_ids in batches:
            captured = _collect_rms2_outputs(base_model, input_ids)
            seq_shape = captured[LAYERS[0]].shape

            # Encode all source layers i <= target_layer_idx from clean inputs
            flat_inputs = [captured[LAYERS[i]].reshape(-1, clt.cfg.input_size) for i in range(target_layer_idx + 1)]
            source_acts = [clt.encode_layer(flat_inputs[i], i) for i in range(target_layer_idx + 1)]

            # Decode only the target layer
            recon = clt.b_dec[target_layer_idx].unsqueeze(0).expand(source_acts[0].shape[0], -1).clone()
            for i in range(target_layer_idx + 1):
                recon = recon + source_acts[i] @ clt.W_dec[i][target_layer_idx - i]
            recon_shaped = recon.reshape(seq_shape)

            def _make_const(tensor):
                return lambda *a, **kw: tensor

            with patched_forward(base_model.h[LAYERS[target_layer_idx]].mlp, _make_const(recon_shaped)):
                layer_ce += compute_ce_loss(base_model, input_ids)

        total_ce += layer_ce / len(batches)
    return total_ce / clt.cfg.n_layers


# =============================================================================
# SPD eval modes
# =============================================================================


def compute_ce_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)).item()


@torch.no_grad()
def eval_spd_all(spd_model, batches, module_names, threshold: float) -> tuple[float, float]:
    """Evaluate SPD with all modules masked. Returns (ce, l0_per_module)."""
    from spd.models.components import make_mask_infos

    total_ce = 0.0
    module_l0_totals = {name: 0.0 for name in module_names}
    for input_ids in batches:
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
    n = len(batches)
    avg_l0 = sum(v / n for v in module_l0_totals.values()) / len(module_names)
    return total_ce / n, avg_l0


@torch.no_grad()
def eval_spd_single_mlp(spd_model, batches, threshold: float) -> float:
    """Evaluate SPD masking one MLP at a time, average CE over all layers."""
    from spd.models.components import make_mask_infos

    total_ce = 0.0
    for layer_idx in LAYERS:
        layer_module_names = [f"h.{layer_idx}.mlp.c_fc", f"h.{layer_idx}.mlp.down_proj"]
        layer_ce = 0.0
        for input_ids in batches:
            out = spd_model(input_ids, cache_type="input")
            ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
            masks = {}
            for mod_name in layer_module_names:
                ci_post = ci.lower_leaky[mod_name]
                masks[mod_name] = (ci_post > threshold).float()
            mask_infos = make_mask_infos(masks)
            logits = spd_model(input_ids, mask_infos=mask_infos)
            layer_ce += compute_ce_from_logits(logits, input_ids)
        total_ce += layer_ce / len(batches)
    return total_ce / len(LAYERS)


# =============================================================================
# Discover and load all e2e models
# =============================================================================


def discover_tc_models() -> list[tuple[str, int, dict[int, Path]]]:
    """Find all tc_{mode}_k{k}_layer{i}_final checkpoint sets.

    Returns [(mode, top_k, {layer: path}), ...] sorted by mode then k.
    """
    models = {}  # (mode, k) -> {layer: path}
    for mode in ("cascading", "parallel", "independent"):
        for k in (16, 32, 64):
            layer_paths = {}
            for layer in LAYERS:
                p = CHECKPOINT_DIR / f"tc_{mode}_k{k}_layer{layer}_final"
                if p.exists() and (p / "encoder.pt").exists():
                    layer_paths[layer] = p
            if len(layer_paths) == len(LAYERS):
                models[(mode, k)] = layer_paths
    return [(mode, k, paths) for (mode, k), paths in sorted(models.items())]


def discover_clt_models() -> list[tuple[str, int, Path]]:
    """Find all clt_{mode}_k{k}_final checkpoints.

    Returns [(mode, top_k, path), ...] sorted by mode then k.
    """
    models = []
    for mode in ("cascading", "parallel"):
        for k in (16, 32, 64):
            p = CHECKPOINT_DIR / f"clt_{mode}_k{k}_final"
            if p.exists() and (p / "encoder.pt").exists():
                models.append((mode, k, p))
    return sorted(models)


# =============================================================================
# SPD model loading (local, no wandb API needed)
# =============================================================================


def _load_spd_model_local(config_path: Path, checkpoint_path: Path, base_model_cache: Path):
    """Load SPD ComponentModel from local cached files."""
    import yaml
    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig
    from spd.models.component_model import ComponentModel, handle_deprecated_state_dict_keys_
    from spd.configs import ModulePatternInfoConfig, GlobalCiConfig
    from spd.utils.module_utils import expand_module_patterns

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Load base model from cache
    import json as json_mod
    with open(base_model_cache / "config.json") as f:
        model_cfg = LlamaSimpleMLPConfig(**json_mod.load(f))
    target_model = LlamaSimpleMLP(model_cfg)
    state_dict = torch.load(base_model_cache / "state_dict.pt", map_location="cpu", weights_only=True)
    target_model.load_state_dict(state_dict)
    target_model.eval()
    target_model.requires_grad_(False)

    module_info = [
        ModulePatternInfoConfig(module_pattern=m["module_pattern"], C=m["C"])
        for m in raw_config["module_info"]
    ]
    module_path_info = expand_module_patterns(target_model, module_info)

    ci_config = GlobalCiConfig(**raw_config["ci_config"])

    comp_model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_config=ci_config,
        pretrained_model_output_attr=raw_config.get("pretrained_model_output_attr", "idx_0"),
        sigmoid_type=raw_config.get("sigmoid_type", "leaky_hard"),
    )

    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    handle_deprecated_state_dict_keys_(weights)
    comp_model.load_state_dict(weights)

    return comp_model


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate e2e models in 3 CE settings")
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--spd_thresholds", type=float, nargs="+", default=[0.5, 0.0])
    parser.add_argument("--spd_only", action="store_true",
                        help="Only run SPD eval, load TC/CLT results from existing results.json")
    args = parser.parse_args()

    # Load base model
    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

    print("Loading base model...")
    model_cache = Path("experiments/exp_018_e2e_sweep/model_cache")
    import json as json_mod
    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLPConfig
    with open(model_cache / "config.json") as f:
        model_cfg = LlamaSimpleMLPConfig(**json_mod.load(f))
    base_model = LlamaSimpleMLP(model_cfg)
    state_dict = torch.load(model_cache / "state_dict.pt", map_location="cpu", weights_only=True)
    base_model.load_state_dict(state_dict)
    base_model.to(DEVICE)
    base_model.eval()

    # Load eval data
    print(f"Loading {args.n_eval_batches} eval batches...")
    batches = get_eval_batches(args.n_eval_batches, args.batch_size, args.seq_len)

    # Baseline CE
    print("Computing baseline CE...")
    baseline_ce = sum(compute_ce_loss(base_model, b) for b in batches) / len(batches)
    print(f"  Baseline CE: {baseline_ce:.4f}")

    output_path = Path("experiments/exp_019_eval_e2e/output/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.spd_only:
        # Load existing TC/CLT results
        with open(output_path) as f:
            existing = json.load(f)
        results = existing["results"]
        assert baseline_ce - existing["baseline_ce"] < 0.01, "Baseline CE mismatch"
        print(f"Loaded {len(results)} existing TC/CLT results")
    else:
        results = []

        # Evaluate transcoders
        tc_models = discover_tc_models()
        print(f"\nFound {len(tc_models)} transcoder model sets")
        for mode, k, layer_paths in tc_models:
            print(f"\n--- TC {mode} k={k} ---")
            transcoders = {layer: load_transcoder(layer_paths[layer]) for layer in LAYERS}

            l0 = compute_tc_l0(transcoders, base_model, batches)
            ce_cascading = eval_tc_all_cascading(base_model, transcoders, batches)
            ce_parallel = eval_tc_all_parallel(base_model, transcoders, batches)
            ce_single = eval_tc_single_mlp(base_model, transcoders, batches)

            print(f"  L0 (per MLP):              {l0:.1f}")
            print(f"  All-replace cascading:     {ce_cascading:.4f}  (delta: {ce_cascading - baseline_ce:.4f})")
            print(f"  All-replace parallel:      {ce_parallel:.4f}  (delta: {ce_parallel - baseline_ce:.4f})")
            print(f"  Single-MLP (avg):          {ce_single:.4f}  (delta: {ce_single - baseline_ce:.4f})")

            results.append({
                "type": "tc", "mode": mode, "top_k": k, "l0": l0,
                "ce_cascading": ce_cascading, "ce_parallel": ce_parallel, "ce_single": ce_single,
            })

            del transcoders
            torch.cuda.empty_cache()

        # Evaluate CLTs
        clt_models = discover_clt_models()
        print(f"\nFound {len(clt_models)} CLT models")
        for mode, k, path in clt_models:
            print(f"\n--- CLT {mode} k={k} ---")
            clt = load_clt(path)

            l0 = compute_clt_l0(clt, base_model, batches)
            ce_cascading = eval_clt_all_cascading(base_model, clt, batches)
            ce_parallel = eval_clt_all_parallel(base_model, clt, batches)
            ce_single = eval_clt_single_mlp(base_model, clt, batches)

            print(f"  L0 (per layer):            {l0:.1f}")
            print(f"  All-replace cascading:     {ce_cascading:.4f}  (delta: {ce_cascading - baseline_ce:.4f})")
            print(f"  All-replace parallel:      {ce_parallel:.4f}  (delta: {ce_parallel - baseline_ce:.4f})")
            print(f"  Single-MLP (avg):          {ce_single:.4f}  (delta: {ce_single - baseline_ce:.4f})")

            results.append({
                "type": "clt", "mode": mode, "top_k": k, "l0": l0,
                "ce_cascading": ce_cascading, "ce_parallel": ce_parallel, "ce_single": ce_single,
            })

            del clt
            torch.cuda.empty_cache()

    # Evaluate SPD
    print("\nLoading SPD model...")
    spd_model = _load_spd_model_local(
        config_path=Path("../spd/wandb/s-275c8f21/files/final_config.yaml"),
        checkpoint_path=Path("../spd/wandb/s-275c8f21/files/model_400000.pth"),
        base_model_cache=Path("experiments/exp_018_e2e_sweep/model_cache"),
    )
    spd_model.to(DEVICE)
    all_module_names = [f"h.{l}.mlp.c_fc" for l in LAYERS] + [f"h.{l}.mlp.down_proj" for l in LAYERS]

    spd_results = []
    for threshold in args.spd_thresholds:
        label = f"CI>{threshold}"
        print(f"\n--- SPD ({label}) ---")

        ce_all, l0 = eval_spd_all(spd_model, batches, all_module_names, threshold)
        ce_single = eval_spd_single_mlp(spd_model, batches, threshold)

        print(f"  L0 (per module):           {l0:.1f}")
        print(f"  All-replace (parallel):    {ce_all:.4f}  (delta: {ce_all - baseline_ce:.4f})")
        print(f"  Single-MLP (avg):          {ce_single:.4f}  (delta: {ce_single - baseline_ce:.4f})")

        spd_results.append({
            "type": "spd", "mode": label, "threshold": threshold, "l0": l0,
            "ce_all": ce_all, "ce_single": ce_single,
        })

    del spd_model
    torch.cuda.empty_cache()

    # Evaluate jose (SPD s-55ea3f9b, different target model t-9d2b8f02)
    jose_cache = Path("experiments/exp_019_eval_e2e/jose_model_cache")
    if jose_cache.exists():
        print("\n--- Loading jose target model (t-9d2b8f02) ---")
        import json as json_mod
        from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLPConfig
        with open(jose_cache / "config.json") as f:
            jose_model_cfg = LlamaSimpleMLPConfig(**json_mod.load(f))
        jose_base = LlamaSimpleMLP(jose_model_cfg)
        jose_sd = torch.load(jose_cache / "state_dict.pt", map_location="cpu", weights_only=True)
        jose_base.load_state_dict(jose_sd)
        jose_base.to(DEVICE)
        jose_base.eval()
        del jose_sd

        jose_baseline_ce = sum(compute_ce_loss(jose_base, b) for b in batches) / len(batches)
        print(f"  Jose baseline CE: {jose_baseline_ce:.4f}")

        print("Loading jose SPD model...")
        jose_spd = _load_spd_model_local(
            config_path=Path("../spd/wandb/s-55ea3f9b/final_config.yaml"),
            checkpoint_path=Path("../spd/wandb/s-55ea3f9b/model_400000.pth"),
            base_model_cache=jose_cache,
        )
        jose_spd.to(DEVICE)

        jose_module_names = [f"h.{l}.mlp.c_fc" for l in LAYERS] + [f"h.{l}.mlp.down_proj" for l in LAYERS]

        jose_results = []
        for threshold in args.spd_thresholds:
            label = f"CI>{threshold}"
            print(f"\n--- Jose ({label}) ---")

            ce_all, l0 = eval_spd_all(jose_spd, batches, jose_module_names, threshold)
            ce_single = eval_spd_single_mlp(jose_spd, batches, threshold)

            print(f"  L0 (per module):           {l0:.1f}")
            print(f"  All-replace (parallel):    {ce_all:.4f}  (delta: {ce_all - jose_baseline_ce:.4f})")
            print(f"  Single-MLP (avg):          {ce_single:.4f}  (delta: {ce_single - jose_baseline_ce:.4f})")

            jose_results.append({
                "type": "jose", "mode": label, "threshold": threshold, "l0": l0,
                "ce_all": ce_all, "ce_single": ce_single,
            })

        del jose_spd, jose_base
        torch.cuda.empty_cache()
    else:
        jose_results = []
        jose_baseline_ce = None
        print("\nSkipping jose (no model cache found)")

    # Save results
    with open(output_path, "w") as f:
        save_data = {
            "baseline_ce": baseline_ce, "results": results,
            "spd_results": spd_results, "jose_results": jose_results,
        }
        if jose_baseline_ce is not None:
            save_data["jose_baseline_ce"] = jose_baseline_ce
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'='*85}")
    print(f"{'Model':<25} {'k':>4}  {'Cascading':>10}  {'Parallel':>10}  {'Single':>10}  | {'Casc Δ':>8}  {'Para Δ':>8}  {'Sing Δ':>8}")
    print(f"{'-'*85}")
    for r in results:
        label = f"{r['type']}_{r['mode']}"
        print(
            f"{label:<25} {r['top_k']:>4}"
            f"  {r['ce_cascading']:>10.4f}  {r['ce_parallel']:>10.4f}  {r['ce_single']:>10.4f}"
            f"  | {r['ce_cascading'] - baseline_ce:>8.4f}  {r['ce_parallel'] - baseline_ce:>8.4f}  {r['ce_single'] - baseline_ce:>8.4f}"
        )
    print(f"{'-'*85}")
    for r in spd_results:
        label = f"spd_{r['mode']}"
        print(
            f"{label:<25} {'':>4}"
            f"  {'n/a':>10}  {r['ce_all']:>10.4f}  {r['ce_single']:>10.4f}"
            f"  | {'n/a':>8}  {r['ce_all'] - baseline_ce:>8.4f}  {r['ce_single'] - baseline_ce:>8.4f}"
        )
    print(f"\nBaseline CE: {baseline_ce:.4f}")


if __name__ == "__main__":
    main()
