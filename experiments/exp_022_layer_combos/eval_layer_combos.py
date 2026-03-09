"""Evaluate CE degradation for every combination of replaced layers.

For SPD and TC models, replaces subsets of layers {0}, {1}, ..., {0,1}, {0,2}, ..., {0,1,2,3}
and measures CE for each combination. This reveals how errors compound across layers.

Usage:
    python experiments/exp_022_layer_combos/eval_layer_combos.py
    python experiments/exp_022_layer_combos/eval_layer_combos.py --skip_tc
"""

import json
import sys
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
JOSE_MODEL_CACHE = Path("experiments/exp_019_eval_e2e/jose_model_cache")
CHECKPOINT_DIR = Path("checkpoints/jose")
OUTPUT_DIR = Path("experiments/exp_022_layer_combos/output")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp_019_eval_e2e"))
from eval_e2e import (
    get_eval_batches,
    patched_forward,
    compute_ce_loss,
    compute_ce_from_logits,
    load_transcoder,
    _load_spd_model_local,
    _collect_rms2_outputs,
)


ALL_COMBOS = []
for r in range(1, len(LAYERS) + 1):
    for combo in combinations(LAYERS, r):
        ALL_COMBOS.append(combo)


@torch.no_grad()
def eval_spd_layer_combo(spd_model, batches, combo: tuple[int, ...], threshold: float) -> float:
    """Evaluate SPD masking only the MLP modules in the given layer combo."""
    from spd.models.components import make_mask_infos

    module_names = []
    for layer in combo:
        module_names.extend([f"h.{layer}.mlp.c_fc", f"h.{layer}.mlp.down_proj"])

    total_ce = 0.0
    for input_ids in batches:
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        masks = {}
        for mod_name in module_names:
            ci_post = ci.lower_leaky[mod_name]
            masks[mod_name] = (ci_post > threshold).float()
        mask_infos = make_mask_infos(masks)
        logits = spd_model(input_ids, mask_infos=mask_infos)
        total_ce += compute_ce_from_logits(logits, input_ids)
    return total_ce / len(batches)


@torch.no_grad()
def eval_tc_layer_combo_cascading(base_model, transcoders: dict[int, BatchTopKTranscoder],
                                   batches, combo: tuple[int, ...]) -> float:
    """Replace only the layers in combo (cascading: modified residual stream)."""
    total_ce = 0.0
    for input_ids in batches:
        hooks = []
        for layer_idx in combo:
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--spd_thresholds", type=float, nargs="+", default=[0.0])
    parser.add_argument("--skip_tc", action="store_true")
    parser.add_argument("--skip_spd", action="store_true")
    parser.add_argument("--tc_k", type=int, default=32, help="Top-k for TC models")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load base model
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

    results = {"baseline_ce": baseline_ce, "combos": {}}

    combo_keys = [str(list(c)) for c in ALL_COMBOS]
    print(f"\nWill evaluate {len(ALL_COMBOS)} layer combinations: {combo_keys}")

    # SPD evaluation
    if not args.skip_spd:
        for threshold in args.spd_thresholds:
            label = f"spd_ci{threshold}"
            print(f"\n=== SPD (CI>{threshold}) ===")
            torch.cuda.empty_cache()
            jose_spd = _load_spd_model_local(
                config_path=Path("../spd/wandb/s-55ea3f9b/final_config.yaml"),
                checkpoint_path=Path("../spd/wandb/s-55ea3f9b/model_400000.pth"),
                base_model_cache=JOSE_MODEL_CACHE,
            )
            jose_spd.to(DEVICE)

            spd_combo_results = {}
            for combo in ALL_COMBOS:
                ce = eval_spd_layer_combo(jose_spd, batches, combo, threshold)
                delta = ce - baseline_ce
                combo_str = str(list(combo))
                spd_combo_results[combo_str] = {"ce": ce, "delta": delta}
                print(f"  Layers {combo_str}: CE={ce:.4f}  Δ={delta:.4f}")

            results[label] = spd_combo_results

            del jose_spd
            torch.cuda.empty_cache()

    # TC evaluations
    if not args.skip_tc:
        k = args.tc_k
        for mode in ("cascading", "parallel", "independent"):
            label = f"tc_{mode}_k{k}"
            layer_paths = {}
            for layer in LAYERS:
                candidates = list(CHECKPOINT_DIR.glob(f"tc_{mode}_k{k}_*layer{layer}_final"))
                if candidates and (candidates[0] / "encoder.pt").exists():
                    layer_paths[layer] = candidates[0]
            if len(layer_paths) != len(LAYERS):
                print(f"\n=== TC {mode} k={k}: MISSING (need all 4 layers) ===")
                continue

            print(f"\n=== TC {mode} k={k} ===")
            transcoders = {layer: load_transcoder(layer_paths[layer]) for layer in LAYERS}

            tc_combo_results = {}
            for combo in ALL_COMBOS:
                ce = eval_tc_layer_combo_cascading(base_model, transcoders, batches, combo)
                delta = ce - baseline_ce
                combo_str = str(list(combo))
                tc_combo_results[combo_str] = {"ce": ce, "delta": delta}
                print(f"  Layers {combo_str}: CE={ce:.4f}  Δ={delta:.4f}")

            results[label] = tc_combo_results

            del transcoders
            torch.cuda.empty_cache()

    # Save results
    out_path = OUTPUT_DIR / "layer_combo_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
