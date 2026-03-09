"""Evaluate e2e-trained transcoders/CLTs on jose's target model (t-9d2b8f02).

Same eval as exp_019 but for jose models from pile_e2e_sweep_jose wandb project.
Downloads finished runs, evaluates in 3 CE settings (cascading, parallel, single-MLP),
includes jose SPD points, and produces plots.

Usage:
    python experiments/exp_021_eval_e2e_jose/eval_e2e_jose.py
    python experiments/exp_021_eval_e2e_jose/eval_e2e_jose.py --n_eval_batches 20
"""

import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig, CLTConfig
from nn_decompositions.clt import CrossLayerTranscoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
WANDB_PROJECT = "mats-sprint/pile_e2e_sweep_jose"
JOSE_MODEL_CACHE = Path("experiments/exp_019_eval_e2e/jose_model_cache")
CHECKPOINT_DIR = Path("checkpoints/jose")
OUTPUT_DIR = Path("experiments/exp_021_eval_e2e_jose/output")


# =============================================================================
# Download from wandb
# =============================================================================


def download_finished_runs():
    """Download all finished model artifacts from the jose sweep."""
    import wandb

    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT)

    downloaded = {"tc": {}, "clt": {}}

    for run in runs:
        if run.state != "finished":
            continue

        arts = [a for a in run.logged_artifacts() if a.type == "model"]
        if not arts:
            continue

        name = run.name  # e.g. "tc_cascading_k16", "clt_parallel_k32"
        parts = name.split("_")
        # Parse: {type}_{mode}_k{k} or {type}_{mode}_k{k} with layer info in artifact
        model_type = parts[0]  # tc or clt
        mode = parts[1]  # cascading, parallel, independent
        k = int(parts[2].removeprefix("k"))

        if model_type == "tc":
            # TC runs have 4 artifacts (one per layer)
            assert len(arts) == 4, f"Expected 4 artifacts for {name}, got {len(arts)}"
            layer_paths = {}
            for art in arts:
                dest = CHECKPOINT_DIR / f"{name}_{art.name.split(':')[0]}"
                if not (dest / "encoder.pt").exists():
                    art.download(root=str(dest))
                    print(f"  Downloaded {art.name} -> {dest}")
                else:
                    print(f"  Cached {dest}")

                # Parse layer from artifact name (e.g. "tc_cascading_k16_layer0_final:v0")
                art_base = art.name.split(":")[0]
                for p in art_base.split("_"):
                    if p.startswith("layer") and p[5:].isdigit():
                        layer = int(p[5:])
                        layer_paths[layer] = dest
                        break

            if len(layer_paths) == len(LAYERS):
                downloaded["tc"].setdefault((mode, k), {}).update(layer_paths)

        else:  # clt
            assert len(arts) == 1, f"Expected 1 artifact for {name}, got {len(arts)}"
            dest = CHECKPOINT_DIR / f"{name}_final"
            if not (dest / "encoder.pt").exists():
                arts[0].download(root=str(dest))
                print(f"  Downloaded {arts[0].name} -> {dest}")
            else:
                print(f"  Cached {dest}")
            downloaded["clt"][(mode, k)] = dest

    return downloaded


# =============================================================================
# Reuse eval functions from exp_019
# =============================================================================

# Import shared helpers by adding exp_019 to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp_019_eval_e2e"))
from eval_e2e import (
    get_eval_batches,
    patched_forward,
    compute_ce_loss,
    compute_ce_from_logits,
    load_transcoder,
    load_clt,
    compute_tc_l0,
    eval_tc_all_parallel,
    eval_tc_all_cascading,
    eval_tc_single_mlp,
    compute_clt_l0,
    eval_clt_all_parallel,
    eval_clt_all_cascading,
    eval_clt_single_mlp,
    eval_spd_all,
    eval_spd_single_mlp,
    _load_spd_model_local,
    _collect_rms2_outputs,
)


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate jose e2e models in 3 CE settings")
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--spd_thresholds", type=float, nargs="+", default=[0.5, 0.0])
    parser.add_argument("--skip_spd", action="store_true", help="Skip SPD evaluation")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download finished runs from wandb
    if not args.skip_download:
        print("Downloading finished runs from wandb...")
        downloaded = download_finished_runs()
        # Save download manifest
        manifest = {
            "tc": {f"{m}_{k}": {str(l): str(p) for l, p in paths.items()}
                   for (m, k), paths in downloaded["tc"].items()},
            "clt": {f"{m}_{k}": str(p) for (m, k), p in downloaded["clt"].items()},
        }
        with open(OUTPUT_DIR / "download_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    else:
        print("Skipping download, using cached checkpoints")

    # Load jose base model
    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig

    print("\nLoading jose base model...")
    with open(JOSE_MODEL_CACHE / "config.json") as f:
        model_cfg = LlamaSimpleMLPConfig(**json.load(f))
    base_model = LlamaSimpleMLP(model_cfg)
    sd = torch.load(JOSE_MODEL_CACHE / "state_dict.pt", map_location="cpu", weights_only=True)
    base_model.load_state_dict(sd)
    base_model.to(DEVICE)
    base_model.eval()
    del sd

    # Load eval data
    print(f"Loading {args.n_eval_batches} eval batches...")
    batches = get_eval_batches(args.n_eval_batches, args.batch_size, args.seq_len)

    # Baseline CE
    print("Computing baseline CE...")
    baseline_ce = sum(compute_ce_loss(base_model, b) for b in batches) / len(batches)
    print(f"  Baseline CE: {baseline_ce:.4f}")

    results = []

    # Discover downloaded TC models
    tc_models = []
    for mode in ("cascading", "parallel", "independent"):
        for k in (8, 16, 32, 64):
            layer_paths = {}
            for layer in LAYERS:
                candidates = list(CHECKPOINT_DIR.glob(f"tc_{mode}_k{k}_*layer{layer}_final"))
                if candidates and (candidates[0] / "encoder.pt").exists():
                    layer_paths[layer] = candidates[0]
            if len(layer_paths) == len(LAYERS):
                tc_models.append((mode, k, layer_paths))

    print(f"\nFound {len(tc_models)} complete TC model sets")
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

    # Discover downloaded CLT models
    clt_models = []
    for mode in ("cascading", "parallel"):
        for k in (8, 16, 32, 64):
            p = CHECKPOINT_DIR / f"clt_{mode}_k{k}_final"
            if p.exists() and (p / "encoder.pt").exists():
                clt_models.append((mode, k, p))

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

    # Save intermediate results (TC + CLT only) in case SPD OOMs
    intermediate_path = OUTPUT_DIR / "results_tc_clt.json"
    with open(intermediate_path, "w") as f:
        json.dump({"baseline_ce": baseline_ce, "results": results, "spd_results": []}, f, indent=2)
    print(f"\nIntermediate results saved to {intermediate_path}")

    spd_results = []
    if not args.skip_spd:
        # Evaluate jose SPD
        print("\nLoading jose SPD model...")
        torch.cuda.empty_cache()
        jose_spd = _load_spd_model_local(
            config_path=Path("../spd/wandb/s-55ea3f9b/final_config.yaml"),
            checkpoint_path=Path("../spd/wandb/s-55ea3f9b/model_400000.pth"),
            base_model_cache=JOSE_MODEL_CACHE,
        )
        jose_spd.to(DEVICE)
        all_module_names = [f"h.{l}.mlp.c_fc" for l in LAYERS] + [f"h.{l}.mlp.down_proj" for l in LAYERS]

        for threshold in args.spd_thresholds:
            label = f"CI>{threshold}"
            print(f"\n--- Jose SPD ({label}) ---")

            ce_all, l0 = eval_spd_all(jose_spd, batches, all_module_names, threshold)
            ce_single = eval_spd_single_mlp(jose_spd, batches, threshold)

            print(f"  L0 (per module):           {l0:.1f}")
            print(f"  All-replace (parallel):    {ce_all:.4f}  (delta: {ce_all - baseline_ce:.4f})")
            print(f"  Single-MLP (avg):          {ce_single:.4f}  (delta: {ce_single - baseline_ce:.4f})")

            spd_results.append({
                "type": "jose", "mode": label, "threshold": threshold, "l0": l0,
                "ce_all": ce_all, "ce_single": ce_single,
            })

        del jose_spd
        torch.cuda.empty_cache()
    else:
        print("\nSkipping SPD evaluation")

    # Save results
    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump({"baseline_ce": baseline_ce, "results": results, "spd_results": spd_results}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Model':<25} {'k':>4} {'L0':>6}  {'Cascading':>10}  {'Parallel':>10}  {'Single':>10}  | {'Casc Δ':>8}  {'Para Δ':>8}  {'Sing Δ':>8}")
    print(f"{'-'*90}")
    for r in results:
        label = f"{r['type']}_{r['mode']}"
        print(
            f"{label:<25} {r['top_k']:>4} {r['l0']:>6.1f}"
            f"  {r['ce_cascading']:>10.4f}  {r['ce_parallel']:>10.4f}  {r['ce_single']:>10.4f}"
            f"  | {r['ce_cascading'] - baseline_ce:>8.4f}  {r['ce_parallel'] - baseline_ce:>8.4f}  {r['ce_single'] - baseline_ce:>8.4f}"
        )
    print(f"{'-'*90}")
    for r in spd_results:
        label = f"jose_{r['mode']}"
        print(
            f"{label:<25} {'':>4} {r['l0']:>6.1f}"
            f"  {'n/a':>10}  {r['ce_all']:>10.4f}  {r['ce_single']:>10.4f}"
            f"  | {'n/a':>8}  {r['ce_all'] - baseline_ce:>8.4f}  {r['ce_single'] - baseline_ce:>8.4f}"
        )
    print(f"\nBaseline CE: {baseline_ce:.4f}")


if __name__ == "__main__":
    main()
