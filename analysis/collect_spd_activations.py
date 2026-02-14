"""Collect top-K activating examples for SPD components from a wandb run.

Runs the ComponentModel over the Pile dataset (matching the SPD training data)
and saves top activating examples per component. Results are loaded by feature_dashboard.py.

Usage:
    python analysis/collect_spd_activations.py --spd_run goodfire/spd/s-275c8f21
    python analysis/collect_spd_activations.py --spd_run goodfire/spd/s-275c8f21 --modules "h.3.mlp.c_fc"
"""

import argparse
import heapq
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from spd.configs import GlobalCiConfig, ModulePatternInfoConfig
from spd.models.component_model import ComponentModel, handle_deprecated_state_dict_keys_
from spd.utils.general_utils import resolve_class
from spd.utils.module_utils import expand_module_patterns
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_spd_model(wandb_path: str) -> tuple[ComponentModel, dict]:
    """Load an SPD ComponentModel from a wandb run, bypassing full Config validation.

    The run config includes training-only fields (loss configs, autocast_bf16, etc.)
    that may not match the current codebase. We only need the model architecture fields
    to construct the ComponentModel for inference.
    """
    import wandb

    api = wandb.Api()
    run = api.run(wandb_path)
    run_dir = fetch_wandb_run_dir(run.id)

    checkpoint_remote = fetch_latest_wandb_checkpoint(run, prefix="model")
    checkpoint_path = download_wandb_file(run, run_dir, checkpoint_remote.name)
    config_path = download_wandb_file(run, run_dir, "final_config.yaml")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Load the target (base) model
    pretrained_model_name = raw_config.get("pretrained_model_name")
    model_class_path = raw_config["pretrained_model_class"]

    # Import LlamaSimpleMLP directly via importlib to avoid the pretrain __init__
    # chain, which imports gpt2.py -> log0 (renamed to `log` on this branch).
    if model_class_path == "spd.pretrain.models.llama_simple_mlp.LlamaSimpleMLP":
        import importlib

        mod = importlib.import_module("spd.pretrain.models.llama_simple_mlp")
        LlamaSimpleMLP = mod.LlamaSimpleMLP

        target_model = LlamaSimpleMLP.from_pretrained(pretrained_model_name)
    elif model_class_path.startswith("spd.pretrain.models."):
        model_class = resolve_class(model_class_path)
        from spd.pretrain.run_info import PretrainRunInfo

        pretrain_run_info = PretrainRunInfo.from_path(pretrained_model_name)
        if "model_type" not in pretrain_run_info.model_config_dict:
            pretrain_run_info.model_config_dict["model_type"] = model_class_path.split(".")[-1]
        target_model = model_class.from_run_info(pretrain_run_info)
    else:
        model_class = resolve_class(model_class_path)
        target_model = model_class.from_pretrained(pretrained_model_name)

    target_model.eval()
    target_model.requires_grad_(False)

    # Expand module patterns
    module_info = [
        ModulePatternInfoConfig(module_pattern=m["module_pattern"], C=m["C"])
        for m in raw_config["module_info"]
    ]
    module_path_info = expand_module_patterns(target_model, module_info)

    # Build CiConfig from raw YAML (only the fields CiConfig/GlobalCiConfig needs)
    raw_ci = raw_config["ci_config"]
    ci_config = GlobalCiConfig(**raw_ci)

    comp_model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_config=ci_config,
        pretrained_model_output_attr=raw_config.get("pretrained_model_output_attr", "idx_0"),
        sigmoid_type=raw_config.get("sigmoid_type", "leaky_hard"),
    )

    # Load checkpoint weights
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    handle_deprecated_state_dict_keys_(weights)
    comp_model.load_state_dict(weights)

    return comp_model, raw_config


def _make_example(
    token_ids: list[int], activation_values: list[float], max_position: int, max_value: float
) -> dict:
    return {
        "token_ids": token_ids,
        "activation_values": activation_values,
        "max_position": max_position,
        "max_value": max_value,
    }


def _push_to_heap(heap: list, example: dict, max_size: int) -> None:
    val = example["max_value"]
    if len(heap) < max_size:
        heapq.heappush(heap, (val, id(example), example))
    elif val > heap[0][0]:
        heapq.heapreplace(heap, (val, id(example), example))


def get_pile_batches(n_batches: int, batch_size: int, seq_len: int):
    """Load pre-tokenized Pile batches (matching SPD training data)."""
    dataset = load_dataset(
        "danbraunai/pile-uncopyrighted-tok", split="train", streaming=True
    )
    dataset = dataset.shuffle(seed=42, buffer_size=1000)
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


@torch.no_grad()
def collect_spd_activations(
    spd_model: ComponentModel,
    batches: list[torch.Tensor],
    module_names: list[str],
    top_k: int,
) -> dict[str, dict[int, list[dict]]]:
    """Collect top-K activating examples per SPD component by pre-sigmoid CI."""
    heaps: dict[str, dict[int, list]] = {}
    for mod_name in module_names:
        n_components = spd_model.module_to_c[mod_name]
        heaps[mod_name] = {i: [] for i in range(n_components)}

    for input_ids in tqdm(batches, desc="Collecting SPD activations"):
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(
            pre_weight_acts=out.cache, sampling="none", detach_inputs=False
        )

        for mod_name in module_names:
            ci_pre = ci.pre_sigmoid[mod_name]   # (B, S, C)
            ci_post = ci.lower_leaky[mod_name]  # (B, S, C)

            max_vals, max_positions = ci_pre.max(dim=1)  # (B, C)
            batch_indices, comp_indices = torch.where(max_vals > 0)

            for b_idx, c_idx in zip(batch_indices.tolist(), comp_indices.tolist()):
                max_val = max_vals[b_idx, c_idx].item()
                heap = heaps[mod_name][c_idx]
                if len(heap) < top_k or max_val > heap[0][0]:
                    example = _make_example(
                        token_ids=input_ids[b_idx].tolist(),
                        activation_values=ci_post[b_idx, :, c_idx].tolist(),
                        max_position=max_positions[b_idx, c_idx].item(),
                        max_value=max_val,
                    )
                    _push_to_heap(heap, example, top_k)

    results = {}
    for mod_name in module_names:
        results[mod_name] = {
            idx: [ex for _, _, ex in sorted(heap, reverse=True)]
            for idx, heap in heaps[mod_name].items()
            if heap
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help="Comma-separated module names to collect (default: all)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=str(Path(__file__).parent / "activation_cache")
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SPD model from {args.spd_run}...")
    spd_model, raw_config = load_spd_model(args.spd_run)
    spd_model.to(DEVICE)

    all_modules = list(spd_model.module_to_c.keys())
    print(f"Available modules: {all_modules}")
    for mod, c in spd_model.module_to_c.items():
        print(f"  {mod}: {c} components")

    if args.modules:
        module_names = [m.strip() for m in args.modules.split(",")]
        for m in module_names:
            assert m in spd_model.module_to_c, f"Module {m} not found. Available: {all_modules}"
    else:
        module_names = all_modules

    print(f"\nCollecting for modules: {module_names}")

    tokenizer_name = raw_config.get("tokenizer_name", "EleutherAI/gpt-neox-20b")

    print(f"\nLoading {args.n_batches} batches from Pile (pre-tokenized, seq_len={args.seq_len})...")
    batches = get_pile_batches(args.n_batches, args.batch_size, args.seq_len)
    total_tokens = args.n_batches * args.batch_size * args.seq_len
    print(f"  Total tokens: {total_tokens:,}")

    results = collect_spd_activations(spd_model, batches, module_names, args.top_k)

    for mod_name, examples in results.items():
        n_c = spd_model.module_to_c[mod_name]
        print(f"  {mod_name}: {len(examples)}/{n_c} components with examples")

    run_id = args.spd_run.split("/")[-1]
    cache_path = cache_dir / f"spd_{run_id}.pt"
    print(f"\nSaving cache to {cache_path}...")
    torch.save(
        {
            "modules": results,
            "config": {
                "n_batches": args.n_batches,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "top_k": args.top_k,
                "spd_run": args.spd_run,
                "tokenizer_name": tokenizer_name,
                "module_components": dict(spd_model.module_to_c),
            },
        },
        cache_path,
    )
    print("Done!")


if __name__ == "__main__":
    main()
