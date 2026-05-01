"""Shared utilities for the paper's experiment scripts.

These helpers (Pile streaming, CE loss, MLP-input hook, model loaders, …)
are imported by every sibling script under `experiments/`. Keeping them
here means a single canonical implementation per concern, and avoids
the copy-and-drift pattern across scripts.

This module lives in `experiments/` rather than `nn_decompositions/`
because it's paper-experiment scaffolding — the reusable PLT/CLT
library proper is just `nn_decompositions.{transcoder,clt,...}`.

The contents are deliberately small and dependency-light — anything that
needs upstream-`spd`-specific logic lives in `load_vpd_model` so importing
from this module does not pull in the upstream package until the function
is actually called.
"""

from __future__ import annotations

import gc
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from nn_decompositions.clt import CrossLayerTranscoder
from nn_decompositions.config import CLTConfig, EncoderConfig
from nn_decompositions.transcoder import BatchTopKTranscoder

PILE_DATASET = "danbraunai/pile-uncopyrighted-tok"


# =============================================================================
# Pile data
# =============================================================================


def get_pile_batches(
    n_batches: int,
    batch_size: int,
    seq_len: int,
    device: str | torch.device,
    *,
    seed: int = 0,
    shuffle_buffer: int = 10_000,
    desc: str = "Loading batches",
) -> list[torch.Tensor]:
    """Stream `danbraunai/pile-uncopyrighted-tok` into a list of token tensors.

    Note: this dataset is gated on Hugging Face — run `huggingface-cli login`
    once with an account that has access before calling this.
    """
    dataset = load_dataset(PILE_DATASET, split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    data_iter = iter(dataset)
    batches: list[torch.Tensor] = []
    for _ in tqdm(range(n_batches), desc=desc):
        batch_ids = []
        for _ in range(batch_size):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:seq_len])
        batches.append(torch.stack(batch_ids).to(device))
    return batches


# =============================================================================
# CE / patching helpers
# =============================================================================


def compute_ce_loss(model, input_ids: torch.Tensor) -> float:
    logits, _ = model(input_ids)
    return compute_ce_from_logits(logits, input_ids)


def compute_ce_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)
    ).item()


@contextmanager
def patched_forward(module: nn.Module, patched_fn):
    """Temporarily monkey-patch `module.forward` with `patched_fn`."""
    original = module.forward
    module.forward = patched_fn
    try:
        yield
    finally:
        module.forward = original


def collect_mlp_inputs(
    base_model, input_ids: torch.Tensor, layers: list[int]
) -> dict[int, torch.Tensor]:
    """Hook each layer's `rms_2` output (the input to its MLP) for a forward pass."""
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for layer_idx in layers:
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
# Bookkeeping
# =============================================================================


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_free_gpus(min_free_bytes: float) -> list[int]:
    """Return GPU IDs with at least `min_free_bytes` free VRAM."""
    free_gpus = []
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        if free >= min_free_bytes:
            free_gpus.append(i)
    return free_gpus


def parse_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert a string like `"torch.float32"` or `"float16"` to a torch dtype."""
    return getattr(torch, dtype_str.replace("torch.", ""))


# =============================================================================
# Model loaders
# =============================================================================


def load_transcoder(checkpoint_dir: Path, device: str | torch.device) -> BatchTopKTranscoder:
    """Load a single-layer BatchTopK transcoder from a wandb-artifact checkpoint dir."""
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["dtype"] = parse_torch_dtype(cfg_dict.get("dtype", "torch.float32"))
    cfg_dict["device"] = str(device)
    cfg = EncoderConfig(**cfg_dict)
    tc = BatchTopKTranscoder(cfg)
    tc.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=device))
    tc.eval()
    return tc


def load_clt(checkpoint_dir: Path, device: str | torch.device) -> CrossLayerTranscoder:
    """Load a Cross-Layer Transcoder from a wandb-artifact checkpoint dir."""
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["layers"] = json.loads(cfg_dict["layers"])
    cfg_dict["dtype"] = parse_torch_dtype(cfg_dict.get("dtype", "torch.float32"))
    cfg_dict["device"] = str(device)
    cfg = CLTConfig(**cfg_dict)
    clt = CrossLayerTranscoder(cfg)
    clt.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=device))
    clt.eval()
    return clt


def load_vpd_model(wandb_path: str):
    """Load a VPD `ComponentModel` from a wandb run, bypassing full Config validation.

    The run config includes training-only fields (loss configs, autocast_bf16, etc.)
    that may not match the current upstream codebase. This loader extracts only
    the architecture fields needed to reconstruct the `ComponentModel` for
    inference.

    Returns ``(component_model, raw_config_dict)``.
    """
    import wandb
    import yaml

    from spd.configs import GlobalCiConfig, ModulePatternInfoConfig
    from spd.models.component_model import ComponentModel, handle_deprecated_state_dict_keys_
    from spd.utils.general_utils import resolve_class
    from spd.utils.module_utils import expand_module_patterns
    from spd.utils.wandb_utils import (
        download_wandb_file,
        fetch_latest_wandb_checkpoint,
        fetch_wandb_run_dir,
    )

    api = wandb.Api()
    run = api.run(wandb_path)
    run_dir = fetch_wandb_run_dir(run.id)

    checkpoint_remote = fetch_latest_wandb_checkpoint(run, prefix="model")
    checkpoint_path = download_wandb_file(run, run_dir, checkpoint_remote.name)
    config_path = download_wandb_file(run, run_dir, "final_config.yaml")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    pretrained_model_name = raw_config.get("pretrained_model_name")
    model_class_path = raw_config["pretrained_model_class"]

    # Import LlamaSimpleMLP via importlib to avoid the pretrain __init__ chain,
    # which imports gpt2.py -> log0 (renamed to `log` on the upstream `spd`
    # branch we use).
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

    module_info = [
        ModulePatternInfoConfig(module_pattern=m["module_pattern"], C=m["C"])
        for m in raw_config["module_info"]
    ]
    module_path_info = expand_module_patterns(target_model, module_info)

    raw_ci = raw_config["ci_config"]
    ci_config = GlobalCiConfig(**raw_ci)

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

    return comp_model, raw_config
