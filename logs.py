import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from typing import Callable

import torch
import torch.nn as nn
import wandb

from config import EncoderConfig

# (model, tokenizer, input_ids, attention_mask | None) -> float
ComputeLossFn = Callable[[nn.Module, object, torch.Tensor, torch.Tensor | None], float]


# =============================================================================
# Wandb Logging
# =============================================================================

def init_wandb(cfg: EncoderConfig):
    """Initialize wandb run."""
    cfg_dict = asdict(cfg)
    cfg_dict["dtype"] = str(cfg.dtype)
    return wandb.init(
        project=cfg.wandb_project, name=cfg.name, config=cfg_dict, reinit=True
    )


def get_encoder_metrics(output: dict) -> dict:
    """Extract metrics from encoder output as a dict."""
    metrics_to_log = [
        "loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss", "num_dead_features"
    ]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}
    log_dict["n_dead_in_batch"] = (output["feature_acts"].sum(0) == 0).sum().item()
    return log_dict


def log_wandb(output: dict, step: int, wandb_run, suffix: str | None = None):
    """Log encoder metrics to wandb run."""
    log_dict = get_encoder_metrics(output)
    if suffix is not None:
        log_dict = {f"{k}/{suffix}": v for k, v in log_dict.items()}
    wandb_run.log(log_dict, step=step)


# =============================================================================
# Model Performance Evaluation
# =============================================================================

def _compute_loss_hf(model, tokenizer, input_ids, attention_mask):
    """Compute CE loss for a HuggingFace causal LM."""
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100
    return model(input_ids, attention_mask=attention_mask, labels=labels).loss.item()


@contextmanager
def _patched_forward(module: nn.Module, patched_fn):
    """Temporarily replace a module's forward method.

    This monkey-patches module.forward rather than adding a hook, avoiding
    interaction with existing hooks (e.g. from ActivationsStore).
    """
    original_forward = module.forward
    module.forward = patched_fn
    try:
        yield
    finally:
        module.forward = original_forward


@contextmanager
def _multi_patched_forward(modules_and_fns: list[tuple[nn.Module, Callable]]):
    """Temporarily replace multiple modules' forward methods."""
    originals = [(mod, mod.forward) for mod, _ in modules_and_fns]
    for mod, fn in modules_and_fns:
        mod.forward = fn
    try:
        yield
    finally:
        for mod, orig_fn in originals:
            mod.forward = orig_fn


@torch.no_grad()
def get_performance_metrics(
    activation_store,
    encoder,
    batch_tokens: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    compute_loss_fn: ComputeLossFn | None = None,
) -> dict:
    """Compute CE degradation and recovery metrics."""
    cfg = encoder.cfg
    if batch_tokens is None:
        input_ids, attention_mask = activation_store.get_batch_tokens()
        input_ids = input_ids[:cfg.n_eval_seqs]
        if attention_mask is not None:
            attention_mask = attention_mask[:cfg.n_eval_seqs]
    else:
        input_ids, attention_mask = batch_tokens

    model = activation_store.model
    tokenizer = activation_store.tokenizer
    compute_loss = compute_loss_fn or _compute_loss_hf
    loss = lambda ids, mask: compute_loss(model, tokenizer, ids, mask)

    input_acts, output_acts = activation_store.get_activations(input_ids, attention_mask)
    input_acts_flat = input_acts.reshape(-1, cfg.input_size)

    num_layers = activation_store.num_output_layers
    multi = num_layers > 1

    if multi:
        output_acts_flat = output_acts.reshape(-1, num_layers, cfg.output_size)
    else:
        output_acts_flat = output_acts.reshape(-1, cfg.output_size)

    encoder_out = encoder(input_acts_flat, output_acts_flat)["output"].reshape(
        output_acts.shape
    )

    original_loss = loss(input_ids, attention_mask)

    if multi:
        modules = activation_store.output_modules
        original_forwards = [mod.forward for mod in modules]
        # encoder_out: (batch, seq, num_layers, output_size)
        layer_outs = [encoder_out[:, :, i, :] for i in range(num_layers)]

        with _multi_patched_forward(
            [(mod, lambda *a, o=o, **kw: o) for mod, o in zip(modules, layer_outs)]
        ):
            reconstr_loss = loss(input_ids, attention_mask)

        with _multi_patched_forward(
            [(mod, lambda *a, f=f, **kw: torch.zeros_like(f(*a, **kw)))
             for mod, f in zip(modules, original_forwards)]
        ):
            zero_loss = loss(input_ids, attention_mask)

        def _make_mean_fn(orig_fn):
            def mean_forward(*args, **kwargs):
                out = orig_fn(*args, **kwargs)
                return out.mean([0, 1]).expand_as(out)
            return mean_forward

        with _multi_patched_forward(
            [(mod, _make_mean_fn(f)) for mod, f in zip(modules, original_forwards)]
        ):
            mean_loss = loss(input_ids, attention_mask)
    else:
        output_module = activation_store.output_module
        original_forward = output_module.forward

        with _patched_forward(output_module, lambda *a, **kw: encoder_out):
            reconstr_loss = loss(input_ids, attention_mask)

        def zero_forward(*args, **kwargs):
            return torch.zeros_like(original_forward(*args, **kwargs))

        with _patched_forward(output_module, zero_forward):
            zero_loss = loss(input_ids, attention_mask)

        def mean_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            return out.mean([0, 1]).expand_as(out)

        with _patched_forward(output_module, mean_forward):
            mean_loss = loss(input_ids, attention_mask)

    return {
        "performance/original_loss": original_loss,
        "performance/reconstr_loss": reconstr_loss,
        "performance/zero_loss": zero_loss,
        "performance/mean_loss": mean_loss,
        "performance/ce_degradation": reconstr_loss - original_loss,
        "performance/recovery_from_zero": 100 * (reconstr_loss - zero_loss) / (original_loss - zero_loss),
        "performance/recovery_from_mean": 100 * (reconstr_loss - mean_loss) / (original_loss - mean_loss),
    }


@torch.no_grad()
def log_encoder_performance(
    wandb_run,
    step: int,
    activation_store,
    encoder,
    suffix: str | None = None,
    batch_tokens: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    compute_loss_fn: ComputeLossFn | None = None,
):
    """Log model performance metrics to wandb."""
    log_dict = get_performance_metrics(activation_store, encoder, batch_tokens, compute_loss_fn)
    if suffix is not None:
        log_dict = {f"{k}/{suffix}": v for k, v in log_dict.items()}
    wandb_run.log(log_dict, step=step)


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(encoder, cfg: EncoderConfig, step: int | str):
    """Save encoder checkpoint locally."""
    save_dir = f"checkpoints/{cfg.name}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    encoder_path = os.path.join(save_dir, "encoder.pt")
    torch.save(encoder.state_dict(), encoder_path)

    json_safe_cfg = {}
    for key, value in asdict(cfg).items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        else:
            json_safe_cfg[key] = str(value)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    print(f"Checkpoint saved at step {step}: {save_dir}")
