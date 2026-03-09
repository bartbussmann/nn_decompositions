import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from typing import Callable

import torch
import torch.nn as nn
import wandb

from nn_decompositions.config import CLTConfig, EncoderConfig

# (model, tokenizer, input_ids, attention_mask | None) -> float
ComputeLossFn = Callable[[nn.Module, object, torch.Tensor, torch.Tensor | None], float]


# =============================================================================
# Wandb Logging
# =============================================================================

def init_wandb(cfg: EncoderConfig | CLTConfig):
    """Initialize wandb run."""
    cfg_dict = asdict(cfg)
    cfg_dict["dtype"] = str(cfg.dtype)
    return wandb.init(
        project=cfg.wandb_project, name=cfg.name, config=cfg_dict, reinit=True
    )


def get_encoder_metrics(output: dict) -> dict:
    """Extract metrics from encoder output as a dict."""
    metrics_to_log = [
        "loss", "l2_loss", "kl_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss",
        "num_dead_features",
    ]
    log_dict = {
        k: output[k].item() if hasattr(output[k], "item") else output[k]
        for k in metrics_to_log if k in output
    }
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
def patched_forward(module: nn.Module, patched_fn):
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
    output_acts_flat = output_acts.reshape(-1, cfg.output_size)

    encoder_out = encoder(input_acts_flat, output_acts_flat)["output"].reshape(
        output_acts.shape
    )

    output_module = activation_store.output_module
    original_forward = output_module.forward

    original_loss = loss(input_ids, attention_mask)

    with patched_forward(output_module, lambda *a, **kw: encoder_out):
        reconstr_loss = loss(input_ids, attention_mask)

    def zero_forward(*args, **kwargs):
        return torch.zeros_like(original_forward(*args, **kwargs))

    with patched_forward(output_module, zero_forward):
        zero_loss = loss(input_ids, attention_mask)

    def mean_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        return out.mean([0, 1]).expand_as(out)

    with patched_forward(output_module, mean_forward):
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
# CLT Performance Evaluation
# =============================================================================

@torch.no_grad()
def get_clt_performance_metrics(
    activation_store,
    clt,
    batch_tokens: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    compute_loss_fn: ComputeLossFn | None = None,
) -> dict:
    """Compute CE degradation and recovery metrics for a cross-layer transcoder.

    Logs both parallel (all layers from clean activations) and cascading
    (each layer sees modified residual stream) reconstruction losses.
    """
    import torch.nn as nn_module

    cfg = clt.cfg
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

    output_modules = activation_store.output_modules
    original_forwards = [m.forward for m in output_modules]

    original_loss = loss(input_ids, attention_mask)

    # --- Parallel reconstruction: encode from clean activations, patch all at once ---
    input_acts_list, output_acts_list = activation_store.get_activations(
        input_ids, attention_mask,
    )
    output_seq_shape = output_acts_list[0].shape
    flat_inputs = [a.reshape(-1, cfg.input_size) for a in input_acts_list]
    flat_targets = [a.reshape(-1, cfg.output_size) for a in output_acts_list]
    reconstructions = clt(flat_inputs, flat_targets)["output"]
    reconstructions = [r.reshape(output_seq_shape) for r in reconstructions]

    def _make_const_fn(tensor):
        return lambda *a, **kw: tensor

    with ExitStack() as stack:
        for mod, recon in zip(output_modules, reconstructions):
            stack.enter_context(patched_forward(mod, _make_const_fn(recon)))
        parallel_reconstr_loss = loss(input_ids, attention_mask)

    # --- Cascading reconstruction: each layer's encoder sees modified inputs ---
    all_acts: list[torch.Tensor] = []

    def _make_cascading_hook(layer_idx: int):
        def _hook(_module: nn_module.Module, inp: tuple, _output: torch.Tensor):
            encoder_input = inp[0]
            flat = encoder_input.reshape(-1, cfg.input_size)
            acts = clt.encode_layer(flat, layer_idx)
            all_acts.append(acts)
            recon = clt.b_dec[layer_idx].unsqueeze(0).expand(flat.shape[0], -1).clone()
            for i in range(layer_idx + 1):
                recon = recon + all_acts[i] @ clt.W_dec[i][layer_idx - i]
            return recon.reshape(encoder_input.shape)
        return _hook

    hooks = []
    for layer_idx in range(cfg.n_layers):
        h = output_modules[layer_idx].register_forward_hook(
            _make_cascading_hook(layer_idx)
        )
        hooks.append(h)
    cascading_reconstr_loss = loss(input_ids, attention_mask)
    for h in hooks:
        h.remove()

    # --- Baselines ---
    def _make_zero_fn(orig_forward):
        return lambda *a, **kw: torch.zeros_like(orig_forward(*a, **kw))

    with ExitStack() as stack:
        for mod, orig_fwd in zip(output_modules, original_forwards):
            stack.enter_context(patched_forward(mod, _make_zero_fn(orig_fwd)))
        zero_loss = loss(input_ids, attention_mask)

    def _make_mean_fn(orig_forward):
        def mean_fn(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            return out.mean([0, 1]).expand_as(out)
        return mean_fn

    with ExitStack() as stack:
        for mod, orig_fwd in zip(output_modules, original_forwards):
            stack.enter_context(patched_forward(mod, _make_mean_fn(orig_fwd)))
        mean_loss = loss(input_ids, attention_mask)

    return {
        "performance/original_loss": original_loss,
        "performance/cascading_reconstr_loss": cascading_reconstr_loss,
        "performance/parallel_reconstr_loss": parallel_reconstr_loss,
        "performance/zero_loss": zero_loss,
        "performance/mean_loss": mean_loss,
        "performance/cascading_ce_degradation": cascading_reconstr_loss - original_loss,
        "performance/parallel_ce_degradation": parallel_reconstr_loss - original_loss,
        "performance/cascading_recovery_from_zero": 100 * (cascading_reconstr_loss - zero_loss) / (original_loss - zero_loss),
        "performance/parallel_recovery_from_zero": 100 * (parallel_reconstr_loss - zero_loss) / (original_loss - zero_loss),
    }


@torch.no_grad()
def log_clt_performance(
    wandb_run,
    step: int,
    activation_store,
    clt,
    batch_tokens: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    compute_loss_fn: ComputeLossFn | None = None,
):
    """Log CLT performance metrics to wandb."""
    log_dict = get_clt_performance_metrics(
        activation_store, clt, batch_tokens, compute_loss_fn,
    )
    wandb_run.log(log_dict, step=step)


# =============================================================================
# Cascading Independent Transcoders Performance Evaluation
# =============================================================================

@torch.no_grad()
def get_cascading_performance_metrics(
    activation_store,
    encoders: list,
    batch_tokens: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    compute_loss_fn: ComputeLossFn | None = None,
) -> dict:
    """Compute CE degradation and recovery metrics for cascading independent transcoders.

    Uses cascading forward hooks (matching training): each layer's encoder sees
    the residual stream modified by previous layers' reconstructions.
    """
    import torch.nn as nn

    cfg = encoders[0].cfg
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

    output_modules = activation_store.output_modules
    original_forwards = [m.forward for m in output_modules]

    original_loss = loss(input_ids, attention_mask)

    # Cascading reconstruction: each MLP hook encodes + decodes on the fly
    def _make_cascading_hook(layer_idx: int):
        def _hook(_module: nn.Module, inp: tuple, _output: torch.Tensor):
            encoder_input = inp[0]
            flat = encoder_input.reshape(-1, cfg.input_size)
            acts = encoders[layer_idx].encode(flat)
            recon = encoders[layer_idx].decode(acts)
            return recon.reshape(encoder_input.shape)
        return _hook

    hooks = []
    for layer_idx in range(len(encoders)):
        h = output_modules[layer_idx].register_forward_hook(
            _make_cascading_hook(layer_idx)
        )
        hooks.append(h)
    cascading_reconstr_loss = loss(input_ids, attention_mask)
    for h in hooks:
        h.remove()

    # Parallel reconstruction: encode from clean activations, patch all at once
    input_acts_list, _ = activation_store.get_activations(input_ids, attention_mask)
    reconstructions = []
    for enc, inp in zip(encoders, input_acts_list):
        flat = inp.reshape(-1, cfg.input_size)
        acts = enc.encode(flat)
        recon = enc.decode(acts)
        reconstructions.append(recon.reshape(inp.shape))

    def _make_const_fn(tensor):
        return lambda *a, **kw: tensor

    with ExitStack() as stack:
        for mod, recon in zip(output_modules, reconstructions):
            stack.enter_context(patched_forward(mod, _make_const_fn(recon)))
        parallel_reconstr_loss = loss(input_ids, attention_mask)

    # Zero ablation
    def _make_zero_fn(orig_forward):
        return lambda *a, **kw: torch.zeros_like(orig_forward(*a, **kw))

    with ExitStack() as stack:
        for mod, orig_fwd in zip(output_modules, original_forwards):
            stack.enter_context(patched_forward(mod, _make_zero_fn(orig_fwd)))
        zero_loss = loss(input_ids, attention_mask)

    # Mean ablation
    def _make_mean_fn(orig_forward):
        def mean_fn(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            return out.mean([0, 1]).expand_as(out)
        return mean_fn

    with ExitStack() as stack:
        for mod, orig_fwd in zip(output_modules, original_forwards):
            stack.enter_context(patched_forward(mod, _make_mean_fn(orig_fwd)))
        mean_loss = loss(input_ids, attention_mask)

    return {
        "performance/original_loss": original_loss,
        "performance/cascading_reconstr_loss": cascading_reconstr_loss,
        "performance/parallel_reconstr_loss": parallel_reconstr_loss,
        "performance/zero_loss": zero_loss,
        "performance/mean_loss": mean_loss,
        "performance/cascading_ce_degradation": cascading_reconstr_loss - original_loss,
        "performance/parallel_ce_degradation": parallel_reconstr_loss - original_loss,
        "performance/cascading_recovery_from_zero": 100 * (cascading_reconstr_loss - zero_loss) / (original_loss - zero_loss),
        "performance/parallel_recovery_from_zero": 100 * (parallel_reconstr_loss - zero_loss) / (original_loss - zero_loss),
    }


@torch.no_grad()
def log_cascading_performance(
    wandb_run,
    step: int,
    activation_store,
    encoders: list,
    batch_tokens: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    compute_loss_fn: ComputeLossFn | None = None,
):
    """Log cascading independent transcoders performance metrics to wandb."""
    log_dict = get_cascading_performance_metrics(
        activation_store, encoders, batch_tokens, compute_loss_fn,
    )
    wandb_run.log(log_dict, step=step)


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    encoder,
    cfg: EncoderConfig | CLTConfig,
    step: int | str,
    wandb_run=None,
):
    """Save encoder checkpoint locally and optionally upload to wandb."""
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

    if wandb_run is not None:
        artifact = wandb.Artifact(
            name=f"{cfg.name}_checkpoint_{step}",
            type="model",
            metadata={"step": step, "run_name": cfg.name},
        )
        artifact.add_file(encoder_path, name="encoder.pt")
        artifact.add_file(config_path, name="config.json")
        wandb_run.log_artifact(artifact)

    print(f"Checkpoint saved at step {step}: {save_dir}")
