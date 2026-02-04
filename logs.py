import json
import os
from dataclasses import asdict
from functools import partial

import torch
import wandb

from config import EncoderConfig


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

def _reconstr_hook(activation, hook, encoder_out):
    return encoder_out


def _zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)


def _mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)


@torch.no_grad()
def get_performance_metrics(
    model, activations_store, encoder, batch_tokens: torch.Tensor | None = None
) -> dict:
    """Compute model performance metrics and return as dict."""
    cfg = encoder.cfg
    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()[: cfg.n_eval_seqs]

    input_acts, _ = activations_store.get_activations(batch_tokens)
    input_acts_flat = input_acts.reshape(-1, cfg.input_size)

    encoder_out = encoder(input_acts_flat, input_acts_flat)["output"].reshape(
        batch_tokens.shape[0], batch_tokens.shape[1], -1
    )

    hook_point = cfg.eval_hook_point

    original_loss = model(batch_tokens, return_type="loss").item()
    reconstr_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(hook_point, partial(_reconstr_hook, encoder_out=encoder_out))],
        return_type="loss",
    ).item()
    zero_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(hook_point, _zero_abl_hook)],
        return_type="loss",
    ).item()
    mean_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(hook_point, _mean_abl_hook)],
        return_type="loss",
    ).item()

    ce_degradation = original_loss - reconstr_loss
    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    return {
        "performance/ce_degradation": ce_degradation,
        "performance/recovery_from_zero": (reconstr_loss - zero_loss) / zero_degradation,
        "performance/recovery_from_mean": (reconstr_loss - mean_loss) / mean_degradation,
    }


@torch.no_grad()
def log_encoder_performance(
    wandb_run, step: int, model, activations_store, encoder,
    suffix: str | None = None, batch_tokens: torch.Tensor | None = None
):
    """Log model performance metrics to wandb run."""
    log_dict = get_performance_metrics(model, activations_store, encoder, batch_tokens)
    if suffix is not None:
        # performance/ce_degradation -> performance/ce_degradation/topk
        log_dict = {f"{k}/{suffix}": v for k, v in log_dict.items()}
    wandb_run.log(log_dict, step=step)


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(encoder, cfg: EncoderConfig, step: int):
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
