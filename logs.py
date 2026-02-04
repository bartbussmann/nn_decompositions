import json
import os
from dataclasses import asdict
from functools import partial

import torch
import wandb

from config import EncoderConfig


def init_wandb(cfg: EncoderConfig):
    # Convert dataclass to dict for wandb config
    cfg_dict = asdict(cfg)
    cfg_dict["dtype"] = str(cfg.dtype)  # torch.dtype not JSON serializable
    return wandb.init(
        project=cfg.wandb_project, name=cfg.name, config=cfg_dict, reinit=True
    )


def log_wandb(output, step, wandb_run, index=None):
    metrics_to_log = [
        "loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss", "num_dead_features"
    ]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}
    log_dict["n_dead_in_batch"] = (output["feature_acts"].sum(0) == 0).sum().item()

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    return sae_out


def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)


def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)


@torch.no_grad()
def log_model_performance(
    wandb_run, step, model, activations_store, sae, index=None, batch_tokens=None
):
    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()[
            : sae.cfg.batch_size // sae.cfg.seq_len
        ]
    batch = activations_store.get_activations(batch_tokens).reshape(
        -1, sae.cfg.act_size
    )

    sae_output = sae(batch)["output"].reshape(
        batch_tokens.shape[0], batch_tokens.shape[1], -1
    )

    original_loss = model(batch_tokens, return_type="loss").item()
    reconstr_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg.hook_point, partial(reconstr_hook, sae_out=sae_output))],
        return_type="loss",
    ).item()
    zero_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg.hook_point, zero_abl_hook)],
        return_type="loss",
    ).item()
    mean_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg.hook_point, mean_abl_hook)],
        return_type="loss",
    ).item()

    ce_degradation = original_loss - reconstr_loss
    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    log_dict = {
        "performance/ce_degradation": ce_degradation,
        "performance/recovery_from_zero": (reconstr_loss - zero_loss) / zero_degradation,
        "performance/recovery_from_mean": (reconstr_loss - mean_loss) / mean_degradation,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


@torch.no_grad()
def log_transcoder_performance(
    wandb_run, step, model, activations_store, transcoder, index=None, batch_tokens=None
):
    """Log model performance metrics for transcoders (input != output hook point)."""
    cfg = transcoder.cfg
    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()[
            : cfg.batch_size // cfg.seq_len
        ]

    # Get input activations and run transcoder
    input_acts, _ = activations_store.get_activations(batch_tokens)
    input_acts_flat = input_acts.reshape(-1, cfg.input_size)

    # Get transcoder output reshaped for hooks
    transcoder_out = transcoder(input_acts_flat, input_acts_flat)["output"].reshape(
        batch_tokens.shape[0], batch_tokens.shape[1], -1
    )

    original_loss = model(batch_tokens, return_type="loss").item()
    reconstr_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(cfg.output_hook_point, partial(reconstr_hook, sae_out=transcoder_out))],
        return_type="loss",
    ).item()
    zero_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(cfg.output_hook_point, zero_abl_hook)],
        return_type="loss",
    ).item()
    mean_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(cfg.output_hook_point, mean_abl_hook)],
        return_type="loss",
    ).item()

    ce_degradation = original_loss - reconstr_loss
    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    log_dict = {
        "performance/ce_degradation": ce_degradation,
        "performance/recovery_from_zero": (reconstr_loss - zero_loss) / zero_degradation,
        "performance/recovery_from_mean": (reconstr_loss - mean_loss) / mean_degradation,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


def save_checkpoint(wandb_run, sae, cfg: EncoderConfig, step):
    save_dir = f"checkpoints/{cfg.name}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    # Convert dataclass to JSON-safe dict
    json_safe_cfg = {}
    for key, value in asdict(cfg).items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    # Create and log artifact
    artifact = wandb.Artifact(
        name=f"{cfg.name}_{step}",
        type="model",
        description=f"Model checkpoint at step {step}",
    )
    artifact.add_file(sae_path)
    artifact.add_file(config_path)
    wandb_run.log_artifact(artifact)

    print(f"Model and config saved as artifact at step {step}")
