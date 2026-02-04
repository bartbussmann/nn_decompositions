from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import torch
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer


@lru_cache(maxsize=16)
def get_model_config(model_name: str):
    """Get model config without loading weights."""
    return HookedTransformer.from_pretrained(model_name, device="meta").cfg


def get_site_dim(model_name: str, site: str) -> int:
    """Get the dimension of a hook site for a given model."""
    cfg = get_model_config(model_name)
    if site in ("resid_pre", "resid_post", "resid_mid", "mlp_out", "attn_out"):
        return cfg.d_model
    elif site == "mlp_in":
        return cfg.d_mlp
    else:
        raise ValueError(f"Unknown site: {site}. Add it to get_site_dim().")


@dataclass
class EncoderConfig:
    """Base config for encoder architectures (SAE and Transcoder)."""

    # Architecture - set automatically from model in subclasses
    input_size: int = field(init=False)
    output_size: int = field(init=False)
    dict_size: int = 12288

    # Encoder type
    encoder_type: Literal["vanilla", "topk", "batchtopk", "jumprelu"] = "topk"

    # Model
    model_name: str = "gpt2-small"

    # Training
    seed: int = 49
    batch_size: int = 4096
    lr: float = 3e-4
    num_tokens: int = int(1e9)
    l1_coeff: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.99
    max_grad_norm: float = 1.0

    # Device
    device: str = "cuda:0"
    dtype: torch.dtype = field(default=torch.float32)

    # Data
    seq_len: int = 128
    model_batch_size: int = 512
    num_batches_in_buffer: int = 10
    dataset_path: str = "Skylion007/openwebtext"

    # Dead feature tracking
    n_batches_to_dead: int = 50

    # Optional features
    input_unit_norm: bool = False
    pre_enc_bias: bool = False

    # TopK specific
    top_k: int = 32
    top_k_aux: int = 512
    aux_penalty: float = 1 / 32

    # JumpReLU specific
    bandwidth: float = 0.001

    # Logging
    wandb_project: str = "encoders"
    perf_log_freq: int = 1000
    checkpoint_freq: int = 10000
    n_eval_seqs: int = 8


@dataclass
class SAEConfig(EncoderConfig):
    """Config for Sparse Autoencoders."""

    # Hook point (SAE-specific: single site/layer)
    site: str = "resid_pre"
    layer: int = 8

    # Override default
    wandb_project: str = "sparse_autoencoders"

    def __post_init__(self):
        act_size = get_site_dim(self.model_name, self.site)
        self.input_size = act_size
        self.output_size = act_size

    @property
    def act_size(self) -> int:
        return self.input_size

    @property
    def hook_point(self) -> str:
        return utils.get_act_name(self.site, self.layer)

    @property
    def eval_hook_point(self) -> str:
        return self.hook_point

    @property
    def name(self) -> str:
        base = f"{self.model_name}_{self.hook_point}_{self.dict_size}_{self.encoder_type}"
        if self.encoder_type in ("topk", "batchtopk"):
            base += f"_k{self.top_k}"
        return f"{base}_{self.lr}"


@dataclass
class TranscoderConfig(EncoderConfig):
    """Config for Transcoders (input site != output site)."""

    # Hook points (Transcoder-specific: separate input/output sites)
    input_site: str = "resid_mid"
    output_site: str = "mlp_out"
    input_layer: int = 8
    output_layer: int = 8

    # Override default
    wandb_project: str = "transcoders"

    def __post_init__(self):
        self.input_size = get_site_dim(self.model_name, self.input_site)
        self.output_size = get_site_dim(self.model_name, self.output_site)

    @property
    def input_hook_point(self) -> str:
        return utils.get_act_name(self.input_site, self.input_layer)

    @property
    def output_hook_point(self) -> str:
        return utils.get_act_name(self.output_site, self.output_layer)

    @property
    def eval_hook_point(self) -> str:
        return self.output_hook_point

    @property
    def name(self) -> str:
        base = f"transcoder_{self.model_name}_{self.input_hook_point}_to_{self.output_hook_point}_{self.dict_size}_{self.encoder_type}"
        if self.encoder_type in ("topk", "batchtopk"):
            base += f"_k{self.top_k}"
        return base
