from dataclasses import dataclass, field
from typing import Literal

import torch
import transformer_lens.utils as utils


@dataclass
class EncoderConfig:
    """Base config for encoder architectures (SAE and Transcoder)."""

    # Architecture (required)
    input_size: int
    output_size: int
    dict_size: int

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

    # Evaluation
    n_eval_seqs: int = 8  # Number of sequences for performance evaluation


@dataclass
class SAEConfig(EncoderConfig):
    """Config for Sparse Autoencoders.

    SAE reconstructs input, so input_size = output_size = act_size.
    """

    # SAE uses same size for input/output (set in __post_init__)
    input_size: int = field(init=False)
    output_size: int = field(init=False)

    # Encoder type
    encoder_type: Literal["vanilla", "topk", "batchtopk", "jumprelu"] = "topk"

    # Activation size (the main size parameter for SAE)
    act_size: int = 768
    dict_size: int = 12288

    # Hook point specification
    model_name: str = "gpt2-small"
    site: str = "resid_pre"
    layer: int = 8

    # Data
    seq_len: int = 128
    model_batch_size: int = 512
    num_batches_in_buffer: int = 5
    dataset_path: str = "Skylion007/openwebtext"

    # Logging
    wandb_project: str = "sparse_autoencoders"
    perf_log_freq: int = 1000
    checkpoint_freq: int = 10000

    def __post_init__(self):
        # SAE: input_size = output_size = act_size
        self.input_size = self.act_size
        self.output_size = self.act_size

    @property
    def hook_point(self) -> str:
        return utils.get_act_name(self.site, self.layer)

    @property
    def eval_hook_point(self) -> str:
        """Hook point used for performance evaluation (same as hook_point for SAE)."""
        return self.hook_point

    @property
    def name(self) -> str:
        base = f"{self.model_name}_{self.hook_point}_{self.dict_size}_{self.encoder_type}"
        # Only include k for topk variants
        if self.encoder_type in ("topk", "batchtopk"):
            base += f"_k{self.top_k}"
        return f"{base}_{self.lr}"


@dataclass
class TranscoderConfig(EncoderConfig):
    """Config for Transcoders (input != output)."""

    # Architecture (required, no defaults)
    input_size: int
    output_size: int
    dict_size: int

    # Encoder type
    encoder_type: Literal["vanilla", "topk", "batchtopk", "jumprelu"] = "topk"

    # Hook points
    model_name: str = "gpt2-small"
    input_site: str = "resid_pre"
    output_site: str = "resid_post"
    input_layer: int = 8
    output_layer: int = 8

    # Data
    seq_len: int = 128
    model_batch_size: int = 512
    num_batches_in_buffer: int = 10
    dataset_path: str = "Skylion007/openwebtext"

    # Logging
    wandb_project: str = "transcoders"
    perf_log_freq: int = 1000
    checkpoint_freq: int = 10000

    @property
    def input_hook_point(self) -> str:
        return utils.get_act_name(self.input_site, self.input_layer)

    @property
    def output_hook_point(self) -> str:
        return utils.get_act_name(self.output_site, self.output_layer)

    @property
    def eval_hook_point(self) -> str:
        """Hook point used for performance evaluation (output hook for transcoder)."""
        return self.output_hook_point

    @property
    def name(self) -> str:
        base = f"transcoder_{self.model_name}_{self.input_hook_point}_to_{self.output_hook_point}_{self.dict_size}_{self.encoder_type}"
        if self.encoder_type in ("topk", "batchtopk"):
            base += f"_k{self.top_k}"
        return base
