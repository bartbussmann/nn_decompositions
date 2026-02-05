"""Train a transcoder on the SimpleStories 4-layer LlamaSimple model (layer 3 MLP).

This script trains a TopK transcoder to decompose the MLP at layer 3 of the
canonical SimpleStories model used in SPD experiments.

Model: wandb:goodfire/spd/runs/erq48r3w (LlamaSimple 4-layer, 1.25M params)
Architecture: d_model=128, d_mlp=341, 4 layers, SwiGLU MLP
Target: Layer 3 MLP (input: resid_mid, output: mlp_out)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from simple_stories_train.models.llama_simple import LlamaSimple
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from activation_store import GenericActivationsStore, GenericDataConfig
from base import BatchTopK, TopK
from logs import get_encoder_metrics, init_wandb, save_checkpoint
from training import train_encoder


@dataclass
class SSTranscoderConfig:
    """Config for SimpleStories transcoder training."""

    # Model
    model_path: str = "wandb:goodfire/spd/runs/erq48r3w"
    layer: int = 3

    # Architecture (auto-detected from model)
    input_size: int = field(init=False)
    output_size: int = field(init=False)
    dict_size: int = 2048
    encoder_type: str = "topk"

    # Training
    seed: int = 42
    batch_size: int = 128
    lr: float = 3e-4
    num_tokens: int = int(5e7)  # 50M tokens
    beta1: float = 0.9
    beta2: float = 0.99
    max_grad_norm: float = 1.0

    # TopK specific
    top_k: int = 32
    top_k_aux: int = 512
    aux_penalty: float = 1 / 32
    l1_coeff: float = 0.0
    n_batches_to_dead: int = 50

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = field(default=torch.float32)

    # Data
    dataset_name: str = "lennart-finke/SimpleStories"
    tokenizer_name: str = "SimpleStories/test-SimpleStories-gpt2-1.25M"
    seq_len: int = 512
    text_column: str = "story"

    # Logging
    wandb_project: str = "ss_transcoder"
    perf_log_freq: int = 500
    checkpoint_freq: int | str = "final"
    n_eval_seqs: int = 8

    # Optional features (for compatibility with base classes)
    input_unit_norm: bool = False
    pre_enc_bias: bool = False
    bandwidth: float = 0.001

    def __post_init__(self):
        # Load model config to get dimensions
        model = LlamaSimple.from_pretrained(self.model_path)
        self.input_size = model.config.n_embd  # 128
        self.output_size = model.config.n_embd  # 128
        del model

    @property
    def name(self) -> str:
        return f"ss_transcoder_layer{self.layer}_{self.dict_size}_{self.encoder_type}_k{self.top_k}"


def create_activation_store(model: LlamaSimple, cfg: SSTranscoderConfig) -> GenericActivationsStore:
    """Create activation store for the SimpleStories model."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    data_config = GenericDataConfig(
        dataset_name=cfg.dataset_name,
        tokenizer=tokenizer,
        text_column=cfg.text_column,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        device=cfg.device,
        seed=cfg.seed,
        lowercase=True,  # SimpleStories uses lowercase
    )

    # Hook after RMSNorm (MLP input) and after MLP (MLP output)
    input_module = model.h[cfg.layer].rms_2
    output_module = model.h[cfg.layer].mlp

    return GenericActivationsStore(
        model=model,
        input_module=input_module,
        output_module=output_module,
        data_config=data_config,
        input_size=cfg.input_size,
        output_size=cfg.output_size,
    )


def main():
    """Train a transcoder on the SimpleStories model."""
    cfg = SSTranscoderConfig(
        layer=3,
        dict_size=2048,
        top_k=32,
        encoder_type="topk",
        num_tokens=int(5e7),  # 50M tokens
        batch_size=64,
        lr=1e-3,
    )

    print(f"Loading model from {cfg.model_path}...")
    model = LlamaSimple.from_pretrained(cfg.model_path)
    model.to(cfg.dtype).to(cfg.device)
    model.eval()

    print(f"Model config: d_model={model.config.n_embd}, d_mlp={model.config.n_intermediate}")
    print(f"Training transcoder on layer {cfg.layer} MLP")
    print(f"  Input/Output size: {cfg.input_size}")
    print(f"  Dict size: {cfg.dict_size}")
    print(f"  Top-k: {cfg.top_k}")

    # Create activation store using GenericActivationsStore
    activation_store = create_activation_store(model, cfg)

    # Create transcoder
    if cfg.encoder_type == "topk":
        transcoder = TopK(cfg)
    elif cfg.encoder_type == "batchtopk":
        transcoder = BatchTopK(cfg)
    else:
        raise ValueError(f"Unknown encoder type: {cfg.encoder_type}")

    # Train using the standard training loop (model=None skips perf logging)
    train_encoder(transcoder, activation_store, cfg, model=None)


if __name__ == "__main__":
    main()
