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
import torch.nn as nn
import tqdm
from datasets import load_dataset
from simple_stories_train.models.llama_simple import LlamaSimple
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BatchTopK, TopK
from logs import get_encoder_metrics, init_wandb, save_checkpoint


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


class SSActivationStore:
    """Activation store for SimpleStories model using PyTorch hooks."""

    def __init__(self, model: nn.Module, cfg: SSTranscoderConfig):
        self.model = model
        self.cfg = cfg
        self.layer = cfg.layer

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup streaming dataset (tokenize on-the-fly)
        print("Setting up streaming dataset...")
        self.dataset = load_dataset(cfg.dataset_name, split="train", streaming=True)
        self.dataset = self.dataset.shuffle(seed=cfg.seed, buffer_size=10000)
        self.data_iter = iter(self.dataset)

        # Activation storage
        self._mlp_input: torch.Tensor | None = None
        self._mlp_output: torch.Tensor | None = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture MLP input/output."""

        def input_hook(module, input, output):
            # RMSNorm output = MLP input
            self._mlp_input = output.detach()

        def output_hook(module, input, output):
            self._mlp_output = output.detach()

        # Hook after RMSNorm (before MLP) and after MLP
        self.model.h[self.layer].rms_2.register_forward_hook(input_hook)
        self.model.h[self.layer].mlp.register_forward_hook(output_hook)

    def get_batch_tokens(self) -> torch.Tensor:
        """Get a batch of tokens by tokenizing text on-the-fly."""
        texts = []
        for _ in range(self.cfg.batch_size):
            try:
                sample = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataset)
                sample = next(self.data_iter)
            texts.append(sample[self.cfg.text_column].lower())

        tokens = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.cfg.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return tokens["input_ids"].to(self.cfg.device)

    @torch.no_grad()
    def get_activations(self, batch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass and return (mlp_input, mlp_output) activations."""
        self.model(batch_tokens)
        assert self._mlp_input is not None and self._mlp_output is not None
        return self._mlp_input, self._mlp_output

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of (input, target) activations, flattened."""
        batch_tokens = self.get_batch_tokens()
        mlp_in, mlp_out = self.get_activations(batch_tokens)
        # Flatten: (batch, seq, d_model) -> (batch * seq, d_model)
        return mlp_in.reshape(-1, self.cfg.input_size), mlp_out.reshape(-1, self.cfg.output_size)


def train_transcoder(cfg: SSTranscoderConfig):
    """Train a transcoder on the SimpleStories model."""
    print(f"Loading model from {cfg.model_path}...")
    model = LlamaSimple.from_pretrained(cfg.model_path)
    model.to(cfg.dtype).to(cfg.device)
    model.eval()

    print(f"Model config: d_model={model.config.n_embd}, d_mlp={model.config.n_intermediate}")
    print(f"Training transcoder on layer {cfg.layer} MLP")
    print(f"  Input/Output size: {cfg.input_size}")
    print(f"  Dict size: {cfg.dict_size}")
    print(f"  Top-k: {cfg.top_k}")

    # Create activation store
    activation_store = SSActivationStore(model, cfg)

    # Create transcoder
    if cfg.encoder_type == "topk":
        transcoder = TopK(cfg)
    elif cfg.encoder_type == "batchtopk":
        transcoder = BatchTopK(cfg)
    else:
        raise ValueError(f"Unknown encoder type: {cfg.encoder_type}")

    # Training setup
    num_batches = cfg.num_tokens // (cfg.batch_size * cfg.seq_len)
    optimizer = torch.optim.Adam(transcoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)

    for i in pbar:
        x_in, y_target = activation_store.next_batch()
        output = transcoder(x_in, y_target)

        # Log metrics
        log_dict = get_encoder_metrics(output)
        wandb_run.log(log_dict, step=i)

        # Checkpoint
        if cfg.checkpoint_freq != "final" and i % cfg.checkpoint_freq == 0 and i > 0:
            save_checkpoint(transcoder, cfg, i)

        # Training step
        loss = output["loss"]
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "L0": f"{output['l0_norm']:.2f}",
            "L2": f"{output['l2_loss']:.4f}",
        })

        loss.backward()
        torch.nn.utils.clip_grad_norm_(transcoder.parameters(), cfg.max_grad_norm)
        transcoder.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

        del output, x_in, y_target

    save_checkpoint(transcoder, cfg, "final")
    wandb_run.finish()
    print("Training complete!")


if __name__ == "__main__":
    cfg = SSTranscoderConfig(
        layer=3,
        dict_size=2048,
        top_k=32,
        encoder_type="topk",
        num_tokens=int(5e7),  # 50M tokens
        batch_size=64,
        lr=1e-3,
    )
    train_transcoder(cfg)
