# NN Decompositions

A research repository for neural network decomposition methods including Sparse Autoencoders (SAEs), Transcoders, and related techniques.

## Overview

This repository provides implementations of various neural network decomposition approaches for mechanistic interpretability research. The goal is to decompose neural network activations and weights into interpretable components.

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Current Implementations

### Encoder Architectures (`base.py`)

Four encoder variants with unified `forward(x_in, y_target)` signature:

- **Vanilla**: Standard encoder with L1 regularization
- **TopK**: Keeps only the top-k activations per sample
- **BatchTopK**: Keeps top-k activations across the entire batch
- **JumpReLUEncoder**: Uses learnable thresholds with straight-through gradients

### SAEs (`sae.py`)

SAE wrappers that call `forward(x, x)` to reconstruct input:

- `VanillaSAE`, `TopKSAE`, `BatchTopKSAE`, `JumpReLUSAE`

### Transcoders

Use base classes directly for transcoders (input ≠ output):

```python
from transcoder import TopK
from config import TranscoderConfig

cfg = TranscoderConfig(
    input_size=768,
    output_size=768,
    dict_size=6144,
    input_site="resid_mid",
    output_site="mlp_out",
    ...
)
transcoder = TopK(cfg)
output = transcoder(x_in, y_target)
```

## Usage

### Training an SAE

```python
from activation_store import ActivationsStore
from config import SAEConfig
from sae import TopKSAE
from training import train_sae
from transformer_lens import HookedTransformer

cfg = SAEConfig(
    encoder_type="topk",
    act_size=768,
    dict_size=768 * 16,
    model_name="gpt2-small",
    layer=8,
    site="resid_pre",
    top_k=32,
)

model = HookedTransformer.from_pretrained(cfg.model_name).to(cfg.device)
sae = TopKSAE(cfg)
activation_store = ActivationsStore(model, cfg)

train_sae(sae, activation_store, model, cfg)
```

### Training a Transcoder

```python
from activation_store import TranscoderActivationsStore
from transcoder import TopK
from config import TranscoderConfig
from training import train_transcoder
from transformer_lens import HookedTransformer

cfg = TranscoderConfig(
    encoder_type="topk",
    input_size=768,
    output_size=768,
    dict_size=6144,
    input_site="resid_mid",
    output_site="mlp_out",
    input_layer=8,
    output_layer=8,
    top_k=32,
)

model = HookedTransformer.from_pretrained(cfg.model_name).to(cfg.device)
transcoder = TopK(cfg)
activation_store = TranscoderActivationsStore(model, cfg)

train_transcoder(transcoder, activation_store, model, cfg)
```

## Configuration

Configs are dataclasses with type safety and computed properties:

### SAEConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder_type` | `"topk"` | One of: vanilla, topk, batchtopk, jumprelu |
| `act_size` | `768` | Activation dimension (input = output for SAE) |
| `dict_size` | `12288` | Dictionary/latent dimension |
| `top_k` | `32` | Number of active features (TopK variants) |
| `l1_coeff` | `0.0` | L1/sparsity regularization coefficient |
| `site` | `"resid_pre"` | Hook point for activations |
| `layer` | `8` | Layer to extract from |

### TranscoderConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | — | Input activation dimension |
| `output_size` | — | Output activation dimension |
| `input_site` | `"resid_pre"` | Input hook point |
| `output_site` | `"resid_post"` | Output hook point |
| `input_layer` | `8` | Input layer |
| `output_layer` | `8` | Output layer |

## Project Structure

```
nn_decompositions/
├── base.py              # Unified encoder architectures
├── sae.py               # SAE wrappers
├── config.py            # Dataclass configs (SAEConfig, TranscoderConfig)
├── activation_store.py  # Activation collection from models
├── training.py          # Training loops
├── logs.py              # WandB logging and checkpointing
└── main.py              # Example training script
```

## References

- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/)
- [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093)
