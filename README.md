# NN Decompositions

A research repository for neural network decomposition methods including Sparse Autoencoders (SAEs), Transcoders, MOLTs, SPD, and related techniques.

## Overview

This repository provides implementations of various neural network decomposition approaches for mechanistic interpretability research. The goal is to decompose neural network activations and weights into interpretable components.

## Current Implementations

### Sparse Autoencoders (SAEs)

Four SAE variants are implemented in `sae.py`:

- **VanillaSAE**: Standard sparse autoencoder with L1 regularization
- **TopKSAE**: Keeps only the top-k activations per sample
- **BatchTopKSAE**: Keeps top-k activations across the entire batch (more flexible sparsity)
- **JumpReLUSAE**: Uses learnable thresholds with straight-through gradient estimation

All SAE variants share a common `BaseAutoencoder` class with:
- Encoder/decoder weight initialization
- Optional input unit normalization
- Decoder weight normalization
- Dead feature tracking

## Installation

```bash
pip install torch transformer_lens datasets wandb tqdm
```

## Usage

```python
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from sae import TopKSAE
from training import train_sae
from transformer_lens import HookedTransformer

# Configure
cfg = get_default_cfg()
cfg["sae_type"] = "topk"
cfg["model_name"] = "gpt2-small"
cfg["layer"] = 8
cfg["site"] = "resid_pre"
cfg["dict_size"] = 768 * 16
cfg["top_k"] = 32
cfg = post_init_cfg(cfg)

# Initialize
model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["device"])
sae = TopKSAE(cfg)
activation_store = ActivationsStore(model, cfg)

# Train
train_sae(sae, activation_store, model, cfg)
```

## Configuration

Key configuration options (see `config.py` for full list):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"gpt2-small"` | TransformerLens model name |
| `layer` | `8` | Layer to extract activations from |
| `site` | `"resid_pre"` | Hook point (resid_pre, resid_post, mlp_out, etc.) |
| `dict_size` | `12288` | Dictionary/latent dimension |
| `top_k` | `32` | Number of active features (TopK variants) |
| `l1_coeff` | `0` | L1 regularization coefficient |
| `num_tokens` | `1e9` | Total training tokens |
| `batch_size` | `4096` | Training batch size |

## Project Structure

```
nn_decompositions/
├── sae.py              # SAE model implementations
├── activation_store.py # Activation collection from models
├── training.py         # Training loop
├── config.py           # Configuration defaults
├── logs.py             # WandB logging and checkpointing
├── results.py          # Results analysis utilities
└── main.py             # Example training scripts
```

## Logging

Training logs to Weights & Biases with:
- Loss metrics (L0, L1, L2, auxiliary loss)
- Dead feature counts
- Model performance (CE degradation, recovery metrics)
- Periodic checkpoints as artifacts

## References

- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/)
- [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093)
