# NN Decompositions

A research repository for comparing neural network decomposition methods for mechanistic interpretability. Implements Sparse Autoencoders (SAEs), Transcoders, and Cross-Layer Transcoders (CLTs), and benchmarks them against Sparse Parameter Decomposition (SPD).

## Overview

The goal is to decompose transformer MLP computations into sparse, interpretable components and compare methods on reconstruction quality (MSE) vs. sparsity (L0) and downstream task performance (cross-entropy).

The primary target model is a 4-layer **LlamaSimpleMLP** transformer (`n_embd=768`, GELU MLPs with `n_intermediate=3072`), trained on the Pile dataset.

## Installation

**On RunPod** (recommended — handles Python 3.13, CUDA, and SPD):
```bash
bash setup_env.sh
source .venv/bin/activate
```

**Manual:**
```bash
pip install -e ".[dev,analysis]"
pip install -e /path/to/spd  # sibling SPD repo
```

## Model Implementations

### Transcoders (`transcoder.py`)

Map MLP inputs to MLP outputs through a sparse bottleneck (input ≠ output):

- **VanillaTranscoder** — L1 regularization on latent activations
- **TopKTranscoder** — keeps top-k activations per sample
- **BatchTopKTranscoder** — keeps top-k activations across the entire batch
- **JumpReLUTranscoder** — learnable per-feature thresholds with straight-through gradients

### Cross-Layer Transcoder (`clt.py`)

Per-layer encoders with **triangular decoders**: a feature activated at source layer `i` writes to MLP outputs at layers `i, i+1, ..., n-1`. This captures cross-layer structure that per-layer transcoders cannot.

### SAEs (`sae.py`)

Standard sparse autoencoders that reconstruct input activations (input = output). Same four variants: Vanilla, TopK, BatchTopK, JumpReLU.

## Configuration

Configs are dataclasses defined in `config.py`:

- **EncoderConfig** — base config for transcoders (input_size, output_size, dict_size, encoder_type, top_k, training params)
- **SAEConfig** — extends EncoderConfig with `input_size == output_size` constraint
- **CLTConfig** — adds `layers` (list of layer indices) for cross-layer structure

## Experiments

Each experiment lives in `experiments/exp_XXX_<name>/` with outputs in an `output/` subdirectory.

### Training
| ID | Description |
|----|-------------|
| 001 | Train BatchTopKTranscoders on all 4 layers (per-layer top_k matching SPD L0s) |
| 002 | Train transcoders on SimpleStories dataset |
| 003 | Train Cross-Layer Transcoder on all 4 layers |
| 010 | Uniform top_k sweep [8, 16, 32, 64] across all layers |

### Evaluation & Comparison
| ID | Description |
|----|-------------|
| 005 | Pareto comparison on GPT-2 |
| 006 | Single-layer Pareto comparison on Pile |
| 007 | All-layers Pareto: replace all 4 MLPs simultaneously, L0 sweep, 3 x-axis variants |
| 011 | Pareto from naturally-trained checkpoints (transcoders, CLTs, SPD thresholds, neuron baseline) |
| 012 | Comprehensive model comparison table (dict size, alive features, dead %, L0, CE, MSE, params) |
| 013 | Steering evaluation: LLM-judged concept/fluency scores across steering factors |
| 014 | Topic ablation: zero AUROC-selected features, measure selectivity of topic suppression |

### Analysis
| ID | Description |
|----|-------------|
| 008 | Automated interpretability, intruder detection, faithfulness |
| 009 | Diagnose activation tail behavior |

### Pareto plots (exp_007, exp_011)

Compare methods on three x-axis definitions to account for structural differences:
1. **Active components per module** — raw average L0
2. **Active components per MLP reconstruction** — accounts for CLT's cross-layer writes and SPD's dual modules
3. **Total active parameters** — actual parameter count using per-layer L0s

## Project Structure

```
nn_decompositions/
├── sae.py               # SAE implementations (4 variants)
├── transcoder.py         # Transcoder implementations (4 variants)
├── clt.py                # Cross-Layer Transcoder
├── config.py             # EncoderConfig, SAEConfig, CLTConfig
├── activation_store.py   # ActivationsStore + DataConfig
├── training.py           # Training loop (train_encoder)
├── logs.py               # WandB logging and checkpointing
├── main.py               # Example training scripts
├── setup_env.sh          # RunPod environment setup
├── analysis/             # Activation collection and dashboards
│   ├── collect_activations.py
│   ├── collect_spd_activations.py
│   └── feature_dashboard.py
└── experiments/
    ├── exp_001_train_transcoder_pile/
    ├── exp_002_train_transcoder_ss/
    ├── exp_003_train_clt_pile/
    ├── ...
    ├── exp_011_pareto_trained_all_layers/
    ├── exp_013_steering_eval/
    └── exp_014_ablation_eval/
```

## References

- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/)
- [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093)
