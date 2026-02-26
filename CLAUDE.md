# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NN Decompositions is a research repository for neural network decomposition methods for mechanistic interpretability. It implements SAEs, Transcoders, and Cross-Layer Transcoders (CLTs) to decompose transformer MLP activations into sparse, interpretable features. Experiments compare these methods against Sparse Parameter Decomposition (SPD) from the sibling `/workspace/spd` repo.

The primary target model is **LlamaSimpleMLP** (4-layer transformer, `n_embd=768`, `n_intermediate=3072`, GELU MLPs) loaded from `wandb:goodfire/spd/t-32d1bb3b`.

## Model Types

**SAEs** (`sae.py`) — reconstruct input activations (input = output):
- VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE

**Transcoders** (`transcoder.py`) — map MLP input to MLP output (input ≠ output):
- VanillaTranscoder, TopKTranscoder, BatchTopKTranscoder, JumpReLUTranscoder

**Cross-Layer Transcoder** (`clt.py`) — per-layer encoders with triangular decoders. Features at source layer `i` write to MLP outputs at layers `i` through `n-1`.

## Core Files

- `sae.py` — SAE implementations (BaseAutoencoder + 4 variants)
- `transcoder.py` — Transcoder implementations (BaseTranscoder + 4 variants)
- `clt.py` — CrossLayerTranscoder with per-layer encoders and triangular decoders
- `config.py` — EncoderConfig, SAEConfig, CLTConfig dataclasses
- `activation_store.py` — ActivationsStore + DataConfig for streaming activations
- `training.py` — Training loop (`train_encoder`)
- `logs.py` — WandB logging and checkpoint saving
- `main.py` — Example training scripts

## Experiments

Each experiment lives in `experiments/exp_XXX_<name>/` with outputs in `output/`.

| ID | Name | Description |
|----|------|-------------|
| 001 | train_transcoder_pile | Train BatchTopKTranscoders on LlamaSimpleMLP layers 0-3 (Pile dataset, per-layer top_k matching SPD) |
| 002 | train_transcoder_ss | Train transcoders on SimpleStories |
| 003 | train_clt_pile | Train CLT on LlamaSimpleMLP layers 0-3 (Pile dataset) |
| 004 | train_spd_gpt2 | Train SPD on GPT-2 |
| 005 | pareto_gpt2 | Pareto comparison on GPT-2 |
| 006 | pareto_pile | Single-layer Pareto comparison on Pile |
| 007 | pareto_pile_all_layers | All-layers Pareto: replace all 4 MLPs simultaneously, sweep L0, 3 x-axis variants |
| 008 | interp_pile | Autointerp, intruder detection, faithfulness analysis |
| 009 | diagnose_tail | Diagnose activation tail behavior |
| 010 | train_transcoder_pile_sweep | Uniform top_k sweep [8,16,32,64] across all layers (mp.Process batches) |
| 011 | pareto_trained_all_layers | Pareto from naturally-trained checkpoints only (transcoders from wandb, CLTs, SPD thresholds, neuron baseline) |

## Key Patterns

**All-layers patching**: Use `ExitStack` + factory functions to patch all 4 MLPs simultaneously, avoiding closure variable capture bugs:
```python
with ExitStack() as stack:
    for layer_idx in LAYERS:
        def _make_patched(tc_, size_):
            def _patched(x): ...
            return _patched
        stack.enter_context(patched_forward(mlp, _make_patched(tc, size)))
```

**Parallel training**: Use `mp.Process` in batches (not ProcessPoolExecutor) to avoid cascade failures when a worker OOMs.

**WandB artifacts**: Checkpoints contain `encoder.pt` + `config.json`. Override `cfg_dict["device"]` when loading since saved configs may reference training GPUs (e.g. `cuda:3`) that don't exist locally.

## Development Commands

**Setup** (creates Python 3.13 venv on local disk for RunPod):
```bash
bash setup_env.sh
source .venv/bin/activate
```

**Running experiments:**
```bash
python experiments/exp_011_pareto_trained_all_layers/pareto_trained_all_layers.py
```

## Dependencies

- Requires Python 3.13 (for SPD compatibility)
- PyTorch with CUDA 12.4
- SPD repo at `/workspace/spd` (installed editable)
- Key packages: torch, wandb, datasets, transformer-lens, einops, matplotlib

## GitHub

- Use the github cli for issues/PRs (e.g. `gh issue view 1` or `gh pr view 1`)
- Only commit files with relevant changes, don't commit all files
- Use branch names `refactor/X` or `feature/Y` or `fix/Z`
- **Always commit after making code edits** - don't let changes accumulate

## Coding Guidelines

**This is research code, not production. Prioritize simplicity and fail-fast over defensive programming.**

Core principles:
- **Fail fast** - assert assumptions, crash on violations, don't silently recover
- **No backwards compat** - delete unused code, don't deprecate or add migration shims
- **Narrow types** - avoid `| None` unless null is semantically meaningful
- **No try/except for control flow** - check preconditions explicitly, then trust them
- **YAGNI** - don't add abstractions, config options, or flexibility for hypothetical futures
- Assert your invariants. If you're afraid to assert, your program might already be broken
- Write invariants into types as much as possible
- Don't use bare dictionaries for structures with heterogeneous values
- Keep I/O as high up as possible, make functions pure where possible
- Delete unused code. If an argument is always the same value, inline it
- Comments hide sloppy code - prefer clear naming over comments
