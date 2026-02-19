# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NN Decompositions is a research repository for neural network decomposition methods, currently focused on Sparse Autoencoders (SAEs) for mechanistic interpretability. The codebase trains SAEs on transformer activations to decompose them into interpretable features.

**Available SAE types** (defined in `sae.py`):

- **VanillaSAE**: Standard sparse autoencoder with L1 regularization
- **TopKSAE**: Keeps only the top-k activations per sample
- **BatchTopKSAE**: Keeps top-k activations across the entire batch (more flexible sparsity)
- **JumpReLUSAE**: Uses learnable thresholds with straight-through gradient estimation

## Development Commands

**Setup:**
```bash
pip install -r requirements.txt
```

**Running:**
```bash
python main.py  # Run training experiments
```

## Architecture Overview

**Core Files:**
- `sae.py` - SAE model implementations (BaseAutoencoder + 4 variants)
- `activation_store.py` - Streams activations from transformer models via TransformerLens
- `training.py` - Training loop with optimizer setup
- `config.py` - Configuration defaults and post-init processing
- `logs.py` - WandB logging and checkpoint saving
- `results.py` - Results analysis utilities
- `main.py` - Example training scripts with hyperparameter sweeps

**Key Data Flow:**

1. Load a pretrained transformer via TransformerLens
2. ActivationsStore collects activations from a specified hook point
3. SAE trains to reconstruct activations with sparse latent representations
4. Metrics logged to WandB: L0/L1 norms, reconstruction loss, dead features

**Configuration:**

Config is a dictionary with keys like:
- `model_name` - TransformerLens model (e.g., "gpt2-small")
- `layer`, `site` - Which activations to extract (e.g., layer 8, "resid_pre")
- `dict_size` - SAE latent dimension (e.g., 768 * 16)
- `top_k` - Active features for TopKTranscoder variants
- `l1_coeff` - Sparsity penalty coefficient
- `sae_type` - Which SAE variant to use

## GitHub

- Use the github cli for issues/PRs (e.g. `gh issue view 1` or `gh pr view 1`)
- Only commit files with relevant changes, don't commit all files
- Use branch names `refactor/X` or `feature/Y` or `fix/Z`
- **Always commit after making code edits** - don't let changes accumulate
- Often after committing you want to push the changes to the remote branch (not main). You can do this with `git push origin <branch_name>`

## Coding Guidelines

**This is research code, not production. Prioritize simplicity and fail-fast over defensive programming.**

Core principles:
- **Fail fast** - assert assumptions, crash on violations, don't silently recover
- **No backwards compat** - delete unused code, don't deprecate or add migration shims
- **Narrow types** - avoid `| None` unless null is semantically meaningful
- **No try/except for control flow** - check preconditions explicitly, then trust them
- **YAGNI** - don't add abstractions, config options, or flexibility for hypothetical futures

```python
# BAD - defensive, recovers silently, wide types
def get_config(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

# GOOD - fail fast, narrow types, trust preconditions
def get_config(path: Path) -> Config:
    assert path.exists(), f"config not found: {path}"
    with open(path) as f:
        data = json.load(f)
    return Config(**data)
```

## Software Engineering Principles

- Assert your invariants. If you're afraid to assert, your program might already be broken
- Never write: `if everythingIsOk: continueHappyPath()`. Instead do `assert everythingIsOk`
- Write invariants into types as much as possible
- Don't use bare dictionaries for structures with heterogeneous values
- Keep I/O as high up as possible, make functions pure where possible
- Delete unused code. If an argument is always the same value, inline it
- Don't write try/catch blocks unless absolutely necessary
- Comments hide sloppy code - prefer clear naming over comments
