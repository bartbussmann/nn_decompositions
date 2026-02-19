# Pareto Comparison: SPD vs Transcoder

## Setup

Both methods decompose GPT-2 small's layer 8 MLP into sparse components. We evaluate faithfulness by sweeping L0 (number of active components per token) and measuring how well each method reconstructs the original model behavior.

- **Transcoder** (BatchTopKTranscoder, 6144 dictionary features, trained k=32): A single encoder-decoder that replaces the full MLP. At evaluation, we keep only the top-k features by activation magnitude.
- **SPD** (6144 components per weight matrix): Decomposes individual weight matrices (c_fc, c_proj) into rank-1 components with learned causal importance (CI) gates. At evaluation, we keep the top-k components by pre-sigmoid CI score. Each weight matrix is evaluated independently while the other retains original weights.

## Metrics

For each L0 value in {1, 2, 3, 5, 10, 20, 50, 100, 200, 500}:

1. **CE loss**: Full model forward pass with the MLP (or weight matrix) replaced by the sparse reconstruction. Measures downstream impact on next-token prediction.
2. **MLP output MSE**: Direct comparison of the reconstructed MLP output against the original. Measures local reconstruction quality independent of the rest of the model.

## Special points

- **Transcoder (train k)**: The transcoder evaluated at its training-time sparsity (k=32).
- **SPD (CI>0.5)**: SPD with its natural binary gating (post-sigmoid CI > 0.5 threshold). The resulting L0 is data-dependent rather than fixed.

## Baselines

- **Original CE**: Unmodified GPT-2 layer 8 MLP.
- **Zero-ablation CE**: MLP output set to zero (worst case).

## Key question

At matched sparsity, which method better preserves the original model's computation?
