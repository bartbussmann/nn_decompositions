# Jose Target Model Training Runs

All runs target **jose's LlamaSimpleMLP** (`goodfire/spd/runs/t-9d2b8f02`), trained on the Pile dataset (500M tokens), using BatchTopK activation function.

## Local MSE Training

Training mode: per-layer MSE reconstruction loss (no end-to-end KL).

### 4k dict (`pile_local_sweep_jose`)

| Run | Type | Dict Size | Top-k | Status | Artifacts |
|-----|------|-----------|-------|--------|-----------|
| tc_k8 | Transcoder | 4096 | 8 | finished | 4 (per layer) |
| tc_k16 | Transcoder | 4096 | 16 | finished | 4 (per layer) |
| tc_k32 | Transcoder | 4096 | 32 | finished | 4 (per layer) |
| tc_k64 | Transcoder | 4096 | 64 | finished | 4 (per layer) |
| clt_k8 | CLT | 4096 | 8 | finished | 1 |
| clt_k16 | CLT | 4096 | 16 | finished | 1 |
| clt_k32 | CLT | 4096 | 32 | finished | 1 |
| clt_k64 | CLT | 4096 | 64 | finished | 1 |

### 32k dict (`pile_local_sweep_jose_32k`)

| Run | Type | Dict Size | Top-k | Status | Artifacts |
|-----|------|-----------|-------|--------|-----------|
| tc_k8 | Transcoder | 32768 | 8 | finished | 4 (per layer) |
| tc_k16 | Transcoder | 32768 | 16 | finished | 4 (per layer) |
| tc_k32 | Transcoder | 32768 | 32 | finished | 4 (per layer) |
| tc_k64 | Transcoder | 32768 | 64 | **running** | 0 (missing) |
| clt_k8 | CLT | 32768 | 8 | finished | 1 |
| clt_k16 | CLT | 32768 | 16 | finished | 1 |
| clt_k32 | CLT | 32768 | 32 | finished | 1 |
| clt_k64 | CLT | 32768 | 64 | finished | 1 |

## End-to-End KL Training

Training mode: end-to-end KL divergence on logits. Three TC modes (cascading, parallel, independent) and two CLT modes (cascading, parallel).

- **cascading**: each layer's reconstruction feeds into the next layer's encoder
- **parallel**: all layers encode from clean activations, all MLPs patched simultaneously
- **independent**: each layer gets its own KL loss (only its MLP replaced), sequential backward

### 4k dict (`pile_e2e_sweep_jose`)

| Run | Type | Mode | Dict Size | Top-k | Status | Artifacts |
|-----|------|------|-----------|-------|--------|-----------|
| tc_cascading_k8 | Transcoder | cascading | 4096 | 8 | finished | 4 |
| tc_cascading_k16 | Transcoder | cascading | 4096 | 16 | finished | 4 |
| tc_cascading_k32 | Transcoder | cascading | 4096 | 32 | finished | 4 |
| tc_cascading_k64 | Transcoder | cascading | 4096 | 64 | finished | 4 |
| tc_parallel_k8 | Transcoder | parallel | 4096 | 8 | finished | 4 |
| tc_parallel_k16 | Transcoder | parallel | 4096 | 16 | finished | 4 |
| tc_parallel_k32 | Transcoder | parallel | 4096 | 32 | finished | 4 |
| tc_parallel_k64 | Transcoder | parallel | 4096 | 64 | finished | 4 |
| tc_independent_k8 | Transcoder | independent | 4096 | 8 | finished | 4 |
| tc_independent_k16 | Transcoder | independent | 4096 | 16 | finished | 4 |
| tc_independent_k32 | Transcoder | independent | 4096 | 32 | finished | 4 |
| tc_independent_k64 | Transcoder | independent | 4096 | 64 | finished | 4 |
| clt_cascading_k8 | CLT | cascading | 4096 | 8 | finished | 1 |
| clt_cascading_k16 | CLT | cascading | 4096 | 16 | finished | 1 |
| clt_cascading_k32 | CLT | cascading | 4096 | 32 | finished | 1 |
| clt_cascading_k64 | CLT | cascading | 4096 | 64 | finished | 1 |
| clt_parallel_k8 | CLT | parallel | 4096 | 8 | finished | 1 |
| clt_parallel_k16 | CLT | parallel | 4096 | 16 | finished | 1 |
| clt_parallel_k32 | CLT | parallel | 4096 | 32 | finished | 1 |
| clt_parallel_k64 | CLT | parallel | 4096 | 64 | finished | 1 |

### 32k dict (`pile_e2e_sweep_jose_32k`)

| Run | Type | Mode | Dict Size | Top-k | Status | Artifacts |
|-----|------|------|-----------|-------|--------|-----------|
| tc_cascading_k8 | Transcoder | cascading | 32768 | 8 | finished | 4 |
| tc_cascading_k16 | Transcoder | cascading | 32768 | 16 | finished | 4 |
| tc_cascading_k32 | Transcoder | cascading | 32768 | 32 | **running** | 0 |
| tc_cascading_k64 | Transcoder | cascading | 32768 | 64 | **running** | 0 |
| tc_parallel_k8 | Transcoder | parallel | 32768 | 8 | finished | 4 |
| tc_parallel_k16 | Transcoder | parallel | 32768 | 16 | finished | 4 |
| tc_parallel_k32 | Transcoder | parallel | 32768 | 32 | finished | 4 |
| tc_parallel_k64 | Transcoder | parallel | 32768 | 64 | **running** | 0 |
| tc_independent_k8 | Transcoder | independent | 32768 | 8 | finished | 4 |
| tc_independent_k16 | Transcoder | independent | 32768 | 16 | finished | 4 |
| tc_independent_k32 | Transcoder | independent | 32768 | 32 | finished | 4 |
| tc_independent_k64 | Transcoder | independent | 32768 | 64 | **running** | 0 |
| clt_cascading_k8 | CLT | cascading | 32768 | 8 | finished | 1 |
| clt_cascading_k16 | CLT | cascading | 32768 | 16 | **running** | 0 |
| clt_cascading_k32 | CLT | cascading | 32768 | 32 | **running** | 0 |
| clt_parallel_k8 | CLT | parallel | 32768 | 8 | finished | 1 |
| clt_parallel_k16 | CLT | parallel | 32768 | 16 | finished | 1 |
| clt_parallel_k32 | CLT | parallel | 32768 | 32 | **running** | 0 |
| clt_parallel_k64 | CLT | parallel | 32768 | 64 | **running** | 0 |

## Summary

| Project | Dict | Training | Finished | Running | Total |
|---------|------|----------|----------|---------|-------|
| pile_local_sweep_jose | 4k | local MSE | 8 | 0 | 8 |
| pile_local_sweep_jose_32k | 32k | local MSE | 7 | 1 | 8 |
| pile_e2e_sweep_jose | 4k | e2e KL | 20 | 0 | 20 |
| pile_e2e_sweep_jose_32k | 32k | e2e KL | 10 | 9 | 19 |

All runs use: LR=3e-4, batch_size=4096, seq_len=512, 500M tokens, BatchTopK activation.
