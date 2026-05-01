# nn_decompositions — VPD paper replication

This repository contains the per-layer transcoder (PLT) and cross-layer
transcoder (CLT) training and evaluation code used in the VPD paper. It
is a minimal slice that exactly reproduces the four headline figures
(plus their underlying training runs).

The PLT and CLT decomposition baselines are trained here; VPD itself is
loaded from public artifacts on the [SPD repository](https://github.com/goodfire-ai/spd).

## Target model

All experiments target a 4-layer LlamaSimpleMLP transformer
(`d_model = 768`, `d_intermediate = 3072`, GELU MLPs) trained on the
Pile. The base model is published at
[`goodfire/spd/runs/t-9d2b8f02`](https://wandb.ai/goodfire/spd/runs/t-9d2b8f02)
and is auto-downloaded on first use.

## Layout

```
.
├── nn_decompositions/                    # core PLT / CLT package
│   ├── transcoder.py                     # BatchTopKTranscoder
│   ├── clt.py                            # CrossLayerTranscoder
│   ├── config.py                         # EncoderConfig, CLTConfig
│   ├── activation_store.py
│   ├── training.py
│   └── logs.py
├── analysis/
│   └── collect_spd_activations.py        # SPD model loader
├── experiments/
│   ├── jose_base_model/                  # auto-populated cache (gitignored)
│   ├── jose_training_runs.md             # wandb run-id table for every PLT/CLT
│   ├── exp_020_e2e_sweep_jose/           # PLT + CLT end-to-end training (4k)
│   ├── exp_032_local_sweep_jose/         # PLT + CLT local-MSE training (4k or 32k)
│   ├── exp_033_e2e_sweep_jose_32k/       # PLT + CLT end-to-end training (32k)
│   ├── exp_040_eval_e2e_jose_v2/         # CE / L0 evaluation + figure
│   ├── exp_045_pareto_combined/          # Pareto plot (CE / MSE vs capacity)
│   ├── exp_049_spd_feature_splitting/    # alive-component scaling figure
│   └── exp_052_cross_model_heatmaps_t05/ # cross-model feature-matching heatmaps
├── setup_env.sh
└── pyproject.toml
```

## Setup

Requires Python 3.13 and a CUDA-12.4 GPU.

```bash
bash setup_env.sh                # creates .venv, installs torch + spd + this package
source .venv/bin/activate
wandb login                      # needed to pull artifacts and log training
```

## Training the PLTs and CLTs

The paper uses local-MSE-trained PLTs and CLTs at two dictionary sizes
(4 096 and 32 768) and four top-k values (8, 16, 32, 64). End-to-end KL
variants are also trained. All runs go to four wandb projects whose IDs
are listed in `experiments/jose_training_runs.md`.

| project | script | notes |
|---|---|---|
| `pile_local_sweep_jose`     | `exp_032_local_sweep_jose/local_sweep_jose.py` | default `--dict_size 4096` |
| `pile_local_sweep_jose_32k` | `exp_032_local_sweep_jose/local_sweep_jose.py --dict_size 32768` | same script, 32k variant |
| `pile_e2e_sweep_jose`       | `exp_020_e2e_sweep_jose/e2e_sweep_jose.py` | 4k end-to-end KL |
| `pile_e2e_sweep_jose_32k`   | `exp_033_e2e_sweep_jose_32k/e2e_sweep_jose_32k.py` | 32k end-to-end KL |

Each script polls GPU memory and launches one job per free device. Full
sweeps require ≈ 40 GPU-hours on H100s.

## Replicating the four figures

After training (or pointing the scripts at the existing wandb run IDs
above), run each experiment from the repository root:

| Figure | Script | Wall-clock |
|---|---|---|
| **exp_040** — CE / L0 evaluation | `python experiments/exp_040_eval_e2e_jose_v2/eval_e2e_jose_v2.py` then `python experiments/exp_040_eval_e2e_jose_v2/plot_e2e_jose_v2.py` | ≈ 20 min |
| **exp_045** — Pareto plot | `python experiments/exp_045_pareto_combined/pareto_combined.py` | ≈ 30 min |
| **exp_049** — alive-subcomponent scaling | `python experiments/exp_049_spd_feature_splitting/alive_line_plot.py` | ≈ 15 min |
| **exp_052** — cross-model heatmaps | `python experiments/exp_052_cross_model_heatmaps_t05/cross_model_heatmaps.py` | ≈ 25 min |

Each script supports `--plot-only` to re-render figures from the cached
JSON in its `output/` directory without re-running the heavy compute.

The data files (`results_*.json`, `pareto_data.json`, `alive_line_data.json`,
`heatmap_data_*.json`) are the canonical numerical output and are
designed to be re-styled by collaborators without rerunning the
extraction.

## Reference VPD checkpoints

exp_045, exp_049, and exp_052 also load four VPD (SPD) checkpoints from
the public `goodfire/spd` wandb project:

| Capacity | wandb run |
|---|---|
| 0.5x | `goodfire/spd/s-b2b37c4e` |
| 1x   | `goodfire/spd/s-55ea3f9b` |
| 2x   | `goodfire/spd/s-266cb440` |
| 4x   | `goodfire/spd/s-d3834f54` |

These are loaded via `analysis/collect_spd_activations.py:load_spd_model`
and require the [SPD repository](https://github.com/goodfire-ai/spd) on
the branch installed by `setup_env.sh`.

## License

MIT — see `LICENSE`.
