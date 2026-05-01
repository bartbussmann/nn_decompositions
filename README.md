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
├── experiments/
│   ├── paper_runs.py                     # wandb / VPD run-IDs used in the paper
│   ├── eval_utils.py                     # shared helpers (data, hooks, CE, model loaders)
│   ├── llm_base_model/                   # auto-populated cache (gitignored)
│   ├── train_local_mse/                  # PLT + CLT local-MSE training (4k or 32k)
│   ├── train_e2e/                        # PLT + CLT end-to-end training (4k or 32k)
│   ├── pareto_plot_local/                # local-MSE Pareto plot (CE / L0)
│   ├── pareto_plot_e2e/                  # end-to-end Pareto plot (CE / MSE vs capacity)
│   ├── alive_subcomponents/              # alive-subcomponent scaling figure
│   └── feature_splitting_heatmap/        # cross-model feature-splitting heatmaps
├── setup_env.sh
└── pyproject.toml
```

## Setup

Requires a CUDA-12.4-compatible GPU and `git`. Python 3.13 itself does
**not** need to be pre-installed — `setup_env.sh` uses [`uv`](https://astral.sh/uv)
to fetch a 3.13 interpreter on the fly.

```bash
bash setup_env.sh                # installs uv + Python 3.13, creates .venv,
                                 # installs torch + spd + this package
source .venv/bin/activate
wandb login                      # needed to pull artifacts and log training
huggingface-cli login            # `danbraunai/pile-uncopyrighted-tok` is gated
```

## Training the PLTs and CLTs

The paper uses local-MSE-trained PLTs and CLTs at two dictionary sizes
(4 096 and 32 768) and four top-k values (8, 16, 32, 64). End-to-end KL
variants are also trained. The four wandb projects are:

| project | script | notes |
|---|---|---|
| `pile_local_sweep_jose`     | `train_local_mse/train_local_mse.py` | default `--dict_size 4096` |
| `pile_local_sweep_jose_32k` | `train_local_mse/train_local_mse.py --dict_size 32768` | same script, 32k variant |
| `pile_e2e_sweep_jose`       | `train_e2e/train_e2e.py` | default `--dict_size 4096` |
| `pile_e2e_sweep_jose_32k`   | `train_e2e/train_e2e.py --dict_size 32768` | same script, 32k variant |

Each script polls GPU memory and launches one job per free device. Full
sweeps require ≈ 40 GPU-hours on H100s.

## Replicating the four figures

After training (or pointing the scripts at the existing wandb run IDs
above), run each experiment from the repository root:

| Figure | Script | Output JSON | Wall-clock |
|---|---|---|---|
| Local-MSE Pareto | `python experiments/pareto_plot_local/pareto_plot_local.py` then `python experiments/pareto_plot_local/plot.py` | `pareto_plot_local/output/results_{4k,32k}.json` | ≈ 20 min |
| End-to-end Pareto | `python experiments/pareto_plot_e2e/pareto_plot_e2e.py` | `pareto_plot_e2e/output/pareto_data.json` | ≈ 30 min |
| Alive-subcomponent scaling | `python experiments/alive_subcomponents/alive_subcomponents.py` | `alive_subcomponents/output/alive_line_data.json` | ≈ 15 min |
| Feature-splitting heatmaps | `python experiments/feature_splitting_heatmap/feature_splitting_heatmap.py` | `feature_splitting_heatmap/output/heatmap_data_{input,output,matrix}_t0p5.json` | ≈ 25 min |

Each script supports `--plot-only` to re-render figures from the cached
JSON in its `output/` directory without re-running the heavy compute.

The data files (`results_*.json`, `pareto_data.json`, `alive_line_data.json`,
`heatmap_data_*.json`) are the canonical numerical output and are
designed to be re-styled by collaborators without rerunning the
extraction.

## Reference VPD checkpoints

`pareto_plot_local`, `pareto_plot_e2e`, `alive_subcomponents`, and `feature_splitting_heatmap` also load four VPD checkpoints from
the public `goodfire/spd` wandb project:

| Capacity | wandb run |
|---|---|
| 0.5x | `goodfire/spd/s-b2b37c4e` |
| 1x   | `goodfire/spd/s-55ea3f9b` |
| 2x   | `goodfire/spd/s-266cb440` |
| 4x   | `goodfire/spd/s-d3834f54` |

These are loaded via `experiments.eval_utils.load_vpd_model` and
require the [`spd` repository](https://github.com/goodfire-ai/spd) on the
branch installed by `setup_env.sh`. All run IDs (VPD baselines, headline
PLT/CLT runs, base-model path, wandb projects) live in
`experiments/paper_runs.py` — edit there to swap in your own checkpoints.

## License

MIT — see `LICENSE`.
