"""Cross-model feature matching heatmap.

For each pair of models, count how many features in model A have a match
(cosine > threshold) in model B. Display as a heatmap.

Uses precomputed alive_indices from cosine_capacity.json (exp_044) to skip
data loading — only loads model weights to extract direction vectors.

Supports two spaces:
  --space output (default): SPD U (down_proj), TC/CLT W_dec
  --space input:            SPD V (c_fc), TC/CLT W_enc

Usage:
    python experiments/exp_044_cosine_splitting/cross_model_heatmap.py
    python experiments/exp_044_cosine_splitting/cross_model_heatmap.py --space input
    python experiments/exp_044_cosine_splitting/cross_model_heatmap.py --plot-only
    python experiments/exp_044_cosine_splitting/cross_model_heatmap.py --thresholds 0.3 0.5 0.7
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

OUTPUT_DIR = Path("experiments/exp_044_cosine_splitting/output")
CACHE_DIR = OUTPUT_DIR / "direction_cache"
ALIVE_CAPACITY_PATH = OUTPUT_DIR / "cosine_capacity.json"
ALIVE_BETA_PATH = OUTPUT_DIR / "cosine_beta.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]

# All models to compare — "alive_key" maps to JSON keys, "alive_file" selects which JSON
MODELS = {
    "VPD 0.5x": {"type": "spd", "run": "goodfire/spd/s-b2b37c4e", "alive_key": "spd_0.5x", "alive_file": "capacity"},
    "VPD 1x": {"type": "spd", "run": "goodfire/spd/s-55ea3f9b", "alive_key": "spd_1x", "alive_file": "capacity"},
    "VPD 2x": {"type": "spd", "run": "goodfire/spd/s-266cb440", "alive_key": "spd_2x", "alive_file": "capacity"},
    "VPD 4x": {"type": "spd", "run": "goodfire/spd/s-d3834f54", "alive_key": "spd_4x", "alive_file": "capacity"},
    "VPD b=0.0": {"type": "spd", "run": "goodfire/spd/s-68b5d00f", "alive_key": "beta=0.0", "alive_file": "beta"},
    "VPD b=0.25": {"type": "spd", "run": "goodfire/spd/s-2d710891", "alive_key": "beta=0.25", "alive_file": "beta"},
    "VPD b=0.5": {"type": "spd", "run": "goodfire/spd/s-cebfd0fb", "alive_key": "beta=0.5", "alive_file": "beta"},
    "VPD b=1.0": {"type": "spd", "run": "goodfire/spd/s-c4895972", "alive_key": "beta=1.0", "alive_file": "beta"},
    "PLT 4k": {"type": "tc", "project": "mats-sprint/pile_local_sweep_jose", "run_id": "4ziu27fn", "alive_key": "tc_4096", "alive_file": "capacity"},
    "PLT 32k": {"type": "tc", "project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "c4o8i98k", "alive_key": "tc_32768", "alive_file": "capacity"},
    "CLT 4k": {"type": "clt", "project": "mats-sprint/pile_local_sweep_jose", "run_id": "77sgz1pe", "alive_key": "clt_4096", "alive_file": "capacity"},
    "CLT 32k": {"type": "clt", "project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "j20m9hzr", "alive_key": "clt_32768", "alive_file": "capacity"},
}


# =============================================================================
# Subprocess workers — load model weights + precomputed alive indices only
# space_arg is passed as sys.argv: "output" or "input"
# =============================================================================

SPD_EXTRACT = r"""
import sys, json, torch, torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/spd')

run_path = sys.argv[1]
alive_json_path = sys.argv[2]
alive_key = sys.argv[3]
out_path = sys.argv[4]
space = sys.argv[5]  # "output" or "input"
LAYERS = [0, 1, 2, 3]

from analysis.collect_spd_activations import load_spd_model

with open(alive_json_path) as f:
    alive_data = json.load(f)[alive_key]

spd_model, _ = load_spd_model(run_path)

results = {}
for layer in LAYERS:
    if space == 'output':
        mod_name = f'h.{layer}.mlp.down_proj'
    else:
        mod_name = f'h.{layer}.mlp.c_fc'
    if mod_name not in alive_data:
        continue
    indices = torch.tensor(alive_data[mod_name]['alive_indices'], dtype=torch.long)
    if space == 'output':
        vecs = spd_model.components[mod_name].U.float()  # (C, d_out)
        vecs_alive = F.normalize(vecs[indices], dim=1).cpu()
    else:
        V = spd_model.components[mod_name].V.float()  # (d_in, C)
        vecs_alive = F.normalize(V[:, indices].T, dim=1).cpu()  # (n_alive, d_in)
    results[layer] = vecs_alive
    print(f'  L{layer}: {len(indices)} alive, shape={vecs_alive.shape}')

torch.save(results, out_path)
print(f'DONE: saved to {out_path}')
"""

TC_EXTRACT = r"""
import sys, json, torch, torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/spd')
import wandb

project = sys.argv[1]
run_id = sys.argv[2]
alive_json_path = sys.argv[3]
alive_key = sys.argv[4]
out_path = sys.argv[5]
space = sys.argv[6]  # "output" or "input"
DEVICE = 'cpu'
LAYERS = [0, 1, 2, 3]

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig

with open(alive_json_path) as f:
    alive_data = json.load(f)[alive_key]

def download_artifact(proj, art_name, dest):
    dest = Path(dest)
    if dest.exists() and (dest / 'encoder.pt').exists():
        return dest
    api = wandb.Api()
    artifact = api.artifact(f'{proj}/{art_name}')
    artifact.download(root=str(dest))
    return dest

def load_tc(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / 'config.json') as f:
        cfg_dict = json.load(f)
    cfg_dict['dtype'] = getattr(torch, cfg_dict.get('dtype', 'torch.float32').replace('torch.', ''))
    cfg_dict['device'] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    tc = BatchTopKTranscoder(cfg)
    tc.load_state_dict(torch.load(checkpoint_dir / 'encoder.pt', map_location=DEVICE))
    tc.eval()
    return tc

api = wandb.Api()
run = api.run(f'{project}/runs/{run_id}')
arts = [a for a in run.logged_artifacts() if a.type == 'model']
transcoders = {}
for a in arts:
    aname = a.name.split(':')[0]
    for li in LAYERS:
        if f'layer{li}_final' in aname:
            dest = Path(f'checkpoints/heatmap_tc_{run_id}_layer{li}')
            download_artifact(project, a.name, dest)
            transcoders[li] = load_tc(dest)

results = {}
for layer_idx in LAYERS:
    mod_name = f'h.{layer_idx}.mlp.down_proj'
    indices = torch.tensor(alive_data[mod_name]['alive_indices'], dtype=torch.long)
    tc = transcoders[layer_idx]
    if space == 'output':
        W = F.normalize(tc.W_dec.float(), dim=1)  # (dict_size, output_size)
    else:
        W = F.normalize(tc.W_enc.float().T, dim=1)  # (dict_size, input_size)
    W_alive = W[indices].cpu()
    results[layer_idx] = W_alive
    print(f'  L{layer_idx}: {len(indices)} alive, shape={W_alive.shape}')

torch.save(results, out_path)
print(f'DONE: saved to {out_path}')
"""

CLT_EXTRACT = r"""
import sys, json, torch, torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/spd')
import wandb

project = sys.argv[1]
run_id = sys.argv[2]
alive_json_path = sys.argv[3]
alive_key = sys.argv[4]
out_path = sys.argv[5]
space = sys.argv[6]  # "output" or "input"
DEVICE = 'cpu'
LAYERS = [0, 1, 2, 3]

from nn_decompositions.clt import CrossLayerTranscoder
from nn_decompositions.config import CLTConfig

with open(alive_json_path) as f:
    alive_data = json.load(f)[alive_key]

def download_artifact(proj, art_name, dest):
    dest = Path(dest)
    if dest.exists() and (dest / 'encoder.pt').exists():
        return dest
    api = wandb.Api()
    artifact = api.artifact(f'{proj}/{art_name}')
    artifact.download(root=str(dest))
    return dest

def load_clt(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / 'config.json') as f:
        cfg_dict = json.load(f)
    cfg_dict['layers'] = json.loads(cfg_dict['layers'])
    cfg_dict['dtype'] = getattr(torch, cfg_dict.get('dtype', 'torch.float32').replace('torch.', ''))
    cfg_dict['device'] = DEVICE
    cfg = CLTConfig(**cfg_dict)
    clt = CrossLayerTranscoder(cfg)
    clt.load_state_dict(torch.load(checkpoint_dir / 'encoder.pt', map_location=DEVICE))
    clt.eval()
    return clt

api = wandb.Api()
run = api.run(f'{project}/runs/{run_id}')
arts = [a for a in run.logged_artifacts() if a.type == 'model']
final_arts = [a for a in arts if 'final' in a.name]
dest = Path(f'checkpoints/heatmap_clt_{run_id}')
download_artifact(project, final_arts[0].name, dest)
clt = load_clt(dest)

results = {}
for i in range(clt.cfg.n_layers):
    mod_name = f'h.{i}.mlp.down_proj'
    indices = torch.tensor(alive_data[mod_name]['alive_indices'], dtype=torch.long)
    if space == 'output':
        W = F.normalize(clt.W_dec[i][0].float(), dim=1)  # (dict_size, output_size)
    else:
        W = F.normalize(clt.W_enc[i].float().T, dim=1)  # (dict_size, input_size)
    W_alive = W[indices].cpu()
    results[i] = W_alive
    print(f'  L{i}: {len(indices)} alive, shape={W_alive.shape}')

torch.save(results, out_path)
print(f'DONE: saved to {out_path}')
"""


def extract_directions(name: str, info: dict, space: str) -> Path:
    """Extract alive directions for a model using precomputed alive indices."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    cache_path = CACHE_DIR / f"{safe_name}_{space}.pt"

    if cache_path.exists():
        print(f"  Using cached {cache_path}")
        return cache_path

    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    alive_file = info.get("alive_file", "capacity")
    alive_path = str(ALIVE_CAPACITY_PATH if alive_file == "capacity" else ALIVE_BETA_PATH)
    alive_key = info["alive_key"]

    if info["type"] == "spd":
        cmd = [sys.executable, "-c", SPD_EXTRACT, info["run"], alive_path, alive_key, str(cache_path), space]
    elif info["type"] == "tc":
        cmd = [sys.executable, "-c", TC_EXTRACT, info["project"], info["run_id"], alive_path, alive_key, str(cache_path), space]
    elif info["type"] == "clt":
        cmd = [sys.executable, "-c", CLT_EXTRACT, info["project"], info["run_id"], alive_path, alive_key, str(cache_path), space]
    else:
        raise ValueError(f"Unknown type: {info['type']}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    stderr_tail = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
    print(stderr_tail, end="")

    if not cache_path.exists():
        raise RuntimeError(
            f"Failed to extract {name}:\n{result.stdout[-500:]}\n{result.stderr[-500:]}"
        )
    return cache_path


def count_matches(dirs_from: torch.Tensor, dirs_to: torch.Tensor,
                  threshold: float, exclude_self: bool = False) -> np.ndarray:
    """For each direction in dirs_from, count how many in dirs_to have cosine > threshold."""
    dirs_to_gpu = dirs_to.to(DEVICE)
    chunk_size = 512
    counts = []

    for i in range(0, dirs_from.shape[0], chunk_size):
        chunk = dirs_from[i:i + chunk_size].to(DEVICE)
        cosine = chunk @ dirs_to_gpu.T  # (chunk, n_to)
        counts.append((cosine > threshold).sum(dim=1).cpu().numpy())

    result = np.concatenate(counts)
    if exclude_self:
        result = np.maximum(result - 1, 0)
    return result


def compute_and_plot(thresholds: list[float], space: str = "output"):
    """Extract all directions, compute cross-model matches, plot heatmaps."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_names = list(MODELS.keys())

    # Extract directions for all models
    all_dirs = {}  # {name: {layer: tensor(n_alive, d)}}
    for name, info in MODELS.items():
        print(f"\n{'='*60}\n  Extracting: {name} ({space} space)\n{'='*60}")
        cache_path = extract_directions(name, info, space)
        all_dirs[name] = torch.load(cache_path, map_location="cpu", weights_only=True)
        for layer, dirs in all_dirs[name].items():
            print(f"    L{layer}: {dirs.shape[0]} alive directions")

    # For each threshold, compute heatmaps
    for threshold in thresholds:
        print(f"\n{'='*60}\n  Threshold = {threshold}\n{'='*60}")

        n = len(model_names)
        # Aggregate across all layers: mean matches per feature, averaged over layers
        mean_matches_matrix = np.zeros((n, n))
        pct_split_matrix = np.zeros((n, n))  # % with >1 match
        pct_novel_matrix = np.zeros((n, n))  # % with 0 matches

        for i, name_from in enumerate(model_names):
            for j, name_to in enumerate(model_names):
                layer_means = []
                layer_splits = []
                layer_novels = []
                exclude_self = (name_from == name_to)

                for layer in LAYERS:
                    dirs_from = all_dirs[name_from][layer]
                    dirs_to = all_dirs[name_to][layer]
                    counts = count_matches(dirs_from, dirs_to, threshold, exclude_self)

                    if len(counts) > 0:
                        layer_means.append(float(counts.mean()))
                        layer_splits.append(100.0 * (counts > 1).sum() / len(counts))
                        layer_novels.append(100.0 * (counts == 0).sum() / len(counts))

                mean_matches_matrix[i, j] = np.mean(layer_means) if layer_means else 0
                pct_split_matrix[i, j] = np.mean(layer_splits) if layer_splits else 0
                pct_novel_matrix[i, j] = np.mean(layer_novels) if layer_novels else 0

                print(f"  {name_from:>10} → {name_to:<10}: "
                      f"mean={mean_matches_matrix[i,j]:.2f}, "
                      f"split={pct_split_matrix[i,j]:.1f}%, "
                      f"novel={pct_novel_matrix[i,j]:.1f}%")

        # Plot
        space_label = "output (decoder/U)" if space == "output" else "input (encoder/V)"
        _plot_heatmaps(model_names, mean_matches_matrix, pct_split_matrix,
                       pct_novel_matrix, threshold, space, space_label)

        # Save data
        suffix = f"_{threshold}".replace(".", "p")
        space_suffix = f"_{space}" if space != "output" else ""
        data = {
            "threshold": threshold,
            "space": space,
            "models": model_names,
            "mean_matches": mean_matches_matrix.tolist(),
            "pct_split": pct_split_matrix.tolist(),
            "pct_novel": pct_novel_matrix.tolist(),
        }
        with open(OUTPUT_DIR / f"heatmap_data{suffix}{space_suffix}.json", "w") as f:
            json.dump(data, f, indent=2)


def _plot_heatmaps(model_names, mean_matches, pct_split, pct_novel, threshold,
                   space="output", space_label="output (decoder/U)"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    suffix = f"_{threshold}".replace(".", "p")
    space_suffix = f"_{space}" if space != "output" else ""
    n = len(model_names)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    for ax, matrix, title, fmt, cmap in [
        (axes[0], mean_matches, "Mean # matches", ".2f", "YlOrRd"),
        (axes[1], pct_split, "% components with >1 match (split)", ".1f", "YlOrRd"),
        (axes[2], pct_novel, "% components with 0 matches (novel)", ".1f", "YlGnBu"),
    ]:
        im = ax.imshow(matrix, cmap=cmap, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(model_names, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(model_names, fontsize=8)
        ax.set_xlabel("To model")
        ax.set_ylabel("From model")
        ax.set_title(title, fontsize=11)

        for i in range(n):
            for j in range(n):
                color = "white" if matrix[i, j] > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{matrix[i, j]:{fmt}}", ha="center", va="center",
                        fontsize=7, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Cross-Model Feature Matching — {space_label} (cosine > {threshold})", fontsize=13)
    fig.tight_layout()
    path = OUTPUT_DIR / f"cross_model_heatmap{suffix}{space_suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_only(thresholds: list[float], space: str = "output"):
    space_suffix = f"_{space}" if space != "output" else ""
    for threshold in thresholds:
        suffix = f"_{threshold}".replace(".", "p")
        data_path = OUTPUT_DIR / f"heatmap_data{suffix}{space_suffix}.json"
        if not data_path.exists():
            print(f"No data for threshold={threshold}, space={space}")
            continue
        with open(data_path) as f:
            data = json.load(f)
        space_label = "output (decoder/U)" if space == "output" else "input (encoder/V)"
        _plot_heatmaps(
            data["models"],
            np.array(data["mean_matches"]),
            np.array(data["pct_split"]),
            np.array(data["pct_novel"]),
            threshold,
            space,
            space_label,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--space", choices=["output", "input"], default="output",
                        help="output: SPD U / TC,CLT W_dec; input: SPD V / TC,CLT W_enc")
    args = parser.parse_args()

    if args.plot_only:
        plot_only(args.thresholds, args.space)
    else:
        compute_and_plot(args.thresholds, args.space)


if __name__ == "__main__":
    main()
