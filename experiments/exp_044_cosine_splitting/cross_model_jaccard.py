"""Cross-model feature matching heatmap via Jaccard similarity.

For each pair of models, compute max Jaccard similarity of binary activation
patterns. A feature "fires" if CI > 0 (SPD) or act > 0 (TC/CLT).

Collects binary masks via subprocesses (one model at a time to manage memory),
then computes cross-model Jaccard in the main process.

Usage:
    python experiments/exp_044_cosine_splitting/cross_model_jaccard.py
    python experiments/exp_044_cosine_splitting/cross_model_jaccard.py --plot-only
    python experiments/exp_044_cosine_splitting/cross_model_jaccard.py --n-tokens 100000
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

OUTPUT_DIR = Path("experiments/exp_044_cosine_splitting/output")
CACHE_DIR = OUTPUT_DIR / "jaccard_cache"
ALIVE_CAPACITY_PATH = OUTPUT_DIR / "cosine_capacity.json"
ALIVE_BETA_PATH = OUTPUT_DIR / "cosine_beta.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]

MODELS = {
    "SPD 0.5x": {"type": "spd", "run": "goodfire/spd/s-b2b37c4e", "alive_key": "spd_0.5x", "alive_file": "capacity"},
    "SPD 1x": {"type": "spd", "run": "goodfire/spd/s-55ea3f9b", "alive_key": "spd_1x", "alive_file": "capacity"},
    "SPD 2x": {"type": "spd", "run": "goodfire/spd/s-266cb440", "alive_key": "spd_2x", "alive_file": "capacity"},
    "SPD 4x": {"type": "spd", "run": "goodfire/spd/s-d3834f54", "alive_key": "spd_4x", "alive_file": "capacity"},
    "SPD b=0.0": {"type": "spd", "run": "goodfire/spd/s-68b5d00f", "alive_key": "beta=0.0", "alive_file": "beta"},
    "SPD b=0.25": {"type": "spd", "run": "goodfire/spd/s-2d710891", "alive_key": "beta=0.25", "alive_file": "beta"},
    "SPD b=0.5": {"type": "spd", "run": "goodfire/spd/s-cebfd0fb", "alive_key": "beta=0.5", "alive_file": "beta"},
    "SPD b=1.0": {"type": "spd", "run": "goodfire/spd/s-c4895972", "alive_key": "beta=1.0", "alive_file": "beta"},
    "TC 4k": {"type": "tc", "project": "mats-sprint/pile_local_sweep_jose", "run_id": "4ziu27fn", "alive_key": "tc_4096", "alive_file": "capacity"},
    "TC 32k": {"type": "tc", "project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "c4o8i98k", "alive_key": "tc_32768", "alive_file": "capacity"},
    "CLT 4k": {"type": "clt", "project": "mats-sprint/pile_local_sweep_jose", "run_id": "77sgz1pe", "alive_key": "clt_4096", "alive_file": "capacity"},
    "CLT 32k": {"type": "clt", "project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "j20m9hzr", "alive_key": "clt_32768", "alive_file": "capacity"},
}


# =============================================================================
# Subprocess workers — collect binary activation masks per layer
# Saves {layer: (n_alive, n_positions) bool tensor} to disk
# =============================================================================

SPD_COLLECT = r"""
import sys, json, torch
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/spd')

run_path = sys.argv[1]
alive_json_path = sys.argv[2]
alive_key = sys.argv[3]
out_path = sys.argv[4]
n_tokens = int(sys.argv[5])

DEVICE = 'cuda'
LAYERS = [0, 1, 2, 3]
BATCH_SIZE = 4
SEQ_LEN = 512
N_BATCHES = n_tokens // (BATCH_SIZE * SEQ_LEN)

from datasets import load_dataset
from tqdm import tqdm
from analysis.collect_spd_activations import load_spd_model

with open(alive_json_path) as f:
    alive_data = json.load(f)[alive_key]

dataset = load_dataset('danbraunai/pile-uncopyrighted-tok', split='train', streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=10000)
data_iter = iter(dataset)
batches = []
for _ in tqdm(range(N_BATCHES), desc='Loading batches'):
    batch_ids = []
    for _ in range(BATCH_SIZE):
        sample = next(data_iter)
        ids = sample['input_ids']
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        batch_ids.append(ids[:SEQ_LEN])
    batches.append(torch.stack(batch_ids))

spd_model, _ = load_spd_model(run_path)
spd_model.to(DEVICE)
torch.set_grad_enabled(False)

results = {}
for layer in LAYERS:
    mod_name = f'h.{layer}.mlp.down_proj'
    if mod_name not in alive_data or 'alive_indices' not in alive_data[mod_name]:
        continue
    indices = alive_data[mod_name]['alive_indices']
    C = alive_data[mod_name]['total']
    idx_tensor = torch.tensor(indices, dtype=torch.long)

    all_binary = []
    for input_ids_cpu in tqdm(batches, desc=f'SPD L{layer}'):
        input_ids = input_ids_cpu.to(DEVICE)
        out = spd_model(input_ids, cache_type='input')
        ci = spd_model.calc_causal_importances(out.cache, sampling='continuous')
        ci_vals = ci.lower_leaky[mod_name]  # (B, S, C)
        binary = (ci_vals > 0).reshape(-1, C)[:, idx_tensor].cpu()  # (B*S, n_alive)
        all_binary.append(binary)

    binary_cat = torch.cat(all_binary, dim=0).T.contiguous()  # (n_alive, n_positions)
    results[layer] = binary_cat
    print(f'  L{layer}: {binary_cat.shape[0]} alive, {binary_cat.shape[1]} positions, '
          f'density={binary_cat.float().mean():.4f}')

torch.save(results, out_path)
print(f'DONE: saved to {out_path}')
"""

TC_COLLECT = r"""
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
n_tokens = int(sys.argv[6])

DEVICE = 'cuda'
LAYERS = [0, 1, 2, 3]
BATCH_SIZE = 4
SEQ_LEN = 512
N_BATCHES = n_tokens // (BATCH_SIZE * SEQ_LEN)

from datasets import load_dataset
from tqdm import tqdm
from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig
from analysis.collect_spd_activations import load_spd_model

with open(alive_json_path) as f:
    alive_data = json.load(f)[alive_key]

spd_model, _ = load_spd_model('goodfire/spd/s-55ea3f9b')
spd_model.to(DEVICE)
base_model = spd_model.target_model
base_model.eval()
del spd_model; torch.cuda.empty_cache()

dataset = load_dataset('danbraunai/pile-uncopyrighted-tok', split='train', streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=10000)
data_iter = iter(dataset)
batches = []
for _ in tqdm(range(N_BATCHES), desc='Loading batches'):
    batch_ids = []
    for _ in range(BATCH_SIZE):
        sample = next(data_iter)
        ids = sample['input_ids']
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        batch_ids.append(ids[:SEQ_LEN])
    batches.append(torch.stack(batch_ids).to(DEVICE))

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
            dest = Path(f'checkpoints/jaccard_tc_{run_id}_layer{li}')
            download_artifact(project, a.name, dest)
            transcoders[li] = load_tc(dest)

torch.set_grad_enabled(False)

results = {}
for layer_idx in LAYERS:
    mod_name = f'h.{layer_idx}.mlp.down_proj'
    indices = alive_data[mod_name]['alive_indices']
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    tc = transcoders[layer_idx]

    all_binary = []
    for input_ids in tqdm(batches, desc=f'TC L{layer_idx}'):
        captured = {}
        def _hook(_mod, _inp, out, _li=layer_idx):
            captured[_li] = out.detach()
        hook = base_model.h[layer_idx].rms_2.register_forward_hook(_hook)
        base_model(input_ids)
        hook.remove()
        flat = captured[layer_idx].reshape(-1, tc.cfg.input_size)
        acts = tc.encode(flat)  # (B*S, dict_size)
        binary = (acts > 0)[:, idx_tensor].cpu()  # (B*S, n_alive)
        all_binary.append(binary)

    binary_cat = torch.cat(all_binary, dim=0).T.contiguous()  # (n_alive, n_positions)
    results[layer_idx] = binary_cat
    print(f'  L{layer_idx}: {binary_cat.shape[0]} alive, {binary_cat.shape[1]} positions, '
          f'density={binary_cat.float().mean():.4f}')

torch.save(results, out_path)
print(f'DONE: saved to {out_path}')
"""

CLT_COLLECT = r"""
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
n_tokens = int(sys.argv[6])

DEVICE = 'cuda'
LAYERS = [0, 1, 2, 3]
BATCH_SIZE = 4
SEQ_LEN = 512
N_BATCHES = n_tokens // (BATCH_SIZE * SEQ_LEN)

from datasets import load_dataset
from tqdm import tqdm
from nn_decompositions.clt import CrossLayerTranscoder
from nn_decompositions.config import CLTConfig
from analysis.collect_spd_activations import load_spd_model

with open(alive_json_path) as f:
    alive_data = json.load(f)[alive_key]

spd_model, _ = load_spd_model('goodfire/spd/s-55ea3f9b')
spd_model.to(DEVICE)
base_model = spd_model.target_model
base_model.eval()
del spd_model; torch.cuda.empty_cache()

dataset = load_dataset('danbraunai/pile-uncopyrighted-tok', split='train', streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=10000)
data_iter = iter(dataset)
batches = []
for _ in tqdm(range(N_BATCHES), desc='Loading batches'):
    batch_ids = []
    for _ in range(BATCH_SIZE):
        sample = next(data_iter)
        ids = sample['input_ids']
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        batch_ids.append(ids[:SEQ_LEN])
    batches.append(torch.stack(batch_ids).to(DEVICE))

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
dest = Path(f'checkpoints/jaccard_clt_{run_id}')
download_artifact(project, final_arts[0].name, dest)
clt = load_clt(dest)

torch.set_grad_enabled(False)

results = {}
for i in range(clt.cfg.n_layers):
    mod_name = f'h.{i}.mlp.down_proj'
    indices = alive_data[mod_name]['alive_indices']
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)

    all_binary = []
    for input_ids in tqdm(batches, desc=f'CLT L{i}'):
        captured = {}
        def _hook(_mod, _inp, out, _li=i):
            captured[_li] = out.detach()
        hook = base_model.h[i].rms_2.register_forward_hook(_hook)
        base_model(input_ids)
        hook.remove()
        flat = captured[i].reshape(-1, clt.cfg.input_size)
        pre_acts = F.relu(flat @ clt.W_enc[i] + clt.b_enc[i])
        k = clt.cfg.top_k
        n_keep = k * pre_acts.shape[0]
        if n_keep < pre_acts.numel():
            topk = torch.topk(pre_acts.flatten(), n_keep, dim=-1)
            acts = torch.zeros_like(pre_acts.flatten()).scatter(-1, topk.indices, topk.values).reshape(pre_acts.shape)
        else:
            acts = pre_acts
        binary = (acts > 0)[:, idx_tensor].cpu()  # (B*S, n_alive)
        all_binary.append(binary)

    binary_cat = torch.cat(all_binary, dim=0).T.contiguous()  # (n_alive, n_positions)
    results[i] = binary_cat
    print(f'  L{i}: {binary_cat.shape[0]} alive, {binary_cat.shape[1]} positions, '
          f'density={binary_cat.float().mean():.4f}')

torch.save(results, out_path)
print(f'DONE: saved to {out_path}')
"""


def collect_binary_masks(name: str, info: dict, n_tokens: int) -> Path:
    """Collect binary activation masks for a model via subprocess."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    cache_path = CACHE_DIR / f"{safe_name}_{n_tokens}.pt"

    if cache_path.exists():
        print(f"  Using cached {cache_path}")
        return cache_path

    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    alive_file = info.get("alive_file", "capacity")
    alive_path = str(ALIVE_CAPACITY_PATH if alive_file == "capacity" else ALIVE_BETA_PATH)
    alive_key = info["alive_key"]

    if info["type"] == "spd":
        cmd = [sys.executable, "-c", SPD_COLLECT, info["run"], alive_path, alive_key,
               str(cache_path), str(n_tokens)]
    elif info["type"] == "tc":
        cmd = [sys.executable, "-c", TC_COLLECT, info["project"], info["run_id"],
               alive_path, alive_key, str(cache_path), str(n_tokens)]
    elif info["type"] == "clt":
        cmd = [sys.executable, "-c", CLT_COLLECT, info["project"], info["run_id"],
               alive_path, alive_key, str(cache_path), str(n_tokens)]
    else:
        raise ValueError(f"Unknown type: {info['type']}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    stderr_tail = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
    print(stderr_tail, end="")
    for line in result.stdout.split("\n"):
        if line.strip():
            print(line)

    if not cache_path.exists():
        raise RuntimeError(
            f"Failed to collect {name}:\n{result.stdout[-500:]}\n{result.stderr[-500:]}"
        )
    return cache_path


CHUNK_SIZE = 256
TARGET_CHUNK_SIZE = 8192  # chunk target too if > this many features


def max_jaccard_chunked(query: torch.Tensor, target: torch.Tensor,
                        exclude_self: bool = False) -> np.ndarray:
    """For each row in query, find max Jaccard with any row in target.

    query/target: (n_features, n_positions) bool tensors on CPU.
    Returns (n_query,) array of max Jaccard values.
    """
    n_query = query.shape[0]
    n_target = target.shape[0]
    best = np.zeros(n_query, dtype=np.float32)

    # Precompute target sums on CPU
    target_sums_cpu = target.float().sum(dim=1)  # (n_target,)

    for i in range(0, n_query, CHUNK_SIZE):
        q_chunk = query[i:i + CHUNK_SIZE].float().to(DEVICE)
        q_sums = q_chunk.sum(dim=1)  # (chunk_q,)
        chunk_best = torch.full((q_chunk.shape[0],), -1.0, device=DEVICE)

        # Chunk target to avoid OOM
        for j in range(0, n_target, TARGET_CHUNK_SIZE):
            t_chunk = target[j:j + TARGET_CHUNK_SIZE].float().to(DEVICE)
            t_sums = target_sums_cpu[j:j + TARGET_CHUNK_SIZE].to(DEVICE)

            intersection = q_chunk @ t_chunk.T  # (chunk_q, chunk_t)
            union = q_sums.unsqueeze(1) + t_sums.unsqueeze(0) - intersection
            jaccard = intersection / union.clamp(min=1)

            if exclude_self:
                for local_qi in range(jaccard.shape[0]):
                    global_qi = i + local_qi
                    # Check if global_qi falls in this target chunk
                    local_ti = global_qi - j
                    if 0 <= local_ti < jaccard.shape[1]:
                        jaccard[local_qi, local_ti] = 0.0

            chunk_max = jaccard.max(dim=1).values
            chunk_best = torch.maximum(chunk_best, chunk_max)

        best[i:i + CHUNK_SIZE] = chunk_best.cpu().numpy()

    return best


def compute_and_plot(n_tokens: int):
    """Collect binary masks, compute cross-model Jaccard, plot heatmaps."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_names = list(MODELS.keys())

    # Collect binary masks for all models
    all_masks = {}  # {name: {layer: (n_alive, n_positions) bool}}
    for name, info in MODELS.items():
        print(f"\n{'='*60}\n  Collecting: {name}\n{'='*60}")
        cache_path = collect_binary_masks(name, info, n_tokens)
        all_masks[name] = torch.load(cache_path, map_location="cpu", weights_only=True)
        for layer, mask in all_masks[name].items():
            print(f"    L{layer}: {mask.shape[0]} alive, {mask.shape[1]} positions")

    # Compute cross-model max Jaccard
    n = len(model_names)
    mean_max_jaccard = np.zeros((n, n))
    pct_matched = np.zeros((n, n))  # % with max_jaccard > 0.1
    pct_strong = np.zeros((n, n))  # % with max_jaccard > 0.3

    print(f"\n{'='*60}\n  Computing Jaccard\n{'='*60}")

    for i, name_from in enumerate(model_names):
        for j, name_to in enumerate(model_names):
            layer_jaccards = []
            layer_matched = []
            layer_strong = []
            exclude_self = (name_from == name_to)

            for layer in LAYERS:
                masks_from = all_masks[name_from][layer]
                masks_to = all_masks[name_to][layer]
                max_j = max_jaccard_chunked(masks_from, masks_to, exclude_self)

                if len(max_j) > 0:
                    layer_jaccards.append(float(max_j.mean()))
                    layer_matched.append(100.0 * (max_j > 0.1).sum() / len(max_j))
                    layer_strong.append(100.0 * (max_j > 0.3).sum() / len(max_j))

            mean_max_jaccard[i, j] = np.mean(layer_jaccards) if layer_jaccards else 0
            pct_matched[i, j] = np.mean(layer_matched) if layer_matched else 0
            pct_strong[i, j] = np.mean(layer_strong) if layer_strong else 0

            print(f"  {name_from:>12} → {name_to:<12}: "
                  f"mean_max_J={mean_max_jaccard[i,j]:.4f}, "
                  f"matched(>0.1)={pct_matched[i,j]:.1f}%, "
                  f"strong(>0.3)={pct_strong[i,j]:.1f}%")

    # Save data
    data = {
        "n_tokens": n_tokens,
        "models": model_names,
        "mean_max_jaccard": mean_max_jaccard.tolist(),
        "pct_matched_0p1": pct_matched.tolist(),
        "pct_strong_0p3": pct_strong.tolist(),
    }
    with open(OUTPUT_DIR / "jaccard_heatmap_data.json", "w") as f:
        json.dump(data, f, indent=2)

    _plot_heatmaps(model_names, mean_max_jaccard, pct_matched, pct_strong)


def _plot_heatmaps(model_names, mean_max_jaccard, pct_matched, pct_strong):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    n = len(model_names)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for ax, matrix, title, fmt, cmap in [
        (axes[0], mean_max_jaccard, "Mean max Jaccard", ".3f", "YlOrRd"),
        (axes[1], pct_matched, "% features with max J > 0.1", ".1f", "YlOrRd"),
        (axes[2], pct_strong, "% features with max J > 0.3", ".1f", "YlOrRd"),
    ]:
        im = ax.imshow(matrix, cmap=cmap, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(model_names, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(model_names, fontsize=7)
        ax.set_xlabel("To model")
        ax.set_ylabel("From model")
        ax.set_title(title, fontsize=11)

        for i in range(n):
            for j in range(n):
                color = "white" if matrix[i, j] > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{matrix[i, j]:{fmt}}", ha="center", va="center",
                        fontsize=6, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Cross-Model Feature Matching (Jaccard of activation patterns)", fontsize=13)
    fig.tight_layout()
    path = OUTPUT_DIR / "cross_model_jaccard.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_only():
    data_path = OUTPUT_DIR / "jaccard_heatmap_data.json"
    with open(data_path) as f:
        data = json.load(f)
    _plot_heatmaps(
        data["models"],
        np.array(data["mean_max_jaccard"]),
        np.array(data["pct_matched_0p1"]),
        np.array(data["pct_strong_0p3"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--n-tokens", type=int, default=100_000,
                        help="Number of tokens for activation collection (default 100k)")
    args = parser.parse_args()

    if args.plot_only:
        plot_only()
    else:
        compute_and_plot(args.n_tokens)


if __name__ == "__main__":
    main()
