"""Measure feature splitting via max cosine similarity between alive components.

For each alive component in a module, compute the max cosine similarity with any
other alive component in the same module. Report the mean of these max-cosines.

Plot 1 (capacity): x=total component capacity, y=mean max cosine, lines for SPD/TC/CLT
Plot 2 (beta): x=ImpMin beta, y=mean max cosine, lines per layer + total

Usage:
    python experiments/exp_044_cosine_splitting/cosine_splitting.py
    python experiments/exp_044_cosine_splitting/cosine_splitting.py --plot-only
    python experiments/exp_044_cosine_splitting/cosine_splitting.py --step capacity
    python experiments/exp_044_cosine_splitting/cosine_splitting.py --step beta
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("experiments/exp_044_cosine_splitting/output")

LAYERS = [0, 1, 2, 3]
MLP_MODULES = [f"h.{l}.mlp.{m}" for l in LAYERS for m in ("c_fc", "down_proj")]

# ── Models for Plot 1 (capacity) ──
SPD_CAPACITY_RUNS = {
    "0.5x": "goodfire/spd/s-b2b37c4e",
    "1x": "goodfire/spd/s-55ea3f9b",
    "2x": "goodfire/spd/s-266cb440",
    "4x": "goodfire/spd/s-d3834f54",
}

TC_RUNS = {
    4096: {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "4ziu27fn"},
    32768: {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "c4o8i98k"},
}

CLT_RUNS = {
    4096: {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "77sgz1pe"},
    32768: {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "j20m9hzr"},
}

# ── Models for Plot 2 (beta) ──
SPD_BETA_RUNS = {
    "beta=0.0": "goodfire/spd/s-68b5d00f",
    "beta=0.25": "goodfire/spd/s-2d710891",
    "beta=0.5": "goodfire/spd/s-cebfd0fb",
    "beta=1.0": "goodfire/spd/s-c4895972",
}


# =============================================================================
# SPD worker: compute alive masks + max cosine per module via subprocess
# =============================================================================

SPD_WORKER = r"""
import sys, json, torch, torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/spd')

DEVICE = 'cuda'
LAYERS = [0, 1, 2, 3]
BATCH_SIZE = 4
SEQ_LEN = 512
N_TOKENS = int(1e6)
N_BATCHES = N_TOKENS // (BATCH_SIZE * SEQ_LEN)
MLP_MODULE_PATTERNS = ['h.{}.mlp.c_fc', 'h.{}.mlp.down_proj']
THRESHOLD = 1e-6

label = sys.argv[1]
run_path = sys.argv[2]

from datasets import load_dataset
from tqdm import tqdm
from analysis.collect_spd_activations import load_spd_model

dataset = load_dataset('danbraunai/pile-uncopyrighted-tok', split='train', streaming=True)
dataset = dataset.shuffle(seed=0, buffer_size=10000)
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

print(f'Loading {label} ({run_path})...')
spd_model, _ = load_spd_model(run_path)
spd_model.to(DEVICE)

mlp_modules = []
for layer in LAYERS:
    for pattern in MLP_MODULE_PATTERNS:
        mod_name = pattern.format(layer)
        if mod_name in spd_model.module_to_c:
            mlp_modules.append(mod_name)

# Accumulate mean CI
ci_sum = {}
for mod_name in mlp_modules:
    n_c = spd_model.module_to_c[mod_name]
    ci_sum[mod_name] = torch.zeros(n_c, dtype=torch.float64, device='cpu')

torch.set_grad_enabled(False)
n_tokens_total = 0
for input_ids_cpu in tqdm(batches, desc=f'{label} CI'):
    input_ids = input_ids_cpu.to(DEVICE)
    B, S = input_ids.shape
    n_tokens_total += B * S
    out = spd_model(input_ids, cache_type='input')
    ci = spd_model.calc_causal_importances(out.cache, sampling='continuous')
    for mod_name in mlp_modules:
        ci_vals = ci.lower_leaky[mod_name].reshape(-1, spd_model.module_to_c[mod_name])
        ci_sum[mod_name] += ci_vals.double().sum(dim=0).cpu()

# Compute alive masks and max cosine per module
results = {}
for mod_name in mlp_modules:
    mean_ci = ci_sum[mod_name] / n_tokens_total
    alive_mask = mean_ci > THRESHOLD
    n_alive = alive_mask.sum().item()
    n_total = spd_model.module_to_c[mod_name]

    # Get U vectors (output space) for alive components
    U = spd_model.components[mod_name].U.float()  # (C, d_out)
    U_alive = F.normalize(U[alive_mask], dim=1).to(DEVICE)  # (n_alive, d_out)

    # Max cosine within module (exclude self)
    if n_alive > 1:
        cosine = U_alive @ U_alive.T  # (n_alive, n_alive)
        cosine.fill_diagonal_(-float('inf'))
        max_cos = cosine.max(dim=1).values.cpu().numpy()
        mean_max_cos = float(max_cos.mean())
    else:
        mean_max_cos = 0.0

    results[mod_name] = {
        'alive': n_alive,
        'total': n_total,
        'mean_max_cosine': mean_max_cos,
        'alive_indices': alive_mask.nonzero(as_tuple=True)[0].tolist(),
    }
    print(f'  {mod_name}: {n_alive}/{n_total} alive, mean max cos={mean_max_cos:.4f}')

print('RESULT_JSON:' + json.dumps(results))
"""


# =============================================================================
# TC worker: compute alive masks + max cosine per layer via subprocess
# =============================================================================

TC_WORKER = r"""
import sys, json, torch, torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/spd')

import wandb

DEVICE = 'cuda'
LAYERS = [0, 1, 2, 3]
BATCH_SIZE = 4
SEQ_LEN = 512
N_TOKENS = int(1e6)
N_BATCHES = N_TOKENS // (BATCH_SIZE * SEQ_LEN)
THRESHOLD = 1e-6

label = sys.argv[1]
project = sys.argv[2]
run_id = sys.argv[3]

from datasets import load_dataset
from tqdm import tqdm
from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig
from analysis.collect_spd_activations import load_spd_model

# Load base model
spd_model, _ = load_spd_model('goodfire/spd/s-55ea3f9b')
spd_model.to(DEVICE)
base_model = spd_model.target_model
base_model.eval()
del spd_model
torch.cuda.empty_cache()

# Load data
dataset = load_dataset('danbraunai/pile-uncopyrighted-tok', split='train', streaming=True)
dataset = dataset.shuffle(seed=0, buffer_size=10000)
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

# Download TC artifacts
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
    for layer_idx in LAYERS:
        if f'layer{layer_idx}_final' in aname:
            dest = Path(f'checkpoints/cos_tc_{run_id}_layer{layer_idx}')
            download_artifact(project, a.name, dest)
            transcoders[layer_idx] = load_tc(dest)
            break

# Count alive and compute max cosine per layer
torch.set_grad_enabled(False)

results = {}
for layer_idx in LAYERS:
    tc = transcoders[layer_idx]
    dict_size = tc.cfg.dict_size
    fire_count = torch.zeros(dict_size, dtype=torch.int64, device=DEVICE)
    n_tokens = 0

    for input_ids in tqdm(batches, desc=f'TC L{layer_idx} alive'):
        captured = {}
        def _hook(_mod, _inp, out, _li=layer_idx):
            captured[_li] = out.detach()
        hook = base_model.h[layer_idx].rms_2.register_forward_hook(_hook)
        base_model(input_ids)
        hook.remove()

        flat = captured[layer_idx].reshape(-1, tc.cfg.input_size)
        acts = tc.encode(flat)
        fire_count += (acts > 0).sum(dim=0).to(torch.int64)
        n_tokens += flat.shape[0]

    sparsity = fire_count.float() / n_tokens
    alive_mask = sparsity > THRESHOLD
    n_alive = alive_mask.sum().item()

    # Max cosine on decoder vectors
    W_dec = F.normalize(tc.W_dec.float(), dim=1)
    W_dec_alive = W_dec[alive_mask].to(DEVICE)

    if n_alive > 1:
        cosine = W_dec_alive @ W_dec_alive.T
        cosine.fill_diagonal_(-float('inf'))
        max_cos = cosine.max(dim=1).values.cpu().numpy()
        mean_max_cos = float(max_cos.mean())
    else:
        mean_max_cos = 0.0

    # Store for both c_fc and down_proj module names (TC maps input->output for one layer)
    # TC decoder is in output space, corresponding to down_proj
    mod_name = f'h.{layer_idx}.mlp.down_proj'
    results[mod_name] = {
        'alive': n_alive,
        'total': dict_size,
        'mean_max_cosine': mean_max_cos,
    }
    print(f'  {mod_name}: {n_alive}/{dict_size} alive, mean max cos={mean_max_cos:.4f}')

print('RESULT_JSON:' + json.dumps(results))
"""


# =============================================================================
# CLT worker
# =============================================================================

CLT_WORKER = r"""
import sys, json, torch, torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/spd')

import wandb

DEVICE = 'cuda'
LAYERS = [0, 1, 2, 3]
BATCH_SIZE = 4
SEQ_LEN = 512
N_TOKENS = int(1e6)
N_BATCHES = N_TOKENS // (BATCH_SIZE * SEQ_LEN)
THRESHOLD = 1e-6

label = sys.argv[1]
project = sys.argv[2]
run_id = sys.argv[3]

from datasets import load_dataset
from tqdm import tqdm
from nn_decompositions.clt import CrossLayerTranscoder
from nn_decompositions.config import CLTConfig
from analysis.collect_spd_activations import load_spd_model

# Load base model
spd_model, _ = load_spd_model('goodfire/spd/s-55ea3f9b')
spd_model.to(DEVICE)
base_model = spd_model.target_model
base_model.eval()
del spd_model
torch.cuda.empty_cache()

# Load data
dataset = load_dataset('danbraunai/pile-uncopyrighted-tok', split='train', streaming=True)
dataset = dataset.shuffle(seed=0, buffer_size=10000)
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

# Download CLT artifact
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
dest = Path(f'checkpoints/cos_clt_{run_id}')
download_artifact(project, final_arts[0].name, dest)
clt = load_clt(dest)

# Count alive and compute max cosine per source layer
torch.set_grad_enabled(False)

results = {}
n_layers = clt.cfg.n_layers
dict_size = clt.cfg.dict_size

for i in range(n_layers):
    fire_count = torch.zeros(dict_size, dtype=torch.int64, device=DEVICE)
    n_tokens = 0

    for input_ids in tqdm(batches, desc=f'CLT L{i} alive'):
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
        fire_count += (acts > 0).sum(dim=0).to(torch.int64)
        n_tokens += flat.shape[0]

    sparsity = fire_count.float() / n_tokens
    alive_mask = sparsity > THRESHOLD
    n_alive = alive_mask.sum().item()

    # Same-layer decoder vectors: W_dec[i][0] has shape (dict_size, output_size)
    W_dec_layer = F.normalize(clt.W_dec[i][0].float(), dim=1)
    W_dec_alive = W_dec_layer[alive_mask].to(DEVICE)

    if n_alive > 1:
        cosine = W_dec_alive @ W_dec_alive.T
        cosine.fill_diagonal_(-float('inf'))
        max_cos = cosine.max(dim=1).values.cpu().numpy()
        mean_max_cos = float(max_cos.mean())
    else:
        mean_max_cos = 0.0

    mod_name = f'h.{i}.mlp.down_proj'
    results[mod_name] = {
        'alive': n_alive,
        'total': dict_size,
        'mean_max_cosine': mean_max_cos,
    }
    print(f'  {mod_name}: {n_alive}/{dict_size} alive, mean max cos={mean_max_cos:.4f}')

print('RESULT_JSON:' + json.dumps(results))
"""


# =============================================================================
# Runner
# =============================================================================


def run_worker(script: str, label: str, *extra_args) -> dict:
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cmd = [sys.executable, "-c", script, label] + list(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    print(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr, end="")
    for line in result.stdout.split("\n"):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(
        f"No result from {label}:\n{result.stdout[-500:]}\n{result.stderr[-500:]}"
    )


def compute_capacity():
    """Compute max cosine for all capacity models (SPD 0.5x-4x, TC 4k/32k, CLT 4k/32k)."""
    data_file = OUTPUT_DIR / "cosine_capacity.json"
    if data_file.exists():
        with open(data_file) as f:
            all_data = json.load(f)
    else:
        all_data = {}

    # SPD capacity variants
    for cap_label, run_path in SPD_CAPACITY_RUNS.items():
        key = f"spd_{cap_label}"
        if key in all_data:
            print(f"  Skipping {key} (cached)")
            continue
        print(f"\n{'='*60}\n  SPD {cap_label}\n{'='*60}")
        result = run_worker(SPD_WORKER, cap_label, run_path)
        all_data[key] = result
        with open(data_file, "w") as f:
            json.dump(all_data, f, indent=2)

    # TC
    for dict_size, run_info in TC_RUNS.items():
        key = f"tc_{dict_size}"
        if key in all_data:
            print(f"  Skipping {key} (cached)")
            continue
        print(f"\n{'='*60}\n  TC {dict_size}\n{'='*60}")
        result = run_worker(TC_WORKER, f"tc_{dict_size}", run_info["project"], run_info["run_id"])
        all_data[key] = result
        with open(data_file, "w") as f:
            json.dump(all_data, f, indent=2)

    # CLT
    for dict_size, run_info in CLT_RUNS.items():
        key = f"clt_{dict_size}"
        if key in all_data:
            print(f"  Skipping {key} (cached)")
            continue
        print(f"\n{'='*60}\n  CLT {dict_size}\n{'='*60}")
        result = run_worker(CLT_WORKER, f"clt_{dict_size}", run_info["project"], run_info["run_id"])
        all_data[key] = result
        with open(data_file, "w") as f:
            json.dump(all_data, f, indent=2)

    return all_data


def compute_beta():
    """Compute max cosine for SPD beta variants."""
    data_file = OUTPUT_DIR / "cosine_beta.json"
    if data_file.exists():
        with open(data_file) as f:
            all_data = json.load(f)
    else:
        all_data = {}

    for beta_label, run_path in SPD_BETA_RUNS.items():
        key = beta_label
        if key in all_data:
            print(f"  Skipping {key} (cached)")
            continue
        print(f"\n{'='*60}\n  SPD {beta_label}\n{'='*60}")
        result = run_worker(SPD_WORKER, beta_label, run_path)
        all_data[key] = result
        with open(data_file, "w") as f:
            json.dump(all_data, f, indent=2)

    return all_data


# =============================================================================
# Plotting
# =============================================================================


def _mean_max_cos_total(model_data: dict) -> float:
    """Average mean_max_cosine across all modules in model_data."""
    vals = [v["mean_max_cosine"] for v in model_data.values()]
    return sum(vals) / len(vals) if vals else 0.0


def plot_capacity(data: dict | None = None):
    if data is None:
        with open(OUTPUT_DIR / "cosine_capacity.json") as f:
            data = json.load(f)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    # SPD points
    spd_caps = ["0.5x", "1x", "2x", "4x"]
    spd_totals = []
    spd_cos = []
    for cap in spd_caps:
        key = f"spd_{cap}"
        if key not in data:
            continue
        total = sum(v["total"] for v in data[key].values())
        cos = _mean_max_cos_total(data[key])
        spd_totals.append(total)
        spd_cos.append(cos)
    ax.plot(spd_totals, spd_cos, "o-", color="#7b3294", markersize=7, lw=2, label="VPD", zorder=3)

    # TC points
    tc_totals = []
    tc_cos = []
    for ds in sorted(TC_RUNS.keys()):
        key = f"tc_{ds}"
        if key not in data:
            continue
        total = sum(v["total"] for v in data[key].values())
        # TC has 4 layers × dict_size total features
        total = ds * len(LAYERS)
        cos = _mean_max_cos_total(data[key])
        tc_totals.append(total)
        tc_cos.append(cos)
    ax.plot(tc_totals, tc_cos, "s-", color="#e66101", markersize=7, lw=2, label="PLT (k=16)", zorder=3)

    # CLT points
    clt_totals = []
    clt_cos = []
    for ds in sorted(CLT_RUNS.keys()):
        key = f"clt_{ds}"
        if key not in data:
            continue
        total = ds * len(LAYERS)
        cos = _mean_max_cos_total(data[key])
        clt_totals.append(total)
        clt_cos.append(cos)
    ax.plot(clt_totals, clt_cos, "^-", color="#1b9e77", markersize=7, lw=2, label="CLT (k=16)", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Total component capacity")
    ax.set_ylabel("Mean max cosine similarity")
    ax.set_title("Feature Splitting: Cosine Similarity vs. Capacity")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, linewidth=0.5, which="both")

    fig.tight_layout()
    save_path = OUTPUT_DIR / "cosine_splitting_capacity.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Capacity plot saved to {save_path}")


def plot_beta(data: dict | None = None):
    if data is None:
        with open(OUTPUT_DIR / "cosine_beta.json") as f:
            data = json.load(f)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    beta_labels = ["beta=0.0", "beta=0.25", "beta=0.5", "beta=1.0"]
    beta_labels = [b for b in beta_labels if b in data]
    beta_vals = [float(b.split("=")[1]) for b in beta_labels]

    layer_colors = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#984ea3"}

    fig, ax = plt.subplots(figsize=(7, 5))

    # Per-layer lines
    for layer in LAYERS:
        cos_vals = []
        for bl in beta_labels:
            # Average c_fc and down_proj for this layer
            mods = [f"h.{layer}.mlp.c_fc", f"h.{layer}.mlp.down_proj"]
            vals = [data[bl][m]["mean_max_cosine"] for m in mods if m in data[bl]]
            cos_vals.append(sum(vals) / len(vals) if vals else 0.0)
        ax.plot(beta_vals, cos_vals, "o-", color=layer_colors[layer],
                markersize=7, lw=2, label=f"Layer {layer}", zorder=3)

    # Total line
    total_cos = [_mean_max_cos_total(data[bl]) for bl in beta_labels]
    ax.plot(beta_vals, total_cos, "s-", color="black", markersize=7, lw=2,
            label="Total", zorder=3)

    ax.set_xlabel("ImpMin beta")
    ax.set_ylabel("Mean max cosine similarity")
    ax.set_title("Feature Splitting: Cosine Similarity vs. Frequency Minimality")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, linewidth=0.5)

    fig.tight_layout()
    save_path = OUTPUT_DIR / "cosine_splitting_beta.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Beta plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--step", choices=["capacity", "beta"], default=None,
                        help="Run only one step")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        plot_capacity()
        plot_beta()
        return

    if args.step is None or args.step == "capacity":
        cap_data = compute_capacity()
        plot_capacity(cap_data)

    if args.step is None or args.step == "beta":
        beta_data = compute_beta()
        plot_beta(beta_data)


if __name__ == "__main__":
    main()
