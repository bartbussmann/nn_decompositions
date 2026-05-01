"""Matrix-space cosine splitting.

For each alive feature j we form the rank-1 matrix M_j = enc_j outer dec_j and
compute pairwise cosine similarity in matrix space. By the outer-product
identity, cos(M_i, M_j) = cos(enc_i, enc_j) * cos(dec_i, dec_j), so we compute
encoder-side and decoder-side pairwise cosines and multiply them elementwise.

This mirrors exp_044's capacity sweep (SPD 0.5x/1x/2x/4x, TC 4k/32k, CLT 4k/32k)
but with one extra plot: the matrix-space variant.

Usage:
    python experiments/exp_051_cosine_matrix_splitting/cosine_matrix_splitting.py
    python experiments/exp_051_cosine_matrix_splitting/cosine_matrix_splitting.py --plot-only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path("experiments/exp_051_cosine_matrix_splitting/output")
LAYERS = [0, 1, 2, 3]

SPD_RUNS = {
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


# =============================================================================
# SPD worker — V is (d_in, C), U is (C, d_out); matrix M_j = V[:, j] outer U[j, :]
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

ci_sum = {m: torch.zeros(spd_model.module_to_c[m], dtype=torch.float64, device='cpu') for m in mlp_modules}

torch.set_grad_enabled(False)
n_tokens_total = 0
for input_ids_cpu in tqdm(batches, desc=f'{label} CI'):
    input_ids = input_ids_cpu.to(DEVICE)
    B, S = input_ids.shape
    n_tokens_total += B * S
    out = spd_model(input_ids, cache_type='input')
    ci = spd_model.calc_causal_importances(out.cache, sampling='continuous')
    for m in mlp_modules:
        ci_vals = ci.lower_leaky[m].reshape(-1, spd_model.module_to_c[m])
        ci_sum[m] += ci_vals.double().sum(dim=0).cpu()

results = {}
for m in mlp_modules:
    mean_ci = ci_sum[m] / n_tokens_total
    alive_mask = mean_ci > THRESHOLD
    n_alive = int(alive_mask.sum().item())
    n_total = int(spd_model.module_to_c[m])

    comp = spd_model.components[m]
    V = comp.V.float()  # (d_in, C)
    U = comp.U.float()  # (C, d_out)

    enc = V.T[alive_mask].to(DEVICE)            # (n_alive, d_in)
    dec = U[alive_mask].to(DEVICE)              # (n_alive, d_out)
    enc_n = F.normalize(enc, dim=1)
    dec_n = F.normalize(dec, dim=1)

    if n_alive > 1:
        cos_enc = enc_n @ enc_n.T
        cos_dec = dec_n @ dec_n.T
        cos_mat = cos_enc * cos_dec
        cos_mat.fill_diagonal_(-float('inf'))
        max_cos_mat = float(cos_mat.max(dim=1).values.mean().item())

        cos_enc.fill_diagonal_(-float('inf'))
        max_cos_enc = float(cos_enc.max(dim=1).values.mean().item())

        cos_dec.fill_diagonal_(-float('inf'))
        max_cos_dec = float(cos_dec.max(dim=1).values.mean().item())
    else:
        max_cos_mat = max_cos_enc = max_cos_dec = 0.0

    results[m] = {
        'alive': n_alive,
        'total': n_total,
        'mean_max_cosine_matrix': max_cos_mat,
        'mean_max_cosine_encoder': max_cos_enc,
        'mean_max_cosine_decoder': max_cos_dec,
    }
    print(f'  {m}: {n_alive}/{n_total} alive, mat={max_cos_mat:.4f} enc={max_cos_enc:.4f} dec={max_cos_dec:.4f}')

print('RESULT_JSON:' + json.dumps(results))
"""


# =============================================================================
# TC worker — W_enc (d_in, dict_size), W_dec (dict_size, d_out)
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

spd_model, _ = load_spd_model('goodfire/spd/s-55ea3f9b')
spd_model.to(DEVICE)
base_model = spd_model.target_model
base_model.eval()
del spd_model
torch.cuda.empty_cache()

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
            dest = Path(f'checkpoints/cosmat_tc_{run_id}_layer{layer_idx}')
            download_artifact(project, a.name, dest)
            transcoders[layer_idx] = load_tc(dest)
            break

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
    n_alive = int(alive_mask.sum().item())

    W_enc = tc.W_enc.float()  # (d_in, dict_size)
    W_dec = tc.W_dec.float() if tc.W_dec.shape[0] == dict_size else tc.W_dec.float().T
    # W_dec is (dict_size, d_out)

    enc = W_enc.T[alive_mask].to(DEVICE)
    dec = W_dec[alive_mask].to(DEVICE)
    enc_n = F.normalize(enc, dim=1)
    dec_n = F.normalize(dec, dim=1)

    if n_alive > 1:
        cos_enc = enc_n @ enc_n.T
        cos_dec = dec_n @ dec_n.T
        cos_mat = cos_enc * cos_dec
        cos_mat.fill_diagonal_(-float('inf'))
        max_cos_mat = float(cos_mat.max(dim=1).values.mean().item())

        cos_enc.fill_diagonal_(-float('inf'))
        max_cos_enc = float(cos_enc.max(dim=1).values.mean().item())

        cos_dec.fill_diagonal_(-float('inf'))
        max_cos_dec = float(cos_dec.max(dim=1).values.mean().item())
    else:
        max_cos_mat = max_cos_enc = max_cos_dec = 0.0

    mod_name = f'h.{layer_idx}.mlp.down_proj'
    results[mod_name] = {
        'alive': n_alive,
        'total': dict_size,
        'mean_max_cosine_matrix': max_cos_mat,
        'mean_max_cosine_encoder': max_cos_enc,
        'mean_max_cosine_decoder': max_cos_dec,
    }
    print(f'  {mod_name}: {n_alive}/{dict_size} alive, mat={max_cos_mat:.4f} enc={max_cos_enc:.4f} dec={max_cos_dec:.4f}')

print('RESULT_JSON:' + json.dumps(results))
"""


# =============================================================================
# CLT worker — same-layer: W_enc[i] (d_in, dict_size), W_dec[i][0] (dict_size, d_out)
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

spd_model, _ = load_spd_model('goodfire/spd/s-55ea3f9b')
spd_model.to(DEVICE)
base_model = spd_model.target_model
base_model.eval()
del spd_model
torch.cuda.empty_cache()

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
dest = Path(f'checkpoints/cosmat_clt_{run_id}')
download_artifact(project, final_arts[0].name, dest)
clt = load_clt(dest)

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
    n_alive = int(alive_mask.sum().item())

    W_enc_layer = clt.W_enc[i].float()       # (d_in, dict_size)
    W_dec_layer = clt.W_dec[i][0].float()    # (dict_size, d_out)

    enc = W_enc_layer.T[alive_mask].to(DEVICE)
    dec = W_dec_layer[alive_mask].to(DEVICE)
    enc_n = F.normalize(enc, dim=1)
    dec_n = F.normalize(dec, dim=1)

    if n_alive > 1:
        cos_enc = enc_n @ enc_n.T
        cos_dec = dec_n @ dec_n.T
        cos_mat = cos_enc * cos_dec
        cos_mat.fill_diagonal_(-float('inf'))
        max_cos_mat = float(cos_mat.max(dim=1).values.mean().item())

        cos_enc.fill_diagonal_(-float('inf'))
        max_cos_enc = float(cos_enc.max(dim=1).values.mean().item())

        cos_dec.fill_diagonal_(-float('inf'))
        max_cos_dec = float(cos_dec.max(dim=1).values.mean().item())
    else:
        max_cos_mat = max_cos_enc = max_cos_dec = 0.0

    mod_name = f'h.{i}.mlp.down_proj'
    results[mod_name] = {
        'alive': n_alive,
        'total': dict_size,
        'mean_max_cosine_matrix': max_cos_mat,
        'mean_max_cosine_encoder': max_cos_enc,
        'mean_max_cosine_decoder': max_cos_dec,
    }
    print(f'  {mod_name}: {n_alive}/{dict_size} alive, mat={max_cos_mat:.4f} enc={max_cos_enc:.4f} dec={max_cos_dec:.4f}')

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
    raise RuntimeError(f"No result from {label}:\n{result.stdout[-500:]}\n{result.stderr[-500:]}")


def compute_all() -> dict:
    data_file = OUTPUT_DIR / "cosine_matrix.json"
    all_data = json.loads(data_file.read_text()) if data_file.exists() else {}

    for cap_label, run_path in SPD_RUNS.items():
        key = f"spd_{cap_label}"
        if key in all_data:
            print(f"  Skipping {key} (cached)")
            continue
        print(f"\n{'=' * 60}\n  SPD {cap_label}\n{'=' * 60}")
        all_data[key] = run_worker(SPD_WORKER, cap_label, run_path)
        data_file.write_text(json.dumps(all_data, indent=2))

    for dict_size, run_info in TC_RUNS.items():
        key = f"tc_{dict_size}"
        if key in all_data:
            print(f"  Skipping {key} (cached)")
            continue
        print(f"\n{'=' * 60}\n  TC {dict_size}\n{'=' * 60}")
        all_data[key] = run_worker(TC_WORKER, f"tc_{dict_size}", run_info["project"], run_info["run_id"])
        data_file.write_text(json.dumps(all_data, indent=2))

    for dict_size, run_info in CLT_RUNS.items():
        key = f"clt_{dict_size}"
        if key in all_data:
            print(f"  Skipping {key} (cached)")
            continue
        print(f"\n{'=' * 60}\n  CLT {dict_size}\n{'=' * 60}")
        all_data[key] = run_worker(CLT_WORKER, f"clt_{dict_size}", run_info["project"], run_info["run_id"])
        data_file.write_text(json.dumps(all_data, indent=2))

    return all_data


# =============================================================================
# Plotting
# =============================================================================


def _mean(model_data: dict, key: str) -> float:
    vals = [v[key] for v in model_data.values()]
    return sum(vals) / len(vals) if vals else 0.0


def plot(data: dict | None = None) -> None:
    if data is None:
        with open(OUTPUT_DIR / "cosine_matrix.json") as f:
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

    spd_caps = ["0.5x", "1x", "2x", "4x"]
    spd_totals, spd_cos = [], []
    for cap in spd_caps:
        key = f"spd_{cap}"
        if key not in data:
            continue
        total = sum(v["total"] for v in data[key].values())
        spd_totals.append(total)
        spd_cos.append(_mean(data[key], "mean_max_cosine_matrix"))
    ax.plot(spd_totals, spd_cos, "o-", color="#7b3294", markersize=7, lw=2, label="VPD", zorder=3)

    tc_totals, tc_cos = [], []
    for ds in sorted(TC_RUNS.keys()):
        key = f"tc_{ds}"
        if key not in data:
            continue
        tc_totals.append(ds * len(LAYERS))
        tc_cos.append(_mean(data[key], "mean_max_cosine_matrix"))
    ax.plot(tc_totals, tc_cos, "s-", color="#e66101", markersize=7, lw=2, label="PLT (k=16)", zorder=3)

    clt_totals, clt_cos = [], []
    for ds in sorted(CLT_RUNS.keys()):
        key = f"clt_{ds}"
        if key not in data:
            continue
        clt_totals.append(ds * len(LAYERS))
        clt_cos.append(_mean(data[key], "mean_max_cosine_matrix"))
    ax.plot(clt_totals, clt_cos, "^-", color="#1b9e77", markersize=7, lw=2, label="CLT (k=16)", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Total component capacity")
    ax.set_ylabel("Mean max cosine similarity (matrix space)")
    ax.set_title("Feature Splitting in Matrix Space (encoder-cos × decoder-cos)")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, linewidth=0.5, which="both")

    fig.tight_layout()
    save_path = OUTPUT_DIR / "cosine_matrix_capacity.png"
    pdf_path = OUTPUT_DIR / "cosine_matrix_capacity.pdf"
    fig.savefig(save_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Plot saved to {save_path} and {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot()
        return

    data = compute_all()
    plot(data)


if __name__ == "__main__":
    main()
