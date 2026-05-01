"""Comprehensive model comparison table: evaluate ALL models from scratch.

Downloads every model checkpoint, computes all metrics per model:
  - L0 (avg active features per token)
  - CE degradation in 3 eval modes (cascading, parallel, single-MLP)
  - MLP reconstruction MSE
  - Alive feature count (firing rate > 1e-6 over 1M tokens)
  - Mean max cosine similarity between alive decoder vectors

Results are cached incrementally in output/results_cache.json.

Usage:
    python experiments/exp_048_model_table/model_table.py
    python experiments/exp_048_model_table/model_table.py --table-only
"""

import argparse
import json
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig, CLTConfig
from nn_decompositions.clt import CrossLayerTranscoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
OUTPUT_DIR = Path("experiments/exp_048_model_table/output")
CACHE_PATH = OUTPUT_DIR / "results_cache.json"

ALIVE_THRESHOLD = 1e-6
# Eval config
EVAL_BATCH_SIZE = 8
EVAL_SEQ_LEN = 512
N_ALIVE_BATCHES = 245  # ~1M tokens for alive counting
N_CE_BATCHES = 50      # ~200k tokens for CE/MSE eval

# ═══════════════════════════════════════════════════════════════════════
# All model definitions
# ═══════════════════════════════════════════════════════════════════════

LOCAL_MSE_RUNS = {
    # 4k dict — project: mats-sprint/pile_local_sweep_jose
    ("PLT", "4k", "local_mse", 8):  {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "1fgdzkza", "type": "tc"},
    ("PLT", "4k", "local_mse", 16): {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "4ziu27fn", "type": "tc"},
    ("PLT", "4k", "local_mse", 32): {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "r7jmo7tn", "type": "tc"},
    ("PLT", "4k", "local_mse", 64): {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "ms2gzfro", "type": "tc"},
    ("CLT", "4k", "local_mse", 8):  {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "kn2wny4z", "type": "clt"},
    ("CLT", "4k", "local_mse", 16): {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "77sgz1pe", "type": "clt"},
    ("CLT", "4k", "local_mse", 32): {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "hxyj1pdx", "type": "clt"},
    ("CLT", "4k", "local_mse", 64): {"project": "mats-sprint/pile_local_sweep_jose", "run_id": "8g87bvon", "type": "clt"},
    # 32k dict — project: mats-sprint/pile_local_sweep_jose_32k
    ("PLT", "32k", "local_mse", 8):  {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "bu4on2g8", "type": "tc"},
    ("PLT", "32k", "local_mse", 16): {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "c4o8i98k", "type": "tc"},
    ("PLT", "32k", "local_mse", 32): {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "73cidp53", "type": "tc"},
    # tc_k64 32k still running — skip
    ("CLT", "32k", "local_mse", 8):  {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "ty1ofzrw", "type": "clt"},
    ("CLT", "32k", "local_mse", 16): {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "j20m9hzr", "type": "clt"},
    ("CLT", "32k", "local_mse", 32): {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "tzsnndn7", "type": "clt"},
    ("CLT", "32k", "local_mse", 64): {"project": "mats-sprint/pile_local_sweep_jose_32k", "run_id": "ywqw69cj", "type": "clt"},
}

E2E_4K_RUNS = {
    # TC cascading
    ("PLT", "4k", "cascading", 8):    {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "wbz3ud8u", "type": "tc"},
    ("PLT", "4k", "cascading", 16):   {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "bq2n1t9m", "type": "tc"},
    ("PLT", "4k", "cascading", 32):   {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "7qblz2fn", "type": "tc"},
    ("PLT", "4k", "cascading", 64):   {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "v4gyqbrd", "type": "tc"},
    # TC parallel
    ("PLT", "4k", "parallel", 8):     {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "u8bz4xov", "type": "tc"},
    ("PLT", "4k", "parallel", 16):    {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "8sq5enhf", "type": "tc"},
    ("PLT", "4k", "parallel", 32):    {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "6d6yme3j", "type": "tc"},
    ("PLT", "4k", "parallel", 64):    {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "cgbuoclg", "type": "tc"},
    # TC independent
    ("PLT", "4k", "independent", 8):  {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "4dl1b5zq", "type": "tc"},
    ("PLT", "4k", "independent", 16): {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "rjhk4uat", "type": "tc"},
    ("PLT", "4k", "independent", 32): {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "9v7q1zc6", "type": "tc"},
    ("PLT", "4k", "independent", 64): {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "n53887yb", "type": "tc"},
    # CLT cascading
    ("CLT", "4k", "cascading", 8):    {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "2t7c7oml", "type": "clt"},
    ("CLT", "4k", "cascading", 16):   {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "0lvptasy", "type": "clt"},
    ("CLT", "4k", "cascading", 32):   {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "75w3puee", "type": "clt"},
    ("CLT", "4k", "cascading", 64):   {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "g65tcjl6", "type": "clt"},
    # CLT parallel
    ("CLT", "4k", "parallel", 8):     {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "p8c3mqql", "type": "clt"},
    ("CLT", "4k", "parallel", 16):    {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "k88ilqfu", "type": "clt"},
    ("CLT", "4k", "parallel", 32):    {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "oiikseki", "type": "clt"},
    ("CLT", "4k", "parallel", 64):    {"project": "mats-sprint/pile_e2e_sweep_jose", "run_id": "fz4b8yx0", "type": "clt"},
}

E2E_32K_RUNS = {
    # TC cascading (k32, k64 still running)
    ("PLT", "32k", "cascading", 8):    {"project": "mats-sprint/pile_e2e_sweep_jose_32k", "run_id": "3fqbtuq7", "type": "tc"},
    ("PLT", "32k", "cascading", 16):   {"project": "mats-sprint/pile_e2e_sweep_jose_32k", "run_id": "i8ij10e5", "type": "tc"},
    # TC parallel (k64 still running)
    ("PLT", "32k", "parallel", 8):     {"project": "mats-sprint/pile_e2e_sweep_jose_32k", "run_id": "1vrbgs47", "type": "tc"},
    ("PLT", "32k", "parallel", 16):    {"project": "mats-sprint/pile_e2e_sweep_jose_32k", "run_id": "p1q596pr", "type": "tc"},
    ("PLT", "32k", "parallel", 32):    {"project": "mats-sprint/pile_e2e_sweep_jose_32k", "run_id": "g5cleafb", "type": "tc"},
    # TC independent (k64 still running)
    ("PLT", "32k", "independent", 8):  {"project": "mats-sprint/pile_e2e_sweep_jose_32k", "run_id": "l8asugzq", "type": "tc"},
    ("PLT", "32k", "independent", 16): {"project": "mats-sprint/pile_e2e_sweep_jose_32k", "run_id": "1c2mu991", "type": "tc"},
    ("PLT", "32k", "independent", 32): {"project": "mats-sprint/pile_e2e_sweep_jose_32k", "run_id": "dr5q43yq", "type": "tc"},
    # CLT cascading (k16+ still running)
    ("CLT", "32k", "cascading", 8):    {"project": "mats-sprint/pile_e2e_sweep_jose_32k", "run_id": "2uzec2cd", "type": "clt"},
    # CLT parallel (k32, k64 still running)
    ("CLT", "32k", "parallel", 8):     {"project": "mats-sprint/pile_e2e_sweep_jose_32k", "run_id": "f7yjyafh", "type": "clt"},
    ("CLT", "32k", "parallel", 16):    {"project": "mats-sprint/pile_e2e_sweep_jose_32k", "run_id": "e9pih3sr", "type": "clt"},
}

ALL_RUNS = {**LOCAL_MSE_RUNS, **E2E_4K_RUNS, **E2E_32K_RUNS}


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

@contextmanager
def patched_forward(module, new_forward):
    orig = module.forward
    module.forward = new_forward
    try:
        yield
    finally:
        module.forward = orig


def compute_ce_loss(model, input_ids):
    logits = model(input_ids)
    if hasattr(logits, "logits"):
        logits = logits.logits
    if isinstance(logits, tuple):
        logits = logits[0]
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
        input_ids[:, 1:].contiguous().view(-1),
    ).item()


def collect_rms2(model, input_ids):
    captured = {}
    hooks = []
    for li in LAYERS:
        def _make(i):
            def _hook(_m, _inp, out):
                captured[i] = out.detach()
            return _hook
        hooks.append(model.h[li].rms_2.register_forward_hook(_make(li)))
    model(input_ids)
    for h in hooks:
        h.remove()
    return captured


def download_artifact(project, art_name, dest):
    dest = Path(dest)
    if dest.exists() and (dest / "encoder.pt").exists():
        return dest
    api = wandb.Api()
    artifact = api.artifact(f"{project}/{art_name}")
    artifact.download(root=str(dest))
    return dest


def load_transcoder(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["dtype"] = getattr(torch, cfg_dict.get("dtype", "torch.float32").replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    tc = BatchTopKTranscoder(cfg)
    tc.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    tc.eval()
    return tc


def load_clt(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["layers"] = json.loads(cfg_dict["layers"])
    cfg_dict["dtype"] = getattr(torch, cfg_dict.get("dtype", "torch.float32").replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = CLTConfig(**cfg_dict)
    clt = CrossLayerTranscoder(cfg)
    clt.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    clt.eval()
    return clt


def get_eval_batches(n_batches):
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)
    batches = []
    for _ in tqdm(range(n_batches), desc="Loading batches"):
        batch_ids = []
        for _ in range(EVAL_BATCH_SIZE):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:EVAL_SEQ_LEN])
        batches.append(torch.stack(batch_ids).to(DEVICE))
    return batches


# ═══════════════════════════════════════════════════════════════════════
# Download model checkpoints
# ═══════════════════════════════════════════════════════════════════════

def download_tc_artifacts(run_info):
    api = wandb.Api()
    run = api.run(f"{run_info['project']}/runs/{run_info['run_id']}")
    arts = [a for a in run.logged_artifacts() if a.type == "model"]
    layer_paths = {}
    for a in arts:
        aname = a.name.split(":")[0]
        for li in LAYERS:
            if f"layer{li}_final" in aname:
                dest = Path(f"checkpoints/table_{run_info['run_id']}_layer{li}")
                download_artifact(run_info["project"], a.name, dest)
                layer_paths[li] = dest
                break
    assert set(layer_paths.keys()) == set(LAYERS), f"Missing layers for {run_info['run_id']}: got {set(layer_paths.keys())}"
    return layer_paths


def download_clt_artifact(run_info):
    api = wandb.Api()
    run = api.run(f"{run_info['project']}/runs/{run_info['run_id']}")
    arts = [a for a in run.logged_artifacts() if a.type == "model"]
    final_arts = [a for a in arts if "final" in a.name]
    assert len(final_arts) == 1, f"Expected 1 final artifact for {run_info['run_id']}, got {len(final_arts)}"
    dest = Path(f"checkpoints/table_clt_{run_info['run_id']}")
    download_artifact(run_info["project"], final_arts[0].name, dest)
    return dest


# ═══════════════════════════════════════════════════════════════════════
# Transcoder evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_tc(transcoders, base_model, batches, alive_batches):
    """Evaluate transcoders: CE in 3 modes, MSE, alive count, cosine similarity."""
    dict_size = next(iter(transcoders.values())).cfg.dict_size

    # --- Alive counting (use all alive_batches) ---
    fire_count = {l: torch.zeros(dict_size, dtype=torch.int64, device=DEVICE) for l in LAYERS}
    n_alive_tokens = 0
    for input_ids in tqdm(alive_batches, desc="  alive count"):
        rms = collect_rms2(base_model, input_ids)
        B, S = input_ids.shape
        n_alive_tokens += B * S
        for li in LAYERS:
            tc = transcoders[li]
            flat = rms[li].reshape(-1, tc.cfg.input_size)
            acts = tc.encode(flat)
            fire_count[li] += (acts > 0).sum(dim=0).to(torch.int64)

    total_features = dict_size * len(LAYERS)
    alive_count = 0
    alive_masks = {}
    for li in LAYERS:
        sparsity = fire_count[li].float() / n_alive_tokens
        mask = sparsity > ALIVE_THRESHOLD
        alive_masks[li] = mask
        alive_count += mask.sum().item()

    # --- Mean max cosine on decoder vectors of alive features ---
    cos_per_layer = []
    for li in LAYERS:
        tc = transcoders[li]
        W_dec = F.normalize(tc.W_dec.float(), dim=1)
        W_alive = W_dec[alive_masks[li]]
        n_alive = W_alive.shape[0]
        if n_alive > 1:
            cosine = W_alive @ W_alive.T
            cosine.fill_diagonal_(-float("inf"))
            cos_per_layer.append(cosine.max(dim=1).values.mean().item())
        else:
            cos_per_layer.append(0.0)
    mean_max_cosine = sum(cos_per_layer) / len(cos_per_layer)

    # --- CE eval in 3 modes + MSE (use first N_CE_BATCHES) ---
    ce_batches = batches[:N_CE_BATCHES]

    # Precompute MLP ground truth outputs
    mlp_outputs = {l: [] for l in LAYERS}
    for input_ids in ce_batches:
        rms = collect_rms2(base_model, input_ids)
        for li in LAYERS:
            mlp_out = base_model.h[li].mlp(rms[li])
            mlp_outputs[li].append(mlp_out.detach())

    # CE cascading: all MLPs patched, errors propagate
    ce_casc = 0.0
    total_mse = 0.0
    l0_sum = 0.0
    for bi, input_ids in enumerate(ce_batches):
        rms = collect_rms2(base_model, input_ids)
        recons = {}
        batch_l0 = 0.0
        for li in LAYERS:
            tc = transcoders[li]
            flat = rms[li].reshape(-1, tc.cfg.input_size)
            acts = tc.encode(flat)
            recons[li] = tc.decode(acts).reshape(rms[li].shape)
            batch_l0 += (acts > 0).float().sum(-1).mean().item()
        l0_sum += batch_l0 / len(LAYERS)

        with ExitStack() as stack:
            for li in LAYERS:
                r = recons[li]
                def _make(r_):
                    def _p(x): return r_
                    return _p
                stack.enter_context(patched_forward(base_model.h[li].mlp, _make(r)))
            ce_casc += compute_ce_loss(base_model, input_ids)

        batch_mse = 0.0
        for li in LAYERS:
            batch_mse += F.mse_loss(recons[li], mlp_outputs[li][bi]).item()
        total_mse += batch_mse / len(LAYERS)

    # CE parallel: all MLPs patched with clean-input reconstructions
    ce_par = 0.0
    for bi, input_ids in enumerate(ce_batches):
        rms = collect_rms2(base_model, input_ids)
        recons = {}
        for li in LAYERS:
            tc = transcoders[li]
            flat = rms[li].reshape(-1, tc.cfg.input_size)
            recons[li] = tc.decode(tc.encode(flat)).reshape(rms[li].shape)
        with ExitStack() as stack:
            for li in LAYERS:
                r = recons[li]
                def _make(r_):
                    def _p(x): return r_
                    return _p
                stack.enter_context(patched_forward(base_model.h[li].mlp, _make(r)))
            ce_par += compute_ce_loss(base_model, input_ids)

    # CE single: one MLP at a time, average over layers
    ce_single = 0.0
    for li in LAYERS:
        layer_ce = 0.0
        for bi, input_ids in enumerate(ce_batches):
            rms = collect_rms2(base_model, input_ids)
            tc = transcoders[li]
            flat = rms[li].reshape(-1, tc.cfg.input_size)
            recon = tc.decode(tc.encode(flat)).reshape(rms[li].shape)
            def _make(r_):
                def _p(x): return r_
                return _p
            with patched_forward(base_model.h[li].mlp, _make(recon)):
                layer_ce += compute_ce_loss(base_model, input_ids)
        ce_single += layer_ce / len(ce_batches)
    ce_single /= len(LAYERS)

    n = len(ce_batches)
    return {
        "l0": l0_sum / n,
        "ce_cascading": ce_casc / n,
        "ce_parallel": ce_par / n,
        "ce_single": ce_single,
        "mse": total_mse / n,
        "total_components": total_features,
        "alive_components": alive_count,
        "alive_pct": 100.0 * alive_count / total_features,
        "mean_max_cosine": mean_max_cosine,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLT evaluation
# ═══════════════════════════════════════════════════════════════════════

def _clt_encode(clt, clt_inputs, k):
    all_acts = []
    for i in range(clt.cfg.n_layers):
        pre = F.relu(clt_inputs[i] @ clt.W_enc[i] + clt.b_enc[i])
        n_keep = k * pre.shape[0]
        if n_keep < pre.numel():
            topk = torch.topk(pre.flatten(), n_keep)
            a = torch.zeros_like(pre.flatten()).scatter(-1, topk.indices, topk.values).reshape(pre.shape)
        else:
            a = pre
        all_acts.append(a)
    return all_acts


@torch.no_grad()
def eval_clt(clt, base_model, batches, alive_batches):
    """Evaluate CLT: CE in 3 modes, MSE, alive count, cosine similarity."""
    dict_size = clt.cfg.dict_size
    n_layers = clt.cfg.n_layers
    k = clt.cfg.top_k

    # --- Alive counting ---
    fire_count = {i: torch.zeros(dict_size, dtype=torch.int64, device=DEVICE) for i in range(n_layers)}
    n_alive_tokens = 0
    for input_ids in tqdm(alive_batches, desc="  alive count"):
        rms = collect_rms2(base_model, input_ids)
        B, S = input_ids.shape
        n_alive_tokens += B * S
        clt_inputs = [rms[l].reshape(-1, clt.cfg.input_size) for l in LAYERS]
        all_acts = _clt_encode(clt, clt_inputs, k)
        for i, a in enumerate(all_acts):
            fire_count[i] += (a > 0).sum(dim=0).to(torch.int64)

    total_features = dict_size * n_layers
    alive_count = 0
    alive_masks = {}
    for i in range(n_layers):
        sparsity = fire_count[i].float() / n_alive_tokens
        mask = sparsity > ALIVE_THRESHOLD
        alive_masks[i] = mask
        alive_count += mask.sum().item()

    # --- Mean max cosine ---
    cos_per_layer = []
    for i in range(n_layers):
        W_dec = F.normalize(clt.W_dec[i][0].float(), dim=1)
        W_alive = W_dec[alive_masks[i]]
        n_alive = W_alive.shape[0]
        if n_alive > 1:
            cosine = W_alive @ W_alive.T
            cosine.fill_diagonal_(-float("inf"))
            cos_per_layer.append(cosine.max(dim=1).values.mean().item())
        else:
            cos_per_layer.append(0.0)
    mean_max_cosine = sum(cos_per_layer) / len(cos_per_layer)

    # --- CE eval + MSE ---
    ce_batches = batches[:N_CE_BATCHES]

    mlp_outputs = {l: [] for l in LAYERS}
    for input_ids in ce_batches:
        rms = collect_rms2(base_model, input_ids)
        for li in LAYERS:
            mlp_outputs[li].append(base_model.h[li].mlp(rms[li]).detach())

    # CE cascading
    ce_casc = 0.0
    total_mse = 0.0
    l0_sum = 0.0
    for bi, input_ids in enumerate(ce_batches):
        rms = collect_rms2(base_model, input_ids)
        clt_inputs = [rms[l].reshape(-1, clt.cfg.input_size) for l in LAYERS]
        all_acts = _clt_encode(clt, clt_inputs, k)
        recons = clt.decode(all_acts)
        recons_shaped = [r.reshape(rms[LAYERS[0]].shape) for r in recons]

        for i, a in enumerate(all_acts):
            l0_sum += (a > 0).float().sum(-1).mean().item() / n_layers

        with ExitStack() as stack:
            for i, li in enumerate(LAYERS):
                r = recons_shaped[i]
                def _make(r_):
                    def _p(x): return r_
                    return _p
                stack.enter_context(patched_forward(base_model.h[li].mlp, _make(r)))
            ce_casc += compute_ce_loss(base_model, input_ids)

        batch_mse = sum(F.mse_loss(recons[i], mlp_outputs[LAYERS[i]][bi].reshape(-1, mlp_outputs[LAYERS[i]][bi].shape[-1])).item() for i in range(n_layers)) / n_layers
        total_mse += batch_mse

    # CE parallel
    ce_par = 0.0
    for bi, input_ids in enumerate(ce_batches):
        rms = collect_rms2(base_model, input_ids)
        clt_inputs = [rms[l].reshape(-1, clt.cfg.input_size) for l in LAYERS]
        all_acts = _clt_encode(clt, clt_inputs, k)
        recons = clt.decode(all_acts)
        recons_shaped = [r.reshape(rms[LAYERS[0]].shape) for r in recons]
        with ExitStack() as stack:
            for i, li in enumerate(LAYERS):
                r = recons_shaped[i]
                def _make(r_):
                    def _p(x): return r_
                    return _p
                stack.enter_context(patched_forward(base_model.h[li].mlp, _make(r)))
            ce_par += compute_ce_loss(base_model, input_ids)

    # CE single
    ce_single = 0.0
    for target_layer_idx, target_layer in enumerate(LAYERS):
        layer_ce = 0.0
        for bi, input_ids in enumerate(ce_batches):
            rms = collect_rms2(base_model, input_ids)
            clt_inputs = [rms[l].reshape(-1, clt.cfg.input_size) for l in LAYERS]
            all_acts = _clt_encode(clt, clt_inputs, k)
            recons = clt.decode(all_acts)
            recon = recons[target_layer_idx].reshape(rms[LAYERS[0]].shape)
            def _make(r_):
                def _p(x): return r_
                return _p
            with patched_forward(base_model.h[target_layer].mlp, _make(recon)):
                layer_ce += compute_ce_loss(base_model, input_ids)
        ce_single += layer_ce / len(ce_batches)
    ce_single /= len(LAYERS)

    n = len(ce_batches)
    return {
        "l0": l0_sum / n,
        "ce_cascading": ce_casc / n,
        "ce_parallel": ce_par / n,
        "ce_single": ce_single,
        "mse": total_mse / n,
        "total_components": total_features,
        "alive_components": alive_count,
        "alive_pct": 100.0 * alive_count / total_features,
        "mean_max_cosine": mean_max_cosine,
    }


# ═══════════════════════════════════════════════════════════════════════
# VPD evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_vpd(spd_model, batches, threshold):
    from spd.models.components import make_mask_infos

    module_names = [n for n in spd_model.module_to_c if "mlp" in n]
    ce_batches = batches[:N_CE_BATCHES]

    # Alive counting (use all batches)
    # A component "fires" on a token if its CI > the operating threshold.
    # It is "alive" if it fires on > 1e-6 fraction of tokens.
    fire_count = {}
    for mod_name in module_names:
        n_c = spd_model.module_to_c[mod_name]
        fire_count[mod_name] = torch.zeros(n_c, dtype=torch.int64, device="cpu")
    n_tokens = 0

    for input_ids in tqdm(batches, desc=f"  VPD CI>{threshold}"):
        B, S = input_ids.shape
        n_tokens += B * S
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        for mod_name in module_names:
            ci_vals = ci.lower_leaky[mod_name].reshape(-1, spd_model.module_to_c[mod_name])
            fire_count[mod_name] += (ci_vals > threshold).sum(dim=0).cpu().to(torch.int64)

    total_components = 0
    alive_components = 0
    alive_masks = {}
    cos_vals = []
    for mod_name in module_names:
        firing_rate = fire_count[mod_name].float() / n_tokens
        mask = firing_rate > ALIVE_THRESHOLD
        n_total = spd_model.module_to_c[mod_name]
        n_alive = mask.sum().item()
        total_components += n_total
        alive_components += n_alive
        alive_masks[mod_name] = mask

        U = spd_model.components[mod_name].U.float()
        U_alive = F.normalize(U[mask], dim=1).to(DEVICE)
        if n_alive > 1:
            cosine = U_alive @ U_alive.T
            cosine.fill_diagonal_(-float("inf"))
            cos_vals.append(cosine.max(dim=1).values.mean().item())
        else:
            cos_vals.append(0.0)

    mean_max_cosine = sum(cos_vals) / len(cos_vals) if cos_vals else 0.0

    # CE eval
    total_ce_casc = 0.0
    total_ce_par = 0.0
    total_mse = 0.0
    l0_sum = 0.0

    for bi, input_ids in enumerate(tqdm(ce_batches, desc=f"  VPD CE CI>{threshold}")):
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        masks = {}
        for mod_name in module_names:
            ci_post = ci.lower_leaky[mod_name]
            mask = (ci_post > threshold).float()
            masks[mod_name] = mask
            l0_sum += mask.sum(-1).mean().item() / len(module_names)

        mask_infos = make_mask_infos(masks)
        logits = spd_model(input_ids, mask_infos=mask_infos)
        if hasattr(logits, "logits"):
            logits = logits.logits
        total_ce_casc += F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            input_ids[:, 1:].contiguous().view(-1),
        ).item()

        # MSE
        def capture_mlp(model, ids, **kwargs):
            cap = {}
            hooks = []
            for li in LAYERS:
                def _mk(i):
                    def _h(_m, _inp, o): cap[i] = o.detach()
                    return _h
                hooks.append(model.target_model.h[li].mlp.register_forward_hook(_mk(li)))
            model(ids, **kwargs)
            for h in hooks:
                h.remove()
            return cap

        orig = capture_mlp(spd_model, input_ids)
        masked = capture_mlp(spd_model, input_ids, mask_infos=mask_infos)
        total_mse += sum(F.mse_loss(masked[l], orig[l]).item() for l in LAYERS) / len(LAYERS)

    # CE single: one layer at a time
    ce_single = 0.0
    for target_layer in LAYERS:
        target_mods = [m for m in module_names if f"h.{target_layer}." in m]
        layer_ce = 0.0
        for input_ids in ce_batches:
            out = spd_model(input_ids, cache_type="input")
            ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
            masks = {}
            for mod_name in module_names:
                ci_post = ci.lower_leaky[mod_name]
                if mod_name in target_mods:
                    masks[mod_name] = (ci_post > threshold).float()
                else:
                    masks[mod_name] = torch.ones_like(ci_post)
            mask_infos = make_mask_infos(masks)
            logits = spd_model(input_ids, mask_infos=mask_infos)
            if hasattr(logits, "logits"):
                logits = logits.logits
            layer_ce += F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                input_ids[:, 1:].contiguous().view(-1),
            ).item()
        ce_single += layer_ce / len(ce_batches)
    ce_single /= len(LAYERS)

    # CE parallel = same as cascading for VPD (no distinction)
    n = len(ce_batches)
    return {
        "l0": l0_sum / n,
        "ce_cascading": total_ce_casc / n,
        "ce_parallel": total_ce_casc / n,  # VPD doesn't distinguish
        "ce_single": ce_single,
        "mse": total_mse / n,
        "total_components": total_components,
        "alive_components": alive_components,
        "alive_pct": 100.0 * alive_components / total_components,
        "mean_max_cosine": mean_max_cosine,
    }


# ═══════════════════════════════════════════════════════════════════════
# Neuron baseline
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_neurons(base_model, batches, k):
    ce_batches = batches[:N_CE_BATCHES]
    d_hidden = base_model.h[0].mlp.c_fc.weight.shape[0]

    mlp_outputs = {l: [] for l in LAYERS}
    for input_ids in ce_batches:
        rms = collect_rms2(base_model, input_ids)
        for li in LAYERS:
            mlp_outputs[li].append(base_model.h[li].mlp(rms[li]).detach())

    def neuron_recon(mlp, x, k_):
        h = mlp.gelu(mlp.c_fc(x))
        if k_ < h.shape[-1]:
            topk = torch.topk(h.abs(), k_, dim=-1)
            mask = torch.zeros_like(h)
            mask.scatter_(-1, topk.indices, 1.0)
            h = h * mask
        return mlp.down_proj(h)

    ce_casc = 0.0
    total_mse = 0.0
    for bi, input_ids in enumerate(ce_batches):
        with ExitStack() as stack:
            for li in LAYERS:
                mlp = base_model.h[li].mlp
                def _make(m, k_):
                    def _p(x): return neuron_recon(m, x, k_)
                    return _p
                stack.enter_context(patched_forward(mlp, _make(mlp, k)))
            ce_casc += compute_ce_loss(base_model, input_ids)

        rms = collect_rms2(base_model, input_ids)
        batch_mse = 0.0
        for li in LAYERS:
            recon = neuron_recon(base_model.h[li].mlp, rms[li], k)
            batch_mse += F.mse_loss(recon, mlp_outputs[li][bi]).item()
        total_mse += batch_mse / len(LAYERS)

    # CE single
    ce_single = 0.0
    for li in LAYERS:
        layer_ce = 0.0
        for input_ids in ce_batches:
            mlp = base_model.h[li].mlp
            def _make(m, k_):
                def _p(x): return neuron_recon(m, x, k_)
                return _p
            with patched_forward(mlp, _make(mlp, k)):
                layer_ce += compute_ce_loss(base_model, input_ids)
        ce_single += layer_ce / len(ce_batches)
    ce_single /= len(LAYERS)

    n = len(ce_batches)
    return {
        "l0": float(k),
        "ce_cascading": ce_casc / n,
        "ce_parallel": ce_casc / n,
        "ce_single": ce_single,
        "mse": total_mse / n,
        "total_components": d_hidden * len(LAYERS),
        "alive_components": None,
        "alive_pct": None,
        "mean_max_cosine": None,
    }


# ═══════════════════════════════════════════════════════════════════════
# Cache management
# ═══════════════════════════════════════════════════════════════════════

def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def format_table(baseline_ce, cache):
    def fmt(v, decimals=2):
        if v is None:
            return "—"
        return f"{v:.{decimals}f}"

    rows = []
    model_items = [(k, v) for k, v in cache.items() if k != "baseline_ce"]
    for key_str, r in sorted(model_items, key=lambda kv: (
        {"VPD": 0, "PLT": 1, "CLT": 2, "Neurons": 3}.get(kv[1].get("method", ""), 99),
        {"1x": 0, "4k": 1, "32k": 2, "—": 3}.get(kv[1].get("dict_size", ""), 99),
        kv[1].get("training_mode", ""), kv[1].get("top_k") or 0,
    )):
        k_str = str(r["top_k"]) if r.get("top_k") is not None else "—"
        total = f"{r['total_components']:,}" if r.get("total_components") is not None else "—"
        alive = f"{r['alive_components']:,}" if r.get("alive_components") is not None else "—"

        delta_all = r["ce_cascading"] - baseline_ce if r.get("ce_cascading") else None
        delta_single = r["ce_single"] - baseline_ce if r.get("ce_single") else None

        rows.append(
            f"| {r['method']} | {r.get('dict_size', '—')} | {r.get('training_mode', '—')} | {k_str} "
            f"| {fmt(r.get('l0'), 1)} | {fmt(delta_all, 3)} "
            f"| {fmt(delta_single, 3)} | {fmt(r.get('mse'), 4)} "
            f"| {total} | {alive} | {fmt(r.get('alive_pct'), 1)} | {fmt(r.get('mean_max_cosine'), 3)} |"
        )

    lines = [
        f"Baseline CE: {baseline_ce:.4f}. All ΔCE values are CE_patched − CE_baseline (lower is better).\n",
        "| Method | Dict | Training | k | L0 | ΔCE all-MLP | ΔCE single | MSE | Components | Alive | Alive % | Mean max cos |",
        "|--------|------|----------|---:|----|-------------|------------|-----|------------|-------|---------|--------------|",
    ] + rows

    table_md = "\n".join(lines)
    with open(OUTPUT_DIR / "model_table.md", "w") as f:
        f.write(table_md + "\n")
    print(f"Saved {OUTPUT_DIR / 'model_table.md'}")
    return table_md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-only", action="store_true", help="Just format table from cache")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()

    if args.table_only:
        baseline_ce = cache.get("baseline_ce", 2.826)
        table_md = format_table(baseline_ce, cache)
        print(f"\n{table_md}")
        return

    torch.set_grad_enabled(False)

    # Load base model
    from analysis.collect_spd_activations import load_spd_model
    print("Loading base model...")
    spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    # Load data
    n_batches = max(N_ALIVE_BATCHES, N_CE_BATCHES)
    print(f"Loading {n_batches} eval batches...")
    batches = get_eval_batches(n_batches)

    # Baseline CE
    if "baseline_ce" not in cache:
        print("Computing baseline CE...")
        baseline_ce = sum(compute_ce_loss(base_model, b) for b in batches[:N_CE_BATCHES]) / N_CE_BATCHES
        cache["baseline_ce"] = baseline_ce
        save_cache(cache)
        print(f"  baseline_ce = {baseline_ce:.4f}")
    baseline_ce = cache["baseline_ce"]

    # Evaluate all TC/CLT models
    for model_key, run_info in ALL_RUNS.items():
        method, dict_size, training_mode, top_k = model_key
        cache_key = f"{method}_{dict_size}_{training_mode}_k{top_k}"

        if cache_key in cache:
            print(f"Skipping {cache_key} (cached)")
            continue

        print(f"\n{'='*60}")
        print(f"  {method} {dict_size} {training_mode} k={top_k} (run {run_info['run_id']})")
        print(f"{'='*60}")

        try:
            if run_info["type"] == "tc":
                layer_paths = download_tc_artifacts(run_info)
                transcoders = {l: load_transcoder(p) for l, p in layer_paths.items()}
                result = eval_tc(transcoders, base_model, batches, batches[:N_ALIVE_BATCHES])
                del transcoders
            elif run_info["type"] == "clt":
                clt_path = download_clt_artifact(run_info)
                clt = load_clt(clt_path)
                result = eval_clt(clt, base_model, batches, batches[:N_ALIVE_BATCHES])
                del clt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        result["method"] = method
        result["dict_size"] = dict_size
        result["training_mode"] = training_mode
        result["top_k"] = top_k
        cache[cache_key] = result
        save_cache(cache)
        print(f"  L0={result['l0']:.1f}, CE_casc={result['ce_cascading']:.4f}, "
              f"MSE={result['mse']:.6f}, alive={result['alive_components']}/{result['total_components']}, "
              f"cos={result['mean_max_cosine']:.4f}")

    # Evaluate VPD
    for threshold in [0.5, 0.0]:
        label = f"CI>{threshold:g}"
        cache_key = f"VPD_1x_{label}"
        if cache_key in cache:
            print(f"Skipping {cache_key} (cached)")
            continue

        print(f"\n{'='*60}")
        print(f"  VPD {label}")
        print(f"{'='*60}")

        # Need spd_model for VPD eval
        if not hasattr(spd_model, "module_to_c"):
            spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
            spd_model.to(DEVICE)

        result = eval_vpd(spd_model, batches, threshold)
        result["method"] = "VPD"
        result["dict_size"] = "1x"
        result["training_mode"] = label
        result["top_k"] = None
        cache[cache_key] = result
        save_cache(cache)
        print(f"  L0={result['l0']:.1f}, CE={result['ce_cascading']:.4f}, alive={result['alive_components']}/{result['total_components']}")

    # Clean up spd_model
    del spd_model
    torch.cuda.empty_cache()

    # Evaluate neurons
    for k in [8, 16, 32, 64, 128]:
        cache_key = f"Neurons_topk_{k}"
        if cache_key in cache:
            print(f"Skipping {cache_key} (cached)")
            continue

        print(f"\n{'='*60}")
        print(f"  Neurons top-k={k}")
        print(f"{'='*60}")

        result = eval_neurons(base_model, batches, k)
        result["method"] = "Neurons"
        result["dict_size"] = "—"
        result["training_mode"] = "top-k"
        result["top_k"] = k
        cache[cache_key] = result
        save_cache(cache)
        print(f"  L0={k}, CE={result['ce_cascading']:.4f}, MSE={result['mse']:.6f}")

    # Format table
    table_md = format_table(baseline_ce, cache)
    print(f"\n{table_md}")


if __name__ == "__main__":
    main()
