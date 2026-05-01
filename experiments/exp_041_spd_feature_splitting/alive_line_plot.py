"""Line plot: total components (x) vs alive components (y) for SPD, TCs, and CLTs.

SPD data comes from the precomputed alive_components.json (0.5x, 1x, 2x, 4x capacity).
TC and CLT alive counts are computed on-the-fly for 4k and 32k dict sizes at k=16.

Usage:
    python experiments/exp_041_spd_feature_splitting/alive_line_plot.py
"""

import json
import sys
from contextlib import ExitStack
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import numpy as np
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
N_TOKENS = int(1e6)
BATCH_SIZE = 8
SEQ_LEN = 512
N_BATCHES = N_TOKENS // (BATCH_SIZE * SEQ_LEN)

OUTPUT_DIR = Path("experiments/exp_041_spd_feature_splitting/output")

# WandB projects and run IDs for local MSE k=16
TC_RUNS = {
    4096: {
        "project": "mats-sprint/pile_local_sweep_jose",
        "run_id": "4ziu27fn",
    },
    32768: {
        "project": "mats-sprint/pile_local_sweep_jose_32k",
        "run_id": "c4o8i98k",
    },
}

CLT_RUNS = {
    4096: {
        "project": "mats-sprint/pile_local_sweep_jose",
        "run_id": "77sgz1pe",
    },
    32768: {
        "project": "mats-sprint/pile_local_sweep_jose_32k",
        "run_id": "j20m9hzr",
    },
}


def download_wandb_artifact(project: str, artifact_name: str, dest: Path) -> Path:
    if dest.exists() and (dest / "encoder.pt").exists():
        print(f"  Using cached {dest}")
        return dest
    api = wandb.Api()
    artifact = api.artifact(f"{project}/{artifact_name}")
    artifact.download(root=str(dest))
    print(f"  Downloaded {artifact_name} -> {dest}")
    return dest


def load_transcoder(checkpoint_dir: Path):
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    encoder = BatchTopKTranscoder(cfg)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    encoder.eval()
    return encoder


def load_clt(checkpoint_dir: Path):
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["layers"] = json.loads(cfg_dict["layers"])
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = CLTConfig(**cfg_dict)
    clt = CrossLayerTranscoder(cfg)
    clt.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    clt.eval()
    return clt


def get_eval_batches(n_batches: int) -> list[torch.Tensor]:
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)
    batches = []
    for _ in tqdm(range(n_batches), desc="Loading batches"):
        batch_ids = []
        for _ in range(BATCH_SIZE):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:SEQ_LEN])
        batches.append(torch.stack(batch_ids).to(DEVICE))
    return batches


def collect_mlp_inputs(base_model, input_ids):
    """Collect post-RMSNorm MLP inputs for each layer."""
    captured = {}
    hooks = []
    for layer_idx in LAYERS:
        rms2 = base_model.h[layer_idx].rms_2

        def _make_hook(li):
            def _hook(_mod, _inp, out):
                captured[li] = out.detach()
            return _hook

        hooks.append(rms2.register_forward_hook(_make_hook(layer_idx)))
    base_model(input_ids)
    for h in hooks:
        h.remove()
    return captured


ALIVE_THRESHOLD = 1e-6  # feature is alive if proportion of tokens where act > 0 exceeds this


@torch.no_grad()
def count_tc_alive(transcoders: dict[int, BatchTopKTranscoder], base_model, batches):
    """Count alive features per layer for transcoders.

    A feature is alive if (n_tokens where act > 0) / n_total_tokens > ALIVE_THRESHOLD.
    Returns (total_features, alive_features) summed over layers.
    """
    dict_size = next(iter(transcoders.values())).cfg.dict_size
    fire_count = {l: torch.zeros(dict_size, dtype=torch.int64, device=DEVICE) for l in LAYERS}
    n_tokens = 0

    for input_ids in tqdm(batches, desc="TC alive count"):
        mlp_inputs = collect_mlp_inputs(base_model, input_ids)
        B, S = input_ids.shape
        n_tokens += B * S
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            flat = mlp_inputs[layer_idx].reshape(-1, tc.cfg.input_size)
            acts = tc.encode(flat)
            # Count tokens where each feature fires (acts has shape [B*S, dict_size])
            fire_count[layer_idx] += (acts > 0).sum(dim=0).to(torch.int64)

    total = dict_size * len(LAYERS)
    alive = 0
    for l in LAYERS:
        sparsity = fire_count[l].float() / n_tokens
        alive += (sparsity > ALIVE_THRESHOLD).sum().item()
    return total, int(alive)


@torch.no_grad()
def count_clt_alive(clt: CrossLayerTranscoder, base_model, batches):
    """Count alive features per encoder layer for CLT.

    A feature is alive if (n_tokens where act > 0) / n_total_tokens > ALIVE_THRESHOLD.
    Returns (total_features, alive_features) summed over layers.
    """
    dict_size = clt.cfg.dict_size
    n_layers = clt.cfg.n_layers
    fire_count = {l: torch.zeros(dict_size, dtype=torch.int64, device=DEVICE) for l in range(n_layers)}
    n_tokens = 0

    for input_ids in tqdm(batches, desc="CLT alive count"):
        mlp_inputs = collect_mlp_inputs(base_model, input_ids)
        B, S = input_ids.shape
        n_tokens += B * S
        for i in range(n_layers):
            flat = mlp_inputs[i].reshape(-1, clt.cfg.input_size)
            pre_acts = F.relu(flat @ clt.W_enc[i] + clt.b_enc[i])
            # Apply BatchTopK
            k = clt.cfg.top_k
            n_keep = k * pre_acts.shape[0]
            if n_keep < pre_acts.numel():
                topk = torch.topk(pre_acts.flatten(), n_keep, dim=-1)
                acts = torch.zeros_like(pre_acts.flatten()).scatter(-1, topk.indices, topk.values).reshape(pre_acts.shape)
            else:
                acts = pre_acts
            fire_count[i] += (acts > 0).sum(dim=0).to(torch.int64)

    total = dict_size * n_layers
    alive = 0
    for l in range(n_layers):
        sparsity = fire_count[l].float() / n_tokens
        alive += (sparsity > ALIVE_THRESHOLD).sum().item()
    return total, int(alive)


def download_tc_artifacts(run_info: dict) -> dict[int, Path]:
    """Download per-layer TC artifacts for a run. Returns {layer: path}."""
    api = wandb.Api()
    run = api.run(f"{run_info['project']}/runs/{run_info['run_id']}")
    arts = [a for a in run.logged_artifacts() if a.type == "model"]

    layer_paths = {}
    for a in arts:
        aname = a.name.split(":")[0]
        for layer_idx in LAYERS:
            if f"layer{layer_idx}_final" in aname:
                dest = Path(f"checkpoints/alive_tc_{run_info['run_id']}_layer{layer_idx}")
                download_wandb_artifact(run_info["project"], a.name, dest)
                layer_paths[layer_idx] = dest
                break

    assert set(layer_paths.keys()) == set(LAYERS), f"Missing layers: got {set(layer_paths.keys())}"
    return layer_paths


def download_clt_artifact(run_info: dict) -> Path:
    """Download CLT artifact for a run."""
    api = wandb.Api()
    run = api.run(f"{run_info['project']}/runs/{run_info['run_id']}")
    arts = [a for a in run.logged_artifacts() if a.type == "model"]
    final_arts = [a for a in arts if "final" in a.name]
    assert len(final_arts) == 1, f"Expected 1 final artifact, got {len(final_arts)}"

    dest = Path(f"checkpoints/alive_clt_{run_info['run_id']}")
    download_wandb_artifact(run_info["project"], final_arts[0].name, dest)
    return dest


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── SPD data (precomputed with mean CI > 1e-6 threshold) ──
    with open(OUTPUT_DIR / "alive_components_mean_ci.json") as f:
        spd_raw = json.load(f)

    spd_points = []  # (total, alive) summed across all layers and both MLP matrices
    for label in ["0.5x", "1x (jose)", "2x", "4x"]:
        total = sum(v["total"] for v in spd_raw[label].values())
        alive = sum(v["alive"] for v in spd_raw[label].values())
        spd_points.append((total, alive))
        print(f"SPD {label}: {alive}/{total} alive ({100*alive/total:.1f}%)")

    # ── Load base model ──
    from analysis.collect_spd_activations import load_spd_model

    # Load SPD to get jose's base LlamaSimpleMLP
    spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()
    del spd_model
    torch.cuda.empty_cache()

    print(f"\nLoading {N_BATCHES} eval batches ({N_TOKENS/1e6:.0f}M tokens)...")
    batches = get_eval_batches(N_BATCHES)

    # ── Transcoder alive counts ──
    tc_points = []  # (total, alive)
    for dict_size, run_info in TC_RUNS.items():
        print(f"\n{'='*60}")
        print(f"  TC dict_size={dict_size} k=16 (run {run_info['run_id']})")
        print(f"{'='*60}")

        layer_paths = download_tc_artifacts(run_info)
        transcoders = {l: load_transcoder(p) for l, p in layer_paths.items()}
        total, alive = count_tc_alive(transcoders, base_model, batches)
        tc_points.append((total, alive))
        print(f"  TC {dict_size}: {alive}/{total} alive ({100*alive/total:.1f}%)")

        del transcoders
        torch.cuda.empty_cache()

    # ── CLT alive counts ──
    clt_points = []  # (total, alive)
    for dict_size, run_info in CLT_RUNS.items():
        print(f"\n{'='*60}")
        print(f"  CLT dict_size={dict_size} k=16 (run {run_info['run_id']})")
        print(f"{'='*60}")

        clt_path = download_clt_artifact(run_info)
        clt = load_clt(clt_path)
        total, alive = count_clt_alive(clt, base_model, batches)
        clt_points.append((total, alive))
        print(f"  CLT {dict_size}: {alive}/{total} alive ({100*alive/total:.1f}%)")

        del clt
        torch.cuda.empty_cache()

    # ── Save results ──
    results = {
        "spd": [{"total": t, "alive": a} for t, a in spd_points],
        "tc": [{"total": t, "alive": a, "dict_size": ds} for (t, a), ds in zip(tc_points, TC_RUNS.keys())],
        "clt": [{"total": t, "alive": a, "dict_size": ds} for (t, a), ds in zip(clt_points, CLT_RUNS.keys())],
    }
    with open(OUTPUT_DIR / "alive_line_data.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nData saved to {OUTPUT_DIR / 'alive_line_data.json'}")

    # ── Plot ──
    plot(spd_points, tc_points, clt_points)


def plot(spd_points=None, tc_points=None, clt_points=None):
    """Generate the line plot. Can be called standalone with saved data."""
    if spd_points is None:
        with open(OUTPUT_DIR / "alive_line_data.json") as f:
            data = json.load(f)
        spd_points = [(d["total"], d["alive"]) for d in data["spd"]]
        tc_points = [(d["total"], d["alive"]) for d in data["tc"]]
        clt_points = [(d["total"], d["alive"]) for d in data["clt"]]

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
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    # Diagonal reference line (y=x) in log-log
    all_totals = [t for t, a in spd_points + tc_points + clt_points]
    lo = min(all_totals) * 0.7
    hi = max(all_totals) * 1.5
    ax.plot([lo, hi], [lo, hi], ls="--", color="#ccc", lw=1, zorder=0, label="y = x (all alive)")

    # SPD
    spd_x = [t for t, a in spd_points]
    spd_y = [a for t, a in spd_points]
    ax.plot(spd_x, spd_y, "o-", color="#6b21a8", markersize=7, lw=2, label="VPD", zorder=3)

    # Per-layer transcoders
    tc_x = [t for t, a in tc_points]
    tc_y = [a for t, a in tc_points]
    ax.plot(tc_x, tc_y, "s-", color="#2b6cb0", markersize=7, lw=2, label="PLT (k=16)", zorder=3)

    # CLTs
    clt_x = [t for t, a in clt_points]
    clt_y = [a for t, a in clt_points]
    ax.plot(clt_x, clt_y, "^-", color="#dd6b20", markersize=7, lw=2, label="CLT (k=16)", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total component capacity")
    ax.set_ylabel("Alive components")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, linewidth=0.5, which="both")

    fig.tight_layout()
    save_path = OUTPUT_DIR / "alive_line_plot.png"
    fig.savefig(save_path)
    fig.savefig(OUTPUT_DIR / "alive_line_plot.pdf")
    plt.close(fig)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Only plot from saved data")
    args = parser.parse_args()

    if args.plot_only:
        plot()
    else:
        main()
