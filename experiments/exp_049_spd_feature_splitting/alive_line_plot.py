"""
Usage:
  python experiments/exp_049_spd_feature_splitting/alive_line_plot.py
  python experiments/exp_049_spd_feature_splitting/alive_line_plot.py --plot-only
  python experiments/exp_049_spd_feature_splitting/alive_line_plot.py --reuse-cache
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
N_TOKENS = int(1e6)
BATCH_SIZE = 8
SEQ_LEN = 512
N_BATCHES = N_TOKENS // (BATCH_SIZE * SEQ_LEN)
ALIVE_THRESHOLD = 1e-6

OUTPUT_DIR = Path("experiments/exp_049_spd_feature_splitting/output")
SPD_ALIVE_FILE = OUTPUT_DIR / "alive_components_mean_ci.json"
LINE_DATA_FILE = OUTPUT_DIR / "alive_line_data.json"

SPD_RUNS = {
    "0.5x": "goodfire/spd/s-b2b37c4e",
    "1x (jose)": "goodfire/spd/s-55ea3f9b",
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

MLP_MODULE_PATTERNS = ["h.{}.mlp.c_fc", "h.{}.mlp.down_proj"]


@dataclass
class SimpleBatchTopKTranscoder:
    input_size: int
    dict_size: int
    top_k: int
    W_enc: torch.Tensor
    b_enc: torch.Tensor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = F.relu(x @ self.W_enc + self.b_enc)
        n_keep = self.top_k * x.shape[0]
        if n_keep < pre_acts.numel():
            topk = torch.topk(pre_acts.flatten(), n_keep, dim=-1)
            acts = torch.zeros_like(pre_acts.flatten()).scatter(-1, topk.indices, topk.values).reshape(pre_acts.shape)
            return acts
        return pre_acts


@dataclass
class SimpleCrossLayerTranscoder:
    input_size: int
    dict_size: int
    top_k: int
    W_enc: list[torch.Tensor]
    b_enc: list[torch.Tensor]

    @property
    def n_layers(self) -> int:
        return len(self.W_enc)

    def encode(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        pre_acts = F.relu(x @ self.W_enc[layer_idx] + self.b_enc[layer_idx])
        n_keep = self.top_k * x.shape[0]
        if n_keep < pre_acts.numel():
            topk = torch.topk(pre_acts.flatten(), n_keep, dim=-1)
            acts = torch.zeros_like(pre_acts.flatten()).scatter(-1, topk.indices, topk.values).reshape(pre_acts.shape)
            return acts
        return pre_acts


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        clean_key = key[len("module.") :] if key.startswith("module.") else key
        normalized[clean_key] = value
    return normalized


def _to_torch_dtype(dtype_str: str) -> torch.dtype:
    return getattr(torch, dtype_str.replace("torch.", ""))


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_eval_batches(n_batches: int) -> list[torch.Tensor]:
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)
    batches: list[torch.Tensor] = []

    for _ in tqdm(range(n_batches), desc="Loading eval batches"):
        batch_ids = []
        for _ in range(BATCH_SIZE):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:SEQ_LEN])
        batches.append(torch.stack(batch_ids).to(DEVICE))
    return batches


def load_transcoder(checkpoint_dir: Path) -> SimpleBatchTopKTranscoder:
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    state_dict = _normalize_state_dict_keys(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    model_dtype = _to_torch_dtype(cfg_dict.get("dtype", "torch.float32"))

    return SimpleBatchTopKTranscoder(
        input_size=int(cfg_dict["input_size"]),
        dict_size=int(cfg_dict["dict_size"]),
        top_k=int(cfg_dict["top_k"]),
        W_enc=state_dict["W_enc"].to(device=DEVICE, dtype=model_dtype),
        b_enc=state_dict["b_enc"].to(device=DEVICE, dtype=model_dtype),
    )


def load_clt(checkpoint_dir: Path) -> SimpleCrossLayerTranscoder:
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    state_dict = _normalize_state_dict_keys(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    model_dtype = _to_torch_dtype(cfg_dict.get("dtype", "torch.float32"))

    w_enc_by_idx: dict[int, torch.Tensor] = {}
    b_enc_by_idx: dict[int, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        w_match = re.match(r"^W_enc\.(\d+)$", key)
        b_match = re.match(r"^b_enc\.(\d+)$", key)
        if w_match:
            w_enc_by_idx[int(w_match.group(1))] = tensor.to(device=DEVICE, dtype=model_dtype)
        elif b_match:
            b_enc_by_idx[int(b_match.group(1))] = tensor.to(device=DEVICE, dtype=model_dtype)

    if not w_enc_by_idx or set(w_enc_by_idx.keys()) != set(b_enc_by_idx.keys()):
        raise RuntimeError("CLT checkpoint is missing W_enc/b_enc tensors")

    sorted_idxs = sorted(w_enc_by_idx.keys())
    return SimpleCrossLayerTranscoder(
        input_size=int(cfg_dict["input_size"]),
        dict_size=int(cfg_dict["dict_size"]),
        top_k=int(cfg_dict["top_k"]),
        W_enc=[w_enc_by_idx[i] for i in sorted_idxs],
        b_enc=[b_enc_by_idx[i] for i in sorted_idxs],
    )


def download_wandb_artifact(project: str, artifact_name: str, dest: Path) -> Path:
    if dest.exists() and (dest / "encoder.pt").exists():
        print(f"  Using cached artifact at {dest}")
        return dest
    api = wandb.Api()
    artifact = api.artifact(f"{project}/{artifact_name}")
    artifact.download(root=str(dest))
    print(f"  Downloaded {artifact_name} -> {dest}")
    return dest


def download_tc_artifacts(run_info: dict) -> dict[int, Path]:
    api = wandb.Api()
    run = api.run(f"{run_info['project']}/runs/{run_info['run_id']}")
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    layer_paths: dict[int, Path] = {}

    for artifact in artifacts:
        artifact_name = artifact.name.split(":")[0]
        for layer_idx in LAYERS:
            if f"layer{layer_idx}_final" in artifact_name:
                dest = Path(f"checkpoints/alive_tc_{run_info['run_id']}_layer{layer_idx}")
                download_wandb_artifact(run_info["project"], artifact.name, dest)
                layer_paths[layer_idx] = dest
                break

    if set(layer_paths.keys()) != set(LAYERS):
        raise RuntimeError(f"Missing TC layers; got {sorted(layer_paths.keys())}")
    return layer_paths


def download_clt_artifact(run_info: dict) -> Path:
    api = wandb.Api()
    run = api.run(f"{run_info['project']}/runs/{run_info['run_id']}")
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    final_artifacts = [a for a in artifacts if "final" in a.name]
    if len(final_artifacts) != 1:
        raise RuntimeError(f"Expected 1 CLT final artifact, got {len(final_artifacts)}")
    dest = Path(f"checkpoints/alive_clt_{run_info['run_id']}")
    download_wandb_artifact(run_info["project"], final_artifacts[0].name, dest)
    return dest


def collect_mlp_inputs(base_model, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    for layer_idx in LAYERS:
        rms2 = base_model.h[layer_idx].rms_2

        def _make_hook(li: int):
            def _hook(_mod, _inp, out):
                captured[li] = out.detach()

            return _hook

        hooks.append(rms2.register_forward_hook(_make_hook(layer_idx)))

    base_model(input_ids)
    for hook in hooks:
        hook.remove()
    return captured


@torch.no_grad()
def count_spd_alive_mean_ci(spd_model, batches: list[torch.Tensor], threshold: float) -> dict[str, dict[str, int]]:
    mlp_modules = []
    for layer in LAYERS:
        for pattern in MLP_MODULE_PATTERNS:
            mod_name = pattern.format(layer)
            if mod_name in spd_model.module_to_c:
                mlp_modules.append(mod_name)

    ci_sum = {
        mod_name: torch.zeros(spd_model.module_to_c[mod_name], dtype=torch.float64, device="cpu")
        for mod_name in mlp_modules
    }
    n_tokens_total = 0

    for input_ids in tqdm(batches, desc="SPD mean-CI alive count"):
        batch = input_ids.to(DEVICE)
        bsz, seq = batch.shape
        n_tokens_total += bsz * seq
        out = spd_model(batch, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        for mod_name in mlp_modules:
            n_c = spd_model.module_to_c[mod_name]
            ci_vals = ci.lower_leaky[mod_name].reshape(-1, n_c)
            ci_sum[mod_name] += ci_vals.double().sum(dim=0).cpu()

    results: dict[str, dict[str, int]] = {}
    for mod_name in mlp_modules:
        mean_ci = ci_sum[mod_name] / n_tokens_total
        alive = int((mean_ci > threshold).sum().item())
        total = int(spd_model.module_to_c[mod_name])
        results[mod_name] = {"alive": alive, "total": total}
    return results


@torch.no_grad()
def count_tc_alive(transcoders: dict[int, SimpleBatchTopKTranscoder], base_model, batches: list[torch.Tensor]) -> tuple[int, int]:
    dict_size = next(iter(transcoders.values())).dict_size
    fire_count = {l: torch.zeros(dict_size, dtype=torch.int64, device=DEVICE) for l in LAYERS}
    n_tokens = 0

    for input_ids in tqdm(batches, desc="TC alive count"):
        mlp_inputs = collect_mlp_inputs(base_model, input_ids)
        bsz, seq = input_ids.shape
        n_tokens += bsz * seq

        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            flat = mlp_inputs[layer_idx].reshape(-1, tc.input_size)
            acts = tc.encode(flat)
            fire_count[layer_idx] += (acts > 0).sum(dim=0).to(torch.int64)

    total = dict_size * len(LAYERS)
    alive = 0
    for layer_idx in LAYERS:
        sparsity = fire_count[layer_idx].float() / n_tokens
        alive += int((sparsity > ALIVE_THRESHOLD).sum().item())
    return total, alive


@torch.no_grad()
def count_clt_alive(clt: SimpleCrossLayerTranscoder, base_model, batches: list[torch.Tensor]) -> tuple[int, int]:
    dict_size = clt.dict_size
    n_layers = clt.n_layers
    fire_count = {l: torch.zeros(dict_size, dtype=torch.int64, device=DEVICE) for l in range(n_layers)}
    n_tokens = 0

    for input_ids in tqdm(batches, desc="CLT alive count"):
        mlp_inputs = collect_mlp_inputs(base_model, input_ids)
        bsz, seq = input_ids.shape
        n_tokens += bsz * seq

        for layer_idx in range(n_layers):
            flat = mlp_inputs[layer_idx].reshape(-1, clt.input_size)
            acts = clt.encode(flat, layer_idx)
            fire_count[layer_idx] += (acts > 0).sum(dim=0).to(torch.int64)

    total = dict_size * n_layers
    alive = 0
    for layer_idx in range(n_layers):
        sparsity = fire_count[layer_idx].float() / n_tokens
        alive += int((sparsity > ALIVE_THRESHOLD).sum().item())
    return total, alive


def compute_spd_results(batches: list[torch.Tensor], reuse_cache: bool) -> dict[str, dict[str, dict[str, int]]]:
    from analysis.collect_spd_activations import load_spd_model

    if reuse_cache and SPD_ALIVE_FILE.exists():
        with open(SPD_ALIVE_FILE) as f:
            existing = json.load(f)
    else:
        existing = {}

    for label, run_path in SPD_RUNS.items():
        if reuse_cache and label in existing:
            print(f"Using cached SPD results for {label}")
            continue

        print(f"\n{'=' * 64}\nSPD {label}: {run_path}\n{'=' * 64}")
        spd_model, _ = load_spd_model(run_path)
        spd_model.to(DEVICE)
        spd_model.eval()
        existing[label] = count_spd_alive_mean_ci(spd_model, batches, threshold=ALIVE_THRESHOLD)

        del spd_model
        cleanup_cuda()

        with open(SPD_ALIVE_FILE, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Saved intermediate SPD results to {SPD_ALIVE_FILE}")

    return existing


def build_plot_points(
    spd_raw: dict[str, dict[str, dict[str, int]]],
) -> list[tuple[int, int]]:
    points = []
    for label in ["0.5x", "1x (jose)", "2x", "4x"]:
        total = sum(v["total"] for v in spd_raw[label].values())
        alive = sum(v["alive"] for v in spd_raw[label].values())
        points.append((total, alive))
        print(f"SPD {label}: {alive}/{total} alive ({100 * alive / total:.1f}%)")
    return points


def plot(spd_points: list[tuple[int, int]], tc_points: list[tuple[int, int]], clt_points: list[tuple[int, int]]) -> None:
    plt.rcParams.update(
        {
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
        }
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    all_totals = [t for t, _ in spd_points + tc_points + clt_points]
    lo = min(all_totals) * 0.7
    hi = max(all_totals) * 1.5
    ax.plot([lo, hi], [lo, hi], ls="--", color="#cccccc", lw=1, zorder=0, label="y = x (all alive)")

    ax.plot([t for t, _ in spd_points], [a for _, a in spd_points], "o-", color="#6b21a8", markersize=7, lw=2, label="VPD", zorder=3)
    ax.plot([t for t, _ in tc_points], [a for _, a in tc_points], "s-", color="#2b6cb0", markersize=7, lw=2, label="PLT (k=16)", zorder=3)
    ax.plot([t for t, _ in clt_points], [a for _, a in clt_points], "^-", color="#dd6b20", markersize=7, lw=2, label="CLT (k=16)", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total subcomponent capacity")
    ax.set_ylabel("Alive subcomponents")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, linewidth=0.5, which="both")

    fig.tight_layout()
    png_path = OUTPUT_DIR / "alive_line_plot.png"
    pdf_path = OUTPUT_DIR / "alive_line_plot.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved plot to {png_path} and {pdf_path}")


def main(plot_only: bool, reuse_cache: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if plot_only:
        with open(LINE_DATA_FILE) as f:
            data = json.load(f)
        spd_points = [(d["total"], d["alive"]) for d in data["spd"]]
        tc_points = [(d["total"], d["alive"]) for d in data["tc"]]
        clt_points = [(d["total"], d["alive"]) for d in data["clt"]]
        plot(spd_points, tc_points, clt_points)
        return

    print(f"Preparing evaluation batches: {N_BATCHES} batches ({N_TOKENS / 1e6:.0f}M tokens)")
    batches = get_eval_batches(N_BATCHES)

    print("\nStage 1/3: SPD mean-CI alive counts")
    spd_raw = compute_spd_results(batches, reuse_cache=reuse_cache)
    spd_points = build_plot_points(spd_raw)

    print("\nStage 2/3: Load base model from Jose SPD checkpoint")
    from analysis.collect_spd_activations import load_spd_model

    spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
    spd_model.to(DEVICE)
    spd_model.eval()
    base_model = spd_model.target_model
    base_model.eval()
    del spd_model
    cleanup_cuda()

    print("\nStage 3/3: TC + CLT alive counts")
    tc_points = []
    for dict_size, run_info in TC_RUNS.items():
        print(f"\nTC dict_size={dict_size}, run={run_info['run_id']}")
        layer_paths = download_tc_artifacts(run_info)
        transcoders = {l: load_transcoder(p) for l, p in layer_paths.items()}
        total, alive = count_tc_alive(transcoders, base_model, batches)
        tc_points.append((total, alive))
        print(f"TC {dict_size}: {alive}/{total} alive ({100 * alive / total:.1f}%)")
        del transcoders
        cleanup_cuda()

    clt_points = []
    for dict_size, run_info in CLT_RUNS.items():
        print(f"\nCLT dict_size={dict_size}, run={run_info['run_id']}")
        clt_path = download_clt_artifact(run_info)
        clt = load_clt(clt_path)
        total, alive = count_clt_alive(clt, base_model, batches)
        clt_points.append((total, alive))
        print(f"CLT {dict_size}: {alive}/{total} alive ({100 * alive / total:.1f}%)")
        del clt
        cleanup_cuda()

    output_data = {
        "spd": [{"total": t, "alive": a} for t, a in spd_points],
        "tc": [{"total": t, "alive": a, "dict_size": ds} for (t, a), ds in zip(tc_points, TC_RUNS.keys())],
        "clt": [{"total": t, "alive": a, "dict_size": ds} for (t, a), ds in zip(clt_points, CLT_RUNS.keys())],
    }
    with open(LINE_DATA_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved line data to {LINE_DATA_FILE}")

    plot(spd_points, tc_points, clt_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Only generate the figure from saved line data")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse cached SPD mean-CI results if present",
    )
    args = parser.parse_args()
    main(plot_only=args.plot_only, reuse_cache=args.reuse_cache)
