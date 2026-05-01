"""Cross-model feature-matching heatmap (output space, threshold > 0.5).

For every ordered pair (A, B) of models and every alive feature j in A,
count how many alive features in B have output-space cosine similarity
above 0.5. The "output direction" of a feature is its decoder vector:

  - VPD     : `U` row of `down_proj`
  - PLT/CLT : `W_dec` row

Models compared: VPD at four capacities (0.5x / 1x / 2x / 4x), PLT
(per-layer transcoders) at 4k and 32k, and CLT at 4k and 32k.

The JSON output stores `pct_split` (averaged over the 4 layers): the
% of features in A with >1 match in B (excluding self when A == B).

Pipeline (single self-contained script, no subprocesses):
  1. Stream 1M Pile tokens.
  2. For each model, compute alive-feature masks per layer and extract
     the normalized output direction of each alive feature.
  3. Cache directions to disk.
  4. Compute the heatmap and save JSON + reference plot.

Usage:
  python experiments/feature_splitting_heatmap/feature_splitting_heatmap.py
  python experiments/feature_splitting_heatmap/feature_splitting_heatmap.py --plot-only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from nn_decompositions.eval_utils import (
    cleanup_cuda,
    collect_mlp_inputs,
    get_pile_batches,
    load_clt,
    load_vpd_model,
    load_transcoder,
)
from experiments.paper_runs import (
    HEADLINE_CLT_RUNS,
    HEADLINE_TC_RUNS,
    VPD_BASELINE_RUN,
    VPD_CAPACITY_RUNS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
THRESHOLD = 0.5
ALIVE_THRESHOLD = 1e-6
N_TOKENS = int(1e6)
BATCH_SIZE = 8
SEQ_LEN = 512
N_BATCHES = N_TOKENS // (BATCH_SIZE * SEQ_LEN)
CHUNK = 512  # rows of A processed per cosine block

OUTPUT_DIR = Path("experiments/feature_splitting_heatmap/output")
DIR_CACHE = OUTPUT_DIR / "direction_cache"
CHECKPOINT_DIR = Path("checkpoints/feature_splitting_heatmap")


@dataclass(frozen=True)
class ModelEntry:
    """One model in the comparison.

    `kind` is "vpd", "tc", or "clt". `display` is the publication-friendly
    label used in the JSON / heatmap.
    """

    kind: str
    display: str
    cache_stem: str
    vpd_run: str | None = None
    project: str | None = None
    run_id: str | None = None


# Order matters: this is also the row/column order of the heatmap.
_PLT_4K_PROJECT, _PLT_4K_RUN_ID = HEADLINE_TC_RUNS[4096]
_PLT_32K_PROJECT, _PLT_32K_RUN_ID = HEADLINE_TC_RUNS[32768]
_CLT_4K_PROJECT, _CLT_4K_RUN_ID = HEADLINE_CLT_RUNS[4096]
_CLT_32K_PROJECT, _CLT_32K_RUN_ID = HEADLINE_CLT_RUNS[32768]

MODELS: list[ModelEntry] = [
    ModelEntry("vpd", "VPD 0.5x", "vpd_0p5x", vpd_run=VPD_CAPACITY_RUNS["0.5x"]),
    ModelEntry("vpd", "VPD 1x",   "vpd_1x",   vpd_run=VPD_CAPACITY_RUNS["1x"]),
    ModelEntry("vpd", "VPD 2x",   "vpd_2x",   vpd_run=VPD_CAPACITY_RUNS["2x"]),
    ModelEntry("vpd", "VPD 4x",   "vpd_4x",   vpd_run=VPD_CAPACITY_RUNS["4x"]),
    ModelEntry("tc",  "PLT 4k",   "tc_4k",    project=_PLT_4K_PROJECT,  run_id=_PLT_4K_RUN_ID),
    ModelEntry("tc",  "PLT 32k",  "tc_32k",   project=_PLT_32K_PROJECT, run_id=_PLT_32K_RUN_ID),
    ModelEntry("clt", "CLT 4k",   "clt_4k",   project=_CLT_4K_PROJECT,  run_id=_CLT_4K_RUN_ID),
    ModelEntry("clt", "CLT 32k",  "clt_32k",  project=_CLT_32K_PROJECT, run_id=_CLT_32K_RUN_ID),
]


# =============================================================================
# Data
# =============================================================================


def get_eval_batches(n_batches: int) -> list[torch.Tensor]:
    return get_pile_batches(n_batches, BATCH_SIZE, SEQ_LEN, device=DEVICE)


def _collect_mlp_inputs(base_model, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
    return collect_mlp_inputs(base_model, input_ids, LAYERS)


# =============================================================================
# Loaders
# =============================================================================


def _download_artifact(project: str, artifact_name: str, dest: Path) -> Path:
    if dest.exists() and (dest / "encoder.pt").exists():
        return dest
    api = wandb.Api()
    artifact = api.artifact(f"{project}/{artifact_name}")
    artifact.download(root=str(dest))
    return dest


def download_tc_layers(entry: ModelEntry) -> dict[int, Path]:
    api = wandb.Api()
    run = api.run(f"{entry.project}/runs/{entry.run_id}")
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    layer_paths: dict[int, Path] = {}
    for artifact in artifacts:
        artifact_name = artifact.name.split(":")[0]
        for layer_idx in LAYERS:
            if f"layer{layer_idx}_final" in artifact_name:
                dest = CHECKPOINT_DIR / f"{entry.cache_stem}_layer{layer_idx}"
                _download_artifact(entry.project, artifact.name, dest)
                layer_paths[layer_idx] = dest
                break
    if set(layer_paths.keys()) != set(LAYERS):
        raise RuntimeError(f"Missing TC layers for {entry.display}: {sorted(layer_paths.keys())}")
    return layer_paths


def download_clt_artifact(entry: ModelEntry) -> Path:
    api = wandb.Api()
    run = api.run(f"{entry.project}/runs/{entry.run_id}")
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    final = [a for a in artifacts if "final" in a.name]
    if len(final) != 1:
        raise RuntimeError(f"Expected 1 CLT final artifact for {entry.display}, got {len(final)}")
    dest = CHECKPOINT_DIR / entry.cache_stem
    _download_artifact(entry.project, final[0].name, dest)
    return dest


# =============================================================================
# Output-direction extraction (one model, all layers)
# =============================================================================


@torch.no_grad()
def extract_directions_vpd(entry: ModelEntry, batches: list[torch.Tensor]
                           ) -> dict[int, torch.Tensor]:
    """For VPD: output directions are `U` rows of each layer's `down_proj`."""
    vpd_model, _ = load_vpd_model(entry.vpd_run)
    vpd_model.to(DEVICE)
    vpd_model.eval()

    down_modules = [f"h.{l}.mlp.down_proj" for l in LAYERS]
    ci_sum = {
        m: torch.zeros(vpd_model.module_to_c[m], dtype=torch.float64, device="cpu")
        for m in down_modules
    }
    n_tokens_total = 0
    for input_ids in tqdm(batches, desc=f"{entry.display} CI"):
        bsz, seq = input_ids.shape
        n_tokens_total += bsz * seq
        out = vpd_model(input_ids, cache_type="input")
        ci = vpd_model.calc_causal_importances(out.cache, sampling="continuous")
        for m in down_modules:
            n_c = vpd_model.module_to_c[m]
            ci_vals = ci.lower_leaky[m].reshape(-1, n_c)
            ci_sum[m] += ci_vals.double().sum(dim=0).cpu()

    dirs: dict[int, torch.Tensor] = {}
    for layer in LAYERS:
        down = f"h.{layer}.mlp.down_proj"
        alive = (ci_sum[down] / n_tokens_total) > ALIVE_THRESHOLD
        U = vpd_model.components[down].U.float()  # (C, d_out)
        dirs[layer] = F.normalize(U[alive], dim=1).cpu()

    del vpd_model
    cleanup_cuda()
    return dirs


@torch.no_grad()
def extract_directions_tc(entry: ModelEntry, base_model, batches: list[torch.Tensor]
                          ) -> dict[int, torch.Tensor]:
    layer_paths = download_tc_layers(entry)
    transcoders = {l: load_transcoder(p, DEVICE) for l, p in layer_paths.items()}

    dict_size = next(iter(transcoders.values())).cfg.dict_size
    fire_count = {l: torch.zeros(dict_size, dtype=torch.int64, device=DEVICE) for l in LAYERS}
    n_tokens = 0
    for input_ids in tqdm(batches, desc=f"{entry.display} fire"):
        captured = _collect_mlp_inputs(base_model, input_ids)
        bsz, seq = input_ids.shape
        n_tokens += bsz * seq
        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            flat = captured[layer_idx].reshape(-1, tc.cfg.input_size)
            acts = tc.encode(flat)
            fire_count[layer_idx] += (acts > 0).sum(dim=0).to(torch.int64)

    dirs: dict[int, torch.Tensor] = {}
    for layer_idx in LAYERS:
        tc = transcoders[layer_idx]
        sparsity = fire_count[layer_idx].float() / n_tokens
        alive = sparsity > ALIVE_THRESHOLD
        # W_dec is (dict_size, d_out).
        dec = F.normalize(tc.W_dec.float(), dim=1)[alive]
        dirs[layer_idx] = dec.cpu()

    del transcoders
    cleanup_cuda()
    return dirs


@torch.no_grad()
def extract_directions_clt(entry: ModelEntry, base_model, batches: list[torch.Tensor]
                           ) -> dict[int, torch.Tensor]:
    clt_path = download_clt_artifact(entry)
    clt = load_clt(clt_path, DEVICE)

    n_layers = clt.cfg.n_layers
    dict_size = clt.cfg.dict_size
    fire_count = {l: torch.zeros(dict_size, dtype=torch.int64, device=DEVICE) for l in range(n_layers)}
    n_tokens = 0
    for input_ids in tqdm(batches, desc=f"{entry.display} fire"):
        captured = _collect_mlp_inputs(base_model, input_ids)
        bsz, seq = input_ids.shape
        n_tokens += bsz * seq
        for layer_idx in range(n_layers):
            flat = captured[LAYERS[layer_idx]].reshape(-1, clt.cfg.input_size)
            acts = clt.encode_layer(flat, layer_idx)
            fire_count[layer_idx] += (acts > 0).sum(dim=0).to(torch.int64)

    dirs: dict[int, torch.Tensor] = {}
    for layer_idx in range(n_layers):
        sparsity = fire_count[layer_idx].float() / n_tokens
        alive = sparsity > ALIVE_THRESHOLD
        # Same-layer W_dec[i][0] is (dict_size, d_out).
        dec = F.normalize(clt.W_dec[layer_idx][0].float(), dim=1)[alive]
        dirs[layer_idx] = dec.cpu()

    del clt
    cleanup_cuda()
    return dirs


def extract_or_load(entry: ModelEntry, base_model_provider) -> dict[int, torch.Tensor]:
    """Return {layer: tensor(n_alive, d) of normalized output directions}, caching to disk."""
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = DIR_CACHE / f"{entry.cache_stem}.pt"
    if cache_path.exists():
        print(f"  [{entry.display}] using cached directions at {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"\n=== Extracting directions: {entry.display} ===")
    batches = base_model_provider.batches
    if entry.kind == "vpd":
        dirs = extract_directions_vpd(entry, batches)
    elif entry.kind == "tc":
        dirs = extract_directions_tc(entry, base_model_provider.base_model, batches)
    elif entry.kind == "clt":
        dirs = extract_directions_clt(entry, base_model_provider.base_model, batches)
    else:
        raise ValueError(entry.kind)
    torch.save(dirs, cache_path)
    print(f"  cached to {cache_path}")
    return dirs


class _BaseModelProvider:
    """Lazily load the base LLM (used by TC/CLT extraction)."""

    def __init__(self, batches: list[torch.Tensor]):
        self.batches = batches
        self._base_model = None

    @property
    def base_model(self):
        if self._base_model is None:
            vpd_model, _ = load_vpd_model(VPD_BASELINE_RUN)
            vpd_model.to(DEVICE)
            vpd_model.eval()
            self._base_model = vpd_model.target_model
            self._base_model.eval()
            del vpd_model
            cleanup_cuda()
        return self._base_model


# =============================================================================
# Pairwise match counting
# =============================================================================


def count_matches(a: torch.Tensor, b: torch.Tensor, threshold: float, exclude_self: bool) -> np.ndarray:
    n_a = a.shape[0]
    if n_a == 0 or b.shape[0] == 0:
        return np.zeros(n_a, dtype=np.int64)
    counts = np.empty(n_a, dtype=np.int64)
    b_dev = b.to(DEVICE)
    for start in range(0, n_a, CHUNK):
        chunk = a[start:start + CHUNK].to(DEVICE)
        cos = chunk @ b_dev.T
        counts[start:start + chunk.shape[0]] = (cos > threshold).sum(dim=1).cpu().numpy()
    return np.maximum(counts - 1, 0) if exclude_self else counts


def _pct_split(counts: np.ndarray) -> float:
    return 100.0 * float((counts > 1).sum()) / counts.size if counts.size else 0.0


def compute_heatmap(dirs: dict[str, dict[int, torch.Tensor]], models: list[ModelEntry],
                    threshold: float) -> dict:
    names = [m.display for m in models]
    n = len(names)
    pct_split = np.zeros((n, n))
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            same = (name_a == name_b)
            per_layer = []
            for layer in LAYERS:
                a = dirs[name_a][layer]
                b = dirs[name_b][layer]
                counts = count_matches(a, b, threshold, same)
                if counts.size:
                    per_layer.append(_pct_split(counts))
            pct_split[i, j] = float(np.mean(per_layer)) if per_layer else 0.0
        print(f"  {name_a:<10} done — diag split={pct_split[i, i]:.1f}%")
    return {"threshold": threshold, "space": "output", "models": names,
            "pct_split": pct_split.tolist()}


# =============================================================================
# Plotting (reference figure — collaborators can restyle from JSON)
# =============================================================================


def plot_heatmap(data: dict, save_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    names = data["models"]
    pct_split = np.array(data["pct_split"])
    n = len(names)
    cell = 0.55
    fig_w = max(5.0, n * cell + 2.0)
    fig_h = max(4.5, n * cell + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(pct_split, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Target model")
    ax.set_ylabel("Source model")

    max_val = float(pct_split.max()) or 1.0
    for i in range(n):
        for j in range(n):
            color = "white" if pct_split[i, j] > 0.6 * max_val else "black"
            ax.text(j, i, f"{pct_split[i, j]:.1f}", ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.85, label="% features with >1 match")
    ax.set_title(
        f"Feature splitting — output space (decoder / U) (cosine > {data['threshold']})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(".png"))
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved {save_path.with_suffix('.png')} and .pdf")


# =============================================================================
# Main
# =============================================================================


def threshold_tag(t: float) -> str:
    return f"t{t}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-render the plot from existing JSON without recomputing")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tag = threshold_tag(THRESHOLD)
    json_path = OUTPUT_DIR / f"heatmap_data_output_{tag}.json"
    plot_stem = OUTPUT_DIR / f"cross_model_heatmap_output_{tag}"

    if args.plot_only:
        with open(json_path) as f:
            data = json.load(f)
        plot_heatmap(data, plot_stem)
        return

    print(f"Loading {N_BATCHES} eval batches ({N_TOKENS / 1e6:.0f}M tokens)...")
    batches = get_eval_batches(N_BATCHES)
    provider = _BaseModelProvider(batches)

    dirs: dict[str, dict[int, torch.Tensor]] = {}
    for entry in MODELS:
        dirs[entry.display] = extract_or_load(entry, provider)

    print(f"\n=== output space (cosine > {THRESHOLD}) ===")
    data = compute_heatmap(dirs, MODELS, THRESHOLD)
    json_path.write_text(json.dumps(data, indent=2))
    plot_heatmap(data, plot_stem)


if __name__ == "__main__":
    main()
