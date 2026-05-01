"""Cross-model feature-matching heatmaps at threshold > 0.5.

For every ordered pair (A, B) of models and every alive feature j in A,
we count how many alive features in B have cosine similarity above 0.5.
We do this in three spaces:

  - input  : encoder direction (SPD V column of c_fc; TC/CLT W_enc column).
  - output : decoder direction (SPD U row of down_proj; TC/CLT W_dec row).
  - matrix : combined / matrix-space cosine = cos_input * cos_output.
             By the outer-product identity this is exactly the cosine of
             the rank-1 matrix M_j = enc_j ⊗ dec_j.

Models compared: VPD (SPD) at four capacities (0.5x / 1x / 2x / 4x),
PLT (per-layer transcoders) at 4k and 32k, and CLT at 4k and 32k.

Note on the matrix-space heatmap: a feature must have *both* an encoder
direction and a decoder direction sharing the same per-feature index for
this cosine to be defined. That holds for transcoders (TC/CLT) but not
for SPD — its c_fc and down_proj components are decomposed independently
and are not paired. The matrix-space heatmap therefore only includes the
TC/CLT models. The input and output heatmaps include all 8 models.

For each (A, B) pair the JSON stores `pct_split` (averaged over the 4
layers): the % of features in A with >1 match in B (excluding self when
A == B). Self-matches are subtracted before counting.

Pipeline (single self-contained script, no subprocesses):
  1. Stream 1M Pile tokens.
  2. For each model, compute alive-feature masks per layer.
  3. Cache normalized alive directions to disk (input + output).
  4. Compute the three heatmaps and save JSON + reference plot.

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

from nn_decompositions.clt import CrossLayerTranscoder
from nn_decompositions.eval_utils import (
    cleanup_cuda,
    collect_mlp_inputs,
    get_pile_batches,
    load_clt,
    load_spd_model,
    load_transcoder,
)
from experiments.paper_runs import (
    HEADLINE_CLT_RUNS,
    HEADLINE_TC_RUNS,
    SPD_BASELINE_RUN,
    SPD_CAPACITY_RUNS,
)
from nn_decompositions.transcoder import BatchTopKTranscoder

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

    `kind` is "spd", "tc", or "clt". `display` is the publication-friendly
    label. `paired` is True iff every alive feature has both an input and
    output direction sharing the same row index — required for matrix-space
    cosine.
    """

    kind: str
    display: str
    cache_stem: str
    paired: bool
    spd_run: str | None = None
    project: str | None = None
    run_id: str | None = None


# Order matters: this is also the row/column order of the heatmap.
_PLT_4K_PROJECT, _PLT_4K_RUN_ID = HEADLINE_TC_RUNS[4096]
_PLT_32K_PROJECT, _PLT_32K_RUN_ID = HEADLINE_TC_RUNS[32768]
_CLT_4K_PROJECT, _CLT_4K_RUN_ID = HEADLINE_CLT_RUNS[4096]
_CLT_32K_PROJECT, _CLT_32K_RUN_ID = HEADLINE_CLT_RUNS[32768]

MODELS: list[ModelEntry] = [
    ModelEntry("spd", "VPD 0.5x", "spd_0p5x", paired=False, spd_run=SPD_CAPACITY_RUNS["0.5x"]),
    ModelEntry("spd", "VPD 1x",   "spd_1x",   paired=False, spd_run=SPD_CAPACITY_RUNS["1x"]),
    ModelEntry("spd", "VPD 2x",   "spd_2x",   paired=False, spd_run=SPD_CAPACITY_RUNS["2x"]),
    ModelEntry("spd", "VPD 4x",   "spd_4x",   paired=False, spd_run=SPD_CAPACITY_RUNS["4x"]),
    ModelEntry("tc",  "PLT 4k",   "tc_4k",    paired=True,
               project=_PLT_4K_PROJECT,  run_id=_PLT_4K_RUN_ID),
    ModelEntry("tc",  "PLT 32k",  "tc_32k",   paired=True,
               project=_PLT_32K_PROJECT, run_id=_PLT_32K_RUN_ID),
    ModelEntry("clt", "CLT 4k",   "clt_4k",   paired=True,
               project=_CLT_4K_PROJECT,  run_id=_CLT_4K_RUN_ID),
    ModelEntry("clt", "CLT 32k",  "clt_32k",  paired=True,
               project=_CLT_32K_PROJECT, run_id=_CLT_32K_RUN_ID),
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
# Direction extraction (one model, both spaces, all layers)
# =============================================================================


@torch.no_grad()
def extract_directions_spd(entry: ModelEntry, batches: list[torch.Tensor]
                           ) -> dict[str, dict[int, torch.Tensor]]:
    """For SPD: 'input' uses c_fc.V columns, 'output' uses down_proj.U rows."""
    spd_model, _ = load_spd_model(entry.spd_run)
    spd_model.to(DEVICE)
    spd_model.eval()

    def _module(layer: int, kind: str) -> str:
        return f"h.{layer}.mlp.{kind}"

    sub_modules = {
        "input": [_module(l, "c_fc") for l in LAYERS],
        "output": [_module(l, "down_proj") for l in LAYERS],
    }
    ci_sum = {
        m: torch.zeros(spd_model.module_to_c[m], dtype=torch.float64, device="cpu")
        for kind in sub_modules for m in sub_modules[kind]
    }
    n_tokens_total = 0
    for input_ids in tqdm(batches, desc=f"{entry.display} CI"):
        bsz, seq = input_ids.shape
        n_tokens_total += bsz * seq
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        for m in ci_sum:
            n_c = spd_model.module_to_c[m]
            ci_vals = ci.lower_leaky[m].reshape(-1, n_c)
            ci_sum[m] += ci_vals.double().sum(dim=0).cpu()

    dirs: dict[str, dict[int, torch.Tensor]] = {"input": {}, "output": {}}
    for layer in LAYERS:
        cfc = _module(layer, "c_fc")
        down = _module(layer, "down_proj")
        cfc_alive = (ci_sum[cfc] / n_tokens_total) > ALIVE_THRESHOLD
        down_alive = (ci_sum[down] / n_tokens_total) > ALIVE_THRESHOLD

        V = spd_model.components[cfc].V.float()      # (d_in, C)
        U = spd_model.components[down].U.float()     # (C, d_out)

        dirs["input"][layer] = F.normalize(V.T[cfc_alive], dim=1).cpu()
        dirs["output"][layer] = F.normalize(U[down_alive], dim=1).cpu()

    del spd_model
    cleanup_cuda()
    return dirs


@torch.no_grad()
def extract_directions_tc(entry: ModelEntry, base_model, batches: list[torch.Tensor]
                          ) -> dict[str, dict[int, torch.Tensor]]:
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

    dirs: dict[str, dict[int, torch.Tensor]] = {"input": {}, "output": {}}
    for layer_idx in LAYERS:
        tc = transcoders[layer_idx]
        sparsity = fire_count[layer_idx].float() / n_tokens
        alive = sparsity > ALIVE_THRESHOLD
        # W_enc is (d_in, dict_size); W_dec is (dict_size, d_out).
        enc = F.normalize(tc.W_enc.float().T, dim=1)[alive]
        dec = F.normalize(tc.W_dec.float(), dim=1)[alive]
        dirs["input"][layer_idx] = enc.cpu()
        dirs["output"][layer_idx] = dec.cpu()

    del transcoders
    cleanup_cuda()
    return dirs


@torch.no_grad()
def extract_directions_clt(entry: ModelEntry, base_model, batches: list[torch.Tensor]
                           ) -> dict[str, dict[int, torch.Tensor]]:
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

    dirs: dict[str, dict[int, torch.Tensor]] = {"input": {}, "output": {}}
    for layer_idx in range(n_layers):
        sparsity = fire_count[layer_idx].float() / n_tokens
        alive = sparsity > ALIVE_THRESHOLD
        # W_enc[i] is (d_in, dict_size); same-layer W_dec[i][0] is (dict_size, d_out).
        enc = F.normalize(clt.W_enc[layer_idx].float().T, dim=1)[alive]
        dec = F.normalize(clt.W_dec[layer_idx][0].float(), dim=1)[alive]
        dirs["input"][layer_idx] = enc.cpu()
        dirs["output"][layer_idx] = dec.cpu()

    del clt
    cleanup_cuda()
    return dirs


def extract_or_load(entry: ModelEntry, base_model_provider) -> dict[str, dict[int, torch.Tensor]]:
    """Return {space: {layer: tensor(n_alive, d) normalized}}, caching to disk."""
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = DIR_CACHE / f"{entry.cache_stem}.pt"
    if cache_path.exists():
        print(f"  [{entry.display}] using cached directions at {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"\n=== Extracting directions: {entry.display} ===")
    batches = base_model_provider.batches
    if entry.kind == "spd":
        dirs = extract_directions_spd(entry, batches)
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
    """Lazily load jose's base model (used by TC/CLT extraction)."""

    def __init__(self, batches: list[torch.Tensor]):
        self.batches = batches
        self._base_model = None

    @property
    def base_model(self):
        if self._base_model is None:
            spd_model, _ = load_spd_model(SPD_BASELINE_RUN)
            spd_model.to(DEVICE)
            spd_model.eval()
            self._base_model = spd_model.target_model
            self._base_model.eval()
            del spd_model
            cleanup_cuda()
        return self._base_model


# =============================================================================
# Pairwise match counting
# =============================================================================


def count_matches_single(a: torch.Tensor, b: torch.Tensor, threshold: float, exclude_self: bool) -> np.ndarray:
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


def count_matches_matrix(a_in, a_out, b_in, b_out, threshold, exclude_self) -> np.ndarray:
    n_a = a_in.shape[0]
    if n_a == 0 or b_in.shape[0] == 0:
        return np.zeros(n_a, dtype=np.int64)
    counts = np.empty(n_a, dtype=np.int64)
    bi = b_in.to(DEVICE)
    bo = b_out.to(DEVICE)
    for start in range(0, n_a, CHUNK):
        ai = a_in[start:start + CHUNK].to(DEVICE)
        ao = a_out[start:start + CHUNK].to(DEVICE)
        cos = (ai @ bi.T) * (ao @ bo.T)
        counts[start:start + ai.shape[0]] = (cos > threshold).sum(dim=1).cpu().numpy()
    return np.maximum(counts - 1, 0) if exclude_self else counts


def _pct_split(counts: np.ndarray) -> float:
    return 100.0 * float((counts > 1).sum()) / counts.size if counts.size else 0.0


def compute_single_space(dirs, models: list[ModelEntry], space: str, threshold: float) -> dict:
    names = [m.display for m in models]
    n = len(names)
    pct_split = np.zeros((n, n))
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            same = (name_a == name_b)
            per_layer = []
            for layer in LAYERS:
                a = dirs[name_a][space][layer]
                b = dirs[name_b][space][layer]
                counts = count_matches_single(a, b, threshold, same)
                if counts.size:
                    per_layer.append(_pct_split(counts))
            pct_split[i, j] = float(np.mean(per_layer)) if per_layer else 0.0
        print(f"  {name_a:<10} done — diag split={pct_split[i, i]:.1f}%")
    return {"threshold": threshold, "space": space, "models": names, "pct_split": pct_split.tolist()}


def compute_matrix_space(dirs, models: list[ModelEntry], threshold: float) -> dict:
    names = [m.display for m in models]
    n = len(names)
    pct_split = np.zeros((n, n))
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            same = (name_a == name_b)
            per_layer = []
            for layer in LAYERS:
                a_in = dirs[name_a]["input"][layer]
                a_out = dirs[name_a]["output"][layer]
                b_in = dirs[name_b]["input"][layer]
                b_out = dirs[name_b]["output"][layer]
                assert a_in.shape[0] == a_out.shape[0], f"{name_a} L{layer} not paired"
                assert b_in.shape[0] == b_out.shape[0], f"{name_b} L{layer} not paired"
                counts = count_matches_matrix(a_in, a_out, b_in, b_out, threshold, same)
                if counts.size:
                    per_layer.append(_pct_split(counts))
            pct_split[i, j] = float(np.mean(per_layer)) if per_layer else 0.0
        print(f"  {name_a:<10} done — diag split={pct_split[i, i]:.1f}%")
    return {"threshold": threshold, "space": "matrix", "models": names, "pct_split": pct_split.tolist()}


# =============================================================================
# Plotting (reference figure — collaborators can restyle from JSON)
# =============================================================================


SPACE_LABELS = {
    "input":  "input space (encoder / V)",
    "output": "output space (decoder / U)",
    "matrix": "matrix space (encoder ⊗ decoder)",
}


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
        f"Feature splitting — {SPACE_LABELS[data['space']]} (cosine > {data['threshold']})",
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
                        help="Re-render plots from existing JSON without recomputing")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tag = threshold_tag(THRESHOLD)
    spaces = ("input", "output", "matrix")

    if args.plot_only:
        for space in spaces:
            json_path = OUTPUT_DIR / f"heatmap_data_{space}_{tag}.json"
            if not json_path.exists():
                print(f"  Skip {space}: {json_path} missing")
                continue
            with open(json_path) as f:
                data = json.load(f)
            plot_heatmap(data, OUTPUT_DIR / f"cross_model_heatmap_{space}_{tag}")
        return

    print(f"Loading {N_BATCHES} eval batches ({N_TOKENS / 1e6:.0f}M tokens)...")
    batches = get_eval_batches(N_BATCHES)
    provider = _BaseModelProvider(batches)

    dirs: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for entry in MODELS:
        dirs[entry.display] = extract_or_load(entry, provider)

    print(f"\n=== input space (cosine > {THRESHOLD}) ===")
    data_in = compute_single_space(dirs, MODELS, "input", THRESHOLD)
    (OUTPUT_DIR / f"heatmap_data_input_{tag}.json").write_text(json.dumps(data_in, indent=2))
    plot_heatmap(data_in, OUTPUT_DIR / f"cross_model_heatmap_input_{tag}")

    print(f"\n=== output space (cosine > {THRESHOLD}) ===")
    data_out = compute_single_space(dirs, MODELS, "output", THRESHOLD)
    (OUTPUT_DIR / f"heatmap_data_output_{tag}.json").write_text(json.dumps(data_out, indent=2))
    plot_heatmap(data_out, OUTPUT_DIR / f"cross_model_heatmap_output_{tag}")

    print(f"\n=== matrix space (cosine > {THRESHOLD}) ===")
    paired = [m for m in MODELS if m.paired]
    print(f"  Restricted to paired models: {[m.display for m in paired]}")
    data_mat = compute_matrix_space(dirs, paired, THRESHOLD)
    (OUTPUT_DIR / f"heatmap_data_matrix_{tag}.json").write_text(json.dumps(data_mat, indent=2))
    plot_heatmap(data_mat, OUTPUT_DIR / f"cross_model_heatmap_matrix_{tag}")


if __name__ == "__main__":
    main()
