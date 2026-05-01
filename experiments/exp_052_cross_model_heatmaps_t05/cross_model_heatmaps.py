"""Cross-model feature-matching heatmaps at threshold > 0.5.

For every ordered pair (A, B) of models and every alive feature j in A,
we count how many alive features in B have cosine similarity above 0.5.
We do this in three spaces:

  - input   : the encoder direction
              (SPD V column of c_fc; TC/CLT W_enc column)
  - output  : the decoder direction
              (SPD U row of down_proj; TC/CLT W_dec row)
  - matrix  : combined / matrix-space cosine = cos_input * cos_output.
              By the outer-product identity this is exactly the cosine of
              the rank-1 matrix M_j = enc_j ⊗ dec_j.

Models compared: VPD (SPD) at four capacities and TC/CLT at two dict
sizes (β-sweep variants are excluded).

Note on the matrix-space heatmap: a feature must have *both* an encoder
direction and a decoder direction sharing the same per-feature index for
this cosine to be defined. That holds for transcoders (TC/CLT) but not
for SPD — its c_fc and down_proj components are decomposed independently
and are not paired. The matrix-space heatmap therefore only includes the
TC/CLT models. The input and output heatmaps include all 8 models.

For each (A, B) pair the JSON stores `pct_split` (averaged over the 4
layers): the % of features in A with >1 match in B (excluding self when
A == B). Self-matches are subtracted before counting.

Outputs (in `output/`):
  heatmap_data_{input,output,matrix}_t0p5.json   # raw matrices + labels
  cross_model_heatmap_{input,output,matrix}_t0p5.{png,pdf}

The JSON files are self-describing so collaborators can apply their own
plotting style without rerunning anything.

Inputs:
  Pre-computed normalized alive directions live in exp_044's
  direction_cache/. Each `.pt` file is `{layer_idx: tensor(n_alive, d)}`
  where rows are L2-normalized in feature index order.

Usage:
  python experiments/exp_052_cross_model_heatmaps_t05/cross_model_heatmaps.py
  python experiments/exp_052_cross_model_heatmaps_t05/cross_model_heatmaps.py --plot-only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
THRESHOLD = 0.5
CHUNK = 512  # rows of A processed per cosine block

DIRECTION_CACHE = Path("experiments/exp_044_cosine_splitting/output/direction_cache")
OUTPUT_DIR = Path("experiments/exp_052_cross_model_heatmaps_t05/output")


@dataclass(frozen=True)
class ModelEntry:
    """One model in the comparison.

    `cache_stem` is the filename stem used by exp_044's direction cache;
    the loader reads `<stem>_input.pt` and `<stem>_output.pt`.
    `display` is the publication-friendly label used in JSON / plots.
    `paired` is True iff every alive feature has both an input and output
    direction sharing the same row index — required for matrix-space cosine.
    """

    cache_stem: str
    display: str
    paired: bool


# Order matters: this is also the row/column order of the heatmap.
MODELS: list[ModelEntry] = [
    ModelEntry("SPD_0.5x",  "VPD 0.5x", paired=False),
    ModelEntry("SPD_1x",    "VPD 1x",   paired=False),
    ModelEntry("SPD_2x",    "VPD 2x",   paired=False),
    ModelEntry("SPD_4x",    "VPD 4x",   paired=False),
    ModelEntry("TC_4k",     "PLT 4k",   paired=True),
    ModelEntry("TC_32k",    "PLT 32k",  paired=True),
    ModelEntry("CLT_4k",    "CLT 4k",   paired=True),
    ModelEntry("CLT_32k",   "CLT 32k",  paired=True),
]


# =============================================================================
# Loading
# =============================================================================


def load_layer_dirs(stem: str, space: str) -> dict[int, torch.Tensor]:
    path = DIRECTION_CACHE / f"{stem}_{space}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing direction cache: {path}.\n"
            f"Run exp_044's cross_model_heatmap.py first to populate it."
        )
    return torch.load(path, map_location="cpu", weights_only=True)


def load_directions(models: list[ModelEntry], spaces: tuple[str, ...]
                    ) -> dict[str, dict[str, dict[int, torch.Tensor]]]:
    """Load alive directions for every (model, space, layer)."""
    dirs: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for entry in models:
        dirs[entry.display] = {space: load_layer_dirs(entry.cache_stem, space) for space in spaces}
    return dirs


# =============================================================================
# Pairwise match counting
# =============================================================================


def count_matches_single_space(
    a: torch.Tensor, b: torch.Tensor, threshold: float, exclude_self: bool,
) -> np.ndarray:
    """For each row in A, count rows in B with cosine > threshold (single space)."""
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


def count_matches_matrix_space(
    a_in: torch.Tensor, a_out: torch.Tensor,
    b_in: torch.Tensor, b_out: torch.Tensor,
    threshold: float, exclude_self: bool,
) -> np.ndarray:
    """For each row in A, count rows in B with (cos_in * cos_out) > threshold."""
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


# =============================================================================
# Heatmap aggregation
# =============================================================================


def _pct_split(counts: np.ndarray) -> float:
    n = counts.size
    return 100.0 * float((counts > 1).sum()) / n if n else 0.0


def compute_single_space_heatmap(
    dirs: dict[str, dict[str, dict[int, torch.Tensor]]],
    models: list[ModelEntry],
    space: str,
    threshold: float,
) -> dict:
    names = [m.display for m in models]
    n = len(names)
    pct_split = np.zeros((n, n))

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            same = (name_a == name_b)
            per_layer: list[float] = []
            for layer in LAYERS:
                a = dirs[name_a][space][layer]
                b = dirs[name_b][space][layer]
                counts = count_matches_single_space(a, b, threshold, same)
                if counts.size:
                    per_layer.append(_pct_split(counts))
            pct_split[i, j] = float(np.mean(per_layer)) if per_layer else 0.0
        print(f"  {name_a:<12} done — diag split={pct_split[i, i]:.1f}%")

    return {
        "threshold": threshold,
        "space": space,
        "models": names,
        "pct_split": pct_split.tolist(),
    }


def compute_matrix_space_heatmap(
    dirs: dict[str, dict[str, dict[int, torch.Tensor]]],
    models: list[ModelEntry],
    threshold: float,
) -> dict:
    names = [m.display for m in models]
    n = len(names)
    pct_split = np.zeros((n, n))

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            same = (name_a == name_b)
            per_layer: list[float] = []
            for layer in LAYERS:
                a_in = dirs[name_a]["input"][layer]
                a_out = dirs[name_a]["output"][layer]
                b_in = dirs[name_b]["input"][layer]
                b_out = dirs[name_b]["output"][layer]
                assert a_in.shape[0] == a_out.shape[0], f"{name_a} L{layer} not paired"
                assert b_in.shape[0] == b_out.shape[0], f"{name_b} L{layer} not paired"
                counts = count_matches_matrix_space(a_in, a_out, b_in, b_out, threshold, same)
                if counts.size:
                    per_layer.append(_pct_split(counts))
            pct_split[i, j] = float(np.mean(per_layer)) if per_layer else 0.0
        print(f"  {name_a:<12} done — diag split={pct_split[i, i]:.1f}%")

    return {
        "threshold": threshold,
        "space": "matrix",
        "models": names,
        "pct_split": pct_split.tolist(),
    }


# =============================================================================
# Plotting (a reference figure — collaborators can restyle from JSON)
# =============================================================================


SPACE_LABELS = {
    "input":  "input space (encoder / V)",
    "output": "output space (decoder / U)",
    "matrix": "matrix space (encoder ⊗ decoder)",
}


def plot_heatmap(data: dict, save_path: Path) -> None:
    """Render the % features with >1 match panel only."""
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
            ax.text(j, i, f"{pct_split[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color=color)
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

    print("Loading direction caches...")
    dirs = load_directions(MODELS, spaces=("input", "output"))
    print(f"  Loaded {len(dirs)} models, layers {LAYERS}\n")

    print(f"=== input space (cosine > {THRESHOLD}) ===")
    data_input = compute_single_space_heatmap(dirs, MODELS, "input", THRESHOLD)
    json_in = OUTPUT_DIR / f"heatmap_data_input_{tag}.json"
    json_in.write_text(json.dumps(data_input, indent=2))
    print(f"  wrote {json_in}")
    plot_heatmap(data_input, OUTPUT_DIR / f"cross_model_heatmap_input_{tag}")

    print(f"\n=== output space (cosine > {THRESHOLD}) ===")
    data_output = compute_single_space_heatmap(dirs, MODELS, "output", THRESHOLD)
    json_out = OUTPUT_DIR / f"heatmap_data_output_{tag}.json"
    json_out.write_text(json.dumps(data_output, indent=2))
    print(f"  wrote {json_out}")
    plot_heatmap(data_output, OUTPUT_DIR / f"cross_model_heatmap_output_{tag}")

    print(f"\n=== matrix space (cosine > {THRESHOLD}) ===")
    paired_models = [m for m in MODELS if m.paired]
    print(f"  Restricted to paired models (TC/CLT): "
          f"{[m.display for m in paired_models]}")
    data_matrix = compute_matrix_space_heatmap(dirs, paired_models, THRESHOLD)
    json_mat = OUTPUT_DIR / f"heatmap_data_matrix_{tag}.json"
    json_mat.write_text(json.dumps(data_matrix, indent=2))
    print(f"  wrote {json_mat}")
    plot_heatmap(data_matrix, OUTPUT_DIR / f"cross_model_heatmap_matrix_{tag}")


if __name__ == "__main__":
    main()
