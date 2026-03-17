"""Plot percentage of dead features per layer for transcoders and CLTs.

Dead = never fires on the eval set (10 batches × 8 seqs × 512 tokens).

Usage:
    python experiments/exp_037_pareto_jose_per_layer/dead_features.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig, CLTConfig
from nn_decompositions.clt import CrossLayerTranscoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
ENCODER_CLASSES = {"batchtopk": BatchTopKTranscoder}


def load_transcoder(checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    encoder = ENCODER_CLASSES[cfg.encoder_type](cfg)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    encoder.eval()
    return encoder


def load_clt(checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
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


def get_eval_batches(n_batches: int, batch_size: int, seq_len: int):
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)
    batches = []
    for _ in tqdm(range(n_batches), desc="Loading batches"):
        batch_ids = []
        for _ in range(batch_size):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:seq_len])
        batches.append(torch.stack(batch_ids).to(DEVICE))
    return batches


@torch.no_grad()
def get_mlp_activations(model, batches):
    all_pairs = {layer_idx: [] for layer_idx in LAYERS}
    for input_ids in batches:
        captured = {}
        hooks = []
        for layer_idx in LAYERS:
            rms2 = model.h[layer_idx].rms_2
            mlp = model.h[layer_idx].mlp

            def _make_hooks(li):
                def _capture_rms2(_mod, _inp, out):
                    captured[f"mlp_in_{li}"] = out.detach()
                def _capture_mlp(_mod, _inp, out):
                    captured[f"mlp_out_{li}"] = out.detach()
                return _capture_rms2, _capture_mlp

            h_rms, h_mlp = _make_hooks(layer_idx)
            hooks.append(rms2.register_forward_hook(h_rms))
            hooks.append(mlp.register_forward_hook(h_mlp))

        model(input_ids)
        for h in hooks:
            h.remove()

        for layer_idx in LAYERS:
            mlp_in = captured[f"mlp_in_{layer_idx}"].reshape(-1, captured[f"mlp_in_{layer_idx}"].shape[-1])
            mlp_out = captured[f"mlp_out_{layer_idx}"].reshape(-1, captured[f"mlp_out_{layer_idx}"].shape[-1])
            all_pairs[layer_idx].append((mlp_in, mlp_out))
    return all_pairs


@torch.no_grad()
def count_dead_features_tc(tc, mlp_inputs: list[torch.Tensor]) -> tuple[int, int]:
    """Returns (n_dead, dict_size)."""
    use_pre_enc_bias = tc.cfg.pre_enc_bias and tc.input_size == tc.output_size
    ever_active = torch.zeros(tc.cfg.dict_size, dtype=torch.bool, device=DEVICE)
    for x_in in mlp_inputs:
        x_enc = x_in - tc.b_dec if use_pre_enc_bias else x_in
        pre_acts = F.relu(x_enc @ tc.W_enc)
        ever_active |= (pre_acts > 0).any(dim=0)
    n_dead = tc.cfg.dict_size - ever_active.sum().item()
    return n_dead, tc.cfg.dict_size


@torch.no_grad()
def count_dead_features_clt(clt, mlp_inputs_per_layer: dict[int, list[torch.Tensor]]) -> dict[int, tuple[int, int]]:
    """Returns {layer: (n_dead, dict_size)}."""
    results = {}
    for i in range(clt.cfg.n_layers):
        ever_active = torch.zeros(clt.cfg.dict_size, dtype=torch.bool, device=DEVICE)
        for x_in in mlp_inputs_per_layer[i]:
            pre_acts = F.relu(x_in @ clt.W_enc[i] + clt.b_enc[i])
            ever_active |= (pre_acts > 0).any(dim=0)
        n_dead = clt.cfg.dict_size - ever_active.sum().item()
        results[i] = (n_dead, clt.cfg.dict_size)
    return results


# =============================================================================
# Plotting
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
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


def main():
    from analysis.collect_spd_activations import load_spd_model

    print("Loading base model...")
    spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    print("Loading eval batches...")
    batches = get_eval_batches(10, 8, 512)

    print("Collecting MLP activations...")
    mlp_activations = get_mlp_activations(base_model, batches)

    del spd_model, base_model
    torch.cuda.empty_cache()

    top_ks = [8, 16, 32, 64]

    # Collect dead feature stats: {method: {layer: [(k, dead_pct)]}}
    tc_dead: dict[int, list[tuple[int, float]]] = {l: [] for l in LAYERS}
    clt_dead: dict[int, list[tuple[int, float]]] = {l: [] for l in LAYERS}

    # Transcoders
    for k in top_ks:
        for layer_idx in LAYERS:
            tc = load_transcoder(f"checkpoints/jose_tc_tc_k{k}_layer{layer_idx}")
            tc.to(DEVICE)
            mlp_inputs = [mlp_activations[layer_idx][i][0] for i in range(len(batches))]
            n_dead, dict_size = count_dead_features_tc(tc, mlp_inputs)
            pct = 100 * n_dead / dict_size
            tc_dead[layer_idx].append((k, pct))
            print(f"  TC k={k} layer={layer_idx}: {n_dead}/{dict_size} dead ({pct:.1f}%)")
            del tc
            torch.cuda.empty_cache()

    # CLTs
    for k in top_ks:
        clt = load_clt(f"checkpoints/jose_clt_clt_k{k}")
        clt.to(DEVICE)
        mlp_inputs_per_layer = {
            l: [mlp_activations[l][i][0] for i in range(len(batches))]
            for l in LAYERS
        }
        layer_results = count_dead_features_clt(clt, mlp_inputs_per_layer)
        for layer_idx in LAYERS:
            n_dead, dict_size = layer_results[layer_idx]
            pct = 100 * n_dead / dict_size
            clt_dead[layer_idx].append((k, pct))
            print(f"  CLT k={k} layer={layer_idx}: {n_dead}/{dict_size} dead ({pct:.1f}%)")
        del clt
        torch.cuda.empty_cache()

    # Plot: 2x2 grid, one per layer
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)

    for layer_idx, ax in zip(LAYERS, axes.flat):
        # Transcoders
        tc_ks = [x[0] for x in tc_dead[layer_idx]]
        tc_pcts = [x[1] for x in tc_dead[layer_idx]]
        ax.plot(tc_ks, tc_pcts, marker="o", color="#1f77b4", linestyle="-",
                linewidth=1.8, markersize=8, label="Transcoders",
                markeredgecolor="white", markeredgewidth=0.8)

        # CLTs
        clt_ks = [x[0] for x in clt_dead[layer_idx]]
        clt_pcts = [x[1] for x in clt_dead[layer_idx]]
        ax.plot(clt_ks, clt_pcts, marker="X", color="#17becf", linestyle="-",
                linewidth=1.8, markersize=9, label="CLT",
                markeredgecolor="white", markeredgewidth=0.8)

        ax.set_title(f"Layer {layer_idx}", fontsize=12)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x)}" if x == int(x) else f"{x:g}"
        ))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.tick_params(direction="in", which="both")

    for ax in axes[1]:
        ax.set_xlabel("Training top-k")
    for ax in axes[:, 0]:
        ax.set_ylabel("Dead features (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = "experiments/exp_037_pareto_jose_per_layer/output/dead_features.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    main()
