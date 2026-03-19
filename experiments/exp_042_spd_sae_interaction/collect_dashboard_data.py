"""Collect data for the interactive SPD-SAE dashboard.

For every SPD MLP component (c_fc + down_proj, all layers), finds the top 10
SAE features with highest combined lift across ALL residual stream SAE layers.

Saves data as JSON for the Gradio dashboard.

Usage:
    python experiments/exp_042_spd_sae_interaction/collect_dashboard_data.py
    python experiments/exp_042_spd_sae_interaction/collect_dashboard_data.py --n_batches 100
"""

import argparse
import heapq
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
SAE_K = 32
OUTPUT_DIR = Path("experiments/exp_042_spd_sae_interaction/output")

BATCH_SIZE = 8
SEQ_LEN = 512
TOP_K_EXAMPLES = 20
MAX_FIRE_RATE = 0.9
TOP_SAE_PER_COMPONENT = 10


def load_sae(checkpoint_dir: Path) -> BatchTopKTranscoder:
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    sae = BatchTopKTranscoder(cfg)
    sae.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    sae.eval()
    return sae


@torch.no_grad()
def collect_all_data(spd_model, saes: dict[int, BatchTopKTranscoder], base_model, tokenizer, n_batches: int):
    """Collect co-activation data for ALL SPD components vs ALL SAE features."""

    # Enumerate all SPD MLP modules
    spd_modules = []
    for layer in LAYERS:
        for mod_type in ["c_fc", "down_proj"]:
            mod_name = f"h.{layer}.mlp.{mod_type}"
            if mod_name in spd_model.module_to_c:
                spd_modules.append((layer, mod_type, mod_name))

    # Build flat SPD component index: (module_name, component_local_idx) -> flat_idx
    spd_flat = []  # list of (layer, mod_type, mod_name, local_idx)
    spd_offsets = {}  # mod_name -> start offset in flat index
    for layer, mod_type, mod_name in spd_modules:
        spd_offsets[mod_name] = len(spd_flat)
        n_c = spd_model.module_to_c[mod_name]
        for c in range(n_c):
            spd_flat.append((layer, mod_type, mod_name, c))
    n_spd_total = len(spd_flat)

    # SAE flat index: (sae_layer, feature_idx) -> flat_idx
    sae_dict = saes[LAYERS[0]].cfg.dict_size
    n_sae_total = len(LAYERS) * sae_dict
    # sae flat idx = sae_layer_idx * sae_dict + feature_idx

    print(f"  Total SPD components: {n_spd_total}")
    print(f"  Total SAE features: {n_sae_total} ({len(LAYERS)} layers x {sae_dict})")

    # Pass 1: firing rates
    print(f"  Pass 1: firing rates over {n_batches} batches...")
    spd_fire_count = torch.zeros(n_spd_total, dtype=torch.float64)
    sae_fire_count = torch.zeros(n_sae_total, dtype=torch.float64)
    n_tokens_total = 0

    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)

    all_input_ids = []
    for _ in tqdm(range(n_batches), desc="Pass 1"):
        batch_ids = []
        for _ in range(BATCH_SIZE):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:SEQ_LEN])
        input_ids = torch.stack(batch_ids).to(DEVICE)
        all_input_ids.append(input_ids)
        B, S = input_ids.shape
        n_tokens_total += B * S

        # SPD
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        for layer, mod_type, mod_name in spd_modules:
            ci_vals = ci.lower_leaky[mod_name].reshape(-1, spd_model.module_to_c[mod_name])
            offset = spd_offsets[mod_name]
            n_c = ci_vals.shape[1]
            spd_fire_count[offset:offset + n_c] += (ci_vals > 0).float().sum(dim=0).cpu().double()

        # SAE (all layers)
        captured = {}
        hooks = []
        for l in LAYERS:
            def _make_hook(li):
                def _hook(_mod, _inp, out):
                    captured[li] = out.detach()
                return _hook
            hooks.append(base_model.h[l].register_forward_hook(_make_hook(l)))
        base_model(input_ids)
        for h in hooks:
            h.remove()

        for li, l in enumerate(LAYERS):
            resid = captured[l].reshape(-1, 768)
            acts = saes[l].encode(resid).float()
            sae_offset = li * sae_dict
            sae_fire_count[sae_offset:sae_offset + sae_dict] += (acts > 0).float().sum(dim=0).cpu().double()

    spd_fire_rate = (spd_fire_count / n_tokens_total).float()
    sae_fire_rate = (sae_fire_count / n_tokens_total).float()

    # Filter by fire rate
    spd_valid = spd_fire_rate <= MAX_FIRE_RATE
    sae_valid = sae_fire_rate <= MAX_FIRE_RATE
    print(f"  Valid: {spd_valid.sum()}/{n_spd_total} SPD, {sae_valid.sum()}/{n_sae_total} SAE")

    # Select top SPD components by fire rate (among valid)
    spd_rates_masked = spd_fire_rate.clone()
    spd_rates_masked[~spd_valid] = -1
    n_top_spd = min(500, spd_valid.sum().item())
    top_spd_flat = spd_rates_masked.topk(n_top_spd).indices  # flat indices

    # Select top SAE features by fire rate (among valid)
    sae_rates_masked = sae_fire_rate.clone()
    sae_rates_masked[~sae_valid] = -1
    n_top_sae = min(500, sae_valid.sum().item())
    top_sae_flat = sae_rates_masked.topk(n_top_sae).indices

    print(f"  Using {n_top_spd} SPD x {n_top_sae} SAE for co-occurrence...")

    # Pass 2: co-occurrence + examples
    print(f"  Pass 2: co-occurrences + top examples...")
    both = torch.zeros(n_top_spd, n_top_sae, dtype=torch.float64)
    spd_on_sae_off = torch.zeros(n_top_spd, n_top_sae, dtype=torch.float64)
    spd_off_sae_on = torch.zeros(n_top_spd, n_top_sae, dtype=torch.float64)
    neither = torch.zeros(n_top_spd, n_top_sae, dtype=torch.float64)

    spd_top_examples = {i: [] for i in range(n_top_spd)}
    sae_top_examples = {i: [] for i in range(n_top_sae)}

    for batch_idx, input_ids in enumerate(tqdm(all_input_ids, desc="Pass 2")):
        B, S = input_ids.shape

        # SPD CI
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        # Build flat SPD activations for this batch
        spd_all = torch.zeros(B * S, n_spd_total, device="cpu")
        for layer, mod_type, mod_name in spd_modules:
            ci_vals = ci.lower_leaky[mod_name].reshape(-1, spd_model.module_to_c[mod_name]).cpu().float()
            offset = spd_offsets[mod_name]
            n_c = ci_vals.shape[1]
            spd_all[:, offset:offset + n_c] = ci_vals

        # SAE activations (all layers)
        captured = {}
        hooks = []
        for l in LAYERS:
            def _make_hook(li):
                def _hook(_mod, _inp, out):
                    captured[li] = out.detach()
                return _hook
            hooks.append(base_model.h[l].register_forward_hook(_make_hook(l)))
        base_model(input_ids)
        for h in hooks:
            h.remove()

        sae_all = torch.zeros(B * S, n_sae_total, device="cpu")
        for li, l in enumerate(LAYERS):
            resid = captured[l].reshape(-1, 768)
            acts = saes[l].encode(resid).float().cpu()
            sae_offset = li * sae_dict
            sae_all[:, sae_offset:sae_offset + sae_dict] = acts

        # Subset to top features
        spd_sub = spd_all[:, top_spd_flat]  # (B*S, n_top_spd)
        sae_sub = sae_all[:, top_sae_flat]  # (B*S, n_top_sae)

        s_on = spd_sub > 0
        a_on = sae_sub > 0

        both += (s_on.float().T @ a_on.float()).double()
        spd_on_sae_off += (s_on.float().T @ (~a_on).float()).double()
        spd_off_sae_on += ((~s_on).float().T @ a_on.float()).double()
        neither += ((~s_on).float().T @ (~a_on).float()).double()

        # Top examples per sequence
        token_ids_cpu = input_ids.cpu()
        spd_seq = spd_sub.reshape(B, S, n_top_spd)
        sae_seq = sae_sub.reshape(B, S, n_top_sae)

        for b in range(B):
            seq_tokens = token_ids_cpu[b].tolist()
            for i in range(n_top_spd):
                acts = spd_seq[b, :, i]
                max_val = acts.max().item()
                if max_val > 0:
                    heap = spd_top_examples[i]
                    entry = (max_val, batch_idx * B + b, seq_tokens, acts.tolist())
                    if len(heap) < TOP_K_EXAMPLES:
                        heapq.heappush(heap, entry)
                    elif max_val > heap[0][0]:
                        heapq.heapreplace(heap, entry)
            for i in range(n_top_sae):
                acts = sae_seq[b, :, i]
                max_val = acts.max().item()
                if max_val > 0:
                    heap = sae_top_examples[i]
                    entry = (max_val, batch_idx * B + b, seq_tokens, acts.tolist())
                    if len(heap) < TOP_K_EXAMPLES:
                        heapq.heappush(heap, entry)
                    elif max_val > heap[0][0]:
                        heapq.heapreplace(heap, entry)

    # Conditional probabilities
    sae_total_on = both + spd_off_sae_on
    sae_total_off = spd_on_sae_off + neither
    spd_total_on = both + spd_on_sae_off
    spd_total_off = spd_off_sae_on + neither

    p_spd_given_sae = both / sae_total_on.clamp(min=1)
    p_spd_given_not_sae = spd_on_sae_off / sae_total_off.clamp(min=1)
    lift_spd = p_spd_given_sae / p_spd_given_not_sae.clamp(min=1e-10)

    p_sae_given_spd = both / spd_total_on.clamp(min=1)
    p_sae_given_not_spd = spd_off_sae_on / spd_total_off.clamp(min=1)
    lift_sae = p_sae_given_spd / p_sae_given_not_spd.clamp(min=1e-10)

    combined_lift = (lift_spd * lift_sae).sqrt()

    def format_examples(heap):
        examples = sorted(heap, key=lambda x: -x[0])
        result = []
        for max_val, _, seq_tokens, act_values in examples:
            per_token = [tokenizer.decode([t]) for t in seq_tokens]
            result.append({
                "max_activation": max_val,
                "tokens": per_token,
                "activations": act_values,
            })
        return result

    def sae_flat_to_label(flat_idx):
        flat_idx = flat_idx.item() if hasattr(flat_idx, 'item') else flat_idx
        sae_layer = flat_idx // sae_dict
        sae_feature = flat_idx % sae_dict
        return LAYERS[sae_layer], sae_feature

    # For each SPD component, find top 10 SAE features
    print(f"  Building per-component top SAE lists...")
    all_components = {}

    for si in range(n_top_spd):
        spd_flat_idx = top_spd_flat[si].item()
        spd_layer, spd_mod_type, spd_mod_name, spd_local_idx = spd_flat[spd_flat_idx]

        lifts_for_this_spd = combined_lift[si]  # (n_top_sae,)
        top_sae_for_spd = lifts_for_this_spd.topk(min(TOP_SAE_PER_COMPONENT, n_top_sae))

        sae_matches = []
        for rank in range(len(top_sae_for_spd.values)):
            ai = top_sae_for_spd.indices[rank].item()
            sae_flat_idx = top_sae_flat[ai].item()
            sae_layer, sae_feature = sae_flat_to_label(sae_flat_idx)

            sae_matches.append({
                "sae_layer": sae_layer,
                "sae_feature": sae_feature,
                "combined_lift": combined_lift[si, ai].item(),
                "p_spd_given_sae": p_spd_given_sae[si, ai].item(),
                "p_spd_given_not_sae": p_spd_given_not_sae[si, ai].item(),
                "lift_spd": lift_spd[si, ai].item(),
                "p_sae_given_spd": p_sae_given_spd[si, ai].item(),
                "p_sae_given_not_spd": p_sae_given_not_spd[si, ai].item(),
                "lift_sae": lift_sae[si, ai].item(),
                "sae_fire_rate": sae_fire_rate[sae_flat_idx].item(),
                "sae_examples": format_examples(sae_top_examples[ai]),
            })

        comp_key = f"L{spd_layer}_{spd_mod_type}[{spd_local_idx}]"
        all_components[comp_key] = {
            "spd_layer": spd_layer,
            "spd_mod_type": spd_mod_type,
            "spd_local_idx": spd_local_idx,
            "spd_fire_rate": spd_fire_rate[spd_flat_idx].item(),
            "spd_examples": format_examples(spd_top_examples[si]),
            "sae_matches": sae_matches,
        }

    return all_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_batches", type=int, default=50)
    args = parser.parse_args()

    from analysis.collect_spd_activations import load_spd_model

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading jose SPD model...")
    spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    saes = {l: load_sae(Path(f"checkpoints/jose_sae_resid_local_k{SAE_K}_layer{l}")) for l in LAYERS}

    all_components = collect_all_data(spd_model, saes, base_model, tokenizer, args.n_batches)

    save_path = OUTPUT_DIR / "dashboard_data_v2.json"
    with open(save_path, "w") as f:
        json.dump(all_components, f)
    print(f"\nSaved {len(all_components)} components to {save_path}")


if __name__ == "__main__":
    main()
