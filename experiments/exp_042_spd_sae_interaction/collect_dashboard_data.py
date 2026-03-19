"""Collect data for the interactive SPD-SAE dashboard.

Saves per-layer data to a .pt file containing:
- Top pairs ranked by combined lift
- Top activating examples with full-sequence activation values
- Conditional probability statistics

Excludes features/components with firing rate > 0.9.

Usage:
    python experiments/exp_042_spd_sae_interaction/collect_dashboard_data.py
    python experiments/exp_042_spd_sae_interaction/collect_dashboard_data.py --layers 1 --n_batches 100
"""

import argparse
import heapq
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import wandb
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
def collect_layer_data(
    spd_model, sae, base_model, tokenizer, layer_idx: int, n_batches: int,
    top_n_pairs: int = 50,
):
    cfc_name = f"h.{layer_idx}.mlp.c_fc"
    n_cfc = spd_model.module_to_c[cfc_name]
    sae_dict = sae.cfg.dict_size

    # Pass 1: firing rates
    print(f"  Pass 1: firing rates...")
    spd_fire_count = torch.zeros(n_cfc, device="cpu")
    sae_fire_count = torch.zeros(sae_dict, device="cpu")
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

        bs = input_ids.shape[0] * input_ids.shape[1]
        n_tokens_total += bs

        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_cfc = ci.lower_leaky[cfc_name].reshape(-1, n_cfc)
        spd_fire_count += (ci_cfc > 0).float().sum(dim=0).cpu()

        captured = {}
        def _make_hook(li):
            def _hook(_mod, _inp, out):
                captured[li] = out.detach()
            return _hook
        h = base_model.h[layer_idx].register_forward_hook(_make_hook(layer_idx))
        base_model(input_ids)
        h.remove()
        resid = captured[layer_idx].reshape(-1, 768)
        sae_acts = sae.encode(resid).float()
        sae_fire_count += (sae_acts > 0).float().sum(dim=0).cpu()

    spd_fire_rate = spd_fire_count / n_tokens_total
    sae_fire_rate = sae_fire_count / n_tokens_total

    # Filter: exclude features with fire rate > MAX_FIRE_RATE
    spd_mask = spd_fire_rate <= MAX_FIRE_RATE
    sae_mask = sae_fire_rate <= MAX_FIRE_RATE
    print(f"  Filtered: {spd_mask.sum()}/{n_cfc} SPD, {sae_mask.sum()}/{sae_dict} SAE (fire rate <= {MAX_FIRE_RATE})")

    # Select top by fire rate among the filtered
    spd_rates_filtered = spd_fire_rate.clone()
    spd_rates_filtered[~spd_mask] = -1
    sae_rates_filtered = sae_fire_rate.clone()
    sae_rates_filtered[~sae_mask] = -1

    top_spd_indices = spd_rates_filtered.topk(min(200, spd_mask.sum().item())).indices
    top_sae_indices = sae_rates_filtered.topk(min(200, sae_mask.sum().item())).indices
    n_spd = len(top_spd_indices)
    n_sae = len(top_sae_indices)

    print(f"  Pass 2: co-occurrences + examples ({n_spd} SPD x {n_sae} SAE)...")

    # Co-occurrence counts
    both_active = torch.zeros(n_spd, n_sae, dtype=torch.float64)
    spd_on_sae_off = torch.zeros(n_spd, n_sae, dtype=torch.float64)
    spd_off_sae_on = torch.zeros(n_spd, n_sae, dtype=torch.float64)
    neither = torch.zeros(n_spd, n_sae, dtype=torch.float64)

    # Top activating examples: store full sequence activations
    # heap entry: (max_act_in_seq, global_token_idx, token_ids_list, act_values_list)
    spd_top = {i: [] for i in range(n_spd)}
    sae_top = {i: [] for i in range(n_sae)}

    for batch_idx, input_ids in enumerate(tqdm(all_input_ids, desc="Pass 2")):
        B, S = input_ids.shape

        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_cfc = ci.lower_leaky[cfc_name]  # (B, S, n_cfc)

        captured = {}
        def _make_hook(li):
            def _hook(_mod, _inp, out):
                captured[li] = out.detach()
            return _hook
        h = base_model.h[layer_idx].register_forward_hook(_make_hook(layer_idx))
        base_model(input_ids)
        h.remove()
        resid = captured[layer_idx].reshape(-1, 768)
        sae_acts = sae.encode(resid).float().reshape(B, S, -1)  # (B, S, sae_dict)

        # Subset
        ci_sub = ci_cfc[:, :, top_spd_indices].cpu()  # (B, S, n_spd)
        sae_sub = sae_acts[:, :, top_sae_indices].cpu()  # (B, S, n_sae)

        # Co-occurrence (flatten over B*S)
        ci_flat = ci_sub.reshape(-1, n_spd)
        sae_flat = sae_sub.reshape(-1, n_sae)
        s_on = ci_flat > 0
        a_on = sae_flat > 0

        # Vectorized co-occurrence
        both_active += (s_on.float().T @ a_on.float()).double()
        spd_on_sae_off += (s_on.float().T @ (~a_on).float()).double()
        spd_off_sae_on += ((~s_on).float().T @ a_on.float()).double()
        neither += ((~s_on).float().T @ (~a_on).float()).double()

        # Top examples: per sequence
        token_ids_cpu = input_ids.cpu()
        for b in range(B):
            seq_tokens = token_ids_cpu[b].tolist()

            for i in range(n_spd):
                acts_seq = ci_sub[b, :, i]  # (S,)
                max_val = acts_seq.max().item()
                if max_val > 0:
                    heap = spd_top[i]
                    entry = (max_val, batch_idx * B + b, seq_tokens, acts_seq.tolist())
                    if len(heap) < TOP_K_EXAMPLES:
                        heapq.heappush(heap, entry)
                    elif max_val > heap[0][0]:
                        heapq.heapreplace(heap, entry)

            for i in range(n_sae):
                acts_seq = sae_sub[b, :, i]  # (S,)
                max_val = acts_seq.max().item()
                if max_val > 0:
                    heap = sae_top[i]
                    entry = (max_val, batch_idx * B + b, seq_tokens, acts_seq.tolist())
                    if len(heap) < TOP_K_EXAMPLES:
                        heapq.heappush(heap, entry)
                    elif max_val > heap[0][0]:
                        heapq.heapreplace(heap, entry)

    # Conditional probabilities
    sae_total_on = both_active + spd_off_sae_on
    sae_total_off = spd_on_sae_off + neither
    spd_total_on = both_active + spd_on_sae_off
    spd_total_off = spd_off_sae_on + neither

    p_spd_given_sae = both_active / sae_total_on.clamp(min=1)
    p_spd_given_not_sae = spd_on_sae_off / sae_total_off.clamp(min=1)
    lift_spd = p_spd_given_sae / p_spd_given_not_sae.clamp(min=1e-10)

    p_sae_given_spd = both_active / spd_total_on.clamp(min=1)
    p_sae_given_not_spd = spd_off_sae_on / spd_total_off.clamp(min=1)
    lift_sae = p_sae_given_spd / p_sae_given_not_spd.clamp(min=1e-10)

    combined_lift = (lift_spd * lift_sae).sqrt()

    # Top pairs
    flat_top = combined_lift.flatten().topk(top_n_pairs)

    def format_examples(heap, tokenizer):
        examples = sorted(heap, key=lambda x: -x[0])
        result = []
        for max_val, _, seq_tokens, act_values in examples:
            decoded = tokenizer.decode(seq_tokens)
            per_token = [tokenizer.decode([t]) for t in seq_tokens]
            result.append({
                "max_activation": max_val,
                "tokens": per_token,
                "activations": act_values,
            })
        return result

    pairs = []
    for val, flat_idx in zip(flat_top.values.tolist(), flat_top.indices.tolist()):
        si = flat_idx // n_sae
        ai = flat_idx % n_sae
        pairs.append({
            "spd_global_idx": top_spd_indices[si].item(),
            "sae_global_idx": top_sae_indices[ai].item(),
            "p_spd_given_sae": p_spd_given_sae[si, ai].item(),
            "p_spd_given_not_sae": p_spd_given_not_sae[si, ai].item(),
            "lift_spd": lift_spd[si, ai].item(),
            "p_sae_given_spd": p_sae_given_spd[si, ai].item(),
            "p_sae_given_not_spd": p_sae_given_not_spd[si, ai].item(),
            "lift_sae": lift_sae[si, ai].item(),
            "combined_lift": val,
            "spd_fire_rate": spd_fire_rate[top_spd_indices[si]].item(),
            "sae_fire_rate": sae_fire_rate[top_sae_indices[ai]].item(),
            "spd_examples": format_examples(spd_top[si], tokenizer),
            "sae_examples": format_examples(sae_top[ai], tokenizer),
        })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--top_n_pairs", type=int, default=50)
    args = parser.parse_args()

    from analysis.collect_spd_activations import load_spd_model

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading jose SPD model...")
    spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    saes = {
        layer: load_sae(Path(f"checkpoints/jose_sae_resid_local_k{SAE_K}_layer{layer}"))
        for layer in args.layers
    }

    all_data = {}
    for layer_idx in args.layers:
        print(f"\n{'='*60}\n  Layer {layer_idx}\n{'='*60}")
        pairs = collect_layer_data(
            spd_model, saes[layer_idx], base_model, tokenizer,
            layer_idx, args.n_batches, args.top_n_pairs,
        )
        all_data[layer_idx] = pairs
        print(f"  Top: SPD[{pairs[0]['spd_global_idx']}] <-> SAE[{pairs[0]['sae_global_idx']}] "
              f"lift={pairs[0]['combined_lift']:.1f}x")

    save_path = OUTPUT_DIR / "dashboard_data.json"
    # Convert int keys to strings for JSON
    json_data = {str(k): v for k, v in all_data.items()}
    with open(save_path, "w") as f:
        json.dump(json_data, f)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
