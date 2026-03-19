"""Build an interactive HTML dashboard for SPD component / SAE feature comparison.

For each layer, collects:
1. Top activating examples for SPD components and SAE features
2. Conditional activation probabilities: P(SPD|SAE), P(SPD|~SAE), P(SAE|SPD), P(SAE|~SPD)
3. Lift ratios for identifying strong associations

Outputs an HTML file with side-by-side comparison of paired SPD components and SAE features.

Usage:
    python experiments/exp_042_spd_sae_interaction/build_dashboard.py
    python experiments/exp_042_spd_sae_interaction/build_dashboard.py --layer 1
    python experiments/exp_042_spd_sae_interaction/build_dashboard.py --n_batches 100
"""

import argparse
import html
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn.functional as F
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
TOP_K_EXAMPLES = 10  # top activating examples per feature


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
def collect_dashboard_data(
    spd_model, sae, base_model, tokenizer, layer_idx: int,
    n_batches: int, top_n_pairs: int = 50,
):
    """Collect all data needed for the dashboard for one layer."""
    cfc_name = f"h.{layer_idx}.mlp.c_fc"
    n_cfc = spd_model.module_to_c[cfc_name]
    sae_dict = sae.cfg.dict_size

    # Streaming accumulators for conditional probabilities
    # We track co-occurrence counts for top SPD components vs all SAE features
    # First pass: identify top SPD components by firing rate
    print(f"  Pass 1: identifying active components...")

    spd_fire_count = torch.zeros(n_cfc, device="cpu")
    sae_fire_count = torch.zeros(sae_dict, device="cpu")
    n_tokens_total = 0

    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)

    # Collect batches and run pass 1
    all_input_ids = []
    for _ in tqdm(range(n_batches), desc="Loading + pass 1"):
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

        # SPD CI
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_cfc = ci.lower_leaky[cfc_name].reshape(-1, n_cfc)  # (B*S, n_cfc)
        spd_fire_count += (ci_cfc > 0).float().sum(dim=0).cpu()

        # SAE activations
        captured = {}
        hooks = []
        def _make_hook(li):
            def _hook(_mod, _inp, out):
                captured[li] = out.detach()
            return _hook
        hooks.append(base_model.h[layer_idx].register_forward_hook(_make_hook(layer_idx)))
        base_model(input_ids)
        for h in hooks:
            h.remove()
        resid = captured[layer_idx].reshape(-1, 768)
        sae_acts = sae.encode(resid).float()
        sae_fire_count += (sae_acts > 0).float().sum(dim=0).cpu()

    # Select top SPD components and SAE features by firing rate
    top_spd_indices = spd_fire_count.topk(min(200, n_cfc)).indices
    top_sae_indices = sae_fire_count.topk(min(200, sae_dict)).indices
    n_spd = len(top_spd_indices)
    n_sae = len(top_sae_indices)

    print(f"  Pass 2: collecting co-occurrences and top examples ({n_spd} SPD x {n_sae} SAE)...")

    # Co-occurrence counts
    both_active = torch.zeros(n_spd, n_sae, dtype=torch.float64)
    spd_active_sae_inactive = torch.zeros(n_spd, n_sae, dtype=torch.float64)
    spd_inactive_sae_active = torch.zeros(n_spd, n_sae, dtype=torch.float64)
    neither_active = torch.zeros(n_spd, n_sae, dtype=torch.float64)

    # Top activating examples: use heaps
    import heapq
    spd_top_examples = {i: [] for i in range(n_spd)}  # index into top_spd_indices
    sae_top_examples = {i: [] for i in range(n_sae)}

    for batch_idx, input_ids in enumerate(tqdm(all_input_ids, desc="Pass 2")):
        # SPD CI
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_cfc = ci.lower_leaky[cfc_name]  # (B, S, n_cfc)

        # SAE
        captured = {}
        hooks = []
        def _make_hook(li):
            def _hook(_mod, _inp, out):
                captured[li] = out.detach()
            return _hook
        hooks.append(base_model.h[layer_idx].register_forward_hook(_make_hook(layer_idx)))
        base_model(input_ids)
        for h in hooks:
            h.remove()
        resid = captured[layer_idx].reshape(-1, 768)
        sae_acts = sae.encode(resid).float()

        ci_flat = ci_cfc.reshape(-1, n_cfc)  # (B*S, n_cfc)

        # Subset to top features
        ci_sub = ci_flat[:, top_spd_indices].cpu()  # (tokens, n_spd)
        sae_sub = sae_acts[:, top_sae_indices].cpu()  # (tokens, n_sae)

        spd_on = (ci_sub > 0)  # (tokens, n_spd)
        sae_on = (sae_sub > 0)  # (tokens, n_sae)

        # Co-occurrence: outer product per token, summed
        for t in range(ci_sub.shape[0]):
            s_on = spd_on[t]  # (n_spd,)
            a_on = sae_on[t]  # (n_sae,)
            both_active += (s_on.unsqueeze(1) & a_on.unsqueeze(0)).double()
            spd_active_sae_inactive += (s_on.unsqueeze(1) & ~a_on.unsqueeze(0)).double()
            spd_inactive_sae_active += (~s_on.unsqueeze(1) & a_on.unsqueeze(0)).double()
            neither_active += (~s_on.unsqueeze(1) & ~a_on.unsqueeze(0)).double()

        # Top activating examples
        token_ids_np = input_ids.cpu()
        B, S = input_ids.shape
        for b in range(B):
            seq_tokens = token_ids_np[b].tolist()
            for s in range(S):
                flat_idx = b * S + s
                # SPD top examples
                for i in range(n_spd):
                    val = ci_sub[flat_idx, i].item()
                    if val > 0:
                        heap = spd_top_examples[i]
                        entry = (val, batch_idx * B * S + flat_idx, seq_tokens, s)
                        if len(heap) < TOP_K_EXAMPLES:
                            heapq.heappush(heap, entry)
                        elif val > heap[0][0]:
                            heapq.heapreplace(heap, entry)
                # SAE top examples
                for i in range(n_sae):
                    val = sae_sub[flat_idx, i].item()
                    if val > 0:
                        heap = sae_top_examples[i]
                        entry = (val, batch_idx * B * S + flat_idx, seq_tokens, s)
                        if len(heap) < TOP_K_EXAMPLES:
                            heapq.heappush(heap, entry)
                        elif val > heap[0][0]:
                            heapq.heapreplace(heap, entry)

    # Compute conditional probabilities
    spd_total_on = both_active + spd_active_sae_inactive  # (n_spd, n_sae) but summed same
    sae_total_on = both_active + spd_inactive_sae_active
    N = float(n_tokens_total)

    # P(SPD active | SAE active) = both / sae_on
    p_spd_given_sae = both_active / sae_total_on.clamp(min=1)
    # P(SPD active | SAE inactive) = spd_on_sae_off / sae_off
    sae_total_off = spd_active_sae_inactive + neither_active
    p_spd_given_not_sae = spd_active_sae_inactive / sae_total_off.clamp(min=1)
    # Lift: P(SPD|SAE) / P(SPD|~SAE)
    lift_spd_given_sae = p_spd_given_sae / p_spd_given_not_sae.clamp(min=1e-10)

    # P(SAE active | SPD active) = both / spd_on
    p_sae_given_spd = both_active / spd_total_on.clamp(min=1)
    # P(SAE active | SPD inactive)
    spd_total_off = spd_inactive_sae_active + neither_active
    p_sae_given_not_spd = spd_inactive_sae_active / spd_total_off.clamp(min=1)
    lift_sae_given_spd = p_sae_given_spd / p_sae_given_not_spd.clamp(min=1e-10)

    # Find top pairs by geometric mean of lifts
    combined_lift = (lift_spd_given_sae * lift_sae_given_spd).sqrt()
    flat_top = combined_lift.flatten().topk(top_n_pairs)

    pairs = []
    for val, flat_idx in zip(flat_top.values.tolist(), flat_top.indices.tolist()):
        si = flat_idx // n_sae
        ai = flat_idx % n_sae
        pairs.append({
            "spd_local_idx": si,
            "sae_local_idx": ai,
            "spd_global_idx": top_spd_indices[si].item(),
            "sae_global_idx": top_sae_indices[ai].item(),
            "p_spd_given_sae": p_spd_given_sae[si, ai].item(),
            "p_spd_given_not_sae": p_spd_given_not_sae[si, ai].item(),
            "lift_spd_given_sae": lift_spd_given_sae[si, ai].item(),
            "p_sae_given_spd": p_sae_given_spd[si, ai].item(),
            "p_sae_given_not_spd": p_sae_given_not_spd[si, ai].item(),
            "lift_sae_given_spd": lift_sae_given_spd[si, ai].item(),
            "combined_lift": val,
            "spd_fire_rate": spd_fire_count[top_spd_indices[si]].item() / n_tokens_total,
            "sae_fire_rate": sae_fire_count[top_sae_indices[ai]].item() / n_tokens_total,
        })

    # Format top examples
    def format_examples(heap, tokenizer):
        examples = sorted(heap, key=lambda x: -x[0])
        result = []
        for val, _, seq_tokens, pos in examples:
            # Show context window around the activating position
            start = max(0, pos - 10)
            end = min(len(seq_tokens), pos + 11)
            tokens = seq_tokens[start:end]
            decoded = [tokenizer.decode([t]) for t in tokens]
            highlight_idx = pos - start
            result.append({
                "activation": val,
                "tokens": decoded,
                "highlight_idx": highlight_idx,
            })
        return result

    for pair in pairs:
        pair["spd_examples"] = format_examples(spd_top_examples[pair["spd_local_idx"]], tokenizer)
        pair["sae_examples"] = format_examples(sae_top_examples[pair["sae_local_idx"]], tokenizer)

    return pairs


def build_html(all_pairs: dict[int, list[dict]], save_path: Path):
    """Build the HTML dashboard."""
    html_parts = ["""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>SPD-SAE Interaction Dashboard</title>
<style>
body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 20px; background: #f8f9fa; }
h1 { color: #1a1a2e; }
h2 { color: #16213e; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }
.layer-section { margin-bottom: 40px; }
.pair-card { background: white; border-radius: 8px; padding: 16px; margin: 12px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.pair-header { display: flex; justify-content: space-between; align-items: center;
               margin-bottom: 12px; }
.pair-header h3 { margin: 0; color: #2d3748; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
.stat-box { background: #f7fafc; border-radius: 6px; padding: 10px; border: 1px solid #e2e8f0; }
.stat-box h4 { margin: 0 0 6px 0; font-size: 13px; color: #718096; }
.stat-row { display: flex; justify-content: space-between; font-size: 13px; margin: 2px 0; }
.stat-value { font-weight: 600; }
.lift-high { color: #22543d; background: #c6f6d5; padding: 1px 6px; border-radius: 3px; }
.lift-low { color: #742a2a; background: #fed7d7; padding: 1px 6px; border-radius: 3px; }
.examples { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.example-col h4 { margin: 0 0 8px 0; font-size: 13px; color: #4a5568; }
.example { font-family: monospace; font-size: 12px; margin: 4px 0; padding: 4px 8px;
           background: #f7fafc; border-radius: 4px; line-height: 1.6; white-space: pre-wrap; }
.highlight { background: #fefcbf; font-weight: bold; border-bottom: 2px solid #d69e2e; }
.act-val { color: #718096; font-size: 11px; }
.tab-container { margin-bottom: 20px; }
.tab-btn { padding: 8px 16px; border: 1px solid #e2e8f0; background: white;
           cursor: pointer; font-size: 14px; border-radius: 6px 6px 0 0; }
.tab-btn.active { background: #4299e1; color: white; border-color: #4299e1; }
.tab-content { display: none; }
.tab-content.active { display: block; }
</style>
<script>
function showLayer(layerIdx) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('btn-' + layerIdx).classList.add('active');
    document.getElementById('layer-' + layerIdx).classList.add('active');
}
</script>
</head><body>
<h1>SPD Component — SAE Feature Interaction Dashboard</h1>
<p>Top pairs by combined lift (geometric mean of P(SPD|SAE)/P(SPD|~SAE) and P(SAE|SPD)/P(SAE|~SPD)).</p>
<div class="tab-container">
"""]

    for layer_idx in sorted(all_pairs.keys()):
        active = "active" if layer_idx == min(all_pairs.keys()) else ""
        html_parts.append(f'<button class="tab-btn {active}" id="btn-{layer_idx}" onclick="showLayer({layer_idx})">Layer {layer_idx}</button>\n')

    html_parts.append('</div>\n')

    for layer_idx in sorted(all_pairs.keys()):
        active = "active" if layer_idx == min(all_pairs.keys()) else ""
        html_parts.append(f'<div class="tab-content {active}" id="layer-{layer_idx}">\n')
        html_parts.append(f'<h2>Layer {layer_idx}</h2>\n')

        for rank, pair in enumerate(all_pairs[layer_idx]):
            lift_class = "lift-high" if pair["combined_lift"] > 10 else "lift-low" if pair["combined_lift"] < 2 else ""
            html_parts.append(f'''<div class="pair-card">
<div class="pair-header">
    <h3>#{rank+1}: SPD c_fc[{pair["spd_global_idx"]}] ↔ SAE[{pair["sae_global_idx"]}]</h3>
    <span class="{lift_class}">Combined lift: {pair["combined_lift"]:.1f}x</span>
</div>
<div class="stats">
    <div class="stat-box">
        <h4>SPD component {pair["spd_global_idx"]} (fire rate: {pair["spd_fire_rate"]:.4f})</h4>
        <div class="stat-row"><span>P(SPD on | SAE on)</span><span class="stat-value">{pair["p_spd_given_sae"]:.4f}</span></div>
        <div class="stat-row"><span>P(SPD on | SAE off)</span><span class="stat-value">{pair["p_spd_given_not_sae"]:.4f}</span></div>
        <div class="stat-row"><span>Lift</span><span class="stat-value">{pair["lift_spd_given_sae"]:.1f}x</span></div>
    </div>
    <div class="stat-box">
        <h4>SAE feature {pair["sae_global_idx"]} (fire rate: {pair["sae_fire_rate"]:.4f})</h4>
        <div class="stat-row"><span>P(SAE on | SPD on)</span><span class="stat-value">{pair["p_sae_given_spd"]:.4f}</span></div>
        <div class="stat-row"><span>P(SAE on | SPD off)</span><span class="stat-value">{pair["p_sae_given_not_spd"]:.4f}</span></div>
        <div class="stat-row"><span>Lift</span><span class="stat-value">{pair["lift_sae_given_spd"]:.1f}x</span></div>
    </div>
</div>
<div class="examples">
    <div class="example-col"><h4>SPD top activations</h4>
''')
            for ex in pair["spd_examples"][:5]:
                tokens_html = ""
                for i, tok in enumerate(ex["tokens"]):
                    escaped = html.escape(tok)
                    if i == ex["highlight_idx"]:
                        tokens_html += f'<span class="highlight">{escaped}</span>'
                    else:
                        tokens_html += escaped
                html_parts.append(f'<div class="example"><span class="act-val">[{ex["activation"]:.3f}]</span> {tokens_html}</div>\n')

            html_parts.append('</div>\n<div class="example-col"><h4>SAE top activations</h4>\n')
            for ex in pair["sae_examples"][:5]:
                tokens_html = ""
                for i, tok in enumerate(ex["tokens"]):
                    escaped = html.escape(tok)
                    if i == ex["highlight_idx"]:
                        tokens_html += f'<span class="highlight">{escaped}</span>'
                    else:
                        tokens_html += escaped
                html_parts.append(f'<div class="example"><span class="act-val">[{ex["activation"]:.3f}]</span> {tokens_html}</div>\n')

            html_parts.append('</div></div></div>\n')

        html_parts.append('</div>\n')

    html_parts.append('</body></html>')

    save_path.write_text("".join(html_parts))
    print(f"Dashboard saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--top_n_pairs", type=int, default=30)
    args = parser.parse_args()

    from analysis.collect_spd_activations import load_spd_model

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading jose SPD model...")
    spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
    spd_model.to(DEVICE)
    base_model = spd_model.target_model

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Load SAEs
    sae_paths = {
        layer: Path(f"checkpoints/jose_sae_resid_local_k{SAE_K}_layer{layer}")
        for layer in args.layers
    }
    saes = {layer: load_sae(sae_paths[layer]) for layer in args.layers}

    all_pairs = {}
    for layer_idx in args.layers:
        print(f"\n{'='*60}")
        print(f"  Layer {layer_idx}")
        print(f"{'='*60}")
        pairs = collect_dashboard_data(
            spd_model, saes[layer_idx], base_model, tokenizer,
            layer_idx, args.n_batches, args.top_n_pairs,
        )
        all_pairs[layer_idx] = pairs
        print(f"  Top pair: SPD[{pairs[0]['spd_global_idx']}] ↔ SAE[{pairs[0]['sae_global_idx']}] "
              f"lift={pairs[0]['combined_lift']:.1f}x")

    build_html(all_pairs, OUTPUT_DIR / "spd_sae_dashboard.html")

    # Also save raw data
    json_data = {}
    for layer_idx, pairs in all_pairs.items():
        json_data[str(layer_idx)] = pairs
    with open(OUTPUT_DIR / "spd_sae_dashboard_data.json", "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Data saved to {OUTPUT_DIR / 'spd_sae_dashboard_data.json'}")


if __name__ == "__main__":
    main()
