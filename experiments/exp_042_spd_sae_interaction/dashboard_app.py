"""Interactive Gradio dashboard for SPD-SAE feature interaction.

Browse by SPD component (c_fc or down_proj, any layer): see its top 10
associated SAE features across ALL residual stream layers, ranked by
combined lift, with side-by-side activating examples showing BOTH features'
activations in two colours (blue=SPD, orange=SAE).

Usage:
    python experiments/exp_042_spd_sae_interaction/dashboard_app.py
    python experiments/exp_042_spd_sae_interaction/dashboard_app.py --share
    python experiments/exp_042_spd_sae_interaction/dashboard_app.py --no-models  # skip model loading
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import gradio as gr
import torch

DATA_PATH = Path("experiments/exp_042_spd_sae_interaction/output/dashboard_data_v2.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]
SAE_K = 32


# =============================================================================
# Model loading
# =============================================================================

def load_sae(checkpoint_dir: Path):
    from nn_decompositions.transcoder import BatchTopKTranscoder
    from nn_decompositions.config import EncoderConfig

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


def load_models():
    from analysis.collect_spd_activations import load_spd_model
    from transformers import AutoTokenizer

    print("Loading SPD model...")
    spd_model, _ = load_spd_model("goodfire/spd/s-55ea3f9b")
    spd_model.to(DEVICE)
    spd_model.eval()

    base_model = spd_model.target_model

    print("Loading SAE models...")
    saes = {l: load_sae(Path(f"checkpoints/jose_sae_resid_local_k{SAE_K}_layer{l}")) for l in LAYERS}

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    print("Models loaded.")
    return spd_model, base_model, saes, tokenizer


# =============================================================================
# Cross-activation computation
# =============================================================================

@torch.no_grad()
def compute_cross_activations(
    examples: list[dict],
    spd_model,
    base_model,
    saes: dict,
    tokenizer,
    spd_layer: int,
    spd_mod_type: str,
    spd_local_idx: int,
    sae_layer: int,
    sae_feature: int,
) -> list[dict]:
    """Run example sequences through both models, return per-token activations for both features."""
    results = []
    mod_name = f"h.{spd_layer}.mlp.{spd_mod_type}"

    for ex in examples:
        text = "".join(ex["tokens"])
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        S = input_ids.shape[1]

        # SPD activations
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_vals = ci.lower_leaky[mod_name].reshape(S, -1)
        spd_acts = ci_vals[:, spd_local_idx].cpu().tolist()

        # SAE activations
        captured = {}
        def _hook(_mod, _inp, out_val):
            captured[0] = out_val.detach()
        hook = base_model.h[sae_layer].register_forward_hook(_hook)
        base_model(input_ids)
        hook.remove()

        resid = captured[0].reshape(S, -1)
        sae_acts_all = saes[sae_layer].encode(resid).float()
        sae_acts = sae_acts_all[:, sae_feature].cpu().tolist()

        results.append({
            "tokens": ex["tokens"][:S],
            "max_activation": ex["max_activation"],
            "spd_activations": spd_acts,
            "sae_activations": sae_acts,
        })

    return results


# =============================================================================
# Dual-colour rendering
# =============================================================================

def format_dual_highlighted_sequence(
    tokens: list[str],
    spd_acts: list[float],
    sae_acts: list[float],
    max_tokens: int = 80,
) -> str:
    """Render tokens with blue top bar (SPD) and orange bottom bar (SAE)."""
    # Centre on peak of whichever has higher max
    spd_peak = max(range(len(spd_acts)), key=lambda i: spd_acts[i])
    sae_peak = max(range(len(sae_acts)), key=lambda i: sae_acts[i])
    peak_pos = spd_peak if max(spd_acts) >= max(sae_acts) else sae_peak
    start = max(0, peak_pos - max_tokens // 2)
    end = min(len(tokens), start + max_tokens)
    start = max(0, end - max_tokens)

    tokens = tokens[start:end]
    spd = spd_acts[start:end]
    sae = sae_acts[start:end]
    max_spd = max(spd) if max(spd) > 0 else 1.0
    max_sae = max(sae) if max(sae) > 0 else 1.0

    parts = []
    for tok, s_act, a_act in zip(tokens, spd, sae):
        escaped = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s_int = min(s_act / max_spd, 1.0) if s_act > 0 else 0
        a_int = min(a_act / max_sae, 1.0) if a_act > 0 else 0

        if s_int == 0 and a_int == 0:
            parts.append(f'<span style="color:#888">{escaped}</span>')
            continue

        # Gradient: top = blue (SPD), bottom = orange (SAE)
        s_alpha = 0.15 + 0.65 * s_int if s_int > 0 else 0
        a_alpha = 0.15 + 0.65 * a_int if a_int > 0 else 0

        bg = (
            f"linear-gradient(to bottom, "
            f"rgba(43,108,176,{s_alpha:.2f}) 50%, "
            f"rgba(192,86,33,{a_alpha:.2f}) 50%)"
        )
        weight = "bold" if max(s_int, a_int) > 0.5 else "normal"
        parts.append(
            f'<span style="background:{bg};font-weight:{weight};'
            f'border-radius:2px;padding:1px 2px;display:inline-block;'
            f'line-height:1.4">{escaped}</span>'
        )
    return "".join(parts)


def render_dual_examples(examples: list[dict], max_examples: int = 10) -> str:
    """Render examples with dual SPD (blue) / SAE (orange) highlighting."""
    legend = (
        '<div style="font-size:11px;margin-bottom:8px;color:#555">'
        '<span style="display:inline-block;width:12px;height:12px;'
        'background:rgba(43,108,176,0.6);border-radius:2px;vertical-align:middle"></span>'
        ' <b>SPD</b> (top half) &nbsp;&nbsp;'
        '<span style="display:inline-block;width:12px;height:12px;'
        'background:rgba(192,86,33,0.6);border-radius:2px;vertical-align:middle"></span>'
        ' <b>SAE</b> (bottom half)</div>'
    )
    html = f'<div style="font-family:monospace;font-size:12px;line-height:1.8">{legend}'
    for i, ex in enumerate(examples[:max_examples]):
        seq_html = format_dual_highlighted_sequence(
            ex["tokens"], ex["spd_activations"], ex["sae_activations"]
        )
        html += (
            f'<div style="margin:6px 0;padding:6px 8px;background:#fafafa;'
            f'border-radius:4px;border-left:3px solid #6b21a8">'
            f'<span style="color:#999;font-size:10px">#{i+1} (max={ex["max_activation"]:.3f})</span>'
            f'<br>{seq_html}</div>'
        )
    html += '</div>'
    return html


def format_highlighted_sequence(tokens: list[str], activations: list[float], max_tokens: int = 80) -> str:
    """Single-colour highlighting (fallback when models not loaded)."""
    peak_pos = max(range(len(activations)), key=lambda i: activations[i])
    start = max(0, peak_pos - max_tokens // 2)
    end = min(len(tokens), start + max_tokens)
    start = max(0, end - max_tokens)

    tokens = tokens[start:end]
    acts = activations[start:end]
    max_act = max(acts) if max(acts) > 0 else 1.0

    parts = []
    for tok, act in zip(tokens, acts):
        escaped = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if act > 0:
            intensity = min(act / max_act, 1.0)
            r, g, b = 255, int(255 - intensity * 100), int(200 - intensity * 200)
            a = 0.3 + 0.7 * intensity
            weight = "bold" if intensity > 0.5 else "normal"
            parts.append(
                f'<span style="background:rgba({r},{g},{b},{a:.2f});'
                f'font-weight:{weight};border-radius:2px;padding:0 1px">{escaped}</span>'
            )
        else:
            parts.append(f'<span style="color:#888">{escaped}</span>')
    return "".join(parts)


def render_examples(examples: list[dict], color: str, max_examples: int = 10) -> str:
    html = '<div style="font-family:monospace;font-size:12px;line-height:1.8">'
    for i, ex in enumerate(examples[:max_examples]):
        seq_html = format_highlighted_sequence(ex["tokens"], ex["activations"])
        html += (
            f'<div style="margin:6px 0;padding:6px 8px;background:#fafafa;'
            f'border-radius:4px;border-left:3px solid {color}">'
            f'<span style="color:#999;font-size:10px">#{i+1} (max={ex["max_activation"]:.3f})</span>'
            f'<br>{seq_html}</div>'
        )
    html += '</div>'
    return html


# =============================================================================
# App builder
# =============================================================================

def build_app(components: dict, models=None):
    has_models = models is not None
    if has_models:
        spd_model, base_model, saes, tokenizer = models

    by_layer = {}
    for comp_key, comp_data in components.items():
        layer = comp_data["spd_layer"]
        by_layer.setdefault(layer, []).append((comp_key, comp_data))
    for layer in by_layer:
        by_layer[layer].sort(key=lambda x: -x[1]["sae_matches"][0]["combined_lift"] if x[1]["sae_matches"] else 0)

    layers = sorted(by_layer.keys())

    def get_component_choices(layer):
        choices = []
        for comp_key, comp_data in by_layer.get(layer, []):
            top_lift = comp_data["sae_matches"][0]["combined_lift"] if comp_data["sae_matches"] else 0
            mod = comp_data["spd_mod_type"]
            idx = comp_data["spd_local_idx"]
            fr = comp_data["spd_fire_rate"]
            choices.append(f"{mod}[{idx}] (fire={fr:.4f}, top_lift={top_lift:.1f}x)")
        return choices

    def get_sae_choices(comp_key):
        comp = components.get(comp_key, {})
        matches = comp.get("sae_matches", [])
        return [
            f"L{m['sae_layer']}_SAE[{m['sae_feature']}] (lift={m['combined_lift']:.1f}x, "
            f"P(SPD|SAE)={m['p_spd_given_sae']:.3f}, P(SAE|SPD)={m['p_sae_given_spd']:.3f})"
            for m in matches
        ]

    def comp_key_from_choice(layer, comp_str):
        mod_and_idx = comp_str.split("(")[0].strip()
        mod = mod_and_idx.split("[")[0]
        idx = int(mod_and_idx.split("[")[1].split("]")[0])
        return f"L{layer}_{mod}[{idx}]"

    def parse_sae_choice(sae_str):
        sae_part = sae_str.split("(")[0].strip()
        sae_layer = int(sae_part.split("_")[0].replace("L", ""))
        sae_feature = int(sae_part.split("[")[1].split("]")[0])
        return sae_layer, sae_feature

    def get_dual_examples(comp, sae_layer, sae_feature):
        """Compute cross-activations for both sets of examples."""
        spd_cross = compute_cross_activations(
            comp["spd_examples"], spd_model, base_model, saes, tokenizer,
            comp["spd_layer"], comp["spd_mod_type"], comp["spd_local_idx"],
            sae_layer, sae_feature,
        )
        # Find the SAE match to get its examples
        for match in comp["sae_matches"]:
            if match["sae_layer"] == sae_layer and match["sae_feature"] == sae_feature:
                sae_cross = compute_cross_activations(
                    match["sae_examples"], spd_model, base_model, saes, tokenizer,
                    comp["spd_layer"], comp["spd_mod_type"], comp["spd_local_idx"],
                    sae_layer, sae_feature,
                )
                return spd_cross, sae_cross
        return spd_cross, []

    def render_pair(comp, sae_layer, sae_feature):
        for match in comp["sae_matches"]:
            if match["sae_layer"] == sae_layer and match["sae_feature"] == sae_feature:
                stats = render_stats(comp, match)
                if has_models:
                    spd_cross, sae_cross = get_dual_examples(comp, sae_layer, sae_feature)
                    spd_html = render_dual_examples(spd_cross)
                    sae_html = render_dual_examples(sae_cross)
                else:
                    spd_html = render_examples(comp["spd_examples"], "#2b6cb0")
                    sae_html = render_examples(match["sae_examples"], "#c05621")
                return stats, spd_html, sae_html
        return "", "", ""

    def on_layer_change(layer_str):
        layer = int(layer_str.split()[-1])
        comp_choices = get_component_choices(layer)
        if not comp_choices:
            return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
        comp_key = comp_key_from_choice(layer, comp_choices[0])
        sae_choices = get_sae_choices(comp_key)
        return (
            gr.update(choices=comp_choices, value=comp_choices[0]),
            gr.update(choices=sae_choices, value=sae_choices[0] if sae_choices else None),
        )

    def on_comp_change(layer_str, comp_str):
        if not comp_str:
            return gr.update(choices=[], value=None), "", "", ""
        layer = int(layer_str.split()[-1])
        comp_key = comp_key_from_choice(layer, comp_str)
        sae_choices = get_sae_choices(comp_key)
        comp = components[comp_key]
        if sae_choices and comp["sae_matches"]:
            match = comp["sae_matches"][0]
            sae_layer, sae_feature = match["sae_layer"], match["sae_feature"]
            stats, spd_html, sae_html = render_pair(comp, sae_layer, sae_feature)
            return gr.update(choices=sae_choices, value=sae_choices[0]), stats, spd_html, sae_html
        spd_html = render_examples(comp["spd_examples"], "#2b6cb0")
        return gr.update(choices=[], value=None), "", spd_html, ""

    def on_sae_change(layer_str, comp_str, sae_str):
        if not comp_str or not sae_str:
            return "", "", ""
        layer = int(layer_str.split()[-1])
        comp_key = comp_key_from_choice(layer, comp_str)
        comp = components[comp_key]
        sae_layer, sae_feature = parse_sae_choice(sae_str)
        return render_pair(comp, sae_layer, sae_feature)

    def render_stats(comp, match):
        return f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:12px 0">
            <div style="background:#f7fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px">
                <h4 style="margin:0 0 8px 0;color:#2b6cb0">
                    SPD L{comp['spd_layer']} {comp['spd_mod_type']}[{comp['spd_local_idx']}]
                </h4>
                <div style="font-size:13px">
                    <div>Fire rate: <b>{comp['spd_fire_rate']:.4f}</b></div>
                    <div>P(SPD on | SAE on): <b>{match['p_spd_given_sae']:.4f}</b></div>
                    <div>P(SPD on | SAE off): <b>{match['p_spd_given_not_sae']:.4f}</b></div>
                    <div>Lift: <b style="color:#22543d">{match['lift_spd']:.1f}x</b></div>
                </div>
            </div>
            <div style="background:#f7fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px">
                <h4 style="margin:0 0 8px 0;color:#c05621">
                    SAE L{match['sae_layer']} feature[{match['sae_feature']}]
                </h4>
                <div style="font-size:13px">
                    <div>Fire rate: <b>{match['sae_fire_rate']:.4f}</b></div>
                    <div>P(SAE on | SPD on): <b>{match['p_sae_given_spd']:.4f}</b></div>
                    <div>P(SAE on | SPD off): <b>{match['p_sae_given_not_spd']:.4f}</b></div>
                    <div>Lift: <b style="color:#22543d">{match['lift_sae']:.1f}x</b></div>
                </div>
            </div>
        </div>
        <div style="text-align:center;font-size:15px;margin:8px 0">
            Combined lift: <b style="font-size:18px;color:#6b21a8">{match['combined_lift']:.1f}x</b>
        </div>
        """

    # Build reverse index
    reverse_index = defaultdict(list)
    for comp_key, comp in components.items():
        for match in comp["sae_matches"]:
            sae_key = f"L{match['sae_layer']}_F{match['sae_feature']}"
            reverse_index[sae_key].append((comp_key, comp, match))
    for sae_key in reverse_index:
        reverse_index[sae_key].sort(key=lambda x: -x[2]["combined_lift"])

    def get_reverse_sae_choices():
        items = sorted(reverse_index.items(), key=lambda x: -x[1][0][2]["combined_lift"])
        return [
            f"{sae_key} ({len(comps)} SPD matches, top_lift={comps[0][2]['combined_lift']:.1f}x)"
            for sae_key, comps in items[:200]
        ]

    def on_reverse_sae_change(sae_choice_str):
        if not sae_choice_str:
            return "", ""
        sae_key = sae_choice_str.split(" (")[0]
        matches = reverse_index.get(sae_key, [])
        if not matches:
            return "No matches", ""

        first_match = matches[0][2]
        sae_html = render_examples(first_match["sae_examples"], "#c05621")

        table = '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        table += '<tr style="background:#f0f0f0"><th style="padding:6px">SPD Component</th><th>Lift</th><th>P(SPD|SAE)</th><th>P(SAE|SPD)</th><th>SPD fire rate</th></tr>'
        for comp_key, comp, match in matches:
            table += (
                f'<tr style="border-bottom:1px solid #eee">'
                f'<td style="padding:6px"><b>L{comp["spd_layer"]} {comp["spd_mod_type"]}[{comp["spd_local_idx"]}]</b></td>'
                f'<td>{match["combined_lift"]:.1f}x</td>'
                f'<td>{match["p_spd_given_sae"]:.4f}</td>'
                f'<td>{match["p_sae_given_spd"]:.4f}</td>'
                f'<td>{comp["spd_fire_rate"]:.4f}</td></tr>'
            )
        table += '</table>'
        return table, sae_html

    def search_components(min_lift, max_matches):
        min_lift = float(min_lift)
        max_matches = int(max_matches)
        results = []
        for comp_key, comp in components.items():
            high_matches = [m for m in comp["sae_matches"] if m["combined_lift"] >= min_lift]
            if 1 <= len(high_matches) <= max_matches:
                top = high_matches[0]
                results.append(
                    f"**{comp_key}** → L{top['sae_layer']}_SAE[{top['sae_feature']}] "
                    f"(lift={top['combined_lift']:.1f}x, {len(high_matches)} matches ≥ {min_lift})"
                )
        results.sort(key=lambda x: -float(x.split("lift=")[1].split("x")[0]))
        return "\n\n".join(results[:100]) if results else "No matches found"

    with gr.Blocks(title="SPD-SAE Interaction Dashboard") as app:
        gr.Markdown("# SPD Component — SAE Feature Interaction Dashboard")
        if has_models:
            gr.Markdown("*Cross-activations computed on the fly (blue=SPD, orange=SAE)*")
        else:
            gr.Markdown("*Running without models — showing pre-computed single-feature activations*")

        with gr.Tabs():
            with gr.Tab("Browse by SPD Component"):
                gr.Markdown("Select a **layer** and **SPD component** to see its "
                            "top 10 associated SAE features across all residual stream layers.")
                with gr.Row():
                    layer_dd = gr.Dropdown(
                        choices=[f"Layer {l}" for l in layers],
                        value=f"Layer {layers[0]}",
                        label="Layer", scale=1,
                    )
                    comp_dd = gr.Dropdown(
                        choices=get_component_choices(layers[0]),
                        value=get_component_choices(layers[0])[0] if get_component_choices(layers[0]) else None,
                        label="SPD Component", scale=2,
                    )
                    sae_dd = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="SAE Feature (top 10 by lift, any layer)", scale=3,
                    )

                stats_out = gr.HTML("")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### SPD Component — Top Activating Sequences")
                        spd_out = gr.HTML("")
                    with gr.Column():
                        gr.Markdown("### SAE Feature — Top Activating Sequences")
                        sae_out = gr.HTML("")

                layer_dd.change(on_layer_change, [layer_dd], [comp_dd, sae_dd])
                comp_dd.change(on_comp_change, [layer_dd, comp_dd], [sae_dd, stats_out, spd_out, sae_out])
                sae_dd.change(on_sae_change, [layer_dd, comp_dd, sae_dd], [stats_out, spd_out, sae_out])

            with gr.Tab("Reverse Lookup (SAE → SPD)"):
                gr.Markdown("Select an **SAE feature** to see which SPD components map to it.")
                rev_sae_dd = gr.Dropdown(
                    choices=get_reverse_sae_choices(),
                    value=None,
                    label="SAE Feature",
                )
                rev_table = gr.HTML("")
                rev_sae_examples = gr.HTML("")
                rev_sae_dd.change(on_reverse_sae_change, [rev_sae_dd], [rev_table, rev_sae_examples])

            with gr.Tab("Search / Filter"):
                gr.Markdown("Find SPD components by their matching properties.")
                with gr.Row():
                    search_lift = gr.Number(value=50, label="Minimum combined lift")
                    search_max = gr.Number(value=3, label="Max # matches at this lift")
                    search_btn = gr.Button("Search")
                search_results = gr.Markdown("")
                search_btn.click(search_components, [search_lift, search_max], [search_results])

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--no-models", action="store_true", help="Skip model loading (no cross-activations)")
    args = parser.parse_args()

    data_path = Path(args.data)
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        components = json.load(f)
    print(f"Loaded {len(components)} SPD components")

    models = None
    if not args.no_models:
        models = load_models()

    app = build_app(components, models=models)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
