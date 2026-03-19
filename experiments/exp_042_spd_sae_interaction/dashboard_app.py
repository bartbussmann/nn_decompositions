"""Interactive Gradio dashboard for SPD-SAE feature interaction.

Browse by SPD component (c_fc or down_proj, any layer): see its top 10
associated SAE features across ALL residual stream layers, ranked by
combined lift, with side-by-side activating examples.

Usage:
    python experiments/exp_042_spd_sae_interaction/dashboard_app.py
    python experiments/exp_042_spd_sae_interaction/dashboard_app.py --share
"""

import argparse
import json
from pathlib import Path

import gradio as gr

DATA_PATH = Path("experiments/exp_042_spd_sae_interaction/output/dashboard_data_v2.json")


def format_highlighted_sequence(tokens: list[str], activations: list[float], max_tokens: int = 80) -> str:
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


def build_app(components: dict):
    # Group components by layer for easier browsing
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
        # "c_fc[123] (...)" or "down_proj[456] (...)"
        mod_and_idx = comp_str.split("(")[0].strip()
        mod = mod_and_idx.split("[")[0]
        idx = int(mod_and_idx.split("[")[1].split("]")[0])
        return f"L{layer}_{mod}[{idx}]"

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
        spd_html = render_examples(comp["spd_examples"], "#2b6cb0")
        if sae_choices and comp["sae_matches"]:
            match = comp["sae_matches"][0]
            stats = render_stats(comp, match)
            sae_html = render_examples(match["sae_examples"], "#c05621")
            return gr.update(choices=sae_choices, value=sae_choices[0]), stats, spd_html, sae_html
        return gr.update(choices=[], value=None), "", spd_html, ""

    def on_sae_change(layer_str, comp_str, sae_str):
        if not comp_str or not sae_str:
            return "", "", ""
        layer = int(layer_str.split()[-1])
        comp_key = comp_key_from_choice(layer, comp_str)
        comp = components[comp_key]

        # Parse SAE selection: "L1_SAE[456] (...)"
        sae_part = sae_str.split("(")[0].strip()
        sae_layer = int(sae_part.split("_")[0].replace("L", ""))
        sae_feature = int(sae_part.split("[")[1].split("]")[0])

        for match in comp["sae_matches"]:
            if match["sae_layer"] == sae_layer and match["sae_feature"] == sae_feature:
                stats = render_stats(comp, match)
                spd_html = render_examples(comp["spd_examples"], "#2b6cb0")
                sae_html = render_examples(match["sae_examples"], "#c05621")
                return stats, spd_html, sae_html
        return "", "", ""

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

    with gr.Blocks(title="SPD-SAE Interaction Dashboard") as app:
        gr.Markdown("# SPD Component — SAE Feature Interaction Dashboard")
        gr.Markdown(
            "Select a **layer** and **SPD component** (c_fc or down_proj) to see its "
            "top 10 associated SAE features across all residual stream layers."
        )

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

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    args = parser.parse_args()

    data_path = Path(args.data)
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        components = json.load(f)
    print(f"Loaded {len(components)} SPD components")

    app = build_app(components)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
