"""Interactive Gradio dashboard for SPD-SAE feature interaction.

Loads pre-computed data from collect_dashboard_data.py and displays:
- Pair selection by layer and rank
- Conditional probability stats
- Side-by-side top activating examples with all activating tokens highlighted

Usage:
    python experiments/exp_042_spd_sae_interaction/dashboard_app.py
    python experiments/exp_042_spd_sae_interaction/dashboard_app.py --port 7861
"""

import argparse
import json
from pathlib import Path

import gradio as gr

DATA_PATH = Path("experiments/exp_042_spd_sae_interaction/output/dashboard_data.json")


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def format_highlighted_sequence(tokens: list[str], activations: list[float], max_tokens: int = 80) -> str:
    """Format a token sequence as HTML with activation-based highlighting."""
    # Find the peak position and show a window around it
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
            # Yellow-orange gradient
            r = 255
            g = int(255 - intensity * 100)
            b = int(200 - intensity * 200)
            a = 0.3 + 0.7 * intensity
            weight = "bold" if intensity > 0.5 else "normal"
            parts.append(
                f'<span style="background:rgba({r},{g},{b},{a:.2f});'
                f'font-weight:{weight};border-radius:2px;padding:0 1px">{escaped}</span>'
            )
        else:
            parts.append(f'<span style="color:#888">{escaped}</span>')

    return "".join(parts)


def render_pair(data, layer_idx, pair_idx):
    layers = sorted(data.keys())
    if layer_idx not in data:
        return "No data for this layer", "", "", ""

    pairs = data[layer_idx]
    if pair_idx >= len(pairs):
        return "Pair index out of range", "", "", ""

    pair = pairs[pair_idx]

    # Stats
    stats_html = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:12px 0">
        <div style="background:#f7fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px">
            <h4 style="margin:0 0 8px 0;color:#2b6cb0">SPD c_fc[{pair['spd_global_idx']}]</h4>
            <div style="font-size:13px">
                <div>Fire rate: <b>{pair['spd_fire_rate']:.4f}</b></div>
                <div>P(SPD on | SAE on): <b>{pair['p_spd_given_sae']:.4f}</b></div>
                <div>P(SPD on | SAE off): <b>{pair['p_spd_given_not_sae']:.4f}</b></div>
                <div>Lift: <b style="color:#22543d">{pair['lift_spd']:.1f}x</b></div>
            </div>
        </div>
        <div style="background:#f7fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px">
            <h4 style="margin:0 0 8px 0;color:#c05621">SAE[{pair['sae_global_idx']}]</h4>
            <div style="font-size:13px">
                <div>Fire rate: <b>{pair['sae_fire_rate']:.4f}</b></div>
                <div>P(SAE on | SPD on): <b>{pair['p_sae_given_spd']:.4f}</b></div>
                <div>P(SAE on | SPD off): <b>{pair['p_sae_given_not_spd']:.4f}</b></div>
                <div>Lift: <b style="color:#22543d">{pair['lift_sae']:.1f}x</b></div>
            </div>
        </div>
    </div>
    <div style="text-align:center;font-size:15px;margin:8px 0">
        Combined lift: <b style="font-size:18px;color:#6b21a8">{pair['combined_lift']:.1f}x</b>
    </div>
    """

    # SPD examples
    spd_examples_html = '<div style="font-family:monospace;font-size:12px;line-height:1.8">'
    for i, ex in enumerate(pair["spd_examples"][:10]):
        seq_html = format_highlighted_sequence(ex["tokens"], ex["activations"])
        spd_examples_html += f'<div style="margin:6px 0;padding:6px 8px;background:#fafafa;border-radius:4px;border-left:3px solid #2b6cb0"><span style="color:#999;font-size:10px">#{i+1} (max={ex["max_activation"]:.3f})</span><br>{seq_html}</div>'
    spd_examples_html += '</div>'

    # SAE examples
    sae_examples_html = '<div style="font-family:monospace;font-size:12px;line-height:1.8">'
    for i, ex in enumerate(pair["sae_examples"][:10]):
        seq_html = format_highlighted_sequence(ex["tokens"], ex["activations"])
        sae_examples_html += f'<div style="margin:6px 0;padding:6px 8px;background:#fafafa;border-radius:4px;border-left:3px solid #c05621"><span style="color:#999;font-size:10px">#{i+1} (max={ex["max_activation"]:.3f})</span><br>{seq_html}</div>'
    sae_examples_html += '</div>'

    title = f"Layer {layer_idx} — Pair #{pair_idx+1}: SPD c_fc[{pair['spd_global_idx']}] ↔ SAE[{pair['sae_global_idx']}]"

    return title, stats_html, spd_examples_html, sae_examples_html


def build_app(data):
    layers = sorted(data.keys())

    def get_pair_choices(layer_idx):
        pairs = data.get(layer_idx, [])
        return [
            f"#{i+1}: SPD[{p['spd_global_idx']}] ↔ SAE[{p['sae_global_idx']}] (lift={p['combined_lift']:.1f}x)"
            for i, p in enumerate(pairs)
        ]

    def on_layer_change(layer_str):
        layer_idx = int(layer_str.split()[-1])
        choices = get_pair_choices(layer_idx)
        return gr.update(choices=choices, value=choices[0] if choices else None)

    def on_pair_select(layer_str, pair_str):
        if not pair_str:
            return "", "", "", ""
        layer_idx = int(layer_str.split()[-1])
        pair_idx = int(pair_str.split(":")[0].strip("#")) - 1
        return render_pair(data, layer_idx, pair_idx)

    with gr.Blocks(title="SPD-SAE Interaction Dashboard") as app:
        gr.Markdown("# SPD Component — SAE Feature Interaction Dashboard")
        gr.Markdown("Explore how SPD components relate to residual stream SAE features. "
                    "Pairs ranked by combined lift (geometric mean of conditional probability ratios).")

        with gr.Row():
            layer_dropdown = gr.Dropdown(
                choices=[f"Layer {l}" for l in layers],
                value=f"Layer {layers[0]}",
                label="Layer",
                scale=1,
            )
            pair_dropdown = gr.Dropdown(
                choices=get_pair_choices(layers[0]),
                value=get_pair_choices(layers[0])[0] if get_pair_choices(layers[0]) else None,
                label="Pair",
                scale=3,
            )

        title_md = gr.Markdown("", elem_classes=["pair-title"])
        stats_html = gr.HTML("")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### SPD Component — Top Activating Sequences")
                spd_html = gr.HTML("")
            with gr.Column():
                gr.Markdown("### SAE Feature — Top Activating Sequences")
                sae_html = gr.HTML("")

        layer_dropdown.change(on_layer_change, [layer_dropdown], [pair_dropdown])
        pair_dropdown.change(on_pair_select, [layer_dropdown, pair_dropdown],
                             [title_md, stats_html, spd_html, sae_html])

        # Initial render
        app.load(
            on_pair_select,
            [layer_dropdown, pair_dropdown],
            [title_md, stats_html, spd_html, sae_html],
        )

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
        raw = json.load(f)
    data = {int(k): v for k, v in raw.items()}
    print(f"Loaded {sum(len(v) for v in data.values())} pairs across {len(data)} layers")

    app = build_app(data)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
