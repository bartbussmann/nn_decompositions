"""Interactive Gradio dashboard for SPD-SAE feature interaction.

Browse by SPD component: select a layer and component, see its top 10 associated
SAE features ranked by combined lift, with side-by-side activating examples.

Usage:
    python experiments/exp_042_spd_sae_interaction/dashboard_app.py
    python experiments/exp_042_spd_sae_interaction/dashboard_app.py --share
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import gradio as gr

DATA_PATH = Path("experiments/exp_042_spd_sae_interaction/output/dashboard_data.json")


def load_and_index(data_path: Path):
    """Load data and build index: {layer: {spd_idx: [pairs sorted by lift]}}."""
    with open(data_path) as f:
        raw = json.load(f)
    data = {int(k): v for k, v in raw.items()}

    index = {}
    for layer_idx, pairs in data.items():
        by_spd = defaultdict(list)
        for pair in pairs:
            by_spd[pair["spd_global_idx"]].append(pair)
        # Sort each SPD component's pairs by combined lift descending
        for spd_idx in by_spd:
            by_spd[spd_idx].sort(key=lambda p: -p["combined_lift"])
        index[layer_idx] = dict(by_spd)

    return data, index


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


def render_pair_detail(pair: dict) -> tuple[str, str, str]:
    """Render stats + SPD examples + SAE examples for one pair."""
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
    spd_html = render_examples(pair["spd_examples"], "#2b6cb0")
    sae_html = render_examples(pair["sae_examples"], "#c05621")
    return stats_html, spd_html, sae_html


def build_app(data, index):
    layers = sorted(data.keys())

    def get_spd_choices(layer_idx):
        spd_components = sorted(index.get(layer_idx, {}).keys())
        choices = []
        for spd_idx in spd_components:
            pairs = index[layer_idx][spd_idx]
            top_lift = pairs[0]["combined_lift"]
            fire_rate = pairs[0]["spd_fire_rate"]
            choices.append(f"c_fc[{spd_idx}] (fire={fire_rate:.4f}, top_lift={top_lift:.1f}x, {len(pairs)} SAE matches)")
        return choices

    def get_sae_choices(layer_idx, spd_idx):
        pairs = index.get(layer_idx, {}).get(spd_idx, [])
        return [
            f"SAE[{p['sae_global_idx']}] (lift={p['combined_lift']:.1f}x, "
            f"P(SPD|SAE)={p['p_spd_given_sae']:.3f}, P(SAE|SPD)={p['p_sae_given_spd']:.3f})"
            for p in pairs[:10]
        ]

    def parse_spd_idx(spd_str):
        # "c_fc[123] (...)" -> 123
        return int(spd_str.split("[")[1].split("]")[0])

    def parse_sae_rank(sae_str):
        # "SAE[456] (...)" -> find rank in the list
        sae_idx = int(sae_str.split("[")[1].split("]")[0])
        return sae_idx

    def on_layer_change(layer_str):
        layer_idx = int(layer_str.split()[-1])
        spd_choices = get_spd_choices(layer_idx)
        sae_choices = get_sae_choices(layer_idx, parse_spd_idx(spd_choices[0])) if spd_choices else []
        return (
            gr.update(choices=spd_choices, value=spd_choices[0] if spd_choices else None),
            gr.update(choices=sae_choices, value=sae_choices[0] if sae_choices else None),
        )

    def on_spd_change(layer_str, spd_str):
        if not spd_str:
            return gr.update(choices=[], value=None), "", "", ""
        layer_idx = int(layer_str.split()[-1])
        spd_idx = parse_spd_idx(spd_str)
        sae_choices = get_sae_choices(layer_idx, spd_idx)
        # Auto-select first SAE and render
        if sae_choices:
            pair = index[layer_idx][spd_idx][0]
            stats, spd_ex, sae_ex = render_pair_detail(pair)
            return gr.update(choices=sae_choices, value=sae_choices[0]), stats, spd_ex, sae_ex
        return gr.update(choices=[], value=None), "", "", ""

    def on_sae_change(layer_str, spd_str, sae_str):
        if not spd_str or not sae_str:
            return "", "", ""
        layer_idx = int(layer_str.split()[-1])
        spd_idx = parse_spd_idx(spd_str)
        sae_idx = parse_sae_rank(sae_str)
        # Find the pair
        for pair in index.get(layer_idx, {}).get(spd_idx, []):
            if pair["sae_global_idx"] == sae_idx:
                return render_pair_detail(pair)
        return "", "", ""

    with gr.Blocks(title="SPD-SAE Interaction Dashboard") as app:
        gr.Markdown("# SPD Component — SAE Feature Interaction Dashboard")
        gr.Markdown(
            "Select a **layer** and **SPD component** to see its top associated SAE features. "
            "Then select an **SAE feature** to compare activating examples side by side."
        )

        with gr.Row():
            layer_dd = gr.Dropdown(
                choices=[f"Layer {l}" for l in layers],
                value=f"Layer {layers[0]}",
                label="Layer", scale=1,
            )
            spd_dd = gr.Dropdown(
                choices=get_spd_choices(layers[0]),
                value=get_spd_choices(layers[0])[0] if get_spd_choices(layers[0]) else None,
                label="SPD Component", scale=2,
            )
            sae_dd = gr.Dropdown(
                choices=get_sae_choices(
                    layers[0],
                    parse_spd_idx(get_spd_choices(layers[0])[0]) if get_spd_choices(layers[0]) else -1,
                ),
                value=None,
                label="SAE Feature (top 10 by lift)", scale=3,
            )

        stats_out = gr.HTML("")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### SPD Component — Top Activating Sequences")
                spd_out = gr.HTML("")
            with gr.Column():
                gr.Markdown("### SAE Feature — Top Activating Sequences")
                sae_out = gr.HTML("")

        layer_dd.change(on_layer_change, [layer_dd], [spd_dd, sae_dd])
        spd_dd.change(on_spd_change, [layer_dd, spd_dd], [sae_dd, stats_out, spd_out, sae_out])
        sae_dd.change(on_sae_change, [layer_dd, spd_dd, sae_dd], [stats_out, spd_out, sae_out])

        # Initial render
        init_spd_choices = get_spd_choices(layers[0])
        if init_spd_choices:
            init_spd_idx = parse_spd_idx(init_spd_choices[0])
            init_sae_choices = get_sae_choices(layers[0], init_spd_idx)
            if init_sae_choices:
                sae_dd.value = init_sae_choices[0]

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    args = parser.parse_args()

    data_path = Path(args.data)
    print(f"Loading data from {data_path}...")
    data, index = load_and_index(data_path)
    for layer_idx in sorted(data.keys()):
        n_spd = len(index[layer_idx])
        n_pairs = sum(len(v) for v in index[layer_idx].values())
        print(f"  Layer {layer_idx}: {n_spd} SPD components, {n_pairs} pairs")

    app = build_app(data, index)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
