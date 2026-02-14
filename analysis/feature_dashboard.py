"""Streamlit dashboard for browsing Transcoder features and SPD components.

Supports two cache formats:
  1. Legacy (collect_activations.py): keys "transcoder", "spd_cfc", "spd_cproj"
  2. SPD-specific (collect_spd_activations.py): key "modules" with dynamic module names

Usage:
    streamlit run analysis/feature_dashboard.py
    streamlit run analysis/feature_dashboard.py -- --cache activation_cache/spd_s-275c8f21.pt
"""

import argparse
import html as html_lib
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer

DEFAULT_CACHE = Path(__file__).parent / "activation_cache" / "activation_examples.pt"

# Legacy format keys
LEGACY_MODEL_KEYS = {
    "Transcoder": "transcoder",
    "SPD (c_fc)": "spd_cfc",
    "SPD (c_proj)": "spd_cproj",
}

LEGACY_VALUE_LABELS = {
    "transcoder": "activation",
    "spd_cfc": "CI",
    "spd_cproj": "CI",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default=None, help="Path to cache .pt file")
    return parser.parse_args()


@st.cache_resource
def load_data(cache_path: str):
    p = Path(cache_path)
    assert p.exists(), f"Cache not found at {p}. Run collect_spd_activations.py first."
    return torch.load(p, weights_only=False)


@st.cache_resource
def load_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name)


def render_highlighted_tokens(
    tokenizer,
    token_ids: list[int],
    activation_values: list[float],
    max_position: int,
    context_window: int,
) -> str:
    """Render tokens with background color scaled by activation magnitude."""
    start = max(0, max_position - context_window)
    end = min(len(token_ids), max_position + context_window + 1)

    window_ids = token_ids[start:end]
    window_acts = activation_values[start:end]

    max_act = max(abs(a) for a in window_acts) if window_acts else 1.0
    if max_act == 0:
        max_act = 1.0

    parts = []
    for i, (tid, act_val) in enumerate(zip(window_ids, window_acts)):
        token_str = tokenizer.decode([tid])
        token_str = html_lib.escape(token_str)
        token_str = token_str.replace("\n", "<br>")
        token_str = token_str.replace(" ", "&nbsp;")

        intensity = max(0, act_val) / max_act
        r, g, b = 66, 133, 244
        alpha = intensity * 0.8

        pos_in_seq = start + i
        is_max = pos_in_seq == max_position
        border = "border: 2px solid #333;" if is_max else ""

        style = (
            f"background-color: rgba({r},{g},{b},{alpha:.2f}); "
            f"padding: 2px 1px; border-radius: 3px; {border}"
        )
        title = f"pos={pos_in_seq} val={act_val:.4f}"
        parts.append(f'<span style="{style}" title="{title}">{token_str}</span>')

    return (
        '<div style="font-family: monospace; font-size: 14px; line-height: 2.0;">'
        + "".join(parts)
        + "</div>"
    )


def main():
    st.set_page_config(page_title="Feature Dashboard", layout="wide")

    args = parse_args()

    # Detect cache file: CLI arg > query param > default
    cache_path = args.cache
    if cache_path is None:
        # Check for SPD cache files
        cache_dir = Path(__file__).parent / "activation_cache"
        spd_caches = sorted(cache_dir.glob("spd_*.pt"))
        if spd_caches:
            cache_path = str(spd_caches[-1])  # Most recent
        else:
            cache_path = str(DEFAULT_CACHE)

    data = load_data(cache_path)
    config = data.get("config", {})

    # Detect format and build model/module selector
    is_spd_format = "modules" in data

    if is_spd_format:
        tokenizer_name = config.get("tokenizer_name", "EleutherAI/gpt-neox-20b")
        modules_data = data["modules"]
        module_names = list(modules_data.keys())
        module_components = config.get("module_components", {})
    else:
        tokenizer_name = config.get("tokenizer_name", "openai-community/gpt2")
        module_names = None

    tokenizer = load_tokenizer(tokenizer_name)

    # Sidebar
    st.sidebar.title("Feature Browser")

    if is_spd_format:
        spd_run = config.get("spd_run", "unknown")
        st.sidebar.caption(f"SPD run: `{spd_run}`")

        module_name = st.sidebar.selectbox("Module", module_names)
        val_label = "CI"
        examples_dict = modules_data[module_name]
        n_total = module_components.get(module_name, "?")
        display_title = module_name
    else:
        model_type = st.sidebar.selectbox("Model", list(LEGACY_MODEL_KEYS.keys()))
        cache_key = LEGACY_MODEL_KEYS[model_type]
        val_label = LEGACY_VALUE_LABELS[cache_key]
        examples_dict = data[cache_key]
        n_total = max(examples_dict.keys()) + 1 if examples_dict else 0
        display_title = model_type

    active_indices = sorted(examples_dict.keys())
    st.sidebar.caption(f"{len(active_indices)} components with examples (of {n_total})")

    only_active = st.sidebar.toggle("Only show components with examples", value=False)

    if only_active and active_indices:
        pos = st.sidebar.number_input(
            f"Position (0\u2013{len(active_indices) - 1})",
            min_value=0,
            max_value=len(active_indices) - 1,
            value=0,
            step=1,
        )
        feature_idx = active_indices[pos]
        st.sidebar.caption(f"Component index: {feature_idx}")
    else:
        feature_idx = st.sidebar.number_input(
            "Component Index", min_value=0, value=0, step=1
        )
    n_examples = st.sidebar.slider("Examples to show", 1, 20, 10)
    context_window = st.sidebar.slider("Context window (tokens each side)", 3, 50, 20)

    if config:
        st.sidebar.markdown("---")
        st.sidebar.caption(
            f"Cache: {config.get('n_batches', '?')} batches x "
            f"{config.get('batch_size', '?')} seqs x "
            f"{config.get('seq_len', '?')} tokens"
        )
        st.sidebar.caption(f"Tokenizer: `{tokenizer_name}`")

    # Main area
    st.title(f"{display_title} \u2014 Component {feature_idx}")

    examples = examples_dict.get(feature_idx, [])

    if not examples:
        st.warning(f"No activating examples found for component {feature_idx}.")
        nearby = sorted(
            [k for k in examples_dict if abs(k - feature_idx) < 20],
            key=lambda k: abs(k - feature_idx),
        )[:10]
        if nearby:
            st.info(f"Nearby components with examples: {nearby}")
        return

    # Summary
    display_vals = [ex["activation_values"][ex["max_position"]] for ex in examples]
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Max {val_label}", f"{max(display_vals):.4f}")
    col2.metric(f"Min {val_label} (in top-K)", f"{min(display_vals):.4f}")
    col3.metric("Examples found", len(examples))

    st.markdown("---")

    # Examples
    for i, ex in enumerate(examples[:n_examples]):
        display_val = ex["activation_values"][ex["max_position"]]
        st.markdown(
            f"**Example {i + 1}** &mdash; {val_label} = `{display_val:.4f}` "
            f"at position {ex['max_position']}"
        )
        html = render_highlighted_tokens(
            tokenizer,
            ex["token_ids"],
            ex["activation_values"],
            ex["max_position"],
            context_window,
        )
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("")


if __name__ == "__main__":
    main()
