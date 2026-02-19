"""Interactive intruder detection dashboard.

See exactly the raw prompt the LLM sees and try to spot the intruder yourself.

Usage:
    cd /workspace/nn_decompositions
    .venv/bin/streamlit run experiments/intruder_dashboard.py -- --n_latents 10

    # Skip heavy SPD computation for quick testing:
    .venv/bin/streamlit run experiments/intruder_dashboard.py -- --n_latents 5 --skip_spd_cfc --skip_spd_down
"""

import argparse
import random
import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/workspace/spd")))
sys.path.insert(0, str(Path("/workspace/SAEBench")))

from autointerp_pile import (
    DEVICE,
    LAYER,
    TokenizerWrapper,
    collect_spd_ci_activations,
    collect_transcoder_activations,
    compute_sparsity,
    load_spd_model,
    load_tokenized_dataset,
    load_transcoder,
    select_alive_latents,
)
from transcoder import BatchTopKTranscoder, TopKTranscoder
from intruder_detection import IntruderConfig, EvalTask, prepare_tasks


# =============================================================================
# Parse CLI args (passed after --)
# =============================================================================


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcoder_path", default="checkpoints/4096_batchtopk_k24_0.0003_final")
    parser.add_argument("--spd_run", default="goodfire/spd/s-275c8f21")
    parser.add_argument("--n_latents", type=int, default=10)
    parser.add_argument("--total_tokens", type=int, default=2_000_000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_transcoder", action="store_true")
    parser.add_argument("--skip_spd_cfc", action="store_true")
    parser.add_argument("--skip_spd_down", action="store_true")
    return parser.parse_args()


# =============================================================================
# Data loading (cached so it only runs once)
# =============================================================================


@st.cache_resource(show_spinner="Loading models and preparing tasks...")
def load_all_tasks(
    transcoder_path: str,
    spd_run: str,
    n_latents: int,
    total_tokens: int,
    seq_len: int,
    batch_size: int,
    skip_transcoder: bool,
    skip_spd_cfc: bool,
    skip_spd_down: bool,
) -> dict[str, list[dict]]:
    """Load models, compute activations, prepare all intruder tasks."""
    torch.set_grad_enabled(False)
    random.seed(42)
    torch.manual_seed(42)

    cfg = IntruderConfig(
        transcoder_path=transcoder_path,
        spd_run=spd_run,
        n_latents=n_latents,
        total_tokens=total_tokens,
        seq_len=seq_len,
        batch_size=batch_size,
    )

    tokenized_dataset = load_tokenized_dataset(total_tokens, seq_len)
    tokenizer_wrapper = TokenizerWrapper()

    methods: dict[str, list[dict]] = {}

    # Load SPD model (needed for both transcoder and SPD)
    spd_model, _ = load_spd_model(spd_run)
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    # --- Transcoder ---
    if not skip_transcoder:
        transcoder = load_transcoder(transcoder_path)
        transcoder.to(DEVICE)
        dict_size = transcoder.cfg.dict_size

        def tc_acts_fn(input_ids):
            captured = {}
            def _hook(mod, inp, out):
                captured["mlp_input"] = out.detach()
            hook = base_model.h[LAYER].rms_2.register_forward_hook(_hook)
            base_model(input_ids)
            hook.remove()
            mlp_input = captured["mlp_input"]
            use_pre_enc_bias = (
                transcoder.cfg.pre_enc_bias
                and transcoder.input_size == transcoder.output_size
            )
            x_enc = mlp_input - transcoder.b_dec if use_pre_enc_bias else mlp_input
            if isinstance(transcoder, (TopKTranscoder, BatchTopKTranscoder)):
                return F.relu(x_enc @ transcoder.W_enc)
            else:
                return F.relu(x_enc @ transcoder.W_enc + transcoder.b_enc)

        firing_count = compute_sparsity(tc_acts_fn, tokenized_dataset, batch_size, dict_size)
        tc_latents = select_alive_latents(firing_count, n_latents, cfg.dead_latent_threshold, cfg.random_seed)

        tc_acts = collect_transcoder_activations(
            base_model, transcoder, tokenized_dataset, batch_size, tc_latents
        )

        tasks = prepare_tasks(cfg, tokenized_dataset, tc_acts, tc_latents, tokenizer_wrapper)
        methods["transcoder"] = _tasks_to_dicts(tasks)

        del transcoder, tc_acts
        torch.cuda.empty_cache()

    # --- SPD ---
    cfc_name = f"h.{LAYER}.mlp.c_fc"
    down_name = f"h.{LAYER}.mlp.down_proj"

    spd_modules_to_eval = []
    if not skip_spd_cfc:
        spd_modules_to_eval.append(cfc_name)
    if not skip_spd_down:
        spd_modules_to_eval.append(down_name)

    ci_thresholds = [0.0, 0.5]

    if spd_modules_to_eval:
        n_components = {name: spd_model.components[name].C for name in spd_modules_to_eval}

        # Pass 1: sparsity for both thresholds
        spd_firing_counts = {
            thresh: {name: torch.zeros(n_components[name], dtype=torch.long) for name in spd_modules_to_eval}
            for thresh in ci_thresholds
        }
        from tqdm import tqdm
        for i in tqdm(range(0, len(tokenized_dataset), batch_size), desc="SPD sparsity"):
            input_ids = tokenized_dataset[i : i + batch_size].to(DEVICE)
            out = spd_model(input_ids, cache_type="input")
            ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
            for name in spd_modules_to_eval:
                ci_scores = ci.lower_leaky[name].clamp(0, 1)
                for thresh in ci_thresholds:
                    spd_firing_counts[thresh][name] += (ci_scores > thresh).sum(dim=(0, 1)).cpu().long()

        # Select alive latents per threshold
        spd_latents = {thresh: {} for thresh in ci_thresholds}
        for thresh in ci_thresholds:
            for name in spd_modules_to_eval:
                spd_latents[thresh][name] = select_alive_latents(
                    spd_firing_counts[thresh][name], n_latents, cfg.dead_latent_threshold, cfg.random_seed,
                )

        # Union latents
        union_latents = {}
        for name in spd_modules_to_eval:
            all_lats = set()
            for thresh in ci_thresholds:
                all_lats.update(spd_latents[thresh][name])
            union_latents[name] = sorted(all_lats)

        # Pass 2: collect raw CI activations
        spd_acts_raw = collect_spd_ci_activations(
            spd_model, tokenized_dataset, spd_modules_to_eval, batch_size,
            selected_latents=union_latents,
        )

        module_labels = {cfc_name: "spd_cfc", down_name: "spd_down_proj"}
        for thresh in ci_thresholds:
            for name in spd_modules_to_eval:
                label = f"{module_labels[name]}_ci>{thresh}"
                union_list = union_latents[name]
                thresh_lats = spd_latents[thresh][name]
                local_indices = [union_list.index(lat) for lat in thresh_lats]

                acts = spd_acts_raw[name][:, :, local_indices]
                if thresh > 0:
                    acts = acts.clone()
                    acts[acts <= thresh] = 0.0

                tasks = prepare_tasks(cfg, tokenized_dataset, acts, thresh_lats, tokenizer_wrapper)
                methods[label] = _tasks_to_dicts(tasks)

    return methods


def _tasks_to_dicts(tasks: list[EvalTask]) -> list[dict]:
    """Convert EvalTasks to serializable dicts, keeping raw messages."""
    result = []
    for t in tasks:
        result.append({
            "latent": t.latent,
            "decile": t.decile,
            "messages": t.messages,
            "correct_answer": t.correct_answer,
        })
    return result


# =============================================================================
# Streamlit app
# =============================================================================


def main():
    st.set_page_config(page_title="Intruder Detection", layout="wide")

    args = get_args()

    st.title("Intruder Detection")
    st.caption("You see exactly what the LLM sees — the raw prompt. Pick the intruder.")

    # Load data
    all_methods = load_all_tasks(
        args.transcoder_path,
        args.spd_run,
        args.n_latents,
        args.total_tokens,
        args.seq_len,
        args.batch_size,
        args.skip_transcoder,
        args.skip_spd_cfc,
        args.skip_spd_down,
    )

    if not all_methods:
        st.error("No methods loaded. Check your --skip flags.")
        return

    # --- Sidebar ---
    with st.sidebar:
        method = st.selectbox("Method", list(all_methods.keys()))
        tasks = all_methods[method]
        st.write(f"**{len(tasks)}** tasks available")

        # Init session state
        if "scores" not in st.session_state:
            st.session_state.scores = {}
        if "task_idx" not in st.session_state:
            st.session_state.task_idx = {}
        if "answered" not in st.session_state:
            st.session_state.answered = {}
        if "shuffled_order" not in st.session_state:
            st.session_state.shuffled_order = {}

        # Per-method state
        if method not in st.session_state.scores:
            st.session_state.scores[method] = {"correct": 0, "total": 0}
        if method not in st.session_state.task_idx:
            st.session_state.task_idx[method] = 0
            order = list(range(len(tasks)))
            random.shuffle(order)
            st.session_state.shuffled_order[method] = order
        if method not in st.session_state.answered:
            st.session_state.answered[method] = False

        scores = st.session_state.scores[method]
        st.divider()
        st.subheader("Your Score")
        if scores["total"] > 0:
            acc = scores["correct"] / scores["total"]
            st.metric("Accuracy", f"{acc:.0%}", f'{scores["correct"]}/{scores["total"]}')
        else:
            st.write("No answers yet")

        st.write(f"Random baseline: **20%**")

        st.divider()
        if st.button("Reset Score"):
            st.session_state.scores[method] = {"correct": 0, "total": 0}
            st.session_state.task_idx[method] = 0
            order = list(range(len(tasks)))
            random.shuffle(order)
            st.session_state.shuffled_order[method] = order
            st.session_state.answered[method] = False
            st.rerun()

    # --- Main area ---
    idx = st.session_state.task_idx[method]
    order = st.session_state.shuffled_order[method]

    if idx >= len(tasks):
        st.success(f"All {len(tasks)} tasks completed for **{method}**!")
        scores = st.session_state.scores[method]
        if scores["total"] > 0:
            st.metric("Final Accuracy", f"{scores['correct']/scores['total']:.1%}")
        return

    task = tasks[order[idx]]
    messages = task["messages"]
    correct = task["correct_answer"]

    st.write(f"**Task {idx + 1}/{len(tasks)}** | Latent {task['latent']} | Decile {task['decile']}/9")
    st.markdown("---")

    # Show the full prompt exactly as the LLM sees it
    for msg in messages:
        role = msg["role"].upper()
        st.text(f"[{role}]")
        st.code(msg["content"], language=None)

    st.markdown("---")

    # Answer buttons
    answered = st.session_state.answered[method]

    if not answered:
        st.write("**Your answer:**")
        cols = st.columns(5)
        for i in range(5):
            if cols[i].button(f"{i+1}", key=f"btn_{method}_{idx}_{i}"):
                picked = i + 1
                is_correct = picked == correct
                st.session_state.scores[method]["total"] += 1
                if is_correct:
                    st.session_state.scores[method]["correct"] += 1
                st.session_state.answered[method] = picked
                st.rerun()
    else:
        picked = st.session_state.answered[method]
        is_correct = picked == correct

        if is_correct:
            st.success(f"Correct! Example {correct} was the intruder.")
        else:
            st.error(f"Wrong! You picked {picked}, but example {correct} was the intruder.")

        if st.button("Next", key=f"next_{method}_{idx}"):
            st.session_state.task_idx[method] += 1
            st.session_state.answered[method] = False
            st.rerun()


if __name__ == "__main__":
    main()
