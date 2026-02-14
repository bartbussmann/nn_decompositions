"""Intruder detection evaluation for Transcoder and SPD on LlamaSimpleMLP.

For each latent, present 4 activating examples + 1 intruder to an LLM judge.
The LLM must identify which example doesn't belong. High accuracy = interpretable.

Evaluates across 10 activation deciles per latent. Random baseline = 20%.

Usage:
    python experiments/intruder_detection.py --api_key sk-...
    python experiments/intruder_detection.py --api_key sk-... --n_latents 5  # quick test
"""

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
from tabulate import tabulate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
from base import BatchTopK, TopK


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class IntruderConfig:
    # Models
    transcoder_path: str = "checkpoints/4096_batchtopk_k24_0.0003_final"
    spd_run: str = "goodfire/spd/s-275c8f21"

    # Dataset
    total_tokens: int = 2_000_000
    seq_len: int = 128

    # Feature selection
    n_latents: int = 100
    dead_latent_threshold: int = 15
    random_seed: int = 42

    # Intruder detection params
    ctx_len: int = 32
    n_deciles: int = 10
    n_activating: int = 4
    act_threshold_frac: float = 0.01

    # API
    openai_model: str = "gpt-5-mini-2025-08-07"
    api_key: str = ""

    # What to evaluate
    eval_transcoder: bool = True
    eval_spd_cfc: bool = True
    eval_spd_down: bool = True

    # Batching
    batch_size: int = 32

    # Output
    save_path: str = "experiments/intruder_results.json"


# =============================================================================
# Few-shot examples
# =============================================================================

FEW_SHOT = [
    {
        "examples": [
            'the temperature was <<42>> degrees that afternoon',
            'she ran the race in <<12>> minutes and won',
            'we counted <<87>> birds in the tall tree',
            'about <<350>> people attended the outdoor event',
            'the <<cat>> curled up on the warm sofa',
        ],
        "intruder": 5,
    },
    {
        "examples": [
            'on Monday the <<president>> gave a speech about',
            'the new <<CEO>> announced quarterly earnings results today',
            '<<Senator>> Johnson proposed the new legislation last week',
            'we went to the <<beach>> and built sand castles',
            'the <<governor>> signed the new bill into law',
        ],
        "intruder": 4,
    },
]

SYSTEM_PROMPT = (
    "You are evaluating neural network features by identifying intruders. "
    "You will see 5 text examples with highlighted tokens (marked with << >>). "
    "Four examples share a common pattern in their highlighted tokens. "
    "One example is an intruder that does not belong. "
    "Identify the intruder by its number. Respond with just the number (1-5)."
)


# =============================================================================
# Context extraction and formatting
# =============================================================================


def extract_context(
    tokens: torch.Tensor,
    acts: torch.Tensor,
    batch_idx: int,
    seq_idx: int,
    ctx_len: int = 32,
) -> tuple[list[int], list[float]]:
    """Extract a ctx_len-token window centered on (batch_idx, seq_idx)."""
    seq_len = tokens.shape[1]
    half = ctx_len // 2
    start = max(0, seq_idx - half)
    end = min(seq_len, start + ctx_len)
    start = max(0, end - ctx_len)

    ctx_toks = tokens[batch_idx, start:end].tolist()
    ctx_acts = acts[batch_idx, start:end].float().tolist()
    return ctx_toks, ctx_acts


def format_example(
    toks: list[int],
    acts: list[float],
    tokenizer_wrapper: TokenizerWrapper,
    threshold: float,
) -> tuple[str, int]:
    """Format an activating example with <<highlighted>> tokens.

    Returns (formatted_text, n_highlighted_tokens).
    """
    str_toks = tokenizer_wrapper.to_str_tokens(torch.tensor(toks))
    parts = []
    in_hl = False
    n_hl = 0
    for tok, act in zip(str_toks, acts):
        if act > threshold:
            if not in_hl:
                parts.append("<<")
                in_hl = True
            parts.append(tok)
            n_hl += 1
        else:
            if in_hl:
                parts.append(">>")
                in_hl = False
            parts.append(tok)
    if in_hl:
        parts.append(">>")
    text = "".join(parts).replace("\n", "\u21b5").replace("\ufffd", "")
    return text, n_hl


def format_intruder(
    toks: list[int],
    tokenizer_wrapper: TokenizerWrapper,
    n_highlight: int,
    seed: int,
) -> str:
    """Format an intruder example with random highlighted tokens."""
    str_toks = tokenizer_wrapper.to_str_tokens(torch.tensor(toks))
    rng = random.Random(seed)
    n_highlight = max(1, min(n_highlight, len(str_toks)))
    hl_indices = set(rng.sample(range(len(str_toks)), n_highlight))

    parts = []
    in_hl = False
    for i, tok in enumerate(str_toks):
        if i in hl_indices:
            if not in_hl:
                parts.append("<<")
                in_hl = True
            parts.append(tok)
        else:
            if in_hl:
                parts.append(">>")
                in_hl = False
            parts.append(tok)
    if in_hl:
        parts.append(">>")
    return "".join(parts).replace("\n", "\u21b5").replace("\ufffd", "")


def build_few_shot_messages() -> list[dict]:
    messages = []
    for ex in FEW_SHOT:
        user = "Which example is the intruder?\n\n"
        for i, text in enumerate(ex["examples"], 1):
            user += f'{i}. "{text}"\n'
        user += "\nAnswer with just the number."
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": str(ex["intruder"])})
    return messages


def build_prompt(activating_texts: list[str], intruder_text: str, intruder_pos: int) -> list[dict]:
    """Build full prompt. intruder_pos is 0-indexed insertion position."""
    all_texts = list(activating_texts)
    all_texts.insert(intruder_pos, intruder_text)

    user = "Which example is the intruder?\n\n"
    for i, text in enumerate(all_texts, 1):
        user += f'{i}. "{text}"\n'
    user += "\nAnswer with just the number."

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *build_few_shot_messages(),
        {"role": "user", "content": user},
    ]


# =============================================================================
# Evaluation task (one latent × one decile)
# =============================================================================


@dataclass
class EvalTask:
    """One evaluation: a latent at a specific decile."""
    latent: int
    local_idx: int
    decile: int
    messages: list[dict]
    correct_answer: int  # 1-indexed


def prepare_tasks(
    cfg: IntruderConfig,
    tokenized_dataset: torch.Tensor,
    activations: torch.Tensor,
    latent_indices: list[int],
    tokenizer_wrapper: TokenizerWrapper,
) -> list[EvalTask]:
    """Prepare all (latent, decile) evaluation tasks."""
    tasks = []
    rng = random.Random(cfg.random_seed)

    for local_idx, latent in enumerate(latent_indices):
        acts = activations[:, :, local_idx].float()
        nonzero_mask = acts > 1e-6
        nonzero_values = acts[nonzero_mask]

        if len(nonzero_values) < 20:
            continue

        # Decile boundaries
        percentiles = torch.linspace(0, 100, cfg.n_deciles + 1)[1:-1]
        boundaries = torch.quantile(nonzero_values, percentiles / 100)
        nonzero_positions = torch.nonzero(nonzero_mask)
        decile_labels = torch.bucketize(nonzero_values, boundaries)

        max_act = nonzero_values.max().item()
        act_threshold = cfg.act_threshold_frac * max_act

        for decile in range(cfg.n_deciles):
            decile_mask = decile_labels == decile
            decile_positions = nonzero_positions[decile_mask]

            if len(decile_positions) < cfg.n_activating:
                continue

            # Sample activating examples
            perm = torch.randperm(len(decile_positions), generator=torch.Generator().manual_seed(cfg.random_seed + latent * 100 + decile))
            selected = decile_positions[perm[: cfg.n_activating]]

            activating_texts = []
            total_hl = 0
            for pos in selected:
                b, s = pos[0].item(), pos[1].item()
                ctx_toks, ctx_acts = extract_context(
                    tokenized_dataset, acts, b, s, cfg.ctx_len
                )
                text, n_hl = format_example(
                    ctx_toks, ctx_acts, tokenizer_wrapper, act_threshold
                )
                activating_texts.append(text)
                total_hl += n_hl

            avg_hl = max(1, total_hl // cfg.n_activating)

            # Sample intruder from a different latent
            other_idx = local_idx
            for _ in range(20):
                other_idx = rng.randint(0, activations.shape[2] - 1)
                if other_idx != local_idx:
                    break
            other_acts = activations[:, :, other_idx].float()
            other_nonzero = torch.nonzero(other_acts > 1e-6)
            if len(other_nonzero) == 0:
                continue

            intruder_pos_in_other = other_nonzero[rng.randint(0, len(other_nonzero) - 1)]
            ib, is_ = intruder_pos_in_other[0].item(), intruder_pos_in_other[1].item()
            intruder_toks, _ = extract_context(
                tokenized_dataset, other_acts, ib, is_, cfg.ctx_len
            )
            intruder_text = format_intruder(
                intruder_toks, tokenizer_wrapper, avg_hl,
                seed=latent * 1000 + decile,
            )

            # Random position for intruder in the list
            intruder_insert = rng.randint(0, cfg.n_activating)
            messages = build_prompt(activating_texts, intruder_text, intruder_insert)
            correct_answer = intruder_insert + 1  # 1-indexed

            tasks.append(EvalTask(
                latent=latent,
                local_idx=local_idx,
                decile=decile,
                messages=messages,
                correct_answer=correct_answer,
            ))

    return tasks


# =============================================================================
# Runner
# =============================================================================


def call_api(client: OpenAI, model: str, messages: list[dict]) -> str:
    max_retries = 8
    for attempt in range(max_retries):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            content = result.choices[0].message.content
            if content is None:
                return ""
            return content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2**attempt + random.random()
                print(f"\nRate limited, retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def parse_response(response: str) -> int | None:
    for ch in response:
        if ch in "12345":
            return int(ch)
    return None


def run_method(
    cfg: IntruderConfig,
    method_name: str,
    tokenized_dataset: torch.Tensor,
    activations: torch.Tensor,
    latent_indices: list[int],
    tokenizer_wrapper: TokenizerWrapper,
) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Intruder detection: {method_name}")
    print(f"  {len(latent_indices)} features selected")
    print(f"{'='*60}")

    print("Preparing evaluation prompts...")
    tasks = prepare_tasks(cfg, tokenized_dataset, activations, latent_indices, tokenizer_wrapper)
    print(f"  {len(tasks)} prompts across {len(set(t.latent for t in tasks))} latents")

    if not tasks:
        return {"method": method_name, "accuracy": 0.0, "n_features": 0, "per_feature": {}}

    client = OpenAI(api_key=cfg.api_key)
    results_by_latent: dict[int, dict[int, bool]] = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_task = {
            executor.submit(call_api, client, cfg.openai_model, t.messages): t
            for t in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="LLM calls"):
            task = future_to_task[future]
            try:
                response = future.result()
            except Exception as e:
                print(f"\nAPI error for latent {task.latent} decile {task.decile}: {e}")
                continue

            predicted = parse_response(response)
            is_correct = predicted == task.correct_answer if predicted is not None else False

            if task.latent not in results_by_latent:
                results_by_latent[task.latent] = {}
            results_by_latent[task.latent][task.decile] = is_correct

    # Aggregate per-latent
    per_feature = {}
    all_scores = []
    for latent, decile_results in sorted(results_by_latent.items()):
        n_correct = sum(decile_results.values())
        n_total = len(decile_results)
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        per_feature[latent] = {
            "accuracy": accuracy,
            "n_deciles": n_total,
            "decile_results": {str(k): v for k, v in sorted(decile_results.items())},
        }
        all_scores.append(accuracy)

    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    std_dev = (sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)) ** 0.5 if len(all_scores) > 1 else 0.0

    # Per-decile aggregate
    decile_accs = {}
    for d in range(cfg.n_deciles):
        vals = [r[d] for r in results_by_latent.values() if d in r]
        if vals:
            decile_accs[d] = sum(vals) / len(vals)

    print(f"\n{method_name} results:")
    print(f"  Intruder detection accuracy: {mean_score:.4f} +/- {std_dev:.4f}")
    print(f"  Features evaluated: {len(all_scores)}")
    print(f"  Random baseline: 0.2000")

    if decile_accs:
        print(f"\n  Per-decile accuracy (0=lowest activation, {cfg.n_deciles-1}=highest):")
        headers = ["Decile", "Accuracy", "N"]
        table = []
        for d in range(cfg.n_deciles):
            if d in decile_accs:
                n = sum(1 for r in results_by_latent.values() if d in r)
                table.append([d, f"{decile_accs[d]:.3f}", n])
        print(tabulate(table, headers=headers, tablefmt="simple_outline"))

    # Per-feature table
    headers = ["Latent", "Accuracy", "Deciles"]
    table = [
        [lat, f"{pf['accuracy']:.3f}", pf["n_deciles"]]
        for lat, pf in sorted(per_feature.items())
    ]
    print(tabulate(table, headers=headers, tablefmt="simple_outline"))

    return {
        "method": method_name,
        "accuracy": mean_score,
        "std_dev": std_dev,
        "n_features": len(all_scores),
        "per_decile": {str(k): v for k, v in decile_accs.items()},
        "per_feature": per_feature,
    }


# =============================================================================
# Plotting
# =============================================================================


def plot_intruder_results(all_results: dict[str, dict], save_path: str):
    """Bar plot with standard-error bars for intruder detection accuracy."""
    methods = []
    accuracies = []
    std_errors = []

    for method, res in all_results.items():
        per_feature = res.get("per_feature", {})
        scores = [v["accuracy"] for v in per_feature.values()]
        if not scores:
            continue
        n = len(scores)
        mean = sum(scores) / n
        std = (sum((s - mean) ** 2 for s in scores) / n) ** 0.5
        se = std / n**0.5

        methods.append(method)
        accuracies.append(mean)
        std_errors.append(se)

    if not methods:
        print("No results to plot.")
        return

    # Nice labels
    label_map = {
        "transcoder": "Transcoder",
        "spd_cfc_ci>0.0": "SPD c_fc\n(CI > 0)",
        "spd_cfc_ci>0.5": "SPD c_fc\n(CI > 0.5)",
        "spd_down_proj_ci>0.0": "SPD down_proj\n(CI > 0)",
        "spd_down_proj_ci>0.5": "SPD down_proj\n(CI > 0.5)",
    }
    labels = [label_map.get(m, m) for m in methods]

    color_map = {
        "transcoder": "tab:blue",
        "spd_cfc_ci>0.0": "tab:orange",
        "spd_cfc_ci>0.5": "#e8833a",
        "spd_down_proj_ci>0.0": "tab:green",
        "spd_down_proj_ci>0.5": "#5aaa5a",
    }
    colors = [color_map.get(m, "tab:gray") for m in methods]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        x, accuracies, yerr=std_errors, width=0.6,
        color=colors, edgecolor="black", linewidth=0.8,
        capsize=5, error_kw={"linewidth": 1.5, "capthick": 1.5},
    )

    # Value labels on bars
    for bar, acc, se in zip(bars, accuracies, std_errors):
        y = bar.get_height() + se + 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2, y,
            f"{acc:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    # Random baseline
    ax.axhline(0.2, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="Random baseline (20%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Intruder Detection Accuracy", fontsize=12)
    ax.set_title("Intruder Detection: SPD vs Transcoder\n(LlamaSimpleMLP Layer 3 MLP, error bars = SE)", fontsize=13)
    ax.set_ylim(0, min(1.0, max(a + s for a, s in zip(accuracies, std_errors)) + 0.1))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Intruder detection for Transcoder and SPD")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--transcoder_path", type=str, default="checkpoints/4096_batchtopk_k24_0.0003_final")
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument("--n_latents", type=int, default=100)
    parser.add_argument("--total_tokens", type=int, default=2_000_000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="experiments/intruder_results.json")
    parser.add_argument("--skip_transcoder", action="store_true")
    parser.add_argument("--skip_spd_cfc", action="store_true")
    parser.add_argument("--skip_spd_down", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Provide OpenAI API key via --api_key or OPENAI_API_KEY env var")

    cfg = IntruderConfig(
        transcoder_path=args.transcoder_path,
        spd_run=args.spd_run,
        n_latents=args.n_latents,
        total_tokens=args.total_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        api_key=api_key,
        save_path=args.save_path,
        eval_transcoder=not args.skip_transcoder,
        eval_spd_cfc=not args.skip_spd_cfc,
        eval_spd_down=not args.skip_spd_down,
    )

    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.set_grad_enabled(False)

    start_time = time.time()

    # Load tokenized dataset
    print("Loading tokenized dataset...")
    tokenized_dataset = load_tokenized_dataset(cfg.total_tokens, cfg.seq_len)
    print(f"  Shape: {tokenized_dataset.shape}")

    tokenizer_wrapper = TokenizerWrapper()

    all_results = {}

    # ---------- Transcoder ----------
    if cfg.eval_transcoder:
        print("\nLoading transcoder...")
        transcoder = load_transcoder(cfg.transcoder_path)
        transcoder.to(DEVICE)
        dict_size = transcoder.cfg.dict_size

        print("Loading SPD model (to get base model)...")
        spd_model, raw_config = load_spd_model(cfg.spd_run)
        spd_model.to(DEVICE)
        base_model = spd_model.target_model
        base_model.eval()

        print(f"Computing transcoder sparsity ({dict_size} features)...")

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
            if isinstance(transcoder, (TopK, BatchTopK)):
                return F.relu(x_enc @ transcoder.W_enc)
            else:
                return F.relu(x_enc @ transcoder.W_enc + transcoder.b_enc)

        firing_count = compute_sparsity(tc_acts_fn, tokenized_dataset, cfg.batch_size, dict_size)
        tc_latents = select_alive_latents(
            firing_count, cfg.n_latents, cfg.dead_latent_threshold, cfg.random_seed
        )
        print(f"  Selected {len(tc_latents)} alive transcoder features")

        tc_acts = collect_transcoder_activations(
            base_model, transcoder, tokenized_dataset, cfg.batch_size, tc_latents
        )

        all_results["transcoder"] = run_method(
            cfg, "transcoder", tokenized_dataset, tc_acts, tc_latents, tokenizer_wrapper
        )

        del transcoder, tc_acts
        torch.cuda.empty_cache()
    else:
        print("Loading SPD model...")
        spd_model, raw_config = load_spd_model(cfg.spd_run)
        spd_model.to(DEVICE)
        base_model = spd_model.target_model
        base_model.eval()

    # ---------- SPD ----------
    cfc_name = f"h.{LAYER}.mlp.c_fc"
    down_name = f"h.{LAYER}.mlp.down_proj"

    spd_modules_to_eval = []
    if cfg.eval_spd_cfc:
        spd_modules_to_eval.append(cfc_name)
    if cfg.eval_spd_down:
        spd_modules_to_eval.append(down_name)

    ci_thresholds = [0.0, 0.5]

    if spd_modules_to_eval:
        n_components = {}
        for name in spd_modules_to_eval:
            n_components[name] = spd_model.components[name].C
            print(f"  {name}: {n_components[name]} components")

        # Pass 1: compute sparsity for both thresholds
        print(f"Computing SPD CI sparsity (thresholds: {ci_thresholds})...")
        spd_firing_counts = {
            thresh: {name: torch.zeros(n_components[name], dtype=torch.long) for name in spd_modules_to_eval}
            for thresh in ci_thresholds
        }

        for i in tqdm(
            range(0, len(tokenized_dataset), cfg.batch_size),
            desc="SPD sparsity",
        ):
            input_ids = tokenized_dataset[i : i + cfg.batch_size].to(DEVICE)
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
                    spd_firing_counts[thresh][name],
                    cfg.n_latents,
                    cfg.dead_latent_threshold,
                    cfg.random_seed,
                )
                print(f"  {name} (CI>{thresh}): selected {len(spd_latents[thresh][name])} alive components")

        # Union latents for shared activation collection
        union_latents = {}
        for name in spd_modules_to_eval:
            all_lats = set()
            for thresh in ci_thresholds:
                all_lats.update(spd_latents[thresh][name])
            union_latents[name] = sorted(all_lats)
            print(f"  {name}: {len(union_latents[name])} unique latents across thresholds")

        # Pass 2: collect raw CI activations
        spd_acts_raw = collect_spd_ci_activations(
            spd_model,
            tokenized_dataset,
            spd_modules_to_eval,
            cfg.batch_size,
            selected_latents=union_latents,
        )

        # Run intruder detection for each threshold × module
        module_labels = {cfc_name: "spd_cfc", down_name: "spd_down_proj"}
        for thresh in ci_thresholds:
            for name in spd_modules_to_eval:
                base_label = module_labels[name]
                label = f"{base_label}_ci>{thresh}"

                union_list = union_latents[name]
                thresh_lats = spd_latents[thresh][name]
                local_indices = [union_list.index(lat) for lat in thresh_lats]

                acts = spd_acts_raw[name][:, :, local_indices]
                if thresh > 0:
                    acts = acts.clone()
                    acts[acts <= thresh] = 0.0

                all_results[label] = run_method(
                    cfg, label, tokenized_dataset, acts, thresh_lats, tokenizer_wrapper
                )

    # ---------- Summary ----------
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary_table = []
    for method, res in all_results.items():
        summary_table.append(
            [method, f"{res['accuracy']:.4f}", f"{res['std_dev']:.4f}", res["n_features"]]
        )
    print(
        tabulate(
            summary_table,
            headers=["Method", "Accuracy", "Std Dev", "N Features"],
            tablefmt="simple_outline",
        )
    )
    print(f"Random baseline: 0.2000")
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    save_data = {
        "config": {
            "total_tokens": cfg.total_tokens,
            "seq_len": cfg.seq_len,
            "n_latents": cfg.n_latents,
            "ctx_len": cfg.ctx_len,
            "n_deciles": cfg.n_deciles,
            "n_activating": cfg.n_activating,
            "openai_model": cfg.openai_model,
            "timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
    }
    os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)
    with open(cfg.save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Results saved to {cfg.save_path}")

    # Plot
    plot_path = cfg.save_path.replace(".json", ".png")
    plot_intruder_results(all_results, plot_path)


if __name__ == "__main__":
    main()
