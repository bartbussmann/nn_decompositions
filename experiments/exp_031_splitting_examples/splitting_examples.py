"""Find clean feature splitting examples between β=0.5 and β=1.0 SPD runs.

Runs both models on the SAME inputs and shows how a baseline component's activation
pattern gets partitioned across multiple β=1.0 components. For each example, the
source row shows where the baseline component fires, and target rows show where each
split component fires — ideally they tile the source pattern.

Usage:
    python experiments/exp_031_splitting_examples/splitting_examples.py
"""

import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from analysis.collect_spd_activations import load_spd_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 2
ALIVE_THRESHOLD = 1e-5
N_EVAL_BATCHES = 20
BATCH_SIZE = 16
SEQ_LEN = 512
OUTPUT_DIR = Path(__file__).parent / "output"
COVERAGE_THRESHOLD = 0.5
N_EXAMPLES = 8
N_SPLITTING_CASES = 15
CONTEXT_WINDOW = 40

RUN_FROM = ("β=0.5 (baseline)", "goodfire/spd/s-55ea3f9b")
RUN_TO = ("β=1.0 (2×)", "goodfire/spd/s-c20df1cc")

BINARY_CACHE_DIR = Path(__file__).parent.parent / "exp_030_activation_splitting" / "output" / "cached_binary"
DIRS_CACHE_DIR = Path(__file__).parent.parent / "exp_029_feature_splitting" / "output" / "cached_dirs"

MODULE_NAME = f"h.{LAYER}.mlp.down_proj"


def get_eval_batches() -> list[torch.Tensor]:
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)
    batches = []
    for _ in tqdm(range(N_EVAL_BATCHES), desc="Loading batches"):
        batch_ids = []
        for _ in range(BATCH_SIZE):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:SEQ_LEN])
        batches.append(torch.stack(batch_ids).to(DEVICE))
    return batches


def load_binary(name: str) -> torch.Tensor:
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("×", "x")
    return torch.load(BINARY_CACHE_DIR / f"{safe_name}.pt")


def load_dirs(name: str) -> torch.Tensor:
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("×", "x")
    return torch.load(DIRS_CACHE_DIR / f"{safe_name}.pt")


def find_splitting_cases(
    binary_from: torch.Tensor,
    binary_to: torch.Tensor,
    dirs_from: torch.Tensor,
    dirs_to: torch.Tensor,
    coverage_threshold: float,
) -> list[dict]:
    """Find components in run_from that split into multiple components in run_to."""
    n_from = binary_from.shape[0]

    from_sums = binary_from.float().sum(dim=1)
    to_f = binary_to.float().to(DEVICE)
    dirs_to_gpu = F.normalize(dirs_to.float(), dim=1).to(DEVICE)

    cases = []
    chunk_size = 256

    for i in tqdm(range(0, n_from, chunk_size), desc="Finding splits"):
        chunk_bin = binary_from[i:i + chunk_size].float().to(DEVICE)
        chunk_sums = from_sums[i:i + chunk_size].to(DEVICE)
        chunk_dirs = F.normalize(dirs_from[i:i + chunk_size].float(), dim=1).to(DEVICE)

        intersection = chunk_bin @ to_f.T
        coverage = intersection / chunk_sums.unsqueeze(1).clamp(min=1)
        cosines = (chunk_dirs @ dirs_to_gpu.T).abs()

        for local_j in range(chunk_bin.shape[0]):
            global_j = i + local_j
            mask = coverage[local_j] >= coverage_threshold
            n_covering = mask.sum().item()

            if n_covering < 2:
                continue

            to_indices = torch.where(mask)[0].tolist()
            covs = coverage[local_j, mask].cpu().tolist()
            coss = cosines[local_j, mask].cpu().tolist()

            from_density = from_sums[global_j].item() / binary_from.shape[1]
            to_densities = [binary_to[ti].float().sum().item() / binary_to.shape[1] for ti in to_indices]

            cases.append({
                "from_idx": global_j,
                "to_indices": to_indices,
                "coverages": covs,
                "cosines": coss,
                "from_density": from_density,
                "to_densities": to_densities,
                "n_splits": n_covering,
            })

    min_density = 200 / binary_from.shape[1]  # need enough activity for good examples
    cases = [c for c in cases if c["from_density"] >= min_density]
    # Sort by mean cosine (features are related) with bonus for 2-3 way splits
    cases.sort(key=lambda c: -(
        np.mean(c["cosines"]) + 0.1 * (c["n_splits"] <= 3)
    ))
    return cases


@torch.no_grad()
def get_alive_mask(spd_model, batches: list[torch.Tensor]) -> torch.Tensor:
    comp = spd_model.components[MODULE_NAME]
    C = comp.C

    fire_counts = torch.zeros(C, device=DEVICE)
    total_positions = 0

    for input_ids in tqdm(batches, desc="Computing alive mask"):
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_vals = ci.lower_leaky[MODULE_NAME]
        fire_counts += (ci_vals > 0).float().sum(dim=(0, 1))
        total_positions += ci_vals.shape[0] * ci_vals.shape[1]

    density = fire_counts / total_positions
    return (density > ALIVE_THRESHOLD).cpu()


@torch.no_grad()
def collect_joint_examples(
    spd_from,
    spd_to,
    batches: list[torch.Tensor],
    alive_mask_from: torch.Tensor,
    alive_mask_to: torch.Tensor,
    cases: list[dict],
    n_examples: int,
) -> dict[int, list[dict]]:
    """Run both models on same inputs. For each splitting case, find examples where
    the source fires and the targets partition its activation cleanly."""

    alive_positions_from = torch.where(alive_mask_from)[0]
    alive_positions_to = torch.where(alive_mask_to)[0]

    # Collect all global indices we need
    from_alive_to_global = {}
    for c in cases:
        from_alive_to_global[c["from_idx"]] = alive_positions_from[c["from_idx"]].item()

    to_alive_to_global = {}
    for c in cases:
        for ti in c["to_indices"]:
            to_alive_to_global[ti] = alive_positions_to[ti].item()

    # For each case, keep a list of (score, example) tuples
    case_examples: dict[int, list[tuple[float, dict]]] = {
        c["from_idx"]: [] for c in cases
    }

    for input_ids in tqdm(batches, desc="Collecting joint examples"):
        # Run source model on GPU, cache results on CPU
        spd_from.to(DEVICE)
        out_from = spd_from(input_ids, cache_type="input")
        ci_from = spd_from.calc_causal_importances(out_from.cache, sampling="none", detach_inputs=False)
        ci_pre_from = ci_from.pre_sigmoid[MODULE_NAME].cpu()  # (B, S, C)
        spd_from.cpu()
        del out_from, ci_from
        torch.cuda.empty_cache()

        # Run target model on GPU, cache results on CPU
        spd_to.to(DEVICE)
        out_to = spd_to(input_ids, cache_type="input")
        ci_to = spd_to.calc_causal_importances(out_to.cache, sampling="none", detach_inputs=False)
        ci_pre_to = ci_to.pre_sigmoid[MODULE_NAME].cpu()  # (B, S, C)
        spd_to.cpu()
        del out_to, ci_to
        torch.cuda.empty_cache()

        for case in cases:
            from_idx = case["from_idx"]
            global_from = from_alive_to_global[from_idx]

            source_acts = ci_pre_from[:, :, global_from]  # (B, S)
            source_active = (source_acts > 0)  # (B, S) bool

            # Skip BOS
            source_active[:, :2] = False

            # Count source active tokens per batch item
            n_source_active = source_active.sum(dim=1)  # (B,)

            for b in range(input_ids.shape[0]):
                n_src = n_source_active[b].item()
                if n_src < 3:
                    continue

                src_mask = source_active[b]  # (S,) bool

                # Get target activations on this input
                target_acts_list = []
                for ti in case["to_indices"]:
                    global_to = to_alive_to_global[ti]
                    t_acts = ci_pre_to[b, :, global_to]  # (S,)
                    target_acts_list.append(t_acts)

                target_acts = torch.stack(target_acts_list, dim=0)  # (n_targets, S)
                target_active = (target_acts > 0)  # (n_targets, S)
                target_active[:, :2] = False

                # Union of all targets
                target_union = target_active.any(dim=0)  # (S,)

                # How well do targets cover source?
                covered = (src_mask & target_union).sum().item()
                coverage_ratio = covered / n_src

                # How much do targets fire outside source? (lower = better partition)
                outside = (target_union & ~src_mask).sum().item()
                target_total = target_union.sum().item()
                precision = covered / max(target_total, 1)

                # Do targets actually split (not all the same)? Check pairwise overlap
                n_targets = len(case["to_indices"])
                if n_targets == 2:
                    overlap = (target_active[0] & target_active[1] & src_mask).sum().item()
                    overlap_frac = overlap / max(covered, 1)
                else:
                    # Fraction of covered positions where >1 target fires
                    multi_fire = ((target_active & src_mask.unsqueeze(0)).sum(dim=0) > 1)
                    overlap_frac = multi_fire.sum().item() / max(covered, 1)

                # Score: high coverage, low overlap between targets, reasonable precision
                score = coverage_ratio * (1 - overlap_frac) * min(precision, 1.0)

                if score < 0.3:
                    continue

                example = {
                    "token_ids": input_ids[b].cpu().tolist(),
                    "source_acts": source_acts[b].cpu().tolist(),
                    "target_acts": [ta.cpu().tolist() for ta in target_acts_list],
                    "coverage_ratio": coverage_ratio,
                    "precision": precision,
                    "overlap_frac": overlap_frac,
                    "score": score,
                    "n_source_active": n_src,
                }

                heap = case_examples[from_idx]
                if len(heap) < n_examples:
                    heap.append((score, example))
                    heap.sort(key=lambda x: x[0])
                elif score > heap[0][0]:
                    heap[0] = (score, example)
                    heap.sort(key=lambda x: x[0])

    return {
        from_idx: [ex for _, ex in sorted(exs, key=lambda x: -x[0])]
        for from_idx, exs in case_examples.items()
        if exs
    }


COLORS = [
    ("#1565C0", "#BBDEFB"),  # blue
    ("#C62828", "#FFCDD2"),  # red
    ("#2E7D32", "#C8E6C9"),  # green
    ("#6A1B9A", "#E1BEE7"),  # purple
    ("#E65100", "#FFE0B2"),  # orange
    ("#00695C", "#B2DFDB"),  # teal
]


def render_token_row(token_ids, acts, tokenizer, start, end, max_act, color_dark, color_light, label):
    """Render one row of tokens with activation highlighting."""
    html = f'<div style="margin: 2px 0; line-height: 2.0;">'
    html += f'<span style="color: {color_dark}; font-size: 11px; font-weight: bold; display: inline-block; width: 180px;">{label}</span>'

    for i in range(start, end):
        tid = token_ids[i]
        act = acts[i]
        token_str = tokenizer.decode([tid]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if token_str.startswith(" "):
            token_str = " " + token_str[1:]
        if not token_str.strip():
            token_str = token_str if token_str else "\u25A1"

        if act > 0:
            intensity = min(act / max_act, 1.0)
            # Use bold coloring — even low-intensity should be visible
            intensity = max(intensity, 0.4)  # floor so any active token is clearly colored
            r0 = int(color_light[1:3], 16)
            g0 = int(color_light[3:5], 16)
            b0 = int(color_light[5:7], 16)
            r = int(255 + (r0 - 255) * intensity)
            g = int(255 + (g0 - 255) * intensity)
            b = int(255 + (b0 - 255) * intensity)
            bg = f"rgb({r},{g},{b})"
            border = f"border-bottom: 2px solid {color_dark};"
        else:
            bg = "transparent"
            border = ""

        html += (
            f'<span style="background-color: {bg}; {border} '
            f'padding: 1px 2px; font-family: monospace; font-size: 13px;" '
            f'title="pos={i} act={act:.3f}">{token_str}</span>'
        )

    html += '</div>'
    return html


def render_joint_example_html(
    example: dict,
    case: dict,
    tokenizer,
) -> str:
    """Render a single example showing source + all targets on the same tokens."""
    token_ids = example["token_ids"]
    source_acts = example["source_acts"]
    target_acts_list = example["target_acts"]

    # Find the window with most source activity (only count positive/active tokens)
    src_arr = np.array(source_acts)
    src_positive = np.maximum(src_arr, 0)
    src_positive[:2] = 0  # ignore BOS

    # Sliding window to find densest region of active tokens
    best_start = 2
    best_score = -1
    window = CONTEXT_WINDOW
    for s in range(2, max(3, len(src_arr) - window)):
        w = src_positive[s:s + window]
        n_active = (w > 0).sum()
        score = n_active * 1000 + w.sum()  # prioritize count, then magnitude
        if score > best_score:
            best_score = score
            best_start = s

    start = max(0, best_start)
    end = min(len(token_ids), start + window)

    # Global max for consistent color scaling
    max_act = max(max(source_acts[start:end]), *(max(ta[start:end]) for ta in target_acts_list), 1e-8)

    cov = example["coverage_ratio"]
    prec = example["precision"]
    olap = example["overlap_frac"]

    html = f'<div style="border: 1px solid #eee; padding: 8px; margin: 8px 0; border-radius: 4px; background: #fafafa;">\n'
    html += f'<span style="color: #888; font-size: 11px;">coverage={cov:.0%}, precision={prec:.0%}, overlap={olap:.0%}, src_active={example["n_source_active"]}</span>\n'

    # Source row
    html += render_token_row(
        token_ids, source_acts, tokenizer, start, end, max_act,
        "#333", "#FFE082", f"Source ({RUN_FROM[0][:7]})"
    )

    # Target rows
    for j, (ti, ta) in enumerate(zip(case["to_indices"], target_acts_list)):
        color_idx = j % len(COLORS)
        dark, light = COLORS[color_idx]
        cov_j = case["coverages"][j]
        cos_j = case["cosines"][j]
        html += render_token_row(
            token_ids, ta, tokenizer, start, end, max_act,
            dark, light, f"Target {ti} (cov={cov_j:.0%})"
        )

    html += '</div>\n'
    return html


def render_splitting_case_html(
    case: dict,
    joint_examples: dict[int, list[dict]],
    tokenizer,
    case_idx: int,
) -> str:
    from_idx = case["from_idx"]

    html = f'<div style="border: 1px solid #ddd; padding: 16px; margin: 20px 0; border-radius: 8px;">\n'
    html += f'<h3>Splitting Case #{case_idx + 1}: Component {from_idx} &rarr; {case["n_splits"]} components</h3>\n'

    # Target summary
    html += '<p style="color: #666; margin: 4px 0;">Targets: '
    sorted_targets = sorted(
        zip(case["to_indices"], case["coverages"], case["cosines"], case["to_densities"]),
        key=lambda x: -x[1],
    )
    for ti, cov, cos, td in sorted_targets:
        html += f'<b>comp {ti}</b> (cov={cov:.0%}, cos={cos:.2f}) &nbsp; '
    html += '</p>\n'
    html += f'<p style="color: #666; margin: 4px 0;">Source density: {case["from_density"]:.4f}</p>\n'

    if from_idx in joint_examples:
        for ex in joint_examples[from_idx][:N_EXAMPLES]:
            html += render_joint_example_html(ex, case, tokenizer)
    else:
        html += '<p style="color: #999;">No examples found</p>\n'

    html += "</div>\n"
    return html


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached binary activations...")
    binary_from = load_binary(RUN_FROM[0])
    binary_to = load_binary(RUN_TO[0])
    print(f"  {RUN_FROM[0]}: {binary_from.shape}")
    print(f"  {RUN_TO[0]}: {binary_to.shape}")

    print("Loading cached directions...")
    dirs_from = load_dirs(RUN_FROM[0])
    dirs_to = load_dirs(RUN_TO[0])
    print(f"  {RUN_FROM[0]}: {dirs_from.shape}")
    print(f"  {RUN_TO[0]}: {dirs_to.shape}")

    print(f"\nFinding splitting cases (coverage >= {COVERAGE_THRESHOLD:.0%})...")
    cases = find_splitting_cases(binary_from, binary_to, dirs_from, dirs_to, COVERAGE_THRESHOLD)
    print(f"Found {len(cases)} splitting cases")

    for i, c in enumerate(cases[:5]):
        print(f"  #{i}: from={c['from_idx']}, n_splits={c['n_splits']}, "
              f"min_cov={min(c['coverages']):.2f}, cosines={[f'{x:.2f}' for x in c['cosines']]}")

    selected = cases[:N_SPLITTING_CASES]

    print("\nLoading eval data...")
    batches = get_eval_batches()

    # Load models one at a time for alive masks, keep on CPU
    print(f"\nLoading {RUN_FROM[0]} model...")
    spd_from, config_from = load_spd_model(RUN_FROM[1])
    spd_from.to(DEVICE)
    alive_mask_from = get_alive_mask(spd_from, batches)
    spd_from.cpu()
    torch.cuda.empty_cache()

    print(f"\nLoading {RUN_TO[0]} model...")
    spd_to, config_to = load_spd_model(RUN_TO[1])
    spd_to.to(DEVICE)
    alive_mask_to = get_alive_mask(spd_to, batches)
    spd_to.cpu()
    torch.cuda.empty_cache()

    print("\nCollecting joint examples (both models on same inputs, swapping GPU)...")
    joint_examples = collect_joint_examples(
        spd_from, spd_to, batches, alive_mask_from, alive_mask_to,
        selected, N_EXAMPLES,
    )

    del spd_from, spd_to
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer_name = config_from.get("tokenizer_name", "EleutherAI/gpt-neox-20b")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    n_with_examples = sum(1 for exs in joint_examples.values() if exs)
    print(f"Got joint examples for {n_with_examples}/{len(selected)} cases")

    print("\nRendering HTML report...")
    html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Feature Splitting: Joint Examples</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { color: #333; }
h3 { color: #1565C0; }
</style>
</head><body>
<h1>Feature Splitting: Joint Activation Examples</h1>
<p>Source: <strong>β=0.5 (baseline)</strong> &rarr; Target: <strong>β=1.0 (2&times;)</strong></p>
<p>Each example shows the <b>same input</b> run through both models. The source row (yellow)
shows where the baseline component activates. Target rows (colored) show where each
split component activates. Together, the target components should tile the source
component's activation pattern.</p>
<p><b>Coverage</b> = fraction of source-active tokens covered by union of targets.
<b>Precision</b> = fraction of target-active tokens that overlap source.
<b>Overlap</b> = fraction of covered positions where &gt;1 target fires (lower = cleaner split).</p>
"""

    for i, case in enumerate(selected):
        html += render_splitting_case_html(case, joint_examples, tokenizer, i)

    html += "</body></html>"

    html_path = OUTPUT_DIR / "splitting_examples.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nSaved {html_path}")

    summary = {
        "source_run": RUN_FROM[0],
        "target_run": RUN_TO[0],
        "coverage_threshold": COVERAGE_THRESHOLD,
        "n_total_splitting_cases": len(cases),
        "n_displayed": len(selected),
        "cases": [
            {
                "from_idx": c["from_idx"],
                "n_splits": c["n_splits"],
                "to_indices": c["to_indices"],
                "coverages": c["coverages"],
                "cosines": c["cosines"],
                "from_density": c["from_density"],
                "to_densities": c["to_densities"],
            }
            for c in selected
        ],
    }
    with open(OUTPUT_DIR / "splitting_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {OUTPUT_DIR / 'splitting_summary.json'}")


if __name__ == "__main__":
    main()
