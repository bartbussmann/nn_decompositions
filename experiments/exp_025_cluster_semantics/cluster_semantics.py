"""Investigate whether UMAP/HDBSCAN clusters have semantic similarity.

For each cluster from exp_024, collects top-activating contexts across components
and uses an LLM to label the cluster and judge semantic coherence.

Usage:
    python experiments/exp_025_cluster_semantics/cluster_semantics.py --api_key sk-...
    python experiments/exp_025_cluster_semantics/cluster_semantics.py --api_key sk-... --max_clusters 10
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import hdbscan
import numpy as np
import torch
import torch.nn.functional as F
import umap
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 2
ALIVE_THRESHOLD = 1e-5
N_EVAL_BATCHES = 20
BATCH_SIZE = 16
SEQ_LEN = 512
CTX_LEN = 32  # tokens of context around top activation
N_TOP_EXAMPLES = 5  # top-activating examples per component
N_COMPONENTS_PER_CLUSTER = 5  # max components to sample from each cluster
OUTPUT_DIR = Path(__file__).parent / "output"


@dataclass
class TopExample:
    tokens: list[str]
    acts: list[float]
    center_idx: int  # position of the max-activating token in the window


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


def load_transcoder(checkpoint_dir: str) -> BatchTopKTranscoder:
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["dtype"] = getattr(torch, cfg_dict.get("dtype", "torch.float32").replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    encoder = BatchTopKTranscoder(cfg)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    encoder.eval()
    return encoder


# =========================================================================
# Activation collection + clustering (reused from exp_024)
# =========================================================================


@torch.no_grad()
def compute_spd_activations(spd_model, batches):
    """Returns (directions, densities, alive_mask, all_acts_per_batch)."""
    module_name = f"h.{LAYER}.mlp.down_proj"
    comp = spd_model.components[module_name]
    C = comp.C

    fire_counts = torch.zeros(C, device=DEVICE)
    total_positions = 0
    all_acts = []  # list of (batch, seq, C) tensors

    for input_ids in tqdm(batches, desc="SPD activations"):
        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")
        ci_vals = ci.lower_leaky[module_name]  # (batch, seq, C)
        fire_counts += (ci_vals > 0).float().sum(dim=(0, 1))
        total_positions += ci_vals.shape[0] * ci_vals.shape[1]
        all_acts.append(ci_vals.cpu())

    density = fire_counts / total_positions
    alive_mask = density > ALIVE_THRESHOLD

    U = comp.U.float()
    U_alive = U[alive_mask]
    U_norm = F.normalize(U_alive, dim=1).cpu().numpy()
    density_alive = density[alive_mask].cpu().numpy()

    # Stack all activations: (n_batches * batch_size, seq, C)
    all_acts_cat = torch.cat(all_acts, dim=0)  # (total_batch, seq, C)
    # Keep only alive components
    all_acts_alive = all_acts_cat[:, :, alive_mask.cpu()]

    print(f"SPD: {alive_mask.sum().item()}/{C} alive")
    return U_norm, density_alive, alive_mask, all_acts_alive


@torch.no_grad()
def compute_tc_activations(tc, base_model, batches):
    """Returns (directions, densities, alive_mask, all_acts_per_batch)."""
    dict_size = tc.cfg.dict_size

    fire_counts = torch.zeros(dict_size, device=DEVICE)
    total_positions = 0
    all_acts = []

    for input_ids in tqdm(batches, desc="TC activations"):
        captured = {}
        def _hook(_mod, _inp, out):
            captured["mlp_in"] = out.detach()
        hook = base_model.h[LAYER].rms_2.register_forward_hook(_hook)
        base_model(input_ids)
        hook.remove()

        mlp_in = captured["mlp_in"]
        flat = mlp_in.reshape(-1, mlp_in.shape[-1])

        use_pre_enc_bias = tc.cfg.pre_enc_bias and tc.cfg.input_size == tc.cfg.output_size
        x_enc = flat - tc.b_dec if use_pre_enc_bias else flat
        acts = F.relu(x_enc @ tc.W_enc)  # (n_tokens, dict_size)

        fire_counts += (acts > 0).float().sum(dim=0)
        total_positions += flat.shape[0]

        # Reshape back to (batch, seq, dict_size)
        all_acts.append(acts.reshape(input_ids.shape[0], input_ids.shape[1], -1).cpu())

    density = fire_counts / total_positions
    alive_mask = density > ALIVE_THRESHOLD

    W_dec = tc.W_dec.float()
    W_dec_alive = W_dec[alive_mask]
    W_dec_norm = F.normalize(W_dec_alive, dim=1).cpu().numpy()
    density_alive = density[alive_mask].cpu().numpy()

    all_acts_cat = torch.cat(all_acts, dim=0)
    all_acts_alive = all_acts_cat[:, :, alive_mask.cpu()]

    print(f"TC: {alive_mask.sum().item()}/{dict_size} alive")
    return W_dec_norm, density_alive, alive_mask, all_acts_alive


def cluster_directions(dirs: np.ndarray) -> np.ndarray:
    """HDBSCAN clustering on 10D UMAP (Towards Monosemanticity params)."""
    reducer = umap.UMAP(
        n_components=10, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=42,
    )
    emb_10d = reducer.fit_transform(dirs)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
    labels = clusterer.fit_predict(emb_10d)
    n_clusters = labels.max() + 1
    n_noise = (labels == -1).sum()
    print(f"  HDBSCAN: {n_clusters} clusters, {n_noise}/{len(labels)} noise")
    return labels


# =========================================================================
# Top-activating example extraction
# =========================================================================


def get_top_examples(
    all_acts: torch.Tensor,  # (total_seqs, seq_len, n_components)
    component_idx: int,
    all_input_ids: torch.Tensor,  # (total_seqs, seq_len)
    tokenizer: AutoTokenizer,
    n_top: int = N_TOP_EXAMPLES,
    ctx_len: int = CTX_LEN,
) -> list[TopExample]:
    """Find top-activating contexts for a single component."""
    acts = all_acts[:, :, component_idx]  # (total_seqs, seq_len)
    # Exclude first/last ctx_len//2 tokens to ensure full context window
    half_ctx = ctx_len // 2
    acts_trimmed = acts[:, half_ctx:-half_ctx].contiguous()
    flat = acts_trimmed.reshape(-1)

    k = min(n_top, (flat > 0).sum().item())
    if k == 0:
        return []

    topk = torch.topk(flat, k)
    indices = topk.indices

    examples = []
    seq_len_trimmed = acts_trimmed.shape[1]
    for idx in indices:
        batch_idx = idx // seq_len_trimmed
        seq_idx = (idx % seq_len_trimmed) + half_ctx  # offset back

        start = max(0, seq_idx - half_ctx)
        end = min(all_input_ids.shape[1], seq_idx + half_ctx)
        token_ids = all_input_ids[batch_idx, start:end].tolist()
        act_vals = acts[batch_idx, start:end].tolist()
        center = seq_idx - start

        str_tokens = [tokenizer.decode([t]) for t in token_ids]
        examples.append(TopExample(tokens=str_tokens, acts=act_vals, center_idx=center))

    return examples


def format_example_highlighted(ex: TopExample, threshold_frac: float = 0.01) -> str:
    """Format with <<highlighted>> tokens above threshold."""
    max_act = max(ex.acts)
    threshold = max_act * threshold_frac
    parts = []
    for tok, act in zip(ex.tokens, ex.acts):
        clean = tok.replace("\n", "\u21b5").replace("\ufffd", "")
        if act > threshold:
            parts.append(f"<<{clean}>>")
        else:
            parts.append(clean)
    return "".join(parts)


# =========================================================================
# LLM labeling
# =========================================================================

CLUSTER_LABEL_SYSTEM = """You are analyzing clusters of neural network features. You will be shown top-activating text examples for several features that were grouped into a cluster based on the similarity of their weight vectors.

For each feature, the tokens it activates most strongly on are marked with << >>. Determine whether the features in this cluster share a common semantic theme.

Respond in this exact JSON format:
{"label": "<short label, 3-6 words>", "coherent": <true or false>, "explanation": "<1 sentence explaining why>"}

If the features clearly activate on a shared concept/pattern, set coherent=true and give a descriptive label. If they seem unrelated, set coherent=false and label as "mixed/incoherent"."""


def build_cluster_prompt(cluster_examples: dict[int, list[TopExample]]) -> str:
    """Build user prompt showing examples for each component in the cluster."""
    parts = []
    for comp_idx, examples in cluster_examples.items():
        parts.append(f"--- Feature {comp_idx} ---")
        for i, ex in enumerate(examples, 1):
            parts.append(f"  {i}. {format_example_highlighted(ex)}")
        parts.append("")
    return "\n".join(parts)


def label_cluster(client: OpenAI, model: str, prompt: str) -> dict:
    """Call LLM to label a cluster."""
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CLUSTER_LABEL_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            # Parse JSON from response
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2 ** attempt + random.random())
            else:
                return {"label": "parse_error", "coherent": False, "explanation": str(e)}
    return {"label": "timeout", "coherent": False, "explanation": "Max retries exceeded"}


# =========================================================================
# Main
# =========================================================================


def process_method(
    name: str,
    dirs: np.ndarray,
    all_acts: torch.Tensor,
    all_input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    client: OpenAI,
    model: str,
    max_clusters: int,
) -> dict:
    """Cluster, collect examples, label clusters for one method."""
    print(f"\n{'='*60}")
    print(f"Processing {name}")
    print(f"{'='*60}")

    labels = cluster_directions(dirs)
    n_clusters = labels.max() + 1

    # Sort clusters by size (descending)
    cluster_sizes = {}
    for ci in range(n_clusters):
        cluster_sizes[ci] = (labels == ci).sum()
    sorted_clusters = sorted(cluster_sizes.keys(), key=lambda c: -cluster_sizes[c])

    clusters_to_process = sorted_clusters[:max_clusters]
    print(f"Processing {len(clusters_to_process)} largest clusters (of {n_clusters})")

    results = []
    for ci in tqdm(clusters_to_process, desc=f"{name} clusters"):
        member_indices = np.where(labels == ci)[0]
        size = len(member_indices)

        # Sample components from cluster
        if len(member_indices) > N_COMPONENTS_PER_CLUSTER:
            sampled = np.random.RandomState(ci).choice(
                member_indices, N_COMPONENTS_PER_CLUSTER, replace=False
            )
        else:
            sampled = member_indices

        # Collect top examples for each sampled component
        cluster_examples = {}
        for comp_idx in sampled:
            examples = get_top_examples(
                all_acts, comp_idx, all_input_ids, tokenizer,
            )
            if examples:
                cluster_examples[int(comp_idx)] = examples

        if not cluster_examples:
            continue

        # Build prompt and label
        prompt = build_cluster_prompt(cluster_examples)
        result = label_cluster(client, model, prompt)
        result["cluster_id"] = int(ci)
        result["size"] = int(size)
        result["n_components_sampled"] = len(cluster_examples)

        # Store example texts for inspection
        result["example_texts"] = {
            str(k): [format_example_highlighted(ex) for ex in v]
            for k, v in cluster_examples.items()
        }

        results.append(result)
        label = result.get("label", "?")
        coherent = result.get("coherent", "?")
        print(f"  Cluster {ci} (size={size}): {label} [coherent={coherent}]")

    # Summary stats
    n_coherent = sum(1 for r in results if r.get("coherent"))
    print(f"\n{name} summary: {n_coherent}/{len(results)} clusters coherent")

    return {
        "method": name,
        "n_alive": int(dirs.shape[0]),
        "n_clusters": int(n_clusters),
        "n_evaluated": len(results),
        "n_coherent": n_coherent,
        "coherence_rate": n_coherent / len(results) if results else 0,
        "clusters": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_clusters", type=int, default=50)
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-55ea3f9b")
    parser.add_argument("--tc_checkpoint", type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent
                                    / "checkpoints/jose/tc_independent_k32_tc_independent_k32_checkpoint_layer2_final"))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = args.api_key
    if api_key is None:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
        assert api_key, f"No --api_key and no OPENAI_API_KEY in {env_path}"
    client = OpenAI(api_key=api_key)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Load data
    print("Loading eval data...")
    batches = get_eval_batches()
    all_input_ids = torch.cat(batches, dim=0).cpu()  # (total_seqs, seq_len)

    # Load models
    from analysis.collect_spd_activations import load_spd_model
    print(f"\nLoading SPD model...")
    spd_model, _ = load_spd_model(args.spd_run)
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    # Compute activations
    spd_dirs, spd_dens, _, spd_acts = compute_spd_activations(spd_model, batches)
    tc = load_transcoder(args.tc_checkpoint)
    tc_dirs, tc_dens, _, tc_acts = compute_tc_activations(tc, base_model, batches)

    # Process both methods
    spd_results = process_method(
        "SPD", spd_dirs, spd_acts, all_input_ids, tokenizer,
        client, args.openai_model, args.max_clusters,
    )
    tc_results = process_method(
        "TC", tc_dirs, tc_acts, all_input_ids, tokenizer,
        client, args.openai_model, args.max_clusters,
    )

    # Save
    output = {
        "config": {
            "layer": LAYER,
            "alive_threshold": ALIVE_THRESHOLD,
            "n_eval_batches": N_EVAL_BATCHES,
            "n_top_examples": N_TOP_EXAMPLES,
            "n_components_per_cluster": N_COMPONENTS_PER_CLUSTER,
            "openai_model": args.openai_model,
            "max_clusters": args.max_clusters,
        },
        "spd": spd_results,
        "tc": tc_results,
    }

    out_path = OUTPUT_DIR / "cluster_semantics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'':>20} {'SPD':>10} {'TC':>10}")
    print(f"{'Alive components':>20} {spd_results['n_alive']:>10} {tc_results['n_alive']:>10}")
    print(f"{'Total clusters':>20} {spd_results['n_clusters']:>10} {tc_results['n_clusters']:>10}")
    print(f"{'Evaluated':>20} {spd_results['n_evaluated']:>10} {tc_results['n_evaluated']:>10}")
    print(f"{'Coherent':>20} {spd_results['n_coherent']:>10} {tc_results['n_coherent']:>10}")
    print(f"{'Coherence rate':>20} {spd_results['coherence_rate']:>10.1%} {tc_results['coherence_rate']:>10.1%}")


if __name__ == "__main__":
    main()
