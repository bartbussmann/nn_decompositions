"""Multi-feature steering evaluation: Transcoders vs CLTs vs SPD.

Tests whether features from each method compose well when steering with
multiple features simultaneously, rather than just the single best AUROC
feature (as in exp_013).

For each concept and each K in [1, 3, 5, 10]:
1. Select top-K features by all-layer AUROC (same approach as exp_014)
2. Construct a combined steering vector: sum of factor * normalize(decoder_dir_i)
   for each of the K features, distributed across layers as appropriate
3. Generate text with the combined steering
4. LLM-judge scores (concept + fluency)

Method-specific multi-feature steering:
- Transcoder: group features by layer, sum decoder directions per layer, patch
  each layer's MLP with its aggregated steering vector
- CLT: features from source layer i write to target layers i..n-1; accumulate
  contributions per target layer across all selected features
- SPD: group U-vector directions by layer, sum per layer, patch each layer's MLP

Usage:
    python experiments/exp_016_multi_feature_steering/multi_feature_steering.py \
        --api_key sk-...

    # Quick test:
    python experiments/exp_016_multi_feature_steering/multi_feature_steering.py \
        --api_key sk-... --quick
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from clt import CrossLayerTranscoder
from config import CLTConfig, EncoderConfig
from transcoder import BatchTopKTranscoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]

# Same 10 concepts as exp_013/014.
CONCEPTS = [
    {
        "id": "sports",
        "description": "text about sports, athletics, or competitive physical activities",
        "positive_keywords": ["football", "basketball", "soccer", "tennis", "baseball",
                             "athlete", "championship", "tournament", "scored", "team won",
                             "Olympic", "marathon", "swimming race", "goal keeper", "match"],
        "negative_keywords": ["cooking recipe", "mathematics proof", "software bug",
                             "financial report", "medical diagnosis", "weather forecast",
                             "historical document", "philosophy essay", "chemistry lab",
                             "legal contract"],
    },
    {
        "id": "code",
        "description": "text about programming, software development, or computer code",
        "positive_keywords": ["function", "variable", "python", "javascript", "compile",
                             "debug", "API endpoint", "class definition", "git commit",
                             "database query", "algorithm", "syntax error", "import library",
                             "recursion", "object oriented"],
        "negative_keywords": ["garden flowers", "ocean waves", "mountain hiking",
                             "birthday party", "ancient civilization", "music concert",
                             "animal behavior", "painting techniques", "weather patterns",
                             "cooking ingredients"],
    },
    {
        "id": "science",
        "description": "text about scientific research, experiments, or discoveries",
        "positive_keywords": ["hypothesis", "experiment", "molecules", "quantum physics",
                             "research paper", "laboratory", "DNA sequence", "chemical reaction",
                             "scientific method", "peer review", "Nobel prize", "clinical trial",
                             "particle accelerator", "genome", "photosynthesis"],
        "negative_keywords": ["fashion trends", "celebrity gossip", "real estate market",
                             "cooking show", "travel blog", "sports scores", "movie review",
                             "music lyrics", "social media post", "shopping list"],
    },
    {
        "id": "politics",
        "description": "text about politics, government, or political events",
        "positive_keywords": ["election", "congress", "legislation", "president", "democracy",
                             "political party", "campaign", "voter", "senate", "policy reform",
                             "diplomatic relations", "parliamentary debate", "constitutional",
                             "governor", "bipartisan"],
        "negative_keywords": ["recipe for cake", "guitar chords", "yoga poses",
                             "gardening tips", "pet care", "home decoration",
                             "astronomy photos", "board games", "knitting patterns",
                             "swimming lessons"],
    },
    {
        "id": "food",
        "description": "text about food, cooking, or culinary topics",
        "positive_keywords": ["recipe", "ingredients", "baking", "sauteed", "cuisine",
                             "restaurant", "flavor", "seasoning", "appetizer", "dessert",
                             "chef prepared", "marinated", "culinary", "fresh produce",
                             "breakfast menu"],
        "negative_keywords": ["quantum mechanics", "stock market", "military history",
                             "computer network", "legal precedent", "geological formation",
                             "space exploration", "electrical circuit", "philosophical theory",
                             "surgical procedure"],
    },
    {
        "id": "music",
        "description": "text about music, musicians, or musical performances",
        "positive_keywords": ["melody", "guitar solo", "symphony", "concert", "album release",
                             "singer", "rhythm", "chord progression", "band tour",
                             "musical instrument", "jazz performance", "hip hop beat",
                             "opera singer", "vinyl record", "music festival"],
        "negative_keywords": ["tax return", "plumbing repair", "insurance claim",
                             "traffic accident", "court ruling", "construction site",
                             "warehouse inventory", "dental appointment", "parking ticket",
                             "utility bill"],
    },
    {
        "id": "nature",
        "description": "text about nature, wildlife, or the natural environment",
        "positive_keywords": ["forest", "wildlife", "ecosystem", "mountains", "ocean",
                             "endangered species", "national park", "coral reef", "migration",
                             "biodiversity", "rainforest", "volcanic eruption", "river delta",
                             "arctic ice", "animal habitat"],
        "negative_keywords": ["office meeting", "spreadsheet", "email inbox",
                             "factory production", "highway construction", "shopping mall",
                             "television show", "bank transaction", "phone call",
                             "elevator repair"],
    },
    {
        "id": "history",
        "description": "text about historical events, periods, or figures",
        "positive_keywords": ["ancient Rome", "World War", "medieval", "revolution",
                             "archaeological", "historical figure", "empire fell",
                             "Renaissance era", "colonial period", "treaty signed",
                             "civilization", "dynasty", "founding fathers", "industrial age",
                             "prehistoric"],
        "negative_keywords": ["smartphone app", "social media", "virtual reality",
                             "streaming service", "electric car", "cryptocurrency",
                             "online shopping", "video game", "drone delivery",
                             "artificial intelligence"],
    },
    {
        "id": "health",
        "description": "text about health, medicine, or medical topics",
        "positive_keywords": ["diagnosis", "treatment", "symptoms", "hospital", "medication",
                             "doctor recommended", "blood pressure", "immune system",
                             "surgical procedure", "clinical study", "patient recovery",
                             "vaccine", "mental health", "physical therapy", "nutrition plan"],
        "negative_keywords": ["space shuttle", "deep sea exploration", "mountain climbing",
                             "car racing", "chess tournament", "poetry reading",
                             "architecture design", "film production", "dance performance",
                             "language translation"],
    },
    {
        "id": "finance",
        "description": "text about finance, economics, or financial markets",
        "positive_keywords": ["stock market", "investment", "interest rate", "inflation",
                             "GDP growth", "banking", "portfolio", "dividend", "fiscal policy",
                             "bond yield", "cryptocurrency", "market crash", "revenue",
                             "quarterly earnings", "hedge fund"],
        "negative_keywords": ["butterfly migration", "folk tale", "pottery class",
                             "surfing lesson", "bird watching", "campfire story",
                             "flower arrangement", "puppet show", "ice skating",
                             "kite flying"],
    },
]

PROMPTS = [
    "The most interesting thing about this is that",
    "In recent years, there has been a growing",
    "It is widely known that the",
    "One of the key factors in understanding",
    "According to experts, the main",
    "There are several important reasons why",
    "The latest developments suggest that",
    "Many people have started to notice that",
    "A comprehensive review of the evidence shows",
    "Perhaps the most significant aspect of",
    "When we consider the broader implications,",
    "The relationship between these factors is",
    "New research has revealed that the",
    "Throughout history, the role of",
    "What makes this particularly notable is",
]

TEMPLATES = [
    "This text is about {kw}. It discusses various aspects of {kw} in detail.",
    "The topic of {kw} is explored here, covering recent developments.",
    "An in-depth look at {kw} and its impact on modern society.",
    "{kw} has been a subject of great interest lately.",
    "Recent news about {kw} suggests significant changes ahead.",
    "Understanding {kw} requires knowledge of the broader context.",
    "The field of {kw} has evolved significantly over the years.",
    "Experts in {kw} have noted several important trends.",
]


# =============================================================================
# Config
# =============================================================================


@dataclass
class MultiFeatureSteeringConfig:
    tc_project: str = "mats-sprint/pile_transcoder_sweep3"
    clt_project: str = "mats-sprint/pile_clt"
    spd_run: str = "goodfire/spd/s-275c8f21"

    n_concepts: int = 10
    n_prompts: int = 5
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    steering_factor: float = 4.0

    n_auroc_examples: int = 200
    auroc_seq_len: int = 128
    auroc_batch_size: int = 32

    max_new_tokens: int = 128
    temperature: float = 0.7

    openai_model: str = "gpt-4o-mini"
    api_key: str = ""

    save_dir: str = "experiments/exp_016_multi_feature_steering/output"


# =============================================================================
# Model loading (same patterns as exp_013/014)
# =============================================================================


@contextmanager
def patched_forward(module: nn.Module, patched_fn):
    original = module.forward
    module.forward = patched_fn
    try:
        yield
    finally:
        module.forward = original


def load_spd_model(wandb_path: str):
    from analysis.collect_spd_activations import load_spd_model as _load
    return _load(wandb_path)


def download_wandb_artifact(project: str, artifact_name: str, dest: Path) -> Path:
    import wandb
    if dest.exists() and (dest / "encoder.pt").exists():
        return dest
    api = wandb.Api()
    artifact = api.artifact(f"{project}/{artifact_name}")
    artifact.download(root=str(dest))
    return dest


def load_transcoder(checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = EncoderConfig(**cfg_dict)
    tc = BatchTopKTranscoder(cfg)
    tc.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    tc.eval()
    return tc


def load_clt(checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["layers"] = json.loads(cfg_dict["layers"])
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg_dict["device"] = DEVICE
    cfg = CLTConfig(**cfg_dict)
    clt = CrossLayerTranscoder(cfg)
    clt.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    clt.eval()
    return clt


def download_best_transcoders(project: str) -> dict[int, Path]:
    """Download the transcoder with the highest top_k for each layer."""
    import wandb
    api = wandb.Api()
    runs = api.runs(project)

    best_per_layer: dict[int, tuple[int, object]] = {}
    for run in runs:
        if run.state != "finished":
            continue
        cfg = run.config
        top_k = cfg.get("top_k", 0)
        run_layer = None
        for part in run.name.split("_"):
            if part.startswith("L") and part[1:].isdigit():
                run_layer = int(part[1:])
                break
        if run_layer is None:
            continue
        if run_layer not in best_per_layer or top_k > best_per_layer[run_layer][0]:
            best_per_layer[run_layer] = (top_k, run)

    paths = {}
    for layer in LAYERS:
        assert layer in best_per_layer, f"No transcoder found for layer {layer}"
        top_k, run = best_per_layer[layer]
        arts = [a for a in run.logged_artifacts() if a.type == "model"]
        assert len(arts) == 1
        dest = Path(f"checkpoints/multi_steer_tc_L{layer}_{run.name}")
        download_wandb_artifact(project, arts[0].name, dest)
        paths[layer] = dest
        print(f"  Layer {layer}: top_k={top_k}, run={run.name}")
    return paths


def download_best_clt(project: str) -> Path:
    """Download the CLT with the highest top_k."""
    import wandb
    api = wandb.Api()
    runs = api.runs(project)
    best_run = None
    best_topk = -1
    for run in runs:
        if run.state != "finished":
            continue
        top_k = run.config.get("top_k", 0)
        if top_k > best_topk:
            best_topk = top_k
            best_run = run
    assert best_run is not None, "No finished CLT run found"
    arts = [a for a in best_run.logged_artifacts() if a.type == "model"]
    assert len(arts) == 1
    dest = Path(f"checkpoints/multi_steer_clt_{best_run.name}")
    download_wandb_artifact(project, arts[0].name, dest)
    print(f"  CLT: top_k={best_topk}, run={best_run.name}")
    return dest


# =============================================================================
# AUROC (no sklearn)
# =============================================================================


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    assert labels.shape == scores.shape
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp, fp, auc = 0, 0, 0.0
    prev_fpr, prev_tpr = 0.0, 0.0
    for i in range(n):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += 0.5 * (fpr - prev_fpr) * (tpr + prev_tpr)
        prev_fpr, prev_tpr = fpr, tpr
    return auc


def make_concept_texts(concept: dict, n_per_class: int) -> tuple[list[str], list[str]]:
    pos_keywords = concept["positive_keywords"]
    neg_keywords = concept["negative_keywords"]

    positives = [TEMPLATES[i % len(TEMPLATES)].format(kw=pos_keywords[i % len(pos_keywords)])
                 for i in range(n_per_class)]
    negatives = [TEMPLATES[i % len(TEMPLATES)].format(kw=neg_keywords[i % len(neg_keywords)])
                 for i in range(n_per_class)]
    return positives, negatives


# =============================================================================
# All-layer AUROC computation (from exp_014 approach)
# =============================================================================


@torch.no_grad()
def _collect_rms2_outputs(base_model, input_ids):
    captured = {}
    hooks = []
    for layer_idx in LAYERS:
        def _make_hook(li):
            def _hook(_mod, _inp, out):
                captured[li] = out.detach()
            return _hook
        hooks.append(base_model.h[layer_idx].rms_2.register_forward_hook(_make_hook(layer_idx)))
    base_model(input_ids)
    for h in hooks:
        h.remove()
    return captured


@torch.no_grad()
def compute_all_transcoder_aurocs(
    base_model,
    transcoders: dict[int, BatchTopKTranscoder],
    tokenizer,
    concept: dict,
    cfg: MultiFeatureSteeringConfig,
) -> list[tuple[int, int, float]]:
    """Compute AUROC for every transcoder feature across all layers.

    Returns list of (layer, feature_idx, auroc) sorted by auroc descending.
    """
    positives, negatives = make_concept_texts(concept, cfg.n_auroc_examples)
    all_texts = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives))

    scores_per_layer = {}
    for layer_idx in LAYERS:
        dict_size = transcoders[layer_idx].cfg.dict_size
        scores_per_layer[layer_idx] = np.zeros((len(all_texts), dict_size))

    for i in range(0, len(all_texts), cfg.auroc_batch_size):
        batch_texts = all_texts[i : i + cfg.auroc_batch_size]
        encoded = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=cfg.auroc_seq_len,
        )
        input_ids = encoded["input_ids"].to(DEVICE)

        captured = _collect_rms2_outputs(base_model, input_ids)

        for layer_idx in LAYERS:
            tc = transcoders[layer_idx]
            mlp_in = captured[layer_idx]
            flat = mlp_in.reshape(-1, tc.cfg.input_size)
            use_pre_enc_bias = tc.cfg.pre_enc_bias and tc.input_size == tc.output_size
            x_enc = flat - tc.b_dec if use_pre_enc_bias else flat
            acts = F.relu(x_enc @ tc.W_enc)
            acts_2d = acts.reshape(mlp_in.shape[0], mlp_in.shape[1], -1)
            max_acts = acts_2d.max(dim=1).values
            scores_per_layer[layer_idx][i : i + len(batch_texts)] = max_acts.cpu().numpy()

    results = []
    for layer_idx in LAYERS:
        scores_np = scores_per_layer[layer_idx]
        for j in range(scores_np.shape[1]):
            col = scores_np[:, j]
            if col.max() == col.min():
                continue
            auc = roc_auc_score(labels, col)
            results.append((layer_idx, j, auc))

    results.sort(key=lambda x: -x[2])
    return results


@torch.no_grad()
def compute_all_clt_aurocs(
    base_model,
    clt: CrossLayerTranscoder,
    tokenizer,
    concept: dict,
    cfg: MultiFeatureSteeringConfig,
) -> list[tuple[int, int, float]]:
    """Compute AUROC for every CLT encoder feature across all layers.

    Returns list of (encoder_layer_idx, feature_idx, auroc) sorted descending.
    """
    positives, negatives = make_concept_texts(concept, cfg.n_auroc_examples)
    all_texts = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives))

    n_layers = clt.cfg.n_layers
    dict_size = clt.cfg.dict_size
    scores_per_layer = {i: np.zeros((len(all_texts), dict_size)) for i in range(n_layers)}

    for i in range(0, len(all_texts), cfg.auroc_batch_size):
        batch_texts = all_texts[i : i + cfg.auroc_batch_size]
        encoded = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=cfg.auroc_seq_len,
        )
        input_ids = encoded["input_ids"].to(DEVICE)

        captured = _collect_rms2_outputs(base_model, input_ids)

        for li in range(n_layers):
            actual_layer = clt.cfg.layers[li]
            mlp_in = captured[actual_layer]
            flat = mlp_in.reshape(-1, clt.cfg.input_size)
            acts = F.relu(flat @ clt.W_enc[li] + clt.b_enc[li])
            acts_2d = acts.reshape(mlp_in.shape[0], mlp_in.shape[1], -1)
            max_acts = acts_2d.max(dim=1).values
            scores_per_layer[li][i : i + len(batch_texts)] = max_acts.cpu().numpy()

    results = []
    for li in range(n_layers):
        scores_np = scores_per_layer[li]
        for j in range(dict_size):
            col = scores_np[:, j]
            if col.max() == col.min():
                continue
            auc = roc_auc_score(labels, col)
            results.append((li, j, auc))

    results.sort(key=lambda x: -x[2])
    return results


@torch.no_grad()
def compute_all_spd_aurocs(
    spd_model,
    tokenizer,
    concept: dict,
    cfg: MultiFeatureSteeringConfig,
) -> list[tuple[str, int, float]]:
    """Compute AUROC for every SPD component (c_fc and down_proj) across all layers.

    Returns list of (module_name, component_idx, auroc) sorted descending.
    """
    positives, negatives = make_concept_texts(concept, cfg.n_auroc_examples)
    all_texts = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives))

    module_names = [f"h.{l}.mlp.c_fc" for l in LAYERS] + [f"h.{l}.mlp.down_proj" for l in LAYERS]
    scores_per_module: dict[str, np.ndarray] = {}
    for mod_name in module_names:
        n_comp = spd_model.module_to_c[mod_name]
        scores_per_module[mod_name] = np.zeros((len(all_texts), n_comp))

    for i in range(0, len(all_texts), cfg.auroc_batch_size):
        batch_texts = all_texts[i : i + cfg.auroc_batch_size]
        encoded = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=cfg.auroc_seq_len,
        )
        input_ids = encoded["input_ids"].to(DEVICE)

        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        for mod_name in module_names:
            ci_scores = ci.lower_leaky[mod_name].clamp(0, 1)
            max_scores = ci_scores.max(dim=1).values
            scores_per_module[mod_name][i : i + len(batch_texts)] = max_scores.cpu().numpy()

    results = []
    for mod_name in module_names:
        scores_np = scores_per_module[mod_name]
        for j in range(scores_np.shape[1]):
            col = scores_np[:, j]
            if col.max() == col.min():
                continue
            auc = roc_auc_score(labels, col)
            results.append((mod_name, j, auc))

    results.sort(key=lambda x: -x[2])
    return results


# =============================================================================
# Multi-feature steering vector construction
# =============================================================================


def build_transcoder_layer_vectors(
    transcoders: dict[int, BatchTopKTranscoder],
    selected_features: list[tuple[int, int, float]],
    steering_factor: float,
) -> dict[int, torch.Tensor]:
    """Build per-layer steering vectors for transcoder multi-feature steering.

    Groups selected features by layer. For each layer, sums
    factor * normalize(W_dec[feat_idx]) across all features at that layer.

    Returns {layer_idx: steering_vector} for layers that have at least one feature.
    """
    layer_vectors: dict[int, torch.Tensor] = {}
    for layer_idx, feat_idx, _auroc in selected_features:
        tc = transcoders[layer_idx]
        decoder_dir = tc.W_dec[feat_idx]  # (output_size,)
        contribution = steering_factor * F.normalize(decoder_dir, dim=0)
        if layer_idx in layer_vectors:
            layer_vectors[layer_idx] = layer_vectors[layer_idx] + contribution
        else:
            layer_vectors[layer_idx] = contribution.clone()
    return layer_vectors


def build_clt_layer_vectors(
    clt: CrossLayerTranscoder,
    selected_features: list[tuple[int, int, float]],
    steering_factor: float,
) -> dict[int, torch.Tensor]:
    """Build per-target-layer steering vectors for CLT multi-feature steering.

    Each selected feature (source_layer, feat_idx) writes to target layers
    source_layer..n-1. For each target layer, accumulate:
      factor * normalize(W_dec[source][target - source][feat_idx])

    Returns {actual_target_layer: steering_vector} for layers with contributions.
    """
    n_layers = clt.cfg.n_layers
    layer_vectors: dict[int, torch.Tensor] = {}
    for source_layer_idx, feat_idx, _auroc in selected_features:
        n_targets = clt.W_dec[source_layer_idx].shape[0]  # n_layers - source_layer_idx
        for offset in range(n_targets):
            target_layer_idx = source_layer_idx + offset
            actual_target_layer = clt.cfg.layers[target_layer_idx]
            decoder_dir = clt.W_dec[source_layer_idx][offset, feat_idx]  # (output_size,)
            contribution = steering_factor * F.normalize(decoder_dir, dim=0)
            if actual_target_layer in layer_vectors:
                layer_vectors[actual_target_layer] = layer_vectors[actual_target_layer] + contribution
            else:
                layer_vectors[actual_target_layer] = contribution.clone()
    return layer_vectors


def _extract_layer_from_module_name(module_name: str) -> int:
    """Extract layer index from SPD module name like 'h.2.mlp.down_proj'."""
    parts = module_name.split(".")
    return int(parts[1])


def build_spd_layer_vectors(
    spd_model,
    selected_features: list[tuple[str, int, float]],
    steering_factor: float,
) -> dict[int, torch.Tensor]:
    """Build per-layer steering vectors for SPD multi-feature steering.

    Each selected component (module_name, comp_idx) contributes
    factor * normalize(U[comp_idx]) to the layer indicated by the module name.

    Returns {layer_idx: steering_vector} for layers with at least one component.
    """
    layer_vectors: dict[int, torch.Tensor] = {}
    for module_name, comp_idx, _auroc in selected_features:
        layer_idx = _extract_layer_from_module_name(module_name)
        components = spd_model.components[module_name]
        u_vec = components.U[comp_idx]  # (d_model,)
        contribution = steering_factor * F.normalize(u_vec, dim=0)
        if layer_idx in layer_vectors:
            layer_vectors[layer_idx] = layer_vectors[layer_idx] + contribution
        else:
            layer_vectors[layer_idx] = contribution.clone()
    return layer_vectors


# =============================================================================
# Multi-feature text generation
# =============================================================================


@torch.no_grad()
def generate_steered_text_multi(
    base_model,
    layer_vectors: dict[int, torch.Tensor],
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate text with multi-layer steering via ExitStack + factory pattern.

    layer_vectors: {actual_layer_idx: steering_vector_tensor}
    Each MLP at a layer with a steering vector gets patched to add the vector.
    """
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(DEVICE)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with ExitStack() as stack:
            for layer_idx in LAYERS:
                if layer_idx not in layer_vectors:
                    continue
                mlp = base_model.h[layer_idx].mlp

                def _make_patched(mlp_, sv_):
                    def _patched(hidden_states):
                        h = mlp_.gelu(mlp_.c_fc(hidden_states))
                        out = mlp_.down_proj(h)
                        return out + sv_.unsqueeze(0).unsqueeze(0).expand_as(out)
                    return _patched

                stack.enter_context(patched_forward(mlp, _make_patched(mlp, layer_vectors[layer_idx])))

            logits, _ = base_model(generated)

        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output_ids = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


# =============================================================================
# LLM Judge (same as exp_013)
# =============================================================================


def _parse_1_to_10(score_str: str) -> int | None:
    match = re.search(r"\b(10|[1-9])\b", score_str)
    if match:
        return int(match.group(1))
    return None


def judge_both_scores(
    client: OpenAI, model: str, concept_description: str,
    prompt_text: str, text: str,
) -> tuple[float, float]:
    """Score concept relevance and fluency in a single LLM call (1-10 scale)."""
    prompt = (
        f"Rate the following generated text on two dimensions.\n\n"
        f"The text was generated by continuing the sentence starter '{prompt_text}'.\n"
        f"The target concept is: '{concept_description}'\n\n"
        f"Text: \"{text}\"\n\n"
        f"1. CONCEPT (1-10): How strongly does the text reflect the target concept?\n"
        f"   1 = completely unrelated, 5 = somewhat related, 10 = entirely about the concept\n"
        f"2. FLUENCY (1-10): How fluent and coherent is the text?\n"
        f"   1 = complete gibberish, 5 = understandable but flawed, 10 = completely fluent\n\n"
        f"Respond with ONLY two numbers separated by a comma, e.g. '3,8'"
    )
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=10,
                temperature=0.0,
            )
            score_str = response.choices[0].message.content.strip()
            parts = score_str.split(",")
            concept_score = _parse_1_to_10(parts[0]) if len(parts) >= 1 else None
            fluency_score = _parse_1_to_10(parts[1]) if len(parts) >= 2 else None
            return float(concept_score or 1), float(fluency_score or 1)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2 ** attempt + random.random())
            else:
                raise
    return 1.0, 1.0


# =============================================================================
# Plotting
# =============================================================================


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_STYLES = {
    "Transcoder": dict(marker="o", color="#1f77b4", linestyle="-", linewidth=1.8, markersize=8, zorder=5),
    "CLT": dict(marker="X", color="#17becf", linestyle="-", linewidth=1.8, markersize=9, zorder=5),
    "SPD": dict(marker="P", color="#9467bd", linestyle="-", linewidth=1.8, markersize=9, zorder=5),
}


def plot_score_vs_k(
    k_scores: dict[str, dict[int, dict[str, list[float]]]],
    k_values: list[int],
    save_dir: str,
):
    """Plot concept score and fluency vs K (line plot per method).

    k_scores: {method: {k: {"concept": [...], "fluency": [...]}}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for method in ["Transcoder", "CLT", "SPD"]:
        style = METHOD_STYLES[method]
        cs_means = [float(np.mean(k_scores[method][k]["concept"])) for k in k_values]
        fl_means = [float(np.mean(k_scores[method][k]["fluency"])) for k in k_values]

        axes[0].plot(k_values, cs_means, label=method,
                     markeredgecolor="white", markeredgewidth=0.8, **style)
        axes[1].plot(k_values, fl_means, label=method,
                     markeredgecolor="white", markeredgewidth=0.8, **style)

    axes[0].set_xlabel("K (number of features)")
    axes[0].set_ylabel("Concept Score")
    axes[0].set_title("Concept Score vs K")
    axes[0].legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95)
    axes[0].grid(True, alpha=0.15, linewidth=0.5)
    axes[0].set_xticks(k_values)

    axes[1].set_xlabel("K (number of features)")
    axes[1].set_ylabel("Fluency Score")
    axes[1].set_title("Fluency vs K")
    axes[1].legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95)
    axes[1].grid(True, alpha=0.15, linewidth=0.5)
    axes[1].set_xticks(k_values)
    axes[1].set_ylim(-0.5, 10.5)

    fig.tight_layout()
    path = os.path.join(save_dir, "score_vs_k.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Score vs K plot saved to {path}")


def plot_pareto_concept_vs_fluency(
    k_scores: dict[str, dict[int, dict[str, list[float]]]],
    k_values: list[int],
    save_dir: str,
):
    """Plot concept vs fluency Pareto curve (one curve per method, points labeled by K)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in ["Transcoder", "CLT", "SPD"]:
        style = METHOD_STYLES[method]
        pts = []
        for k in k_values:
            mc = float(np.mean(k_scores[method][k]["concept"]))
            mf = float(np.mean(k_scores[method][k]["fluency"]))
            pts.append((mc, mf, k))

        pts_sorted = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        ks = [p[2] for p in pts_sorted]

        ax.plot(xs, ys, label=method, markeredgecolor="white", markeredgewidth=0.8, **style)

        for x, y, kv in zip(xs, ys, ks):
            ax.annotate(
                f"K={kv}", (x, y), textcoords="offset points", xytext=(5, 5),
                fontsize=7, color=style["color"], alpha=0.8,
            )

    ax.set_xlabel("Concept Score")
    ax.set_ylabel("Fluency Score")
    ax.set_title("Concept vs Fluency Trade-off by K", fontsize=12, pad=10)
    ax.set_ylim(-0.5, 10.5)
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    fig.tight_layout()
    path = os.path.join(save_dir, "pareto_concept_vs_fluency.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Pareto plot saved to {path}")


def plot_per_concept_k_comparison(
    per_concept_scores: dict[str, dict[str, dict[int, dict[str, float]]]],
    k_values: list[int],
    save_dir: str,
):
    """Bar chart: best concept score across K values, per concept, per method."""
    concept_ids = list(per_concept_scores.keys())
    methods = ["Transcoder", "CLT", "SPD"]
    n_concepts = len(concept_ids)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(n_concepts)
    width = 0.25

    # Panel 1: Best concept score across K values
    for m_idx, method in enumerate(methods):
        scores = []
        for cid in concept_ids:
            best_cs = max(
                per_concept_scores[cid][method][k]["concept"]
                for k in k_values
                if k in per_concept_scores[cid][method]
            )
            scores.append(best_cs)
        style = METHOD_STYLES[method]
        axes[0].bar(x + m_idx * width, scores, width, label=method, color=style["color"])

    axes[0].set_xlabel("Concept")
    axes[0].set_ylabel("Best Concept Score (across K)")
    axes[0].set_title("Best Concept Score per Concept")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(concept_ids, rotation=45, ha="right")
    axes[0].legend()

    # Panel 2: Best-K AUROC
    for m_idx, method in enumerate(methods):
        aurocs = []
        for cid in concept_ids:
            aurocs.append(per_concept_scores[cid].get(f"{method}_top_auroc", 0.0))
        style = METHOD_STYLES[method]
        axes[1].bar(x + m_idx * width, aurocs, width, label=method, color=style["color"])

    axes[1].set_xlabel("Concept")
    axes[1].set_ylabel("Top-1 AUROC")
    axes[1].set_title("Best Feature AUROC per Concept")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(concept_ids, rotation=45, ha="right")
    axes[1].legend()
    axes[1].set_ylim(0.4, 1.05)

    fig.tight_layout()
    path = os.path.join(save_dir, "per_concept_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Per-concept plot saved to {path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-feature steering evaluation")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--tc_project", type=str, default="mats-sprint/pile_transcoder_sweep3")
    parser.add_argument("--clt_project", type=str, default="mats-sprint/pile_clt")
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument("--n_concepts", type=int, default=10)
    parser.add_argument("--n_prompts", type=int, default=5)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--steering_factor", type=float, default=4.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--save_dir", type=str,
                        default="experiments/exp_016_multi_feature_steering/output")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 2 concepts, 2 prompts, K=[1,3]")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    assert api_key, "Provide OpenAI API key via --api_key or OPENAI_API_KEY env var"

    if args.quick:
        args.n_concepts = 2
        args.n_prompts = 2
        args.k_values = [1, 3]

    cfg = MultiFeatureSteeringConfig(
        tc_project=args.tc_project,
        clt_project=args.clt_project,
        spd_run=args.spd_run,
        n_concepts=args.n_concepts,
        n_prompts=args.n_prompts,
        k_values=args.k_values,
        steering_factor=args.steering_factor,
        max_new_tokens=args.max_new_tokens,
        openai_model=args.openai_model,
        api_key=api_key,
        save_dir=args.save_dir,
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    client = OpenAI(api_key=cfg.api_key)
    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    concepts = CONCEPTS[:cfg.n_concepts]
    prompts = PROMPTS[:cfg.n_prompts]

    start_time = time.time()

    # =========================================================================
    # Load models
    # =========================================================================
    print("=" * 70)
    print("Loading models...")
    print("=" * 70)

    print("Loading SPD model (includes base LlamaSimpleMLP)...")
    spd_model, _raw_config = load_spd_model(cfg.spd_run)
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nDownloading transcoders from {cfg.tc_project}...")
    tc_paths = download_best_transcoders(cfg.tc_project)
    transcoders = {}
    for layer_idx in LAYERS:
        tc = load_transcoder(str(tc_paths[layer_idx]))
        tc.to(DEVICE)
        transcoders[layer_idx] = tc
        print(f"  Layer {layer_idx}: dict_size={tc.cfg.dict_size}, top_k={tc.cfg.top_k}")

    print(f"\nDownloading CLT from {cfg.clt_project}...")
    clt_path = download_best_clt(cfg.clt_project)
    clt = load_clt(str(clt_path))
    clt.to(DEVICE)
    print(f"  CLT: dict_size={clt.cfg.dict_size}, n_layers={clt.cfg.n_layers}, top_k={clt.cfg.top_k}")

    # =========================================================================
    # Step 1: All-layer AUROC feature selection
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: All-Layer AUROC Feature Selection")
    print("=" * 70)

    max_k = max(cfg.k_values)

    # concept_id -> {method: sorted_features_list}
    concept_features: dict[str, dict[str, list]] = {}

    for concept in tqdm(concepts, desc="AUROC feature selection"):
        cid = concept["id"]
        concept_features[cid] = {}

        tc_aurocs = compute_all_transcoder_aurocs(base_model, transcoders, tokenizer, concept, cfg)
        concept_features[cid]["Transcoder"] = tc_aurocs
        top3 = tc_aurocs[:3]
        print(f"  [{cid}] Transcoder: {len(tc_aurocs)} features, "
              f"top3 AUROC: {[f'{a:.3f}' for _, _, a in top3]}")

        clt_aurocs = compute_all_clt_aurocs(base_model, clt, tokenizer, concept, cfg)
        concept_features[cid]["CLT"] = clt_aurocs
        top3 = clt_aurocs[:3]
        print(f"  [{cid}] CLT: {len(clt_aurocs)} features, "
              f"top3 AUROC: {[f'{a:.3f}' for _, _, a in top3]}")

        spd_aurocs = compute_all_spd_aurocs(spd_model, tokenizer, concept, cfg)
        concept_features[cid]["SPD"] = spd_aurocs
        top3 = spd_aurocs[:3]
        print(f"  [{cid}] SPD: {len(spd_aurocs)} components, "
              f"top3 AUROC: {[f'{a:.3f}' for _, _, a in top3]}")

    # =========================================================================
    # Step 2: Generate steered text for each (concept, method, K)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Multi-Feature Steered Generation")
    print("=" * 70)

    # all_generations[concept_id][method][k] = list of {prompt, text}
    all_generations: dict[str, dict[str, dict[int, list[dict]]]] = {
        c["id"]: {
            method: {k: [] for k in cfg.k_values}
            for method in ["Transcoder", "CLT", "SPD"]
        }
        for c in concepts
    }

    total_gens = len(concepts) * 3 * len(cfg.k_values) * cfg.n_prompts
    pbar = tqdm(total=total_gens, desc="Generating steered text")

    for concept in concepts:
        cid = concept["id"]

        for k in cfg.k_values:
            # --- Transcoder ---
            tc_feats = concept_features[cid]["Transcoder"][:k]
            tc_layer_vecs = build_transcoder_layer_vectors(transcoders, tc_feats, cfg.steering_factor)
            for prompt_text in prompts:
                text = generate_steered_text_multi(
                    base_model, tc_layer_vecs, tokenizer, prompt_text,
                    cfg.max_new_tokens, cfg.temperature,
                )
                all_generations[cid]["Transcoder"][k].append({
                    "prompt": prompt_text, "text": text,
                })
                pbar.update(1)

            # --- CLT ---
            clt_feats = concept_features[cid]["CLT"][:k]
            clt_layer_vecs = build_clt_layer_vectors(clt, clt_feats, cfg.steering_factor)
            for prompt_text in prompts:
                text = generate_steered_text_multi(
                    base_model, clt_layer_vecs, tokenizer, prompt_text,
                    cfg.max_new_tokens, cfg.temperature,
                )
                all_generations[cid]["CLT"][k].append({
                    "prompt": prompt_text, "text": text,
                })
                pbar.update(1)

            # --- SPD ---
            spd_feats = concept_features[cid]["SPD"][:k]
            spd_layer_vecs = build_spd_layer_vectors(spd_model, spd_feats, cfg.steering_factor)
            for prompt_text in prompts:
                text = generate_steered_text_multi(
                    base_model, spd_layer_vecs, tokenizer, prompt_text,
                    cfg.max_new_tokens, cfg.temperature,
                )
                all_generations[cid]["SPD"][k].append({
                    "prompt": prompt_text, "text": text,
                })
                pbar.update(1)

    pbar.close()

    # =========================================================================
    # Step 3: LLM Judge Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: LLM Judge Evaluation")
    print("=" * 70)

    # all_scored[concept_id][method][k] = list of {prompt, text, concept_score, fluency_score}
    all_scored: dict[str, dict[str, dict[int, list[dict]]]] = {
        c["id"]: {
            method: {k: [] for k in cfg.k_values}
            for method in ["Transcoder", "CLT", "SPD"]
        }
        for c in concepts
    }

    # Flatten judge jobs
    judge_jobs = []
    for concept in concepts:
        cid = concept["id"]
        description = concept["description"]
        for method in ["Transcoder", "CLT", "SPD"]:
            for k in cfg.k_values:
                for gen in all_generations[cid][method][k]:
                    judge_jobs.append((cid, description, method, k, gen))

    def _judge_one(job):
        cid, description, method, k, gen = job
        cs, fl = judge_both_scores(
            client, cfg.openai_model, description, gen["prompt"], gen["text"],
        )
        return cid, method, k, {
            "prompt": gen["prompt"],
            "text": gen["text"],
            "concept_score": cs,
            "fluency_score": fl,
        }

    pbar = tqdm(total=len(judge_jobs), desc="LLM judging")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_judge_one, job) for job in judge_jobs]
        for future in as_completed(futures):
            cid, method, k, entry = future.result()
            all_scored[cid][method][k].append(entry)
            pbar.update(1)
    pbar.close()

    # =========================================================================
    # Step 4: Aggregate metrics
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Metrics & Analysis")
    print("=" * 70)

    # k_scores[method][k] = {"concept": [...], "fluency": [...]} aggregated across all concepts
    k_scores: dict[str, dict[int, dict[str, list[float]]]] = {
        method: {k: {"concept": [], "fluency": []} for k in cfg.k_values}
        for method in ["Transcoder", "CLT", "SPD"]
    }

    # per_concept_scores[cid][method][k] = {"concept": mean, "fluency": mean}
    # per_concept_scores[cid][f"{method}_top_auroc"] = float
    per_concept_scores: dict[str, dict] = {}

    for concept in concepts:
        cid = concept["id"]
        per_concept_scores[cid] = {}
        for method in ["Transcoder", "CLT", "SPD"]:
            per_concept_scores[cid][method] = {}
            for k in cfg.k_values:
                entries = all_scored[cid][method][k]
                cs_list = [e["concept_score"] for e in entries]
                fl_list = [e["fluency_score"] for e in entries]
                k_scores[method][k]["concept"].extend(cs_list)
                k_scores[method][k]["fluency"].extend(fl_list)
                per_concept_scores[cid][method][k] = {
                    "concept": float(np.mean(cs_list)) if cs_list else 0.0,
                    "fluency": float(np.mean(fl_list)) if fl_list else 0.0,
                }
            # Top-1 AUROC for this concept
            top_auroc = concept_features[cid][method][0][2] if concept_features[cid][method] else 0.0
            per_concept_scores[cid][f"{method}_top_auroc"] = top_auroc

    # Print summary table
    print(f"\n{'Method':<15} {'K':>4} {'Concept':>10} {'Fluency':>10}")
    print("-" * 45)
    for method in ["Transcoder", "CLT", "SPD"]:
        for k in cfg.k_values:
            mc = float(np.mean(k_scores[method][k]["concept"]))
            mf = float(np.mean(k_scores[method][k]["fluency"]))
            print(f"{method:<15} {k:>4} {mc:>10.3f} {mf:>10.3f}")
        print()

    # Optimal K: best concept score with fluency >= 5
    print("Optimal K per method (best concept score with fluency >= 5):")
    for method in ["Transcoder", "CLT", "SPD"]:
        best_k = None
        best_cs = -1.0
        for k in cfg.k_values:
            mc = float(np.mean(k_scores[method][k]["concept"]))
            mf = float(np.mean(k_scores[method][k]["fluency"]))
            if mf >= 5.0 and mc > best_cs:
                best_cs = mc
                best_k = k
        if best_k is not None:
            mf = float(np.mean(k_scores[method][best_k]["fluency"]))
            print(f"  {method}: K={best_k} (concept={best_cs:.3f}, fluency={mf:.3f})")
        else:
            print(f"  {method}: no K achieves fluency >= 5")

    # =========================================================================
    # Step 5: Plots
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Generating Plots")
    print("=" * 70)

    plot_score_vs_k(k_scores, cfg.k_values, cfg.save_dir)
    plot_pareto_concept_vs_fluency(k_scores, cfg.k_values, cfg.save_dir)
    plot_per_concept_k_comparison(per_concept_scores, cfg.k_values, cfg.save_dir)

    # =========================================================================
    # Save results
    # =========================================================================
    elapsed = time.time() - start_time

    # Serialize feature selections for JSON (truncate to max_k per concept)
    feature_selections_json = {}
    for cid in [c["id"] for c in concepts]:
        feature_selections_json[cid] = {}
        for method in ["Transcoder", "CLT", "SPD"]:
            feats = concept_features[cid][method][:max_k]
            feature_selections_json[cid][method] = [
                {"feature_id": list(f[:2]), "auroc": f[2]} for f in feats
            ]

    # Serialize scored generations
    scored_json = {}
    for cid in [c["id"] for c in concepts]:
        scored_json[cid] = {}
        for method in ["Transcoder", "CLT", "SPD"]:
            scored_json[cid][method] = {}
            for k in cfg.k_values:
                scored_json[cid][method][str(k)] = all_scored[cid][method][k]

    # Aggregate summary
    summary_per_method = {}
    for method in ["Transcoder", "CLT", "SPD"]:
        method_summary = {}
        for k in cfg.k_values:
            mc = float(np.mean(k_scores[method][k]["concept"]))
            mf = float(np.mean(k_scores[method][k]["fluency"]))
            method_summary[str(k)] = {"mean_concept": mc, "mean_fluency": mf}
        # Optimal K
        best_k = None
        best_cs = -1.0
        for k in cfg.k_values:
            mc = float(np.mean(k_scores[method][k]["concept"]))
            mf = float(np.mean(k_scores[method][k]["fluency"]))
            if mf >= 5.0 and mc > best_cs:
                best_cs = mc
                best_k = k
        method_summary["optimal_k_fluency_5"] = best_k
        summary_per_method[method] = method_summary

    results_data = {
        "config": {
            "n_concepts": cfg.n_concepts,
            "n_prompts": cfg.n_prompts,
            "k_values": cfg.k_values,
            "steering_factor": cfg.steering_factor,
            "max_new_tokens": cfg.max_new_tokens,
            "openai_model": cfg.openai_model,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
        },
        "feature_selections": feature_selections_json,
        "scored_generations": scored_json,
        "summary": summary_per_method,
        "per_concept_scores": {
            cid: {
                method: {
                    str(k): per_concept_scores[cid][method][k]
                    for k in cfg.k_values
                }
                for method in ["Transcoder", "CLT", "SPD"]
            }
            for cid in [c["id"] for c in concepts]
        },
    }

    results_path = os.path.join(cfg.save_dir, "multi_feature_steering_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Concepts evaluated: {cfg.n_concepts}")
    print(f"Prompts per concept: {cfg.n_prompts}")
    print(f"K values: {cfg.k_values}")
    print(f"Steering factor: {cfg.steering_factor}")
    print(f"Total time: {elapsed:.1f}s")
    print()

    # Per-concept top-1 AUROC table
    print(f"{'Concept':<12} {'TC AUROC':>10} {'CLT AUROC':>10} {'SPD AUROC':>10}")
    print("-" * 45)
    for concept in concepts:
        cid = concept["id"]
        tc_a = per_concept_scores[cid].get("Transcoder_top_auroc", 0.0)
        clt_a = per_concept_scores[cid].get("CLT_top_auroc", 0.0)
        spd_a = per_concept_scores[cid].get("SPD_top_auroc", 0.0)
        print(f"{cid:<12} {tc_a:>10.3f} {clt_a:>10.3f} {spd_a:>10.3f}")

    # Per-K mean scores
    print()
    print(f"{'K':<6}", end="")
    for method in ["Transcoder", "CLT", "SPD"]:
        print(f" {method + ' CS':>12} {method + ' FL':>12}", end="")
    print()
    print("-" * 80)
    for k in cfg.k_values:
        print(f"{k:<6}", end="")
        for method in ["Transcoder", "CLT", "SPD"]:
            mc = float(np.mean(k_scores[method][k]["concept"]))
            mf = float(np.mean(k_scores[method][k]["fluency"]))
            print(f" {mc:>12.3f} {mf:>12.3f}", end="")
        print()


if __name__ == "__main__":
    main()
