"""Steering specificity evaluation: cross-concept contamination analysis.

Measures how precisely each decomposition method (Transcoder, CLT, SPD) steers
toward a target concept without leaking into unrelated concepts.

For each method:
1. Select best feature per concept via AUROC (same as exp_013)
2. Steer toward each concept at a fixed steering factor
3. Score each generated text against ALL 10 concepts via a multi-concept LLM judge
4. Build a 10x10 specificity matrix: rows = target, columns = scored concept

Key metrics:
- On-target score: mean of diagonal (how well steering hits the target)
- Off-target score: mean of off-diagonal (how much it bleeds into other concepts)
- Specificity ratio: on-target / off-target (higher = more precise steering)

Usage:
    python experiments/exp_015_steering_specificity/steering_specificity.py \\
        --api_key sk-...

    # Quick test with 3 concepts, 2 prompts:
    python experiments/exp_015_steering_specificity/steering_specificity.py \\
        --api_key sk-... --quick
"""

import argparse
import json
import os
import random
import re
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from clt import CrossLayerTranscoder
from config import CLTConfig, EncoderConfig
from transcoder import BatchTopKTranscoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [0, 1, 2, 3]

# Same 10 concepts as exp_013.
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


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SpecificityConfig:
    # Models
    tc_project: str = "mats-sprint/pile_transcoder_sweep3"
    clt_project: str = "mats-sprint/pile_clt"
    spd_run: str = "goodfire/spd/s-275c8f21"

    # Experiment scope
    n_concepts: int = 10
    n_prompts: int = 5
    steering_factor: float = 4.0
    eval_layer: int = 2

    # AUROC feature selection
    n_auroc_examples: int = 1000
    auroc_seq_len: int = 128
    auroc_batch_size: int = 32

    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.7

    # LLM judge
    openai_model: str = "gpt-4o-mini"
    api_key: str = ""

    # Output
    save_dir: str = "experiments/exp_015_steering_specificity/output"


# =============================================================================
# Model loading (same as exp_013)
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


def download_best_transcoder(project: str, layer: int) -> Path:
    """Download the transcoder with the highest top_k for a given layer."""
    import wandb
    api = wandb.Api()
    runs = api.runs(project)
    best_run = None
    best_topk = -1
    for run in runs:
        if run.state != "finished":
            continue
        cfg = run.config
        top_k = cfg.get("top_k", 0)
        run_name = run.name
        run_layer = None
        for part in run_name.split("_"):
            if part.startswith("L") and part[1:].isdigit():
                run_layer = int(part[1:])
                break
        if run_layer == layer and top_k > best_topk:
            best_topk = top_k
            best_run = run
    assert best_run is not None, f"No finished transcoder run found for layer {layer}"
    arts = [a for a in best_run.logged_artifacts() if a.type == "model"]
    assert len(arts) == 1
    dest = Path(f"checkpoints/steering_tc_L{layer}_{best_run.name}")
    download_wandb_artifact(project, arts[0].name, dest)
    return dest


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
    dest = Path(f"checkpoints/steering_clt_{best_run.name}")
    download_wandb_artifact(project, arts[0].name, dest)
    return dest


# =============================================================================
# AUROC computation (no sklearn dependency)
# =============================================================================


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC score for binary labels and continuous scores.

    Uses the trapezoidal rule on the ROC curve built from sorted thresholds.
    """
    assert labels.shape == scores.shape
    n = len(labels)
    n_pos = labels.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Only one class present")

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for i in range(n):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += 0.5 * (fpr - prev_fpr) * (tpr + prev_tpr)
        prev_fpr = fpr
        prev_tpr = tpr

    return auc


# =============================================================================
# Feature selection via AUROC (same as exp_013)
# =============================================================================


def make_concept_texts(concept: dict, n_per_class: int) -> tuple[list[str], list[str]]:
    """Generate positive and negative text samples for a concept."""
    templates = [
        "This text is about {kw}. It discusses various aspects of {kw} in detail.",
        "The topic of {kw} is explored here, covering recent developments.",
        "An in-depth look at {kw} and its impact on modern society.",
        "{kw} has been a subject of great interest lately.",
        "Recent news about {kw} suggests significant changes ahead.",
        "Understanding {kw} requires knowledge of the broader context.",
        "The field of {kw} has evolved significantly over the years.",
        "Experts in {kw} have noted several important trends.",
    ]

    pos_keywords = concept["positive_keywords"]
    neg_keywords = concept["negative_keywords"]

    positives = []
    for i in range(n_per_class):
        kw = pos_keywords[i % len(pos_keywords)]
        template = templates[i % len(templates)]
        positives.append(template.format(kw=kw))

    negatives = []
    for i in range(n_per_class):
        kw = neg_keywords[i % len(neg_keywords)]
        template = templates[i % len(templates)]
        negatives.append(template.format(kw=kw))

    return positives, negatives


@torch.no_grad()
def get_mlp_input_acts(base_model, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    """Get MLP input (post-RMSNorm) activations at a given layer. Returns (B, S, d_model)."""
    captured = {}

    def _hook(_mod, _inp, out):
        captured["act"] = out.detach()

    hook = base_model.h[layer].rms_2.register_forward_hook(_hook)
    base_model(input_ids)
    hook.remove()
    return captured["act"]


@torch.no_grad()
def compute_transcoder_auroc(
    base_model, transcoder, tokenizer, concept: dict, cfg: SpecificityConfig,
) -> tuple[int, float]:
    """Compute AUROC for each transcoder latent on a concept. Returns (best_latent_idx, best_auroc)."""
    positives, negatives = make_concept_texts(concept, cfg.n_auroc_examples)

    all_texts = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)

    dict_size = transcoder.cfg.dict_size
    all_scores = torch.zeros(len(all_texts), dict_size)

    for i in range(0, len(all_texts), cfg.auroc_batch_size):
        batch_texts = all_texts[i : i + cfg.auroc_batch_size]
        encoded = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=cfg.auroc_seq_len,
        )
        input_ids = encoded["input_ids"].to(DEVICE)

        mlp_input = get_mlp_input_acts(base_model, input_ids, cfg.eval_layer)
        flat = mlp_input.reshape(-1, transcoder.cfg.input_size)
        use_pre_enc_bias = transcoder.cfg.pre_enc_bias and transcoder.input_size == transcoder.output_size
        x_enc = flat - transcoder.b_dec if use_pre_enc_bias else flat
        acts = F.relu(x_enc @ transcoder.W_enc)
        acts = acts.reshape(mlp_input.shape[0], mlp_input.shape[1], -1)

        max_acts = acts.max(dim=1).values
        all_scores[i : i + len(batch_texts)] = max_acts.cpu()

    labels_arr = np.array(labels)
    scores_np = all_scores.numpy()

    best_latent = -1
    best_auroc = 0.0
    for j in range(dict_size):
        col = scores_np[:, j]
        if col.max() == col.min():
            continue
        try:
            auc = roc_auc_score(labels_arr, col)
        except ValueError:
            continue
        if auc > best_auroc:
            best_auroc = auc
            best_latent = j

    return best_latent, best_auroc


@torch.no_grad()
def compute_clt_auroc(
    base_model, clt: CrossLayerTranscoder, tokenizer, concept: dict, cfg: SpecificityConfig,
) -> tuple[int, int, float]:
    """Compute AUROC for each CLT encoder latent. Returns (best_layer_idx, best_latent_idx, best_auroc)."""
    positives, negatives = make_concept_texts(concept, cfg.n_auroc_examples)
    all_texts = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)

    dict_size = clt.cfg.dict_size
    n_layers = clt.cfg.n_layers

    all_scores = {i: torch.zeros(len(all_texts), dict_size) for i in range(n_layers)}

    for i in range(0, len(all_texts), cfg.auroc_batch_size):
        batch_texts = all_texts[i : i + cfg.auroc_batch_size]
        encoded = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=cfg.auroc_seq_len,
        )
        input_ids = encoded["input_ids"].to(DEVICE)

        rms2_outputs = {}
        hooks = []
        for layer_idx in range(n_layers):
            actual_layer = clt.cfg.layers[layer_idx]

            def _make_hook(li):
                def _hook(_mod, _inp, out):
                    rms2_outputs[li] = out.detach()
                return _hook

            hooks.append(base_model.h[actual_layer].rms_2.register_forward_hook(_make_hook(layer_idx)))
        base_model(input_ids)
        for h in hooks:
            h.remove()

        for layer_idx in range(n_layers):
            mlp_input = rms2_outputs[layer_idx]
            flat = mlp_input.reshape(-1, clt.cfg.input_size)
            pre_acts = F.relu(flat @ clt.W_enc[layer_idx] + clt.b_enc[layer_idx])
            pre_acts = pre_acts.reshape(mlp_input.shape[0], mlp_input.shape[1], -1)
            max_acts = pre_acts.max(dim=1).values
            all_scores[layer_idx][i : i + len(batch_texts)] = max_acts.cpu()

    labels_arr = np.array(labels)
    best_layer = -1
    best_latent = -1
    best_auroc = 0.0

    for layer_idx in range(n_layers):
        scores_np = all_scores[layer_idx].numpy()
        for j in range(dict_size):
            col = scores_np[:, j]
            if col.max() == col.min():
                continue
            try:
                auc = roc_auc_score(labels_arr, col)
            except ValueError:
                continue
            if auc > best_auroc:
                best_auroc = auc
                best_layer = layer_idx
                best_latent = j

    return best_layer, best_latent, best_auroc


@torch.no_grad()
def compute_spd_auroc(
    spd_model, tokenizer, concept: dict, cfg: SpecificityConfig,
) -> tuple[str, int, float]:
    """Compute AUROC for each SPD component using CI scores.

    Uses the norm of the gated component output as the activation score.
    Returns (best_module_name, best_component_idx, best_auroc).
    """
    positives, negatives = make_concept_texts(concept, cfg.n_auroc_examples)
    all_texts = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)

    layer = cfg.eval_layer
    module_names = [f"h.{layer}.mlp.down_proj"]

    all_scores = {}
    for mod_name in module_names:
        n_comp = spd_model.module_to_c[mod_name]
        all_scores[mod_name] = torch.zeros(len(all_texts), n_comp)

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
            all_scores[mod_name][i : i + len(batch_texts)] = max_scores.cpu()

    labels_arr = np.array(labels)
    best_module = ""
    best_comp = -1
    best_auroc = 0.0

    for mod_name in module_names:
        scores_np = all_scores[mod_name].numpy()
        n_comp = scores_np.shape[1]
        for j in range(n_comp):
            col = scores_np[:, j]
            if col.max() == col.min():
                continue
            try:
                auc = roc_auc_score(labels_arr, col)
            except ValueError:
                continue
            if auc > best_auroc:
                best_auroc = auc
                best_module = mod_name
                best_comp = j

    return best_module, best_comp, best_auroc


# =============================================================================
# Steering generation (same as exp_013)
# =============================================================================


def generate_steered_text_transcoder(
    base_model, transcoder, tokenizer, prompt_text: str,
    latent_idx: int, steering_factor: float,
    layer: int, max_new_tokens: int, temperature: float,
) -> str:
    """Generate text with transcoder steering: add factor * unit_dir to MLP output."""
    decoder_dir = transcoder.W_dec[latent_idx]
    steering_vector = steering_factor * F.normalize(decoder_dir, dim=0)
    mlp = base_model.h[layer].mlp

    def _steered_forward(hidden_states):
        h = mlp.gelu(mlp.c_fc(hidden_states))
        out = mlp.down_proj(h)
        return out + steering_vector.unsqueeze(0).unsqueeze(0).expand_as(out)

    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(DEVICE)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with patched_forward(mlp, _steered_forward):
            logits, _ = base_model(generated)

        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output_ids = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def generate_steered_text_clt(
    base_model, clt: CrossLayerTranscoder, tokenizer, prompt_text: str,
    source_layer_idx: int, latent_idx: int, steering_factor: float,
    max_new_tokens: int, temperature: float,
) -> str:
    """Generate text with CLT steering.

    The selected feature at source layer i writes to layers i through n-1.
    We add factor * unit_dir at each target layer.
    """
    n_target_layers = clt.W_dec[source_layer_idx].shape[0]
    steering_vectors = {}
    for offset in range(n_target_layers):
        target_layer_idx = source_layer_idx + offset
        actual_target_layer = clt.cfg.layers[target_layer_idx]
        d_j = clt.W_dec[source_layer_idx][offset, latent_idx]
        steering_vectors[actual_target_layer] = steering_factor * F.normalize(d_j, dim=0)

    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(DEVICE)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with ExitStack() as stack:
            for target_layer, sv in steering_vectors.items():
                mlp = base_model.h[target_layer].mlp

                def _make_steered(mlp_, sv_):
                    def _steered(hidden_states):
                        h = mlp_.gelu(mlp_.c_fc(hidden_states))
                        out = mlp_.down_proj(h)
                        return out + sv_.unsqueeze(0).unsqueeze(0).expand_as(out)
                    return _steered

                stack.enter_context(patched_forward(mlp, _make_steered(mlp, sv)))

            logits, _ = base_model(generated)

        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output_ids = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


@torch.no_grad()
def get_spd_steering_direction(
    spd_model, module_name: str, component_idx: int,
) -> torch.Tensor:
    """Get the unit-norm steering direction for an SPD down_proj component from its U vector."""
    components = spd_model.components[module_name]
    u_vec = components.U[component_idx]
    return F.normalize(u_vec, dim=0)


def generate_steered_text_spd(
    base_model, steering_direction: torch.Tensor, tokenizer, prompt_text: str,
    steering_factor: float,
    layer: int, max_new_tokens: int, temperature: float,
) -> str:
    """Generate text with SPD steering. steering_direction is already unit-normalized."""
    steering_vector = steering_factor * steering_direction
    mlp = base_model.h[layer].mlp

    def _make_steered(mlp_, sv_):
        def _steered(hidden_states):
            h = mlp_.gelu(mlp_.c_fc(hidden_states))
            out = mlp_.down_proj(h)
            return out + sv_.unsqueeze(0).unsqueeze(0).expand_as(out)
        return _steered

    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(DEVICE)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with patched_forward(mlp, _make_steered(mlp, steering_vector)):
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
# Multi-concept LLM Judge
# =============================================================================


def _parse_1_to_10(score_str: str) -> int | None:
    """Extract an integer 1-10 from a string."""
    match = re.search(r"\b(10|[1-9])\b", score_str)
    if match:
        return int(match.group(1))
    return None


def judge_all_concepts(
    client: OpenAI, model: str, concept_ids: list[str],
    concept_descriptions: list[str], prompt_text: str, text: str,
) -> list[float]:
    """Score text against ALL concepts in a single LLM call.

    Returns a list of scores (1-10), one per concept, in the same order as concept_ids.
    """
    n = len(concept_ids)
    concept_lines = "\n".join(
        f"{i + 1}. {cid} - {desc}"
        for i, (cid, desc) in enumerate(zip(concept_ids, concept_descriptions))
    )

    prompt = (
        f"Rate how strongly the following text relates to each of the listed concepts.\n\n"
        f"The text was generated by continuing the sentence starter '{prompt_text}'.\n\n"
        f"Text: \"{text}\"\n\n"
        f"Concepts:\n{concept_lines}\n\n"
        f"For each concept, rate from 1 to 10:\n"
        f"  1 = completely unrelated\n"
        f"  5 = somewhat related\n"
        f"  10 = entirely about this concept\n\n"
        f"Respond with ONLY {n} comma-separated integers, one per concept in order. "
        f"Example: 3,1,7,2,1,1,4,1,2,1"
    )

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=60,
                temperature=0.0,
            )
            score_str = response.choices[0].message.content.strip()
            parts = score_str.split(",")
            scores = []
            for p in parts:
                s = _parse_1_to_10(p.strip())
                scores.append(float(s) if s is not None else 1.0)
            # Pad or truncate to exactly n
            while len(scores) < n:
                scores.append(1.0)
            scores = scores[:n]
            return scores
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2 ** attempt + random.random())
            else:
                raise
    return [1.0] * n


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

METHOD_COLORS = {
    "Transcoder": "#1f77b4",
    "CLT": "#17becf",
    "SPD": "#9467bd",
}


def plot_specificity_heatmaps(
    matrices: dict[str, np.ndarray],
    concept_ids: list[str],
    save_path: str,
):
    """Plot 3 heatmaps side-by-side showing the specificity matrix per method."""
    methods = ["Transcoder", "CLT", "SPD"]
    n = len(concept_ids)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Shared color scale across all three
    all_vals = np.concatenate([matrices[m].flatten() for m in methods])
    vmin = max(1.0, all_vals.min())
    vmax = min(10.0, all_vals.max())

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        mat = matrices[method]
        im = ax.imshow(mat, cmap="YlOrRd", aspect="equal", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(n))
        ax.set_xticklabels(concept_ids, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(concept_ids, fontsize=8)

        if ax_idx == 0:
            ax.set_ylabel("Target concept (steered toward)")
        ax.set_xlabel("Scored concept")
        ax.set_title(method, fontsize=13, fontweight="bold",
                     color=METHOD_COLORS[method])

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                text_color = "white" if val > (vmin + vmax) / 2 else "black"
                fontweight = "bold" if i == j else "normal"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight=fontweight)

        # Highlight diagonal
        for i in range(n):
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, linewidth=2,
                                 edgecolor="black", facecolor="none")
            ax.add_patch(rect)

        # Re-enable spines for heatmap
        for spine in ax.spines.values():
            spine.set_visible(True)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Concept score (1-10)")
    fig.suptitle("Steering Specificity: Cross-Concept Contamination",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap plot saved to {save_path}")


def plot_specificity_bars(
    metrics: dict[str, dict[str, float]],
    save_path: str,
):
    """Bar chart comparing specificity ratio across methods."""
    methods = ["Transcoder", "CLT", "SPD"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: Specificity ratio
    ratios = [metrics[m]["specificity_ratio"] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    bars = axes[0].bar(methods, ratios, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_ylabel("Specificity Ratio")
    axes[0].set_title("Specificity Ratio (higher = more precise)")
    for bar, val in zip(bars, ratios):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, max(ratios) * 1.25 if max(ratios) > 0 else 2)

    # Panel 2: On-target vs off-target
    x = np.arange(len(methods))
    width = 0.35
    on_target = [metrics[m]["on_target"] for m in methods]
    off_target = [metrics[m]["off_target"] for m in methods]
    axes[1].bar(x - width / 2, on_target, width, label="On-target",
                color=[METHOD_COLORS[m] for m in methods], edgecolor="white")
    axes[1].bar(x + width / 2, off_target, width, label="Off-target",
                color=[METHOD_COLORS[m] for m in methods], alpha=0.4, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("Mean Score (1-10)")
    axes[1].set_title("On-target vs Off-target Scores")
    axes[1].legend()

    # Panel 3: AUROC
    aurocs = [metrics[m]["mean_auroc"] for m in methods]
    bars3 = axes[2].bar(methods, aurocs, color=colors, edgecolor="white", linewidth=0.8)
    axes[2].set_ylabel("Mean AUROC")
    axes[2].set_title("Feature Selection Quality")
    axes[2].set_ylim(0.4, 1.05)
    for bar, val in zip(bars3, aurocs):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle(f"Steering Specificity Comparison (factor=4.0)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Steering specificity evaluation")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--tc_project", type=str, default="mats-sprint/pile_transcoder_sweep3")
    parser.add_argument("--clt_project", type=str, default="mats-sprint/pile_clt")
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument("--n_concepts", type=int, default=10)
    parser.add_argument("--n_prompts", type=int, default=5)
    parser.add_argument("--steering_factor", type=float, default=4.0)
    parser.add_argument("--eval_layer", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--save_dir", type=str,
                        default="experiments/exp_015_steering_specificity/output")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 concepts, 2 prompts")
    args = parser.parse_args()

    if args.quick:
        args.n_concepts = 3
        args.n_prompts = 2

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    assert api_key, "Provide OpenAI API key via --api_key or OPENAI_API_KEY env var"

    cfg = SpecificityConfig(
        tc_project=args.tc_project,
        clt_project=args.clt_project,
        spd_run=args.spd_run,
        n_concepts=args.n_concepts,
        n_prompts=args.n_prompts,
        steering_factor=args.steering_factor,
        eval_layer=args.eval_layer,
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
    concept_ids = [c["id"] for c in concepts]
    concept_descriptions = [c["description"] for c in concepts]
    prompts = PROMPTS[:cfg.n_prompts]
    n_concepts = len(concepts)

    start_time = time.time()

    # =========================================================================
    # Load models
    # =========================================================================
    print("=" * 70)
    print("Loading models...")
    print("=" * 70)

    print("Loading SPD model (includes base LlamaSimpleMLP)...")
    spd_model, raw_config = load_spd_model(cfg.spd_run)
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Downloading transcoder for layer {cfg.eval_layer}...")
    tc_path = download_best_transcoder(cfg.tc_project, cfg.eval_layer)
    transcoder = load_transcoder(str(tc_path))
    transcoder.to(DEVICE)
    print(f"  Loaded transcoder: dict_size={transcoder.cfg.dict_size}, top_k={transcoder.cfg.top_k}")

    print("Downloading CLT...")
    clt_path = download_best_clt(cfg.clt_project)
    clt = load_clt(str(clt_path))
    clt.to(DEVICE)
    print(f"  Loaded CLT: dict_size={clt.cfg.dict_size}, n_layers={clt.cfg.n_layers}, top_k={clt.cfg.top_k}")

    # =========================================================================
    # Step 1: Feature selection via AUROC
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Feature Selection via AUROC")
    print("=" * 70)

    feature_selections = {}

    for concept in tqdm(concepts, desc="AUROC feature selection"):
        concept_id = concept["id"]
        feature_selections[concept_id] = {}

        tc_latent, tc_auroc = compute_transcoder_auroc(
            base_model, transcoder, tokenizer, concept, cfg,
        )
        feature_selections[concept_id]["Transcoder"] = {
            "latent_idx": tc_latent,
            "auroc": tc_auroc,
        }
        print(f"  [{concept_id}] Transcoder: latent={tc_latent}, AUROC={tc_auroc:.3f}")

        clt_layer, clt_latent, clt_auroc = compute_clt_auroc(
            base_model, clt, tokenizer, concept, cfg,
        )
        feature_selections[concept_id]["CLT"] = {
            "source_layer_idx": clt_layer,
            "latent_idx": clt_latent,
            "auroc": clt_auroc,
        }
        actual_layer = clt.cfg.layers[clt_layer] if clt_layer >= 0 else -1
        print(f"  [{concept_id}] CLT: layer={clt_layer}(actual={actual_layer}), latent={clt_latent}, AUROC={clt_auroc:.3f}")

        spd_module, spd_comp, spd_auroc = compute_spd_auroc(
            spd_model, tokenizer, concept, cfg,
        )
        feature_selections[concept_id]["SPD"] = {
            "module_name": spd_module,
            "component_idx": spd_comp,
            "auroc": spd_auroc,
        }
        print(f"  [{concept_id}] SPD: module={spd_module}, comp={spd_comp}, AUROC={spd_auroc:.3f}")

    # =========================================================================
    # Precompute SPD steering directions
    # =========================================================================
    print("\n" + "=" * 70)
    print("Precomputing SPD steering directions...")
    print("=" * 70)

    spd_directions = {}
    for concept in concepts:
        concept_id = concept["id"]
        spd_sel = feature_selections[concept_id]["SPD"]
        spd_directions[concept_id] = get_spd_steering_direction(
            spd_model, spd_sel["module_name"], spd_sel["component_idx"],
        )
        print(f"  [{concept_id}] SPD direction from U[{spd_sel['component_idx']}] ({spd_sel['module_name']})")

    # =========================================================================
    # Step 2: Generate steered text (single factor)
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"Step 2: Generating Steered Text (factor={cfg.steering_factor})")
    print("=" * 70)

    # all_generations[concept_id][method] = list of {prompt, text}
    all_generations = {c["id"]: {"Transcoder": [], "CLT": [], "SPD": []} for c in concepts}

    total_gens = n_concepts * 3 * cfg.n_prompts
    pbar = tqdm(total=total_gens, desc="Generating steered text")

    for concept in concepts:
        concept_id = concept["id"]

        for prompt_text in prompts:
            # Transcoder
            tc_sel = feature_selections[concept_id]["Transcoder"]
            tc_text = generate_steered_text_transcoder(
                base_model, transcoder, tokenizer, prompt_text,
                tc_sel["latent_idx"], cfg.steering_factor,
                cfg.eval_layer, cfg.max_new_tokens, cfg.temperature,
            )
            all_generations[concept_id]["Transcoder"].append({
                "prompt": prompt_text,
                "text": tc_text,
            })
            pbar.update(1)

            # CLT
            clt_sel = feature_selections[concept_id]["CLT"]
            clt_text = generate_steered_text_clt(
                base_model, clt, tokenizer, prompt_text,
                clt_sel["source_layer_idx"], clt_sel["latent_idx"],
                cfg.steering_factor,
                cfg.max_new_tokens, cfg.temperature,
            )
            all_generations[concept_id]["CLT"].append({
                "prompt": prompt_text,
                "text": clt_text,
            })
            pbar.update(1)

            # SPD
            spd_text = generate_steered_text_spd(
                base_model, spd_directions[concept_id], tokenizer, prompt_text,
                cfg.steering_factor,
                cfg.eval_layer, cfg.max_new_tokens, cfg.temperature,
            )
            all_generations[concept_id]["SPD"].append({
                "prompt": prompt_text,
                "text": spd_text,
            })
            pbar.update(1)

    pbar.close()

    # =========================================================================
    # Step 3: Multi-concept LLM Judge
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Multi-Concept LLM Judge Evaluation")
    print("=" * 70)

    # all_scores[concept_id][method] = list of {prompt, text, scores: list[float]}
    # where scores[j] = how much the text relates to concept j
    all_scored = {c["id"]: {"Transcoder": [], "CLT": [], "SPD": []} for c in concepts}

    # Flatten all judge jobs
    judge_jobs = []
    for concept in concepts:
        concept_id = concept["id"]
        for method in ["Transcoder", "CLT", "SPD"]:
            for gen in all_generations[concept_id][method]:
                judge_jobs.append((concept_id, method, gen))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _judge_one(job):
        target_concept_id, method, gen = job
        scores = judge_all_concepts(
            client, cfg.openai_model, concept_ids, concept_descriptions,
            gen["prompt"], gen["text"],
        )
        return target_concept_id, method, {
            "prompt": gen["prompt"],
            "text": gen["text"],
            "scores": scores,
        }

    pbar = tqdm(total=len(judge_jobs), desc="LLM judging (multi-concept)")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_judge_one, job) for job in judge_jobs]
        for future in as_completed(futures):
            target_concept_id, method, entry = future.result()
            all_scored[target_concept_id][method].append(entry)
            pbar.update(1)
    pbar.close()

    # =========================================================================
    # Step 4: Build specificity matrices
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Computing Specificity Matrices")
    print("=" * 70)

    # matrices[method] is an (n_concepts x n_concepts) array.
    # Row i = steering toward concept i, column j = mean score on concept j.
    matrices = {}
    for method in ["Transcoder", "CLT", "SPD"]:
        mat = np.zeros((n_concepts, n_concepts))
        for i, target_concept in enumerate(concepts):
            target_id = target_concept["id"]
            entries = all_scored[target_id][method]
            if not entries:
                continue
            # Average scores across all prompts for this target concept
            all_score_vecs = np.array([e["scores"] for e in entries])  # (n_prompts, n_concepts)
            mat[i, :] = all_score_vecs.mean(axis=0)
        matrices[method] = mat

    # Compute summary metrics per method
    method_metrics = {}
    for method in ["Transcoder", "CLT", "SPD"]:
        mat = matrices[method]
        diag = np.diag(mat)
        on_target = float(diag.mean())

        # Off-diagonal mean
        mask = ~np.eye(n_concepts, dtype=bool)
        off_target = float(mat[mask].mean())

        specificity_ratio = on_target / off_target if off_target > 0 else float("inf")

        aurocs = [feature_selections[c["id"]][method]["auroc"] for c in concepts]
        mean_auroc = float(np.mean(aurocs))

        method_metrics[method] = {
            "on_target": on_target,
            "off_target": off_target,
            "specificity_ratio": specificity_ratio,
            "mean_auroc": mean_auroc,
        }

    # Print summary
    print(f"\n{'Method':<15} {'On-target':>10} {'Off-target':>11} {'Ratio':>8} {'AUROC':>8}")
    print("-" * 55)
    for method in ["Transcoder", "CLT", "SPD"]:
        m = method_metrics[method]
        print(f"{method:<15} {m['on_target']:>10.2f} {m['off_target']:>11.2f} "
              f"{m['specificity_ratio']:>8.2f} {m['mean_auroc']:>8.3f}")

    # Print per-concept diagonal scores
    print(f"\nPer-concept on-target scores (diagonal):")
    print(f"{'Concept':<12}", end="")
    for method in ["Transcoder", "CLT", "SPD"]:
        print(f" {method:>12}", end="")
    print()
    print("-" * 50)
    for i, concept in enumerate(concepts):
        print(f"{concept['id']:<12}", end="")
        for method in ["Transcoder", "CLT", "SPD"]:
            print(f" {matrices[method][i, i]:>12.2f}", end="")
        print()

    # =========================================================================
    # Plotting
    # =========================================================================
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    plot_specificity_heatmaps(
        matrices, concept_ids,
        save_path=os.path.join(cfg.save_dir, "specificity_heatmaps.png"),
    )

    plot_specificity_bars(
        method_metrics,
        save_path=os.path.join(cfg.save_dir, "specificity_bars.png"),
    )

    # =========================================================================
    # Save results
    # =========================================================================
    elapsed = time.time() - start_time

    results_data = {
        "config": {
            "n_concepts": cfg.n_concepts,
            "n_prompts": cfg.n_prompts,
            "steering_factor": cfg.steering_factor,
            "eval_layer": cfg.eval_layer,
            "max_new_tokens": cfg.max_new_tokens,
            "openai_model": cfg.openai_model,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
        },
        "concept_ids": concept_ids,
        "feature_selections": {
            concept_id: {
                method: {k: v for k, v in sel.items() if not isinstance(v, torch.Tensor)}
                for method, sel in sels.items()
            }
            for concept_id, sels in feature_selections.items()
        },
        "specificity_matrices": {
            method: matrices[method].tolist()
            for method in ["Transcoder", "CLT", "SPD"]
        },
        "method_metrics": method_metrics,
        "scored_generations": {
            concept_id: {
                method: [
                    {k: v for k, v in entry.items()}
                    for entry in entries
                ]
                for method, entries in method_data.items()
            }
            for concept_id, method_data in all_scored.items()
        },
    }

    results_path = os.path.join(cfg.save_dir, "steering_specificity_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Concepts evaluated: {n_concepts}")
    print(f"Prompts per concept: {cfg.n_prompts}")
    print(f"Steering factor: {cfg.steering_factor}")
    print(f"Total generations: {total_gens}")
    print(f"Total judge calls: {len(judge_jobs)}")
    print(f"Total time: {elapsed:.1f}s")
    print()

    print(f"{'Method':<15} {'On-target':>10} {'Off-target':>11} {'Specificity':>12}")
    print("-" * 50)
    for method in ["Transcoder", "CLT", "SPD"]:
        m = method_metrics[method]
        print(f"{method:<15} {m['on_target']:>10.2f} {m['off_target']:>11.2f} "
              f"{m['specificity_ratio']:>12.2f}")

    # Print AUROC table
    print()
    print(f"{'Concept':<12} {'TC AUROC':>10} {'CLT AUROC':>10} {'SPD AUROC':>10}")
    print("-" * 45)
    for concept in concepts:
        cid = concept["id"]
        tc_a = feature_selections[cid]["Transcoder"]["auroc"]
        clt_a = feature_selections[cid]["CLT"]["auroc"]
        spd_a = feature_selections[cid]["SPD"]["auroc"]
        print(f"{cid:<12} {tc_a:>10.3f} {clt_a:>10.3f} {spd_a:>10.3f}")


if __name__ == "__main__":
    main()
