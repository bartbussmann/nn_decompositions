"""AxBench-style steering evaluation: Transcoders vs CLTs vs SPD.

Compares three decomposition methods on steering performance:
1. Feature selection via AUROC on concept-labeled examples
2. Steering via decoder-direction addition at multiple strengths
3. LLM-judge scoring of concept presence and fluency
4. Pareto frontier analysis (concept score vs fluency score)

The target model (LlamaSimpleMLP) is a small base LM, not instruction-tuned.
Instead of instruction-following prompts, we use generic sentence starters and
measure whether steering pushes the continuation toward the target concept.
The fluency score judges coherence of the continuation, not instruction-following.

Usage:
    python experiments/exp_013_steering_eval/steering_eval.py \
        --api_key sk-... \
        --n_concepts 10 \
        --steering_factors 0.5 1.0 2.0 4.0 8.0

    # Quick test with 2 concepts:
    python experiments/exp_013_steering_eval/steering_eval.py \
        --api_key sk-... --n_concepts 2 --n_prompts 2
"""

import argparse
import json
import os
import random
import sys
import time
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

# AxBench Concept10 subset — 10 diverse concepts for the minimal viable experiment.
# Each concept has a description and positive/negative example prompts.
# We generate our own labeled examples by using concept-related vs unrelated text.
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

# Generic sentence starters for a base LM (concept-neutral).
# These are partial sentences the model will continue. Steering should push
# the continuation toward the target concept without the prompt itself being
# concept-specific.
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
class SteeringEvalConfig:
    # Models
    tc_project: str = "mats-sprint/pile_transcoder_sweep3"
    clt_project: str = "mats-sprint/pile_clt"
    spd_run: str = "goodfire/spd/s-275c8f21"

    # Experiment scope
    n_concepts: int = 10
    n_prompts: int = 5  # sentence starters per concept
    steering_factors: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    eval_layer: int = 2  # single layer for minimal experiment

    # AUROC feature selection
    n_auroc_examples: int = 1000  # per class (positive/negative)
    auroc_seq_len: int = 128
    auroc_batch_size: int = 32

    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.7

    # LLM judge
    openai_model: str = "gpt-4o-mini"
    api_key: str = ""

    # SPD steering method: "effective_direction" (Option 2) or "ci_clamp" (Option 1)
    spd_steering_method: str = "effective_direction"

    # Output
    save_dir: str = "experiments/exp_013_steering_eval/output"


# =============================================================================
# Model loading (reused patterns from exp_011)
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

    # Sort by descending score
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Walk through thresholds (each unique score value)
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
        # Trapezoidal area
        auc += 0.5 * (fpr - prev_fpr) * (tpr + prev_tpr)
        prev_fpr = fpr
        prev_tpr = tpr

    return auc


# =============================================================================
# Step 1: Feature selection via AUROC
# =============================================================================


def make_concept_texts(concept: dict, n_per_class: int) -> tuple[list[str], list[str]]:
    """Generate positive and negative text samples for a concept.

    Uses the same templates for both classes — the only difference is the keyword,
    so AUROC selects features that respond to concept content, not template structure.
    """
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
    base_model, transcoder, tokenizer, concept: dict, cfg: SteeringEvalConfig,
) -> tuple[int, float]:
    """Compute AUROC for each transcoder latent on a concept. Returns (best_latent_idx, best_auroc)."""
    positives, negatives = make_concept_texts(concept, cfg.n_auroc_examples)

    all_texts = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)

    # Tokenize and get max-pooled pre-TopK activations per sequence
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
        # Pre-TopK activations (dense)
        flat = mlp_input.reshape(-1, transcoder.cfg.input_size)
        use_pre_enc_bias = transcoder.cfg.pre_enc_bias and transcoder.input_size == transcoder.output_size
        x_enc = flat - transcoder.b_dec if use_pre_enc_bias else flat
        acts = F.relu(x_enc @ transcoder.W_enc)  # (tokens, dict_size)
        acts = acts.reshape(mlp_input.shape[0], mlp_input.shape[1], -1)  # (B, S, dict_size)

        # Max-pool across tokens
        max_acts = acts.max(dim=1).values  # (B, dict_size)
        all_scores[i : i + len(batch_texts)] = max_acts.cpu()

    # Compute AUROC per latent
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
    base_model, clt: CrossLayerTranscoder, tokenizer, concept: dict, cfg: SteeringEvalConfig,
) -> tuple[int, int, float]:
    """Compute AUROC for each CLT encoder latent. Returns (best_layer_idx, best_latent_idx, best_auroc)."""
    positives, negatives = make_concept_texts(concept, cfg.n_auroc_examples)
    all_texts = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)

    dict_size = clt.cfg.dict_size
    n_layers = clt.cfg.n_layers

    # Accumulate max-pooled encoder activations per layer
    all_scores = {i: torch.zeros(len(all_texts), dict_size) for i in range(n_layers)}

    for i in range(0, len(all_texts), cfg.auroc_batch_size):
        batch_texts = all_texts[i : i + cfg.auroc_batch_size]
        encoded = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=cfg.auroc_seq_len,
        )
        input_ids = encoded["input_ids"].to(DEVICE)

        # Collect RMS2 outputs at all layers
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
            # Pre-TopK (dense) activations
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
    spd_model, tokenizer, concept: dict, cfg: SteeringEvalConfig,
) -> tuple[str, int, float]:
    """Compute AUROC for each SPD component using CI scores.

    Uses the norm of the gated component output as the activation score.
    Returns (best_module_name, best_component_idx, best_auroc).
    """
    positives, negatives = make_concept_texts(concept, cfg.n_auroc_examples)
    all_texts = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)

    layer = cfg.eval_layer
    module_names = [f"h.{layer}.mlp.c_fc", f"h.{layer}.mlp.down_proj"]

    # Accumulate max-pooled CI scores per module
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
            ci_scores = ci.lower_leaky[mod_name].clamp(0, 1)  # (B, S, C)
            max_scores = ci_scores.max(dim=1).values  # (B, C)
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
# Step 2: Steering
# =============================================================================


@torch.no_grad()
def compute_avg_activation_norm(
    base_model, transcoder, tokenizer, layer: int, latent_idx: int, n_samples: int = 50,
) -> float:
    """Compute the average activation magnitude of a transcoder latent on generic text."""
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=1000)
    data_iter = iter(dataset)

    total_act = 0.0
    count = 0
    for _ in range(n_samples):
        sample = next(data_iter)
        ids = sample["input_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        input_ids = ids[:128].unsqueeze(0).to(DEVICE)

        mlp_input = get_mlp_input_acts(base_model, input_ids, layer)
        flat = mlp_input.reshape(-1, transcoder.cfg.input_size)
        use_pre_enc_bias = transcoder.cfg.pre_enc_bias and transcoder.input_size == transcoder.output_size
        x_enc = flat - transcoder.b_dec if use_pre_enc_bias else flat
        acts = F.relu(x_enc @ transcoder.W_enc)
        act_val = acts[:, latent_idx].mean().item()
        if act_val > 0:
            total_act += act_val
            count += 1

    return total_act / max(count, 1)


def generate_steered_text_transcoder(
    base_model, transcoder, tokenizer, prompt_text: str,
    latent_idx: int, steering_factor: float, avg_act_norm: float,
    layer: int, max_new_tokens: int, temperature: float,
) -> str:
    """Generate text with transcoder steering: add alpha * d_j to MLP output."""
    decoder_dir = transcoder.W_dec[latent_idx]  # (output_size,)
    steering_vector = steering_factor * avg_act_norm * decoder_dir  # (output_size,)
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
    avg_act_norm: float, max_new_tokens: int, temperature: float,
) -> str:
    """Generate text with CLT steering.

    The selected feature at source layer i writes to layers i through n-1.
    We add alpha * d_j^(l) at each target layer.
    """
    # Get decoder directions for all target layers
    n_target_layers = clt.W_dec[source_layer_idx].shape[0]  # n - source_layer_idx
    steering_vectors = {}
    for offset in range(n_target_layers):
        target_layer_idx = source_layer_idx + offset
        actual_target_layer = clt.cfg.layers[target_layer_idx]
        d_j = clt.W_dec[source_layer_idx][offset, latent_idx]  # (output_size,)
        steering_vectors[actual_target_layer] = steering_factor * avg_act_norm * d_j

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
def compute_spd_effective_direction(
    spd_model, base_model, tokenizer, module_name: str, component_idx: int, layer: int,
) -> torch.Tensor:
    """Compute the effective steering direction for an SPD component (Option 2).

    Runs reference inputs through the SPD model with only the selected component
    active vs all components zeroed out. The difference in MLP outputs gives
    the component's contribution direction.
    """
    from spd.models.components import make_mask_infos

    # Load a few reference inputs from Pile for a stable direction estimate
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=123, buffer_size=1000)
    data_iter = iter(dataset)

    all_module_names = list(spd_model.module_to_c.keys())
    direction_accum = torch.zeros(base_model.config.n_embd, device=DEVICE)
    n_ref = 10

    for _ in range(n_ref):
        sample = next(data_iter)
        ids = sample["input_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        input_ids = ids[:64].unsqueeze(0).to(DEVICE)

        # Build masks: only the selected component active
        masks_single = {}
        masks_zero = {}
        for mod_name in all_module_names:
            n_comp = spd_model.module_to_c[mod_name]
            zero_mask = torch.zeros(1, input_ids.shape[1], n_comp, device=DEVICE)
            masks_zero[mod_name] = zero_mask
            if mod_name == module_name:
                single_mask = zero_mask.clone()
                single_mask[:, :, component_idx] = 1.0
                masks_single[mod_name] = single_mask
            else:
                masks_single[mod_name] = zero_mask

        # Capture MLP output with single component active
        captured_single = {}
        hook = spd_model.target_model.h[layer].mlp.register_forward_hook(
            lambda _mod, _inp, out, d=captured_single: d.update({"out": out.detach()})
        )
        spd_model(input_ids, mask_infos=make_mask_infos(masks_single))
        hook.remove()

        # Capture MLP output with all components zeroed
        captured_zero = {}
        hook = spd_model.target_model.h[layer].mlp.register_forward_hook(
            lambda _mod, _inp, out, d=captured_zero: d.update({"out": out.detach()})
        )
        spd_model(input_ids, mask_infos=make_mask_infos(masks_zero))
        hook.remove()

        # Component contribution = difference
        diff = captured_single["out"] - captured_zero["out"]  # (1, S, d_model)
        direction_accum += diff.mean(dim=(0, 1))

    direction = direction_accum / n_ref

    # Normalize to unit vector
    norm = direction.norm()
    if norm > 1e-8:
        direction = direction / norm

    return direction


def generate_steered_text_spd(
    base_model, steering_direction: torch.Tensor, tokenizer, prompt_text: str,
    steering_factor: float, direction_norm: float,
    layer: int, max_new_tokens: int, temperature: float,
) -> str:
    """Generate text with SPD steering using the effective direction (Option 2)."""
    steering_vector = steering_factor * direction_norm * steering_direction

    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(DEVICE)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        mlp = base_model.h[layer].mlp

        def _make_steered(mlp_, sv_):
            def _steered(hidden_states):
                h = mlp_.gelu(mlp_.c_fc(hidden_states))
                out = mlp_.down_proj(h)
                return out + sv_.unsqueeze(0).unsqueeze(0).expand_as(out)
            return _steered

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
# Step 3: LLM Judge Evaluation
# =============================================================================


def judge_both_scores(
    client: OpenAI, model: str, concept_description: str,
    prompt_text: str, text: str,
) -> tuple[float, float]:
    """Score concept relevance and fluency in a single LLM call. Returns (concept, fluency)."""
    prompt = (
        f"Rate the following generated text on two dimensions.\n\n"
        f"The text was generated by continuing the sentence starter '{prompt_text}'.\n"
        f"The target concept is: '{concept_description}'\n\n"
        f"Text: \"{text}\"\n\n"
        f"1. CONCEPT (0-2): How strongly does the text reflect the target concept?\n"
        f"   0 = not at all related, 1 = somewhat related, 2 = strongly about the concept\n"
        f"2. FLUENCY (0-2): How fluent and coherent is the text?\n"
        f"   0 = incoherent/repetitive/nonsensical, 1 = somewhat coherent with issues, "
        f"2 = well-written natural prose\n\n"
        f"Respond with ONLY two numbers separated by a comma, e.g. '1,2'"
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
            digits = [ch for ch in score_str if ch in "012"]
            concept_score = float(digits[0]) if len(digits) >= 1 else 0.0
            fluency_score = float(digits[1]) if len(digits) >= 2 else 0.0
            return concept_score, fluency_score
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2 ** attempt + random.random())
            else:
                raise
    return 0.0, 0.0


# =============================================================================
# Step 4: Pareto Frontier Analysis
# =============================================================================


def compute_pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute the Pareto frontier (maximize both x=concept and y=fluency).

    Returns list of non-dominated (concept_score, fluency_score) points.
    """
    if not points:
        return []

    # Sort by concept score descending
    sorted_pts = sorted(points, key=lambda p: -p[0])
    frontier = []
    max_fluency = -float("inf")

    for concept, fluency in sorted_pts:
        if fluency >= max_fluency:
            frontier.append((concept, fluency))
            max_fluency = fluency

    return sorted(frontier, key=lambda p: p[0])


def compute_auc_pareto(frontier: list[tuple[float, float]]) -> float:
    """Compute the area under the Pareto frontier using the trapezoidal rule."""
    if len(frontier) < 2:
        return 0.0
    auc = 0.0
    for i in range(len(frontier) - 1):
        x0, y0 = frontier[i]
        x1, y1 = frontier[i + 1]
        auc += 0.5 * (y0 + y1) * (x1 - x0)
    return auc


def concept_score_at_fluency_threshold(
    points: list[tuple[float, float]], threshold: float = 1.5,
) -> float:
    """Find the maximum concept score among points with fluency score >= threshold."""
    eligible = [c for c, f in points if f >= threshold]
    return max(eligible) if eligible else 0.0


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


def plot_pareto_frontier(
    method_points: dict[str, list[tuple[float, float]]],
    method_frontiers: dict[str, list[tuple[float, float]]],
    save_path: str,
    title: str = "Steering Pareto Frontier: Concept vs Fluency",
):
    """Plot concept score vs fluency score with Pareto frontiers for each method."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in ["Transcoder", "CLT", "SPD"]:
        pts = method_points.get(method, [])
        frontier = method_frontiers.get(method, [])
        style = METHOD_STYLES[method]

        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, alpha=0.3, color=style["color"], s=30, zorder=3)

        if frontier:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            ax.plot(fx, fy, label=method, markeredgecolor="white", markeredgewidth=0.8, **style)

    ax.set_xlabel("Concept Score (normalized)")
    ax.set_ylabel("Fluency Score")
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 2.05)
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Pareto plot saved to {save_path}")


def plot_per_concept_comparison(
    concept_results: dict,
    save_path: str,
):
    """Plot per-concept steering comparison as a grouped bar chart."""
    concepts = list(concept_results.keys())
    methods = ["Transcoder", "CLT", "SPD"]
    n_concepts = len(concepts)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Best concept score (across steering factors)
    x = np.arange(n_concepts)
    width = 0.25

    for m_idx, method in enumerate(methods):
        scores = []
        for concept_id in concepts:
            concept_data = concept_results[concept_id]
            method_scores = [
                r["concept_score"]
                for r in concept_data.get(method, [])
            ]
            scores.append(max(method_scores) if method_scores else 0.0)
        style = METHOD_STYLES[method]
        axes[0].bar(x + m_idx * width, scores, width, label=method, color=style["color"])

    axes[0].set_xlabel("Concept")
    axes[0].set_ylabel("Best Concept Score")
    axes[0].set_title("Best Concept Score per Concept")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(concepts, rotation=45, ha="right")
    axes[0].legend()

    # Panel 2: AUROC per concept
    for m_idx, method in enumerate(methods):
        aurocs = []
        for concept_id in concepts:
            concept_data = concept_results[concept_id]
            aurocs.append(concept_data.get(f"{method}_auroc", 0.0))
        style = METHOD_STYLES[method]
        axes[1].bar(x + m_idx * width, aurocs, width, label=method, color=style["color"])

    axes[1].set_xlabel("Concept")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Feature Selection AUROC per Concept")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(concepts, rotation=45, ha="right")
    axes[1].legend()
    axes[1].set_ylim(0.4, 1.05)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Per-concept plot saved to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="AxBench-style steering evaluation")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--tc_project", type=str, default="mats-sprint/pile_transcoder_sweep3")
    parser.add_argument("--clt_project", type=str, default="mats-sprint/pile_clt")
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument("--n_concepts", type=int, default=10)
    parser.add_argument("--n_prompts", type=int, default=10)
    parser.add_argument("--steering_factors", type=float, nargs="+", default=[0.0, 0.1, 0.5, 2.0, 10.0, 50.0])
    parser.add_argument("--eval_layer", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--spd_steering_method", type=str, default="effective_direction",
                        choices=["effective_direction", "ci_clamp"])
    parser.add_argument("--save_dir", type=str,
                        default="experiments/exp_013_steering_eval/output")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    assert api_key, "Provide OpenAI API key via --api_key or OPENAI_API_KEY env var"

    cfg = SteeringEvalConfig(
        tc_project=args.tc_project,
        clt_project=args.clt_project,
        spd_run=args.spd_run,
        n_concepts=args.n_concepts,
        n_prompts=args.n_prompts,
        steering_factors=args.steering_factors,
        eval_layer=args.eval_layer,
        max_new_tokens=args.max_new_tokens,
        openai_model=args.openai_model,
        spd_steering_method=args.spd_steering_method,
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
    spd_model, raw_config = load_spd_model(cfg.spd_run)
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    # We need a tokenizer for text -> tokens
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

    feature_selections = {}  # concept_id -> {method -> selection_info}

    for concept in tqdm(concepts, desc="AUROC feature selection"):
        concept_id = concept["id"]
        feature_selections[concept_id] = {}

        # Transcoder
        tc_latent, tc_auroc = compute_transcoder_auroc(
            base_model, transcoder, tokenizer, concept, cfg,
        )
        feature_selections[concept_id]["Transcoder"] = {
            "latent_idx": tc_latent,
            "auroc": tc_auroc,
        }
        print(f"  [{concept_id}] Transcoder: latent={tc_latent}, AUROC={tc_auroc:.3f}")

        # CLT
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

        # SPD
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
    # Precompute steering info
    # =========================================================================
    print("\n" + "=" * 70)
    print("Precomputing steering vectors...")
    print("=" * 70)

    steering_info = {}
    for concept in concepts:
        concept_id = concept["id"]
        steering_info[concept_id] = {}

        # Transcoder: compute average activation norm
        tc_sel = feature_selections[concept_id]["Transcoder"]
        tc_avg_norm = compute_avg_activation_norm(
            base_model, transcoder, tokenizer, cfg.eval_layer, tc_sel["latent_idx"],
        )
        steering_info[concept_id]["Transcoder"] = {"avg_act_norm": tc_avg_norm}
        print(f"  [{concept_id}] Transcoder avg_act_norm={tc_avg_norm:.4f}")

        # CLT: compute average activation norm for selected encoder feature
        clt_sel = feature_selections[concept_id]["CLT"]
        # Reuse the same approach: compute mean activation on generic text
        dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=1000)
        data_iter = iter(dataset)

        total_act = 0.0
        count = 0
        source_layer_idx = clt_sel["source_layer_idx"]
        latent_idx = clt_sel["latent_idx"]
        actual_source_layer = clt.cfg.layers[source_layer_idx] if source_layer_idx >= 0 else 0

        for _ in range(50):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            input_ids = ids[:128].unsqueeze(0).to(DEVICE)

            mlp_input = get_mlp_input_acts(base_model, input_ids, actual_source_layer)
            flat = mlp_input.reshape(-1, clt.cfg.input_size)
            pre_acts = F.relu(flat @ clt.W_enc[source_layer_idx] + clt.b_enc[source_layer_idx])
            act_val = pre_acts[:, latent_idx].mean().item()
            if act_val > 0:
                total_act += act_val
                count += 1

        clt_avg_norm = total_act / max(count, 1)
        steering_info[concept_id]["CLT"] = {"avg_act_norm": clt_avg_norm}
        print(f"  [{concept_id}] CLT avg_act_norm={clt_avg_norm:.4f}")

        # SPD: compute effective direction and its norm
        spd_sel = feature_selections[concept_id]["SPD"]
        spd_direction = compute_spd_effective_direction(
            spd_model, base_model, tokenizer,
            spd_sel["module_name"], spd_sel["component_idx"], cfg.eval_layer,
        )
        # Use decoder weight norm as a scale reference for SPD
        # The direction is already unit-normalized; use a scale comparable to transcoder
        spd_direction_norm = tc_avg_norm  # use transcoder's norm as reference scale
        steering_info[concept_id]["SPD"] = {
            "direction": spd_direction,
            "direction_norm": spd_direction_norm,
        }
        print(f"  [{concept_id}] SPD effective direction computed")

    # =========================================================================
    # Step 2: Generate steered text
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Generating Steered Text")
    print("=" * 70)

    # all_generations[concept_id][method] = list of {prompt, factor, text}
    all_generations = {c["id"]: {"Transcoder": [], "CLT": [], "SPD": []} for c in concepts}

    total_gens = cfg.n_concepts * 3 * len(cfg.steering_factors) * cfg.n_prompts
    pbar = tqdm(total=total_gens, desc="Generating steered text")

    for concept in concepts:
        concept_id = concept["id"]

        for factor in cfg.steering_factors:
            for prompt_text in prompts:
                # --- Transcoder ---
                tc_sel = feature_selections[concept_id]["Transcoder"]
                tc_info = steering_info[concept_id]["Transcoder"]
                tc_text = generate_steered_text_transcoder(
                    base_model, transcoder, tokenizer, prompt_text,
                    tc_sel["latent_idx"], factor, tc_info["avg_act_norm"],
                    cfg.eval_layer, cfg.max_new_tokens, cfg.temperature,
                )
                all_generations[concept_id]["Transcoder"].append({
                    "prompt": prompt_text,
                    "factor": factor,
                    "text": tc_text,
                })
                pbar.update(1)

                # --- CLT ---
                clt_sel = feature_selections[concept_id]["CLT"]
                clt_info = steering_info[concept_id]["CLT"]
                clt_text = generate_steered_text_clt(
                    base_model, clt, tokenizer, prompt_text,
                    clt_sel["source_layer_idx"], clt_sel["latent_idx"],
                    factor, clt_info["avg_act_norm"],
                    cfg.max_new_tokens, cfg.temperature,
                )
                all_generations[concept_id]["CLT"].append({
                    "prompt": prompt_text,
                    "factor": factor,
                    "text": clt_text,
                })
                pbar.update(1)

                # --- SPD ---
                spd_info = steering_info[concept_id]["SPD"]
                spd_text = generate_steered_text_spd(
                    base_model, spd_info["direction"], tokenizer, prompt_text,
                    factor, spd_info["direction_norm"],
                    cfg.eval_layer, cfg.max_new_tokens, cfg.temperature,
                )
                all_generations[concept_id]["SPD"].append({
                    "prompt": prompt_text,
                    "factor": factor,
                    "text": spd_text,
                })
                pbar.update(1)

    pbar.close()

    # =========================================================================
    # Step 3: LLM Judge Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: LLM Judge Evaluation")
    print("=" * 70)

    # Score all generations (parallel)
    all_scored = {c["id"]: {"Transcoder": [], "CLT": [], "SPD": []} for c in concepts}

    # Flatten all jobs
    judge_jobs = []
    for concept in concepts:
        concept_id = concept["id"]
        description = concept["description"]
        for method in ["Transcoder", "CLT", "SPD"]:
            for gen in all_generations[concept_id][method]:
                judge_jobs.append((concept_id, description, method, gen))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _judge_one(job):
        concept_id, description, method, gen = job
        cs, fl = judge_both_scores(
            client, cfg.openai_model, description, gen["prompt"], gen["text"],
        )
        return concept_id, method, {
            "prompt": gen["prompt"],
            "factor": gen["factor"],
            "text": gen["text"],
            "concept_score": cs,
            "fluency_score": fl,
        }

    pbar = tqdm(total=len(judge_jobs), desc="LLM judging")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_judge_one, job) for job in judge_jobs]
        for future in as_completed(futures):
            concept_id, method, entry = future.result()
            all_scored[concept_id][method].append(entry)
            pbar.update(1)
    pbar.close()

    # =========================================================================
    # Normalize concept scores (min-max per concept across all methods/factors)
    # =========================================================================
    print("\nNormalizing concept scores...")

    for concept in concepts:
        concept_id = concept["id"]
        all_concept_scores = []
        for method in ["Transcoder", "CLT", "SPD"]:
            for entry in all_scored[concept_id][method]:
                all_concept_scores.append(entry["concept_score"])

        min_cs = min(all_concept_scores) if all_concept_scores else 0.0
        max_cs = max(all_concept_scores) if all_concept_scores else 1.0
        score_range = max_cs - min_cs

        for method in ["Transcoder", "CLT", "SPD"]:
            for entry in all_scored[concept_id][method]:
                if score_range > 0:
                    entry["concept_score_normalized"] = (entry["concept_score"] - min_cs) / score_range
                else:
                    entry["concept_score_normalized"] = 0.0

    # =========================================================================
    # Step 4: Pareto Frontier Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Pareto Frontier Analysis")
    print("=" * 70)

    # Aggregate points per method (across all concepts and factors)
    # Points: (concept_score_normalized, fluency_score)
    method_points = {"Transcoder": [], "CLT": [], "SPD": []}

    # Also aggregate per (method, factor) for mean scores
    factor_scores = {
        method: {f: {"concept": [], "fluency": []} for f in cfg.steering_factors}
        for method in ["Transcoder", "CLT", "SPD"]
    }

    for concept in concepts:
        concept_id = concept["id"]
        for method in ["Transcoder", "CLT", "SPD"]:
            for entry in all_scored[concept_id][method]:
                cs_norm = entry["concept_score_normalized"]
                fl_score = entry["fluency_score"]
                method_points[method].append((cs_norm, fl_score))
                factor_scores[method][entry["factor"]]["concept"].append(cs_norm)
                factor_scores[method][entry["factor"]]["fluency"].append(fl_score)

    # Compute mean per (method, factor) for Pareto frontier
    method_mean_points = {}
    for method in ["Transcoder", "CLT", "SPD"]:
        pts = []
        for factor in cfg.steering_factors:
            cs_vals = factor_scores[method][factor]["concept"]
            fl_vals = factor_scores[method][factor]["fluency"]
            if cs_vals and fl_vals:
                pts.append((np.mean(cs_vals), np.mean(fl_vals)))
        method_mean_points[method] = pts

    # Compute Pareto frontiers
    method_frontiers = {}
    for method, pts in method_mean_points.items():
        method_frontiers[method] = compute_pareto_frontier(pts)

    # Compute summary metrics
    print("\nSummary Metrics:")
    print(f"{'Method':<15} {'AUC':>8} {'CS@FL>=1.5':>12} {'Avg AUROC':>10}")
    print("-" * 50)

    for method in ["Transcoder", "CLT", "SPD"]:
        frontier = method_frontiers[method]
        auc = compute_auc_pareto(frontier)
        cs_at_thresh = concept_score_at_fluency_threshold(method_mean_points[method], 1.5)

        # Average AUROC across concepts
        aurocs = [feature_selections[c["id"]][method]["auroc"] for c in concepts]
        avg_auroc = np.mean(aurocs)

        print(f"{method:<15} {auc:>8.4f} {cs_at_thresh:>12.4f} {avg_auroc:>10.3f}")

    # =========================================================================
    # Plotting
    # =========================================================================
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    # Pareto frontier plot
    plot_pareto_frontier(
        method_points, method_frontiers,
        save_path=os.path.join(cfg.save_dir, "pareto_frontier.png"),
    )

    # Per-concept comparison
    concept_results_for_plot = {}
    for concept in concepts:
        concept_id = concept["id"]
        concept_results_for_plot[concept_id] = {}
        for method in ["Transcoder", "CLT", "SPD"]:
            concept_results_for_plot[concept_id][method] = all_scored[concept_id][method]
            concept_results_for_plot[concept_id][f"{method}_auroc"] = (
                feature_selections[concept_id][method]["auroc"]
            )

    plot_per_concept_comparison(
        concept_results_for_plot,
        save_path=os.path.join(cfg.save_dir, "per_concept_comparison.png"),
    )

    # Per-factor plot: mean scores vs steering factor
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method in ["Transcoder", "CLT", "SPD"]:
        style = METHOD_STYLES[method]
        factors = sorted(cfg.steering_factors)
        cs_means = [np.mean(factor_scores[method][f]["concept"]) for f in factors]
        fl_means = [np.mean(factor_scores[method][f]["fluency"]) for f in factors]

        axes[0].plot(factors, cs_means, label=method, **style)
        axes[1].plot(factors, fl_means, label=method, **style)

    axes[0].set_xlabel("Steering Factor")
    axes[0].set_ylabel("Concept Score (normalized)")
    axes[0].set_title("Concept Score vs Steering Factor")
    axes[0].legend()
    axes[0].grid(True, alpha=0.15)

    axes[1].set_xlabel("Steering Factor")
    axes[1].set_ylabel("Fluency Score")
    axes[1].set_title("Fluency Score vs Steering Factor")
    axes[1].legend()
    axes[1].grid(True, alpha=0.15)

    fig.tight_layout()
    factor_plot_path = os.path.join(cfg.save_dir, "score_vs_factor.png")
    fig.savefig(factor_plot_path)
    plt.close(fig)
    print(f"Factor plot saved to {factor_plot_path}")

    # =========================================================================
    # Save results
    # =========================================================================
    elapsed = time.time() - start_time

    results_data = {
        "config": {
            "n_concepts": cfg.n_concepts,
            "n_prompts": cfg.n_prompts,
            "steering_factors": cfg.steering_factors,
            "eval_layer": cfg.eval_layer,
            "max_new_tokens": cfg.max_new_tokens,
            "openai_model": cfg.openai_model,
            "spd_steering_method": cfg.spd_steering_method,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
        },
        "feature_selections": {
            concept_id: {
                method: {k: v for k, v in sel.items() if not isinstance(v, torch.Tensor)}
                for method, sel in sels.items()
            }
            for concept_id, sels in feature_selections.items()
        },
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
        "pareto_analysis": {
            method: {
                "frontier": method_frontiers[method],
                "auc": compute_auc_pareto(method_frontiers[method]),
                "concept_at_fluency_1.5": concept_score_at_fluency_threshold(
                    method_mean_points[method], 1.5
                ),
                "mean_points": method_mean_points[method],
            }
            for method in ["Transcoder", "CLT", "SPD"]
        },
    }

    results_path = os.path.join(cfg.save_dir, "steering_eval_results.json")
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
    print(f"Steering factors: {cfg.steering_factors}")
    print(f"SPD steering method: {cfg.spd_steering_method}")
    print(f"Total time: {elapsed:.1f}s")
    print()

    # Print per-concept AUROC table
    print(f"{'Concept':<12} {'TC AUROC':>10} {'CLT AUROC':>10} {'SPD AUROC':>10}")
    print("-" * 45)
    for concept in concepts:
        cid = concept["id"]
        tc_a = feature_selections[cid]["Transcoder"]["auroc"]
        clt_a = feature_selections[cid]["CLT"]["auroc"]
        spd_a = feature_selections[cid]["SPD"]["auroc"]
        print(f"{cid:<12} {tc_a:>10.3f} {clt_a:>10.3f} {spd_a:>10.3f}")

    # Print per-factor mean scores
    print()
    print(f"{'Factor':<8}", end="")
    for method in ["Transcoder", "CLT", "SPD"]:
        print(f" {method+' CS':>12} {method+' FL':>12}", end="")
    print()
    print("-" * 80)
    for factor in cfg.steering_factors:
        print(f"{factor:<8.1f}", end="")
        for method in ["Transcoder", "CLT", "SPD"]:
            cs = np.mean(factor_scores[method][factor]["concept"]) if factor_scores[method][factor]["concept"] else 0
            fl = np.mean(factor_scores[method][factor]["fluency"]) if factor_scores[method][factor]["fluency"] else 0
            print(f" {cs:>12.3f} {fl:>12.3f}", end="")
        print()


if __name__ == "__main__":
    main()
