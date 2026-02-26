"""Negative steering / concept suppression evaluation: Transcoders vs CLTs vs SPD.

Complements exp_013 (positive steering) and exp_014 (ablation) by testing
whether actively pushing AWAY from a concept (negative steering factors) can
suppress it from model outputs while preserving fluency.

Key differences from exp_013:
- Uses concept-laden prompts that naturally lead to the target concept
- Applies NEGATIVE steering factors so the model is pushed AWAY from the concept
- Measures concept suppression (how well the concept is removed) and fluency

The question: which decomposition method can most cleanly suppress a concept
from text that would otherwise contain it, without destroying fluency?

Usage:
    python experiments/exp_017_negative_steering/negative_steering.py \\
        --api_key sk-...

    # Quick test with 2 concepts and 2 prompts:
    python experiments/exp_017_negative_steering/negative_steering.py \\
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

# Concept-relevant prompts: sentence starters that naturally lead to each concept.
# Negative steering should push the continuation AWAY from the concept despite
# the prompt strongly priming for it.
CONCEPT_PROMPTS = {
    "sports": [
        "The championship game was exciting because",
        "The athlete trained hard for",
        "In the final quarter of the match,",
        "The team's strategy for winning",
        "After scoring the winning goal,",
    ],
    "code": [
        "The function returns an error when",
        "To fix the bug in the program,",
        "The API endpoint handles requests by",
        "When debugging the application,",
        "The algorithm's time complexity is",
    ],
    "science": [
        "The experiment demonstrated that molecules",
        "According to the latest research findings,",
        "In the laboratory, the chemical reaction",
        "The hypothesis was supported by evidence that",
        "Under controlled conditions, the researchers observed",
    ],
    "politics": [
        "The election results showed that voters",
        "Congress debated the new legislation on",
        "The president announced a new policy for",
        "The political campaign focused on issues like",
        "After the parliamentary debate, lawmakers decided",
    ],
    "food": [
        "The recipe calls for fresh ingredients like",
        "The chef prepared a delicious appetizer with",
        "When baking the cake, it is important to",
        "The restaurant is famous for its signature",
        "To enhance the flavor of the dish,",
    ],
    "music": [
        "The concert featured a stunning guitar solo by",
        "The new album was praised for its innovative",
        "During the symphony performance, the orchestra",
        "The singer's vocal range allowed her to",
        "The band's tour included stops at major",
    ],
    "nature": [
        "Deep in the forest, the wildlife",
        "The coral reef ecosystem is home to",
        "As the seasons change, the migration of",
        "The national park was established to protect",
        "Along the river delta, the biodiversity",
    ],
    "history": [
        "During the medieval period, the empire",
        "The archaeological discovery revealed that ancient",
        "The revolution began when the people",
        "In the colonial period, trade routes",
        "The founding fathers established a system of",
    ],
    "health": [
        "The doctor recommended treatment for the patient's",
        "Common symptoms of the disease include",
        "The clinical study found that the medication",
        "Physical therapy can help patients recover from",
        "New advances in the immune system research suggest",
    ],
    "finance": [
        "The stock market experienced a sharp decline when",
        "Investors diversified their portfolio by adding",
        "The central bank adjusted the interest rate to",
        "Quarterly earnings reports showed that revenue",
        "The hedge fund's strategy focused on",
    ],
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NegativeSteeringConfig:
    tc_project: str = "mats-sprint/pile_transcoder_sweep3"
    clt_project: str = "mats-sprint/pile_clt"
    spd_run: str = "goodfire/spd/s-275c8f21"

    n_concepts: int = 10
    n_prompts: int = 5
    steering_factors: list[float] = field(
        default_factory=lambda: [0.0, -0.5, -1.0, -2.0, -4.0, -8.0, -16.0]
    )
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

    save_dir: str = "experiments/exp_017_negative_steering/output"


# =============================================================================
# Model loading (same patterns as exp_013)
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
    dest = Path(f"checkpoints/neg_steering_tc_L{layer}_{best_run.name}")
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
    dest = Path(f"checkpoints/neg_steering_clt_{best_run.name}")
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
# Feature selection via AUROC (top-1 best feature per concept)
# =============================================================================


def make_concept_texts(concept: dict, n_per_class: int) -> tuple[list[str], list[str]]:
    """Generate positive and negative text samples for a concept.

    Uses the same templates for both classes -- the only difference is the keyword,
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
    base_model, transcoder, tokenizer, concept: dict, cfg: NegativeSteeringConfig,
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
        acts = F.relu(x_enc @ transcoder.W_enc)  # (tokens, dict_size)
        acts = acts.reshape(mlp_input.shape[0], mlp_input.shape[1], -1)  # (B, S, dict_size)

        max_acts = acts.max(dim=1).values  # (B, dict_size)
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
    base_model, clt: CrossLayerTranscoder, tokenizer, concept: dict, cfg: NegativeSteeringConfig,
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
    spd_model, tokenizer, concept: dict, cfg: NegativeSteeringConfig,
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
# Steering generation (negative factors push AWAY from the concept direction)
# =============================================================================


def generate_steered_text_transcoder(
    base_model, transcoder, tokenizer, prompt_text: str,
    latent_idx: int, steering_factor: float,
    layer: int, max_new_tokens: int, temperature: float,
) -> str:
    """Generate text with transcoder steering: add factor * unit_dir to MLP output.

    With negative factor, this subtracts the concept direction from the MLP output,
    pushing the model AWAY from the concept.
    """
    decoder_dir = transcoder.W_dec[latent_idx]  # (output_size,)
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
    With negative factor, this subtracts the concept direction at each target layer.
    """
    n_target_layers = clt.W_dec[source_layer_idx].shape[0]  # n - source_layer_idx
    steering_vectors = {}
    for offset in range(n_target_layers):
        target_layer_idx = source_layer_idx + offset
        actual_target_layer = clt.cfg.layers[target_layer_idx]
        d_j = clt.W_dec[source_layer_idx][offset, latent_idx]  # (output_size,)
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
    u_vec = components.U[component_idx]  # (d_model,)
    return F.normalize(u_vec, dim=0)


def generate_steered_text_spd(
    base_model, steering_direction: torch.Tensor, tokenizer, prompt_text: str,
    steering_factor: float,
    layer: int, max_new_tokens: int, temperature: float,
) -> str:
    """Generate text with SPD steering. steering_direction is already unit-normalized.

    With negative factor, this subtracts the concept direction from the MLP output.
    """
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
# LLM Judge Evaluation
# =============================================================================


def _parse_1_to_10(score_str: str) -> int | None:
    """Extract an integer 1-10 from a string."""
    match = re.search(r"\b(10|[1-9])\b", score_str)
    if match:
        return int(match.group(1))
    return None


def judge_both_scores(
    client: OpenAI, model: str, concept_description: str,
    prompt_text: str, text: str,
) -> tuple[float, float]:
    """Score concept relevance and fluency in a single LLM call (1-10 scale). Returns (concept, fluency)."""
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
# Analysis helpers
# =============================================================================


def compute_pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute the Pareto frontier (maximize both x=suppression and y=fluency).

    Returns list of non-dominated (suppression_score, fluency_score) points.
    """
    if not points:
        return []

    sorted_pts = sorted(points, key=lambda p: -p[0])
    frontier = []
    max_fluency = -float("inf")

    for suppression, fluency in sorted_pts:
        if fluency >= max_fluency:
            frontier.append((suppression, fluency))
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


def plot_concept_vs_factor(
    factor_scores: dict,
    steering_factors: list[float],
    save_path: str,
):
    """Plot concept score and fluency score vs |steering factor|."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for method in ["Transcoder", "CLT", "SPD"]:
        style = METHOD_STYLES[method]
        factors_sorted = sorted(steering_factors, key=lambda f: abs(f))
        abs_factors = [abs(f) for f in factors_sorted]
        cs_means = []
        fl_means = []
        for f in factors_sorted:
            cs_vals = factor_scores[method][f]["concept"]
            fl_vals = factor_scores[method][f]["fluency"]
            cs_means.append(np.mean(cs_vals) if cs_vals else 0.0)
            fl_means.append(np.mean(fl_vals) if fl_vals else 0.0)

        axes[0].plot(abs_factors, cs_means, label=method, markeredgecolor="white",
                     markeredgewidth=0.8, **style)
        axes[1].plot(abs_factors, fl_means, label=method, markeredgecolor="white",
                     markeredgewidth=0.8, **style)

    axes[0].set_xlabel("|Steering Factor|")
    axes[0].set_ylabel("Concept Score")
    axes[0].set_title("Concept Score vs |Steering Factor|")
    axes[0].legend()
    axes[0].grid(True, alpha=0.15, linewidth=0.5)
    axes[0].set_ylim(0, 10.5)

    axes[1].set_xlabel("|Steering Factor|")
    axes[1].set_ylabel("Fluency Score")
    axes[1].set_title("Fluency Score vs |Steering Factor|")
    axes[1].legend()
    axes[1].grid(True, alpha=0.15, linewidth=0.5)
    axes[1].set_ylim(0, 10.5)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Concept/fluency vs factor plot saved to {save_path}")


def plot_suppression_pareto(
    factor_means: dict[str, list[tuple[float, float, float]]],
    save_path: str,
):
    """Plot suppression score vs fluency score, one point per (method, factor).

    factor_means: {method: [(suppression, fluency, factor), ...]}
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in ["Transcoder", "CLT", "SPD"]:
        pts = factor_means.get(method, [])
        if not pts:
            continue
        style = METHOD_STYLES[method]
        pts = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        factors = [p[2] for p in pts]

        ax.plot(xs, ys, label=method, markeredgecolor="white", markeredgewidth=0.8, **style)

        for x, y, f in zip(xs, ys, factors):
            ax.annotate(
                f"{f:g}", (x, y), textcoords="offset points", xytext=(5, 5),
                fontsize=7, color=style["color"], alpha=0.8,
            )

    ax.set_xlabel("Suppression Score (11 - concept)")
    ax.set_ylabel("Fluency Score")
    ax.set_title("Suppression vs Fluency Trade-off by Steering Factor", fontsize=12, pad=10)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Suppression Pareto plot saved to {save_path}")


def plot_suppression_efficiency(
    efficiency_per_concept: dict[str, dict[str, float]],
    save_path: str,
    factor_label: str,
):
    """Bar chart of suppression efficiency per concept at a fixed factor.

    efficiency_per_concept: {concept_id: {method: efficiency}}
    """
    concepts = list(efficiency_per_concept.keys())
    methods = ["Transcoder", "CLT", "SPD"]
    n_concepts = len(concepts)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_concepts)
    width = 0.25

    for m_idx, method in enumerate(methods):
        effs = []
        for concept_id in concepts:
            effs.append(efficiency_per_concept[concept_id].get(method, 0.0))
        style = METHOD_STYLES[method]
        ax.bar(x + m_idx * width, effs, width, label=method, color=style["color"])

    ax.set_xlabel("Concept")
    ax.set_ylabel("Suppression Efficiency")
    ax.set_title(f"Suppression Efficiency per Concept (factor={factor_label})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(concepts, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.15, linewidth=0.5, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Suppression efficiency plot saved to {save_path}")


def plot_per_concept_comparison(
    concept_results: dict,
    save_path: str,
):
    """Per-concept comparison: AUROC and best suppression score."""
    concepts = list(concept_results.keys())
    methods = ["Transcoder", "CLT", "SPD"]
    n_concepts = len(concepts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(n_concepts)
    width = 0.25

    # Panel 1: Best suppression score across negative factors
    for m_idx, method in enumerate(methods):
        scores = []
        for concept_id in concepts:
            concept_data = concept_results[concept_id]
            method_entries = concept_data.get(method, [])
            # Suppression = 11 - concept_score; pick the max suppression
            supp_scores = [
                11.0 - r["concept_score"]
                for r in method_entries
                if r["factor"] != 0.0
            ]
            scores.append(max(supp_scores) if supp_scores else 0.0)
        style = METHOD_STYLES[method]
        axes[0].bar(x + m_idx * width, scores, width, label=method, color=style["color"])

    axes[0].set_xlabel("Concept")
    axes[0].set_ylabel("Best Suppression Score (11 - concept)")
    axes[0].set_title("Best Suppression Score per Concept")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(concepts, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(True, alpha=0.15, linewidth=0.5, axis="y")

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
    axes[1].grid(True, alpha=0.15, linewidth=0.5, axis="y")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Per-concept plot saved to {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Negative steering / concept suppression evaluation"
    )
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--tc_project", type=str, default="mats-sprint/pile_transcoder_sweep3")
    parser.add_argument("--clt_project", type=str, default="mats-sprint/pile_clt")
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument("--n_concepts", type=int, default=10)
    parser.add_argument("--n_prompts", type=int, default=5)
    parser.add_argument("--steering_factors", type=float, nargs="+",
                        default=[0.0, -0.5, -1.0, -2.0, -4.0, -8.0, -16.0])
    parser.add_argument("--eval_layer", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--save_dir", type=str,
                        default="experiments/exp_017_negative_steering/output")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 2 concepts, 2 prompts, fewer factors")
    args = parser.parse_args()

    if args.quick:
        args.n_concepts = 2
        args.n_prompts = 2
        args.steering_factors = [0.0, -2.0, -8.0]

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    assert api_key, "Provide OpenAI API key via --api_key or OPENAI_API_KEY env var"

    cfg = NegativeSteeringConfig(
        tc_project=args.tc_project,
        clt_project=args.clt_project,
        spd_run=args.spd_run,
        n_concepts=args.n_concepts,
        n_prompts=args.n_prompts,
        steering_factors=args.steering_factors,
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
    # Step 2: Generate steered text with NEGATIVE factors
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Generating Negatively Steered Text")
    print("=" * 70)

    # all_generations[concept_id][method] = list of {prompt, factor, text}
    all_generations = {c["id"]: {"Transcoder": [], "CLT": [], "SPD": []} for c in concepts}

    total_gens = len(concepts) * 3 * len(cfg.steering_factors) * cfg.n_prompts
    pbar = tqdm(total=total_gens, desc="Generating steered text")

    for concept in concepts:
        concept_id = concept["id"]
        # Select concept-relevant prompts for this concept
        concept_prompts = CONCEPT_PROMPTS[concept_id][:cfg.n_prompts]

        for factor in cfg.steering_factors:
            for prompt_text in concept_prompts:
                # --- Transcoder ---
                tc_sel = feature_selections[concept_id]["Transcoder"]
                tc_text = generate_steered_text_transcoder(
                    base_model, transcoder, tokenizer, prompt_text,
                    tc_sel["latent_idx"], factor,
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
                clt_text = generate_steered_text_clt(
                    base_model, clt, tokenizer, prompt_text,
                    clt_sel["source_layer_idx"], clt_sel["latent_idx"],
                    factor,
                    cfg.max_new_tokens, cfg.temperature,
                )
                all_generations[concept_id]["CLT"].append({
                    "prompt": prompt_text,
                    "factor": factor,
                    "text": clt_text,
                })
                pbar.update(1)

                # --- SPD ---
                spd_text = generate_steered_text_spd(
                    base_model, spd_directions[concept_id], tokenizer, prompt_text,
                    factor,
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

    all_scored = {c["id"]: {"Transcoder": [], "CLT": [], "SPD": []} for c in concepts}

    judge_jobs = []
    for concept in concepts:
        concept_id = concept["id"]
        description = concept["description"]
        for method in ["Transcoder", "CLT", "SPD"]:
            for gen in all_generations[concept_id][method]:
                judge_jobs.append((concept_id, description, method, gen))

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
    # Step 4: Metrics computation
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Metrics Computation")
    print("=" * 70)

    # Aggregate per (method, factor) for mean scores
    factor_scores = {
        method: {f: {"concept": [], "fluency": []} for f in cfg.steering_factors}
        for method in ["Transcoder", "CLT", "SPD"]
    }

    for concept in concepts:
        concept_id = concept["id"]
        for method in ["Transcoder", "CLT", "SPD"]:
            for entry in all_scored[concept_id][method]:
                factor_scores[method][entry["factor"]]["concept"].append(entry["concept_score"])
                factor_scores[method][entry["factor"]]["fluency"].append(entry["fluency_score"])

    # Compute mean per (method, factor) -- suppression = 11 - concept_score
    method_factor_means = {}  # method -> [(suppression, fluency, factor)]
    method_mean_points = {}   # method -> [(suppression, fluency)]
    for method in ["Transcoder", "CLT", "SPD"]:
        pts_with_factor = []
        pts = []
        for factor in sorted(cfg.steering_factors, key=lambda f: abs(f)):
            cs_vals = factor_scores[method][factor]["concept"]
            fl_vals = factor_scores[method][factor]["fluency"]
            if cs_vals and fl_vals:
                mc = float(np.mean(cs_vals))
                mf = float(np.mean(fl_vals))
                suppression = 11.0 - mc
                pts_with_factor.append((suppression, mf, factor))
                pts.append((suppression, mf))
        method_factor_means[method] = pts_with_factor
        method_mean_points[method] = pts

    method_frontiers = {m: compute_pareto_frontier(pts) for m, pts in method_mean_points.items()}

    # Compute per-concept baseline (factor=0) concept scores for suppression efficiency
    baseline_concept_per_concept = {}  # concept_id -> {method -> mean_concept_score}
    for concept in concepts:
        concept_id = concept["id"]
        baseline_concept_per_concept[concept_id] = {}
        for method in ["Transcoder", "CLT", "SPD"]:
            baseline_entries = [
                e["concept_score"]
                for e in all_scored[concept_id][method]
                if e["factor"] == 0.0
            ]
            baseline_concept_per_concept[concept_id][method] = (
                float(np.mean(baseline_entries)) if baseline_entries else 0.0
            )

    # Suppression efficiency at factor=-4.0 (or closest available)
    efficiency_factor = -4.0
    if efficiency_factor not in cfg.steering_factors:
        # Pick the negative factor closest in magnitude to 4.0
        neg_factors = [f for f in cfg.steering_factors if f < 0]
        efficiency_factor = min(neg_factors, key=lambda f: abs(abs(f) - 4.0)) if neg_factors else cfg.steering_factors[-1]

    efficiency_per_concept = {}  # concept_id -> {method -> efficiency}
    for concept in concepts:
        concept_id = concept["id"]
        efficiency_per_concept[concept_id] = {}
        for method in ["Transcoder", "CLT", "SPD"]:
            baseline_cs = baseline_concept_per_concept[concept_id][method]
            steered_entries = [
                e["concept_score"]
                for e in all_scored[concept_id][method]
                if e["factor"] == efficiency_factor
            ]
            steered_cs = float(np.mean(steered_entries)) if steered_entries else baseline_cs
            if baseline_cs > 0:
                eff = (baseline_cs - steered_cs) / baseline_cs
            else:
                eff = 0.0
            efficiency_per_concept[concept_id][method] = eff

    # Summary metrics
    print("\nSummary Metrics:")
    print(f"{'Method':<15} {'Pareto AUC':>12} {'Avg AUROC':>10} {'Avg Eff':>10}")
    print("-" * 50)

    for method in ["Transcoder", "CLT", "SPD"]:
        frontier = method_frontiers[method]
        auc = compute_auc_pareto(frontier)
        aurocs = [feature_selections[c["id"]][method]["auroc"] for c in concepts]
        avg_auroc = float(np.mean(aurocs))
        effs = [efficiency_per_concept[c["id"]][method] for c in concepts]
        avg_eff = float(np.mean(effs))
        print(f"{method:<15} {auc:>12.4f} {avg_auroc:>10.3f} {avg_eff:>10.3f}")

    # =========================================================================
    # Plotting
    # =========================================================================
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    # Plot 1: Concept score and fluency vs |steering factor|
    plot_concept_vs_factor(
        factor_scores,
        cfg.steering_factors,
        save_path=os.path.join(cfg.save_dir, "score_vs_factor.png"),
    )

    # Plot 2: Suppression vs fluency Pareto
    plot_suppression_pareto(
        method_factor_means,
        save_path=os.path.join(cfg.save_dir, "suppression_pareto.png"),
    )

    # Plot 3: Per-concept suppression efficiency bar chart at the chosen factor
    plot_suppression_efficiency(
        efficiency_per_concept,
        save_path=os.path.join(cfg.save_dir, "suppression_efficiency.png"),
        factor_label=str(efficiency_factor),
    )

    # Plot 4: Per-concept comparison (best suppression + AUROC)
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
                "mean_points_suppression_fluency": method_mean_points[method],
            }
            for method in ["Transcoder", "CLT", "SPD"]
        },
        "baseline_concept_scores": baseline_concept_per_concept,
        "suppression_efficiency": {
            "factor": efficiency_factor,
            "per_concept": efficiency_per_concept,
        },
    }

    results_path = os.path.join(cfg.save_dir, "negative_steering_results.json")
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
    print(f"Total time: {elapsed:.1f}s")
    print()

    # Per-concept AUROC table
    print(f"{'Concept':<12} {'TC AUROC':>10} {'CLT AUROC':>10} {'SPD AUROC':>10}")
    print("-" * 45)
    for concept in concepts:
        cid = concept["id"]
        tc_a = feature_selections[cid]["Transcoder"]["auroc"]
        clt_a = feature_selections[cid]["CLT"]["auroc"]
        spd_a = feature_selections[cid]["SPD"]["auroc"]
        print(f"{cid:<12} {tc_a:>10.3f} {clt_a:>10.3f} {spd_a:>10.3f}")

    # Per-factor mean scores table
    print()
    print(f"{'Factor':<8}", end="")
    for method in ["Transcoder", "CLT", "SPD"]:
        print(f" {method+' CS':>12} {method+' FL':>12} {method+' Supp':>12}", end="")
    print()
    print("-" * 122)
    for factor in sorted(cfg.steering_factors, key=lambda f: abs(f)):
        print(f"{factor:<8.1f}", end="")
        for method in ["Transcoder", "CLT", "SPD"]:
            cs_vals = factor_scores[method][factor]["concept"]
            fl_vals = factor_scores[method][factor]["fluency"]
            cs = float(np.mean(cs_vals)) if cs_vals else 0.0
            fl = float(np.mean(fl_vals)) if fl_vals else 0.0
            supp = 11.0 - cs
            print(f" {cs:>12.3f} {fl:>12.3f} {supp:>12.3f}", end="")
        print()

    # Per-concept suppression efficiency
    print()
    print(f"Suppression Efficiency at factor={efficiency_factor}:")
    print(f"{'Concept':<12} {'Baseline CS':>12} {'TC Eff':>10} {'CLT Eff':>10} {'SPD Eff':>10}")
    print("-" * 57)
    for concept in concepts:
        cid = concept["id"]
        # Average baseline across methods
        baselines = [baseline_concept_per_concept[cid][m] for m in ["Transcoder", "CLT", "SPD"]]
        avg_baseline = float(np.mean(baselines))
        tc_eff = efficiency_per_concept[cid]["Transcoder"]
        clt_eff = efficiency_per_concept[cid]["CLT"]
        spd_eff = efficiency_per_concept[cid]["SPD"]
        print(f"{cid:<12} {avg_baseline:>12.2f} {tc_eff:>10.3f} {clt_eff:>10.3f} {spd_eff:>10.3f}")


if __name__ == "__main__":
    main()
