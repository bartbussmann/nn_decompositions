"""Topic ablation evaluation: Transcoders vs CLTs vs SPD.

Compares the ability of three decomposition methods to selectively suppress
specific topics from the model's outputs across all 4 layers simultaneously.

For each concept:
1. AUROC feature selection across ALL layers to rank features by topic relevance
2. Ablate (zero) top-N features during reconstruction, sweeping N
3. Measure CE loss on topic-specific text vs general text
4. Compute selectivity: topic CE increase / general CE increase

Usage:
    python experiments/exp_014_ablation_eval/ablation_eval.py

    # Quick test with 2 concepts:
    python experiments/exp_014_ablation_eval/ablation_eval.py --n_concepts 2
"""

import argparse
import json
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from clt import CrossLayerTranscoder
from config import CLTConfig, EncoderConfig
from transcoder import BatchTopKTranscoder
from spd.models.components import make_mask_infos

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


# =============================================================================
# Config
# =============================================================================


@dataclass
class AblationEvalConfig:
    tc_project: str = "mats-sprint/pile_transcoder_sweep3"
    clt_project: str = "mats-sprint/pile_clt"
    spd_run: str = "goodfire/spd/s-275c8f21"

    n_concepts: int = 10
    n_auroc_examples: int = 200
    auroc_seq_len: int = 128
    auroc_batch_size: int = 32

    n_ablated_features: list[int] = field(
        default_factory=lambda: [1, 2, 5, 10, 20, 50, 100, 200, 500]
    )

    n_eval_batches: int = 20
    eval_batch_size: int = 8
    eval_seq_len: int = 512

    # Topic text: scan this many Pile samples to find keyword matches
    topic_scan_limit: int = 50000
    topic_target_sequences: int = 160  # = n_eval_batches * eval_batch_size

    save_dir: str = "experiments/exp_014_ablation_eval/output"


# =============================================================================
# Model loading (self-contained, same patterns as exp_011/exp_013)
# =============================================================================


@contextmanager
def patched_forward(module: nn.Module, patched_fn):
    original = module.forward
    module.forward = patched_fn
    try:
        yield
    finally:
        module.forward = original


def compute_ce_loss(model, input_ids: torch.Tensor) -> float:
    logits, _ = model(input_ids)
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)).item()


def compute_ce_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)).item()


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
        dest = Path(f"checkpoints/ablation_tc_L{layer}_{run.name}")
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
    dest = Path(f"checkpoints/ablation_clt_{best_run.name}")
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


def split_concept_keywords(concept: dict) -> tuple[dict, dict]:
    """Split a concept's keywords into AUROC (selection) and eval (testing) sets.

    Positive keywords (15) -> 8 auroc, 7 eval.
    Negative keywords (10) -> 5 auroc, 5 eval.
    """
    pos = concept["positive_keywords"]
    neg = concept["negative_keywords"]
    pos_split = len(pos) // 2 + (len(pos) % 2)  # ceil division -> 8 for 15
    neg_split = len(neg) // 2

    auroc_concept = {**concept,
                     "positive_keywords": pos[:pos_split],
                     "negative_keywords": neg[:neg_split]}
    eval_concept = {**concept,
                    "positive_keywords": pos[pos_split:],
                    "negative_keywords": neg[neg_split:]}
    return auroc_concept, eval_concept


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


def make_concept_texts(concept: dict, n_per_class: int) -> tuple[list[str], list[str]]:
    pos_keywords = concept["positive_keywords"]
    neg_keywords = concept["negative_keywords"]

    positives = [TEMPLATES[i % len(TEMPLATES)].format(kw=pos_keywords[i % len(pos_keywords)])
                 for i in range(n_per_class)]
    negatives = [TEMPLATES[i % len(TEMPLATES)].format(kw=neg_keywords[i % len(neg_keywords)])
                 for i in range(n_per_class)]
    return positives, negatives


# =============================================================================
# Feature selection: compute AUROC for ALL features across ALL layers
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
    cfg: AblationEvalConfig,
) -> list[tuple[int, int, float]]:
    """Compute AUROC for every transcoder feature across all layers.

    Returns list of (layer, feature_idx, auroc) sorted by auroc descending.
    """
    positives, negatives = make_concept_texts(concept, cfg.n_auroc_examples)
    all_texts = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives))

    # Pre-allocate per-layer score arrays
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
            mlp_in = captured[layer_idx]  # (B, S, d_model)
            flat = mlp_in.reshape(-1, tc.cfg.input_size)
            use_pre_enc_bias = tc.cfg.pre_enc_bias and tc.input_size == tc.output_size
            x_enc = flat - tc.b_dec if use_pre_enc_bias else flat
            acts = F.relu(x_enc @ tc.W_enc)  # (B*S, dict_size)
            acts_2d = acts.reshape(mlp_in.shape[0], mlp_in.shape[1], -1)  # (B, S, dict_size)
            max_acts = acts_2d.max(dim=1).values  # (B, dict_size)
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
    cfg: AblationEvalConfig,
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
            mlp_in = captured[actual_layer]  # (B, S, d_model)
            flat = mlp_in.reshape(-1, clt.cfg.input_size)
            acts = F.relu(flat @ clt.W_enc[li] + clt.b_enc[li])  # (B*S, dict_size)
            acts_2d = acts.reshape(mlp_in.shape[0], mlp_in.shape[1], -1)
            max_acts = acts_2d.max(dim=1).values  # (B, dict_size)
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
    cfg: AblationEvalConfig,
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
            ci_scores = ci.lower_leaky[mod_name].clamp(0, 1)  # (B, S, C)
            max_scores = ci_scores.max(dim=1).values  # (B, C)
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
# Data collection
# =============================================================================


def get_general_eval_batches(n_batches: int, batch_size: int, seq_len: int) -> list[torch.Tensor]:
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)
    batches = []
    for _ in tqdm(range(n_batches), desc="Loading general eval batches"):
        batch_ids = []
        for _ in range(batch_size):
            sample = next(data_iter)
            ids = sample["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:seq_len])
        batches.append(torch.stack(batch_ids).to(DEVICE))
    return batches


def collect_topic_batches(
    tokenizer,
    eval_concept: dict,
    target_n_sequences: int,
    scan_limit: int,
    seq_len: int,
    batch_size: int,
) -> list[torch.Tensor]:
    """Collect Pile sequences containing eval-split keywords (not AUROC keywords).

    Falls back to template-generated text if not enough matches found.
    """
    keywords = [kw.lower() for kw in eval_concept["positive_keywords"]]
    dataset = load_dataset("danbraunai/pile-uncopyrighted-tok", split="train", streaming=True)
    dataset = dataset.shuffle(seed=200, buffer_size=10000)
    data_iter = iter(dataset)

    matched_ids = []
    scanned = 0
    for sample in data_iter:
        if scanned >= scan_limit:
            break
        scanned += 1
        ids = sample["input_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        if len(ids) < seq_len:
            continue
        ids = ids[:seq_len]
        text = tokenizer.decode(ids, skip_special_tokens=True).lower()
        if any(kw in text for kw in keywords):
            matched_ids.append(ids)
            if len(matched_ids) >= target_n_sequences:
                break

    # Fallback: generate template text if not enough matches
    if len(matched_ids) < batch_size:
        print(f"    Only found {len(matched_ids)} matching sequences, supplementing with templates...")
        pos_keywords = eval_concept["positive_keywords"]
        while len(matched_ids) < target_n_sequences:
            idx = len(matched_ids)
            kw = pos_keywords[idx % len(pos_keywords)]
            template = TEMPLATES[idx % len(TEMPLATES)]
            text = template.format(kw=kw)
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len,
                            padding="max_length")["input_ids"][0]
            matched_ids.append(ids)

    # Pack into batches
    batches = []
    for i in range(0, len(matched_ids) - batch_size + 1, batch_size):
        batch = torch.stack(matched_ids[i : i + batch_size]).to(DEVICE)
        batches.append(batch)

    return batches


# =============================================================================
# Ablation evaluation
# =============================================================================


def _transcoder_batchtopk_recon(tc, x_in, k):
    use_pre_enc_bias = tc.cfg.pre_enc_bias and tc.input_size == tc.output_size
    x_enc = x_in - tc.b_dec if use_pre_enc_bias else x_in
    acts = F.relu(x_enc @ tc.W_enc)
    n_keep = k * acts.shape[0]
    if n_keep < acts.numel():
        topk = torch.topk(acts.flatten(), n_keep, dim=-1)
        acts = torch.zeros_like(acts.flatten()).scatter(-1, topk.indices, topk.values).reshape(acts.shape)
    return acts, acts @ tc.W_dec + tc.b_dec


@torch.no_grad()
def eval_transcoder_ablation(
    base_model,
    transcoders: dict[int, BatchTopKTranscoder],
    features_to_zero: list[tuple[int, int, float]],
    n_ablated: int,
    eval_batches: list[torch.Tensor],
) -> float:
    """CE loss after subtracting ablated features' contributions from real MLP outputs.

    For N=0 this returns the unmodified base model's CE.
    For N>0: real_mlp_output - sum(act_j * W_dec[j, :]) for ablated features j.
    """
    zero_per_layer: dict[int, set[int]] = {l: set() for l in LAYERS}
    for layer, feat_idx, _auroc in features_to_zero[:n_ablated]:
        zero_per_layer[layer].add(feat_idx)

    # If nothing to ablate, just run the real model
    if n_ablated == 0:
        total_ce = 0.0
        for input_ids in eval_batches:
            total_ce += compute_ce_loss(base_model, input_ids)
        return total_ce / len(eval_batches)

    total_ce = 0.0
    for input_ids in eval_batches:
        with ExitStack() as stack:
            for layer_idx in LAYERS:
                zeros = zero_per_layer[layer_idx]
                if not zeros:
                    continue
                mlp = base_model.h[layer_idx].mlp
                tc = transcoders[layer_idx]
                original_forward = mlp.forward

                def _make_patched(orig_fwd_, tc_, zeros_, input_size_, k_):
                    def _patched(hidden_states):
                        real_out = orig_fwd_(hidden_states)
                        flat = hidden_states.reshape(-1, input_size_)
                        acts, _ = _transcoder_batchtopk_recon(tc_, flat, k_)
                        # Compute contribution of ablated features only
                        ablated_acts = torch.zeros_like(acts)
                        idx_list = list(zeros_)
                        ablated_acts[:, idx_list] = acts[:, idx_list]
                        ablated_contribution = (ablated_acts @ tc_.W_dec).reshape(real_out.shape)
                        return real_out - ablated_contribution
                    return _patched

                stack.enter_context(patched_forward(
                    mlp, _make_patched(original_forward, tc, zeros, tc.cfg.input_size, tc.cfg.top_k)
                ))
            total_ce += compute_ce_loss(base_model, input_ids)

    return total_ce / len(eval_batches)


def _clt_batchtopk_acts(clt, inputs, k):
    all_acts = []
    for i in range(clt.cfg.n_layers):
        pre_acts = F.relu(inputs[i] @ clt.W_enc[i] + clt.b_enc[i])
        n_keep = k * pre_acts.shape[0]
        if n_keep < pre_acts.numel():
            topk = torch.topk(pre_acts.flatten(), n_keep, dim=-1)
            acts = torch.zeros_like(pre_acts.flatten()).scatter(
                -1, topk.indices, topk.values
            ).reshape(pre_acts.shape)
        else:
            acts = pre_acts
        all_acts.append(acts)
    return all_acts


@torch.no_grad()
def eval_clt_ablation(
    base_model,
    clt: CrossLayerTranscoder,
    features_to_zero: list[tuple[int, int, float]],
    n_ablated: int,
    eval_batches: list[torch.Tensor],
) -> float:
    """CE loss after subtracting ablated CLT features' contributions from real MLP outputs.

    For N=0 this returns the unmodified base model's CE.
    For N>0: real_mlp_output[j] - sum over ablated (i,f) of acts[i][:,f] * W_dec[i][j-i][f,:]
    """
    zero_per_layer: dict[int, set[int]] = {i: set() for i in range(clt.cfg.n_layers)}
    for enc_layer, feat_idx, _auroc in features_to_zero[:n_ablated]:
        zero_per_layer[enc_layer].add(feat_idx)

    if n_ablated == 0:
        total_ce = 0.0
        for input_ids in eval_batches:
            total_ce += compute_ce_loss(base_model, input_ids)
        return total_ce / len(eval_batches)

    k = clt.cfg.top_k
    n_layers = clt.cfg.n_layers
    total_ce = 0.0
    for input_ids in eval_batches:
        # Pass 1: collect RMS2 outputs and real MLP outputs from unpatched model
        captured = _collect_rms2_outputs(base_model, input_ids)
        seq_shape = captured[LAYERS[0]].shape

        # Encode all layers, apply TopK (to determine activations)
        clt_inputs = [captured[l].reshape(-1, clt.cfg.input_size) for l in LAYERS]
        all_acts = _clt_batchtopk_acts(clt, clt_inputs, k)

        # Compute contribution of ablated features to each target layer
        ablated_contrib = [torch.zeros(all_acts[0].shape[0], clt.cfg.output_size, device=DEVICE)
                           for _ in range(n_layers)]
        for i, zeros in zero_per_layer.items():
            if not zeros:
                continue
            idx_list = list(zeros)
            # acts[i][:, idx_list] shape: (batch*seq, len(idx_list))
            ablated_feats = all_acts[i][:, idx_list]
            for j in range(i, n_layers):
                # W_dec[i][j-i] shape: (dict_size, output_size)
                decoder_cols = clt.W_dec[i][j - i][idx_list, :]  # (len(idx_list), output_size)
                ablated_contrib[j] += ablated_feats @ decoder_cols

        ablated_shaped = [c.reshape(seq_shape) for c in ablated_contrib]

        # Pass 2: patch MLPs to subtract ablated contributions from real output
        with ExitStack() as stack:
            for li, layer_idx in enumerate(LAYERS):
                mlp = base_model.h[layer_idx].mlp
                original_forward = mlp.forward

                def _make_patched(orig_fwd_, contrib_):
                    def _patched(hidden_states):
                        return orig_fwd_(hidden_states) - contrib_
                    return _patched

                stack.enter_context(patched_forward(
                    mlp, _make_patched(original_forward, ablated_shaped[li])
                ))
            total_ce += compute_ce_loss(base_model, input_ids)

    return total_ce / len(eval_batches)


@torch.no_grad()
def eval_spd_ablation(
    spd_model,
    features_to_zero: list[tuple[str, int, float]],
    n_ablated: int,
    eval_batches: list[torch.Tensor],
    all_module_names: list[str],
) -> float:
    """CE loss with top-n_ablated SPD components (c_fc and down_proj) zeroed."""
    # Group by module
    zero_per_module: dict[str, set[int]] = {mod: set() for mod in all_module_names}
    for mod_name, comp_idx, _auroc in features_to_zero[:n_ablated]:
        zero_per_module[mod_name].add(comp_idx)

    total_ce = 0.0
    for input_ids in eval_batches:
        # Build masks: all-ones except zeroed components
        masks = {}
        for mod_name in all_module_names:
            n_comp = spd_model.module_to_c[mod_name]
            mask = torch.ones(input_ids.shape[0], input_ids.shape[1], n_comp, device=DEVICE)
            for comp_idx in zero_per_module[mod_name]:
                mask[:, :, comp_idx] = 0.0
            masks[mod_name] = mask

        mask_infos = make_mask_infos(masks)
        logits = spd_model(input_ids, mask_infos=mask_infos)
        total_ce += compute_ce_from_logits(logits, input_ids)

    return total_ce / len(eval_batches)


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


def plot_curves(
    metrics: dict,
    n_values: list[int],
    save_dir: str,
):
    """Plot 3 curves: topic suppression, general degradation, selectivity."""
    methods = ["Transcoder", "CLT", "SPD"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for method in methods:
        style = METHOD_STYLES[method]
        topic_increases = [metrics[method]["mean_topic_ce_increase"].get(n, 0) for n in n_values]
        general_increases = [metrics[method]["mean_general_ce_increase"].get(n, 0) for n in n_values]
        selectivities = [metrics[method]["mean_selectivity"].get(n, 0) for n in n_values]

        axes[0].plot(n_values, topic_increases, label=method,
                     markeredgecolor="white", markeredgewidth=0.8, **style)
        axes[1].plot(n_values, general_increases, label=method,
                     markeredgecolor="white", markeredgewidth=0.8, **style)
        axes[2].plot(n_values, selectivities, label=method,
                     markeredgecolor="white", markeredgewidth=0.8, **style)

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95)

    axes[0].set_xlabel("N features ablated")
    axes[0].set_ylabel("Topic CE increase")
    axes[0].set_title("Topic Suppression")

    axes[1].set_xlabel("N features ablated")
    axes[1].set_ylabel("General CE increase")
    axes[1].set_title("General Degradation")

    axes[2].set_xlabel("N features ablated")
    axes[2].set_ylabel("Selectivity")
    axes[2].set_title("Selectivity (topic / general)")

    fig.tight_layout()
    path = f"{save_dir}/ablation_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Curves saved to {path}")


def plot_per_concept_selectivity(
    metrics: dict,
    concepts: list[dict],
    n_fixed: int,
    save_dir: str,
):
    """Bar chart: selectivity at fixed N for each concept x method."""
    methods = ["Transcoder", "CLT", "SPD"]
    concept_ids = [c["id"] for c in concepts]
    n_concepts = len(concept_ids)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_concepts)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        style = METHOD_STYLES[method]
        vals = []
        for cid in concept_ids:
            sel = metrics[method]["per_concept"].get(cid, {}).get("selectivity", {}).get(n_fixed, 0)
            vals.append(sel)
        ax.bar(x + i * width, vals, width, label=method, color=style["color"], edgecolor="white")

    ax.set_xlabel("Concept")
    ax.set_ylabel(f"Selectivity (N={n_fixed})")
    ax.set_title(f"Per-Concept Selectivity at N={n_fixed}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(concept_ids, rotation=45, ha="right")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=0.95)
    ax.grid(True, alpha=0.15, linewidth=0.5, axis="y")
    fig.tight_layout()
    path = f"{save_dir}/per_concept_selectivity_n{n_fixed}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Per-concept plot saved to {path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Topic ablation evaluation")
    parser.add_argument("--n_concepts", type=int, default=10)
    parser.add_argument("--n_eval_batches", type=int, default=20)
    parser.add_argument("--n_auroc_examples", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="experiments/exp_014_ablation_eval/output")
    args = parser.parse_args()

    cfg = AblationEvalConfig(
        n_concepts=args.n_concepts,
        n_eval_batches=args.n_eval_batches,
        n_auroc_examples=args.n_auroc_examples,
        save_dir=args.save_dir,
    )
    cfg.topic_target_sequences = cfg.n_eval_batches * cfg.eval_batch_size

    import os
    os.makedirs(cfg.save_dir, exist_ok=True)

    concepts = CONCEPTS[: cfg.n_concepts]

    # =========================================================================
    # Load models
    # =========================================================================
    print("=" * 70)
    print("Loading models")
    print("=" * 70)

    print("Loading SPD model...")
    spd_model, _ = load_spd_model(cfg.spd_run)
    spd_model.to(DEVICE)
    base_model = spd_model.target_model
    base_model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_down_names = [f"h.{l}.mlp.down_proj" for l in LAYERS]
    all_cfc_names = [f"h.{l}.mlp.c_fc" for l in LAYERS]
    all_module_names = all_cfc_names + all_down_names

    print(f"\nDownloading transcoders from {cfg.tc_project}...")
    tc_paths = download_best_transcoders(cfg.tc_project)
    transcoders = {}
    for layer_idx in LAYERS:
        tc = load_transcoder(str(tc_paths[layer_idx]))
        tc.to(DEVICE)
        transcoders[layer_idx] = tc

    print(f"\nDownloading CLT from {cfg.clt_project}...")
    clt_path = download_best_clt(cfg.clt_project)
    clt = load_clt(str(clt_path))
    clt.to(DEVICE)

    # =========================================================================
    # Split keywords: AUROC selection vs evaluation (no overlap)
    # =========================================================================
    auroc_concepts = {}
    eval_concepts = {}
    for concept in concepts:
        auroc_c, eval_c = split_concept_keywords(concept)
        auroc_concepts[concept["id"]] = auroc_c
        eval_concepts[concept["id"]] = eval_c
        print(f"  [{concept['id']}] AUROC keywords: {len(auroc_c['positive_keywords'])} pos, "
              f"{len(auroc_c['negative_keywords'])} neg | "
              f"Eval keywords: {len(eval_c['positive_keywords'])} pos, "
              f"{len(eval_c['negative_keywords'])} neg")

    # =========================================================================
    # Load evaluation data (uses eval-split keywords only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Loading evaluation data")
    print("=" * 70)

    general_batches = get_general_eval_batches(cfg.n_eval_batches, cfg.eval_batch_size, cfg.eval_seq_len)

    topic_batches: dict[str, list[torch.Tensor]] = {}
    for concept in concepts:
        cid = concept["id"]
        print(f"  Collecting topic batches for '{cid}'...")
        batches = collect_topic_batches(
            tokenizer, eval_concepts[cid], cfg.topic_target_sequences,
            cfg.topic_scan_limit, cfg.eval_seq_len, cfg.eval_batch_size,
        )
        topic_batches[cid] = batches
        print(f"    Got {len(batches)} batches")

    # =========================================================================
    # AUROC feature selection (uses auroc-split keywords only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: AUROC Feature Selection (all layers)")
    print("=" * 70)

    concept_features: dict[str, dict[str, list]] = {}
    for concept in tqdm(concepts, desc="AUROC feature selection"):
        cid = concept["id"]
        concept_features[cid] = {}
        auroc_c = auroc_concepts[cid]

        tc_aurocs = compute_all_transcoder_aurocs(base_model, transcoders, tokenizer, auroc_c, cfg)
        concept_features[cid]["Transcoder"] = tc_aurocs
        top3 = tc_aurocs[:3]
        print(f"  [{cid}] Transcoder: {len(tc_aurocs)} features, top3 AUROC: {[f'{a:.3f}' for _, _, a in top3]}")

        clt_aurocs = compute_all_clt_aurocs(base_model, clt, tokenizer, auroc_c, cfg)
        concept_features[cid]["CLT"] = clt_aurocs
        top3 = clt_aurocs[:3]
        print(f"  [{cid}] CLT: {len(clt_aurocs)} features, top3 AUROC: {[f'{a:.3f}' for _, _, a in top3]}")

        spd_aurocs = compute_all_spd_aurocs(spd_model, tokenizer, auroc_c, cfg)
        concept_features[cid]["SPD"] = spd_aurocs
        top3 = spd_aurocs[:3]
        print(f"  [{cid}] SPD: {len(spd_aurocs)} components, top3 AUROC: {[f'{a:.3f}' for _, _, a in top3]}")

    # =========================================================================
    # Step 2: Ablation sweep
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Ablation Sweep")
    print("=" * 70)

    # Include n=0 as baseline
    n_values = cfg.n_ablated_features
    all_n = [0] + n_values

    # results[concept_id][method][n] = {"topic_ce": float, "general_ce": float}
    results: dict[str, dict[str, dict[int, dict[str, float]]]] = {}

    total_evals = len(concepts) * 3 * len(all_n) * 2  # 3 methods, 2 datasets
    pbar = tqdm(total=total_evals, desc="Ablation evaluation")

    for concept in concepts:
        cid = concept["id"]
        results[cid] = {}

        for method in ["Transcoder", "CLT", "SPD"]:
            results[cid][method] = {}
            features = concept_features[cid][method]

            for n in all_n:
                # General CE
                if method == "Transcoder":
                    gen_ce = eval_transcoder_ablation(base_model, transcoders, features, n, general_batches)
                elif method == "CLT":
                    gen_ce = eval_clt_ablation(base_model, clt, features, n, general_batches)
                else:
                    gen_ce = eval_spd_ablation(spd_model, features, n, general_batches, all_module_names)
                pbar.update(1)

                # Topic CE
                t_batches = topic_batches[cid]
                if not t_batches:
                    topic_ce = gen_ce
                elif method == "Transcoder":
                    topic_ce = eval_transcoder_ablation(base_model, transcoders, features, n, t_batches)
                elif method == "CLT":
                    topic_ce = eval_clt_ablation(base_model, clt, features, n, t_batches)
                else:
                    topic_ce = eval_spd_ablation(spd_model, features, n, t_batches, all_module_names)
                pbar.update(1)

                results[cid][method][n] = {"topic_ce": topic_ce, "general_ce": gen_ce}
                if n > 0 or method == "Transcoder":
                    baseline = results[cid][method][0]
                    t_inc = topic_ce - baseline["topic_ce"]
                    g_inc = gen_ce - baseline["general_ce"]
                    sel = t_inc / max(g_inc, 1e-6) if g_inc > 0 else 0.0
                    print(f"  [{cid}] {method} n={n}: topic_CE={topic_ce:.4f} gen_CE={gen_ce:.4f} "
                          f"t_inc={t_inc:+.4f} g_inc={g_inc:+.4f} sel={sel:.2f}")

    pbar.close()

    # =========================================================================
    # Step 3: Compute metrics
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Computing Metrics")
    print("=" * 70)

    EPS = 1e-6
    metrics: dict[str, dict] = {}

    for method in ["Transcoder", "CLT", "SPD"]:
        metrics[method] = {
            "mean_topic_ce_increase": {},
            "mean_general_ce_increase": {},
            "mean_selectivity": {},
            "per_concept": {},
        }

        for cid in [c["id"] for c in concepts]:
            baseline = results[cid][method][0]
            per_concept_data: dict[str, dict[int, float]] = {
                "topic_ce_increase": {},
                "general_ce_increase": {},
                "selectivity": {},
            }
            for n in n_values:
                r = results[cid][method][n]
                t_inc = r["topic_ce"] - baseline["topic_ce"]
                g_inc = r["general_ce"] - baseline["general_ce"]
                sel = t_inc / max(g_inc, EPS) if g_inc > 0 else 0.0
                per_concept_data["topic_ce_increase"][n] = t_inc
                per_concept_data["general_ce_increase"][n] = g_inc
                per_concept_data["selectivity"][n] = sel
            metrics[method]["per_concept"][cid] = per_concept_data

        # Compute means across concepts
        concept_ids = [c["id"] for c in concepts]
        for n in n_values:
            t_incs = [metrics[method]["per_concept"][cid]["topic_ce_increase"][n] for cid in concept_ids]
            g_incs = [metrics[method]["per_concept"][cid]["general_ce_increase"][n] for cid in concept_ids]
            sels = [metrics[method]["per_concept"][cid]["selectivity"][n] for cid in concept_ids]
            metrics[method]["mean_topic_ce_increase"][n] = float(np.mean(t_incs))
            metrics[method]["mean_general_ce_increase"][n] = float(np.mean(g_incs))
            metrics[method]["mean_selectivity"][n] = float(np.mean(sels))

    # Print summary
    print(f"\n{'Method':<15} {'N':>5} {'Topic CE inc':>14} {'General CE inc':>15} {'Selectivity':>12}")
    print("-" * 65)
    for method in ["Transcoder", "CLT", "SPD"]:
        for n in n_values:
            t = metrics[method]["mean_topic_ce_increase"][n]
            g = metrics[method]["mean_general_ce_increase"][n]
            s = metrics[method]["mean_selectivity"][n]
            print(f"{method:<15} {n:>5} {t:>14.4f} {g:>15.4f} {s:>12.2f}")

    # =========================================================================
    # Step 4: Plotting
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Generating Plots")
    print("=" * 70)

    plot_curves(metrics, n_values, cfg.save_dir)

    # Per-concept bar chart at N=1 (or nearest available)
    n_fixed = 1
    plot_per_concept_selectivity(metrics, concepts, n_fixed, cfg.save_dir)

    # =========================================================================
    # Save results
    # =========================================================================
    output = {
        "config": {
            "n_concepts": cfg.n_concepts,
            "n_auroc_examples": cfg.n_auroc_examples,
            "n_eval_batches": cfg.n_eval_batches,
            "n_ablated_features": cfg.n_ablated_features,
        },
        "feature_selections": {
            cid: {
                method: [(str(a), int(b), float(c)) for a, b, c in feats[:20]]
                for method, feats in method_feats.items()
            }
            for cid, method_feats in concept_features.items()
        },
        "results": {
            cid: {
                method: {
                    str(n): {"topic_ce": r["topic_ce"], "general_ce": r["general_ce"]}
                    for n, r in n_results.items()
                }
                for method, n_results in method_results.items()
            }
            for cid, method_results in results.items()
        },
        "metrics": {
            method: {
                "mean_topic_ce_increase": {str(k): v for k, v in m["mean_topic_ce_increase"].items()},
                "mean_general_ce_increase": {str(k): v for k, v in m["mean_general_ce_increase"].items()},
                "mean_selectivity": {str(k): v for k, v in m["mean_selectivity"].items()},
            }
            for method, m in metrics.items()
        },
    }

    results_path = f"{cfg.save_dir}/ablation_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
