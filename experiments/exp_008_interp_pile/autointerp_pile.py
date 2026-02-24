"""SAEBench-style autointerp evaluation for Transcoder and SPD on LlamaSimpleMLP.

Runs GPT-4o-mini generation + scoring phases on:
  - Transcoder features (BatchTopKTranscoder, dict_size=4096)
  - SPD c_fc components (4096 components, using CI scores)
  - SPD down_proj components (3072 components, using CI scores)

Reuses SAEBench prompt templates and indexing utilities while handling
our custom LlamaSimpleMLP model and pre-tokenized Pile dataset.

Usage:
    python experiments/autointerp_pile.py --api_key sk-...
    python experiments/autointerp_pile.py --api_key sk-... --n_latents 5  # quick test
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from openai import OpenAI
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))
sys.path.insert(0, str(Path("/workspace/SAEBench")))

from transcoder import BatchTopKTranscoder, TopKTranscoder
from sae_bench.sae_bench_utils.indexing_utils import (
    get_iw_sample_indices,
    get_k_largest_indices,
    index_with_buffer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 3


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AutoInterpConfig:
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

    # Autointerp params (matching SAEBench defaults)
    buffer: int = 10
    no_overlap: bool = True
    act_threshold_frac: float = 0.01

    n_top_ex_for_generation: int = 10
    n_iw_sampled_ex_for_generation: int = 5
    n_top_ex_for_scoring: int = 2
    n_random_ex_for_scoring: int = 10
    n_iw_sampled_ex_for_scoring: int = 2

    max_tokens_in_explanation: int = 3000
    use_demos_in_explanation: bool = True

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
    save_path: str = "experiments/autointerp_results.json"

    @property
    def n_top_ex(self):
        return self.n_top_ex_for_generation + self.n_top_ex_for_scoring

    @property
    def n_iw_sampled_ex(self):
        return self.n_iw_sampled_ex_for_generation + self.n_iw_sampled_ex_for_scoring

    @property
    def n_ex_for_scoring(self):
        return (
            self.n_top_ex_for_scoring
            + self.n_random_ex_for_scoring
            + self.n_iw_sampled_ex_for_scoring
        )

    @property
    def n_correct_for_scoring(self):
        return self.n_top_ex_for_scoring + self.n_iw_sampled_ex_for_scoring

    @property
    def max_tokens_in_prediction(self):
        return 2 * self.n_ex_for_scoring + 5


# =============================================================================
# Data loading
# =============================================================================


def load_tokenized_dataset(total_tokens: int, seq_len: int) -> torch.Tensor:
    """Load pre-tokenized Pile as a flat (N, seq_len) tensor."""
    n_sequences = total_tokens // seq_len
    dataset = load_dataset(
        "danbraunai/pile-uncopyrighted-tok", split="train", streaming=True
    )
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)

    rows = []
    for _ in tqdm(range(n_sequences), desc="Loading tokenized data"):
        sample = next(data_iter)
        ids = sample["input_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        rows.append(ids[:seq_len])

    return torch.stack(rows)  # (N, seq_len)


# =============================================================================
# Model loading (reused from existing code)
# =============================================================================


def load_transcoder(checkpoint_dir: str):
    from config import EncoderConfig

    ENCODER_CLASSES = {
        "vanilla": __import__("transcoder", fromlist=["VanillaTranscoder"]).VanillaTranscoder,
        "topk": TopKTranscoder,
        "batchtopk": BatchTopKTranscoder,
        "jumprelu": __import__("transcoder", fromlist=["JumpReLUTranscoder"]).JumpReLUTranscoder,
    }

    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg = EncoderConfig(**cfg_dict)
    encoder = ENCODER_CLASSES[cfg.encoder_type](cfg)
    encoder.load_state_dict(
        torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE)
    )
    encoder.eval()
    return encoder


def load_spd_model(wandb_path: str):
    from analysis.collect_spd_activations import load_spd_model as _load

    return _load(wandb_path)


# =============================================================================
# Activation collection
# =============================================================================


@torch.no_grad()
def collect_transcoder_activations(
    base_model,
    transcoder,
    tokenized_dataset: torch.Tensor,
    batch_size: int,
    selected_latents: list[int] | None = None,
) -> torch.Tensor:
    """Collect transcoder feature activations.

    Returns (N, seq_len, n_selected) if selected_latents given,
    else (N, seq_len, dict_size).
    """
    all_acts = []

    for i in tqdm(
        range(0, len(tokenized_dataset), batch_size),
        desc="Transcoder activations",
    ):
        input_ids = tokenized_dataset[i : i + batch_size].to(DEVICE)

        # Hook MLP input (post rms_2)
        captured = {}

        def _hook(mod, inp, out):
            captured["mlp_input"] = out.detach()

        hook = base_model.h[LAYER].rms_2.register_forward_hook(_hook)
        base_model(input_ids)
        hook.remove()

        mlp_input = captured["mlp_input"]

        # Encode through transcoder (no pre_enc_bias for transcoder)
        use_pre_enc_bias = (
            transcoder.cfg.pre_enc_bias
            and transcoder.input_size == transcoder.output_size
        )
        x_enc = mlp_input - transcoder.b_dec if use_pre_enc_bias else mlp_input

        if isinstance(transcoder, (TopKTranscoder, BatchTopKTranscoder)):
            acts = F.relu(x_enc @ transcoder.W_enc)
        else:
            acts = F.relu(x_enc @ transcoder.W_enc + transcoder.b_enc)

        if selected_latents is not None:
            acts = acts[:, :, selected_latents]

        all_acts.append(acts.cpu().to(torch.bfloat16))

    return torch.cat(all_acts, dim=0)


@torch.no_grad()
def collect_spd_ci_activations(
    spd_model,
    tokenized_dataset: torch.Tensor,
    module_names: list[str],
    batch_size: int,
    selected_latents: dict[str, list[int] | None] | None = None,
) -> dict[str, torch.Tensor]:
    """Collect SPD CI score activations for specified modules.

    Returns dict of module_name -> (N, seq_len, n_components).
    """
    all_acts = {name: [] for name in module_names}

    for i in tqdm(
        range(0, len(tokenized_dataset), batch_size),
        desc="SPD CI activations",
    ):
        input_ids = tokenized_dataset[i : i + batch_size].to(DEVICE)

        out = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

        for name in module_names:
            ci_scores = ci.lower_leaky[name].clamp(0, 1)
            if selected_latents is not None and selected_latents.get(name) is not None:
                ci_scores = ci_scores[:, :, selected_latents[name]]
            all_acts[name].append(ci_scores.cpu().to(torch.bfloat16))

    return {name: torch.cat(acts, dim=0) for name, acts in all_acts.items()}


@torch.no_grad()
def compute_sparsity(
    activations_fn,
    tokenized_dataset: torch.Tensor,
    batch_size: int,
    n_features: int,
) -> torch.Tensor:
    """Compute per-feature firing counts in a streaming fashion.

    activations_fn(input_ids) -> (batch, seq_len, n_features) tensor of activations.
    Returns firing_count tensor of shape (n_features,).
    """
    firing_count = torch.zeros(n_features, dtype=torch.long)

    for i in tqdm(
        range(0, len(tokenized_dataset), batch_size),
        desc="Computing sparsity",
    ):
        input_ids = tokenized_dataset[i : i + batch_size].to(DEVICE)
        acts = activations_fn(input_ids)  # (batch, seq, n_features)
        firing_count += (acts > 0).sum(dim=(0, 1)).cpu().long()

    return firing_count


def select_alive_latents(
    firing_count: torch.Tensor, n_latents: int, dead_threshold: int, seed: int
) -> list[int]:
    """Select n_latents random alive features."""
    alive = torch.nonzero(firing_count > dead_threshold).squeeze(1).tolist()
    if len(alive) == 0:
        raise ValueError("No alive latents found!")
    if len(alive) < n_latents:
        print(f"WARNING: Only {len(alive)} alive latents, using all of them")
        return alive
    rng = random.Random(seed)
    return sorted(rng.sample(alive, n_latents))


# =============================================================================
# Tokenizer wrapper
# =============================================================================


class TokenizerWrapper:
    """Provides to_str_tokens() for Example compatibility."""

    def __init__(self, tokenizer_name: str = "EleutherAI/gpt-neox-20b"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def to_str_tokens(self, tokens: torch.Tensor) -> list[str]:
        return [self.tokenizer.decode([t]) for t in tokens.tolist()]


# =============================================================================
# AutoInterp core (adapted from SAEBench)
# =============================================================================


class Example:
    """Data for a single example sequence."""

    def __init__(
        self,
        toks: list[int],
        acts: list[float],
        act_threshold: float,
        tokenizer_wrapper: TokenizerWrapper,
    ):
        self.toks = toks
        self.str_toks = tokenizer_wrapper.to_str_tokens(torch.tensor(self.toks))
        self.acts = acts
        self.act_threshold = act_threshold
        self.toks_are_active = [act > act_threshold for act in self.acts]
        self.is_active = any(self.toks_are_active)

    def to_str(self, mark_toks: bool = False) -> str:
        return (
            "".join(
                f"<<{tok}>>" if (mark_toks and is_active) else tok
                for tok, is_active in zip(self.str_toks, self.toks_are_active)
            )
            .replace("\ufffd", "")
            .replace("\n", "\u21b5")
        )


class Examples:
    """Data for multiple example sequences."""

    def __init__(self, examples: list[Example], shuffle: bool = False) -> None:
        self.examples = examples
        if shuffle:
            random.shuffle(self.examples)
        else:
            self.examples = sorted(
                self.examples, key=lambda x: max(x.acts), reverse=True
            )

    def display(self, predictions: list[int] | None = None) -> str:
        return tabulate(
            [
                (
                    [max(ex.acts), ex.to_str(mark_toks=True)]
                    if predictions is None
                    else [
                        max(ex.acts),
                        "Y" if ex.is_active else "",
                        "Y" if i + 1 in predictions else "",
                        ex.to_str(mark_toks=False),
                    ]
                )
                for i, ex in enumerate(self.examples)
            ],
            headers=["Top act"]
            + ([] if predictions is None else ["Active?", "Predicted?"])
            + ["Sequence"],
            tablefmt="simple_outline",
            floatfmt=".3f",
        )

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class AutoInterpRunner:
    """Runs autointerp generation + scoring on pre-computed activations."""

    def __init__(
        self,
        cfg: AutoInterpConfig,
        tokenized_dataset: torch.Tensor,
        activations: torch.Tensor,
        latent_indices: list[int],
        tokenizer_wrapper: TokenizerWrapper,
    ):
        self.cfg = cfg
        self.tokenized_dataset = tokenized_dataset
        self.activations = activations  # (N, seq_len, n_selected)
        self.latent_indices = latent_indices  # original feature indices
        self.tokenizer_wrapper = tokenizer_wrapper
        self.client = OpenAI(api_key=cfg.api_key)

    def gather_data(self) -> tuple[dict[int, Examples], dict[int, Examples]]:
        """Gather generation and scoring examples for each latent."""
        cfg = self.cfg
        dataset_size, seq_len = self.tokenized_dataset.shape

        generation_examples = {}
        scoring_examples = {}

        for local_idx, latent in tqdm(
            enumerate(self.latent_indices), desc="Collecting examples for LLM judge"
        ):
            acts = self.activations[:, :, local_idx].float()  # (N, seq_len)

            # Random examples
            rand_indices = torch.stack(
                [
                    torch.randint(0, dataset_size, (cfg.n_random_ex_for_scoring,)),
                    torch.randint(
                        cfg.buffer,
                        seq_len - cfg.buffer,
                        (cfg.n_random_ex_for_scoring,),
                    ),
                ],
                dim=-1,
            )
            rand_toks = index_with_buffer(
                self.tokenized_dataset, rand_indices, buffer=cfg.buffer
            )

            # Top-scoring examples
            top_indices = get_k_largest_indices(
                acts,
                k=cfg.n_top_ex,
                buffer=cfg.buffer,
                no_overlap=cfg.no_overlap,
            )
            top_toks = index_with_buffer(
                self.tokenized_dataset, top_indices, buffer=cfg.buffer
            )
            top_values = index_with_buffer(acts, top_indices, buffer=cfg.buffer)
            act_threshold = cfg.act_threshold_frac * top_values.max().item()

            # Importance-weighted examples (disjoint from top)
            threshold = top_values[:, cfg.buffer].min().item()
            acts_thresholded = torch.where(acts >= threshold, 0.0, acts)
            if acts_thresholded[:, cfg.buffer : -cfg.buffer].max() < 1e-6:
                continue  # dead feature
            iw_indices = get_iw_sample_indices(
                acts_thresholded, k=cfg.n_iw_sampled_ex, buffer=cfg.buffer
            )
            iw_toks = index_with_buffer(
                self.tokenized_dataset, iw_indices, buffer=cfg.buffer
            )
            iw_values = index_with_buffer(acts, iw_indices, buffer=cfg.buffer)

            # Split into generation vs scoring
            rand_top_split = torch.randperm(cfg.n_top_ex)
            top_gen_idx = rand_top_split[: cfg.n_top_ex_for_generation]
            top_score_idx = rand_top_split[cfg.n_top_ex_for_generation :]

            rand_iw_split = torch.randperm(cfg.n_iw_sampled_ex)
            iw_gen_idx = rand_iw_split[: cfg.n_iw_sampled_ex_for_generation]
            iw_score_idx = rand_iw_split[cfg.n_iw_sampled_ex_for_generation :]

            def create_examples(
                all_toks: torch.Tensor, all_acts: torch.Tensor | None = None
            ) -> list[Example]:
                if all_acts is None:
                    all_acts = torch.zeros_like(all_toks).float()
                return [
                    Example(
                        toks=toks,
                        acts=a,
                        act_threshold=act_threshold,
                        tokenizer_wrapper=self.tokenizer_wrapper,
                    )
                    for toks, a in zip(all_toks.tolist(), all_acts.tolist())
                ]

            generation_examples[latent] = Examples(
                create_examples(top_toks[top_gen_idx], top_values[top_gen_idx])
                + create_examples(iw_toks[iw_gen_idx], iw_values[iw_gen_idx]),
            )
            scoring_examples[latent] = Examples(
                create_examples(top_toks[top_score_idx], top_values[top_score_idx])
                + create_examples(iw_toks[iw_score_idx], iw_values[iw_score_idx])
                + create_examples(rand_toks),
                shuffle=True,
            )

        return generation_examples, scoring_examples

    def get_generation_prompts(self, generation_examples: Examples) -> list[dict]:
        examples_as_str = "\n".join(
            f"{i + 1}. {ex.to_str(mark_toks=True)}"
            for i, ex in enumerate(generation_examples)
        )

        system_prompt = """We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the neuron activates, in order from most strongly activating to least strongly activating. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Try not to be overly specific in your explanation. Note that some neurons will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the neuron to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words."""

        if self.cfg.use_demos_in_explanation:
            system_prompt += """ Some examples: "This neuron activates on the word 'knows' in rhetorical questions", and "This neuron activates on verbs related to decision-making and preferences", and "This neuron activates on the substring 'Ent' at the start of words", and "This neuron activates on text about government economic policy"."""
        else:
            system_prompt += """Your response should be in the form "This neuron activates on..."."""

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"The activating documents are given below:\n\n{examples_as_str}",
            },
        ]

    def get_scoring_prompts(
        self, explanation: str, scoring_examples: Examples
    ) -> list[dict]:
        examples_as_str = "\n".join(
            f"{i + 1}. {ex.to_str(mark_toks=False)}"
            for i, ex in enumerate(scoring_examples)
        )
        n_ex = self.cfg.n_ex_for_scoring
        n_correct = self.cfg.n_correct_for_scoring
        example_response = sorted(random.sample(range(1, 1 + n_ex), k=n_correct))
        example_response_str = ", ".join(str(i) for i in example_response)

        system_prompt = f"""We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this neuron activates for, and then be shown {n_ex} example sequences in random order. You will have to return a comma-separated list of the examples where you think the neuron should activate at least once, on ANY of the words or substrings in the document. For example, your response might look like "{example_response_str}". Try not to be overly specific in your interpretation of the explanation. If you think there are no examples where the neuron will activate, you should just respond with "None". You should include nothing else in your response other than comma-separated numbers or the word "None" - this is important."""

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Here is the explanation: this neuron fires on {explanation}.\n\nHere are the examples:\n\n{examples_as_str}",
            },
        ]

    def get_api_response(
        self, messages: list[dict], max_tokens: int
    ) -> tuple[str, str]:
        max_retries = 8
        for attempt in range(max_retries):
            try:
                result = self.client.chat.completions.create(
                    model=self.cfg.openai_model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    reasoning_effort="low",
                )
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 2 ** attempt + random.random()
                    print(f"\nRate limited, retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Failed after {max_retries} retries")

        response = result.choices[0].message.content.strip()
        logs = tabulate(
            [
                m.values()
                for m in messages + [{"role": "assistant", "content": response}]
            ],
            tablefmt="simple_grid",
            maxcolwidths=[None, 120],
        )
        return response, logs

    @staticmethod
    def parse_explanation(explanation: str) -> str:
        return explanation.split("activates on")[-1].rstrip(".").strip()

    @staticmethod
    def parse_predictions(predictions: str) -> list[int] | None:
        predictions_split = (
            predictions.strip()
            .rstrip(".")
            .replace("and", ",")
            .replace("None", "")
            .split(",")
        )
        predictions_list = [i.strip() for i in predictions_split if i.strip() != ""]
        if not predictions_list:
            return []
        if not all(p.isdigit() for p in predictions_list):
            return None
        return [int(p) for p in predictions_list]

    @staticmethod
    def score_predictions(predictions: list[int], scoring_examples: Examples) -> float:
        classifications = [
            i in predictions for i in range(1, len(scoring_examples) + 1)
        ]
        correct = [ex.is_active for ex in scoring_examples]
        return sum(c == cc for c, cc in zip(classifications, correct)) / len(
            classifications
        )

    async def run_single_feature(
        self,
        executor: ThreadPoolExecutor,
        latent: int,
        gen_examples: Examples,
        score_examples: Examples,
    ) -> dict[str, Any] | None:
        # Generation phase
        gen_prompts = self.get_generation_prompts(gen_examples)
        explanation_raw, gen_logs = await asyncio.get_event_loop().run_in_executor(
            executor, self.get_api_response, gen_prompts, self.cfg.max_tokens_in_explanation
        )
        explanation = self.parse_explanation(explanation_raw)
        results: dict[str, Any] = {
            "latent": latent,
            "explanation": explanation,
            "logs": f"Generation phase\n{gen_logs}\n{gen_examples.display()}",
        }

        # Scoring phase
        score_prompts = self.get_scoring_prompts(explanation, score_examples)
        predictions_raw, score_logs = await asyncio.get_event_loop().run_in_executor(
            executor,
            self.get_api_response,
            score_prompts,
            self.cfg.max_tokens_in_prediction,
        )
        predictions = self.parse_predictions(predictions_raw)
        if predictions is None:
            return None
        score = self.score_predictions(predictions, score_examples)
        results |= {
            "predictions": predictions,
            "correct seqs": [
                i for i, ex in enumerate(score_examples, start=1) if ex.is_active
            ],
            "score": score,
            "logs": results["logs"]
            + f"\nScoring phase\n{score_logs}\n{score_examples.display(predictions)}",
        }
        return results

    async def run(self) -> dict[int, dict[str, Any]]:
        generation_examples, scoring_examples = self.gather_data()
        latents_with_data = sorted(generation_examples.keys())
        n_dead = len(self.latent_indices) - len(latents_with_data)
        if n_dead > 0:
            print(
                f"Found data for {len(latents_with_data)}/{len(self.latent_indices)} alive latents; {n_dead} dead"
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            tasks = [
                self.run_single_feature(
                    executor,
                    latent,
                    generation_examples[latent],
                    scoring_examples[latent],
                )
                for latent in latents_with_data
            ]
            results = {}
            for future in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Calling API (gen & scoring)",
            ):
                result = await future
                if result:
                    results[result["latent"]] = result

        return results


# =============================================================================
# Per-method evaluation
# =============================================================================


def run_method(
    cfg: AutoInterpConfig,
    method_name: str,
    tokenized_dataset: torch.Tensor,
    activations: torch.Tensor,
    latent_indices: list[int],
    tokenizer_wrapper: TokenizerWrapper,
) -> dict[str, Any]:
    """Run autointerp for one method and return results."""
    print(f"\n{'='*60}")
    print(f"Running autointerp: {method_name}")
    print(f"  {len(latent_indices)} features selected")
    print(f"{'='*60}")

    runner = AutoInterpRunner(
        cfg=cfg,
        tokenized_dataset=tokenized_dataset,
        activations=activations,
        latent_indices=latent_indices,
        tokenizer_wrapper=tokenizer_wrapper,
    )
    results = asyncio.run(runner.run())

    all_scores = [r["score"] for r in results.values()]
    if all_scores:
        scores_t = torch.tensor(all_scores)
        mean_score = scores_t.mean().item()
        std_dev = scores_t.std().item() if len(all_scores) > 1 else 0.0
    else:
        mean_score, std_dev = 0.0, 0.0

    print(f"\n{method_name} results:")
    print(f"  Autointerp score: {mean_score:.4f} +/- {std_dev:.4f}")
    print(f"  Features evaluated: {len(all_scores)}")

    # Print summary table
    headers = ["latent", "explanation", "score"]
    table = [
        [results[lat]["latent"], results[lat]["explanation"], results[lat]["score"]]
        for lat in sorted(results.keys())
    ]
    print(
        tabulate(table, headers=headers, tablefmt="simple_outline", floatfmt=".3f")
    )

    return {
        "method": method_name,
        "autointerp_score": mean_score,
        "std_dev": std_dev,
        "n_features": len(all_scores),
        "per_feature": {
            lat: {k: v for k, v in r.items() if k != "logs"}
            for lat, r in results.items()
        },
    }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SAEBench-style autointerp for Transcoder and SPD"
    )
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument(
        "--transcoder_path",
        type=str,
        default="checkpoints/4096_batchtopk_k24_0.0003_final",
    )
    parser.add_argument("--spd_run", type=str, default="goodfire/spd/s-275c8f21")
    parser.add_argument("--n_latents", type=int, default=100)
    parser.add_argument("--total_tokens", type=int, default=2_000_000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--save_path", type=str, default="experiments/exp_008_interp_pile/output/autointerp_results.json"
    )
    parser.add_argument("--skip_transcoder", action="store_true")
    parser.add_argument("--skip_spd_cfc", action="store_true")
    parser.add_argument("--skip_spd_down", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Provide OpenAI API key via --api_key or OPENAI_API_KEY env var"
        )

    cfg = AutoInterpConfig(
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

        # We need the base model for the transcoder
        print("Loading SPD model (to get base model)...")
        spd_model, raw_config = load_spd_model(cfg.spd_run)
        spd_model.to(DEVICE)
        base_model = spd_model.target_model
        base_model.eval()

        # Pass 1: compute sparsity
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
            if isinstance(transcoder, (TopKTranscoder, BatchTopKTranscoder)):
                return F.relu(x_enc @ transcoder.W_enc)
            else:
                return F.relu(x_enc @ transcoder.W_enc + transcoder.b_enc)

        firing_count = compute_sparsity(
            tc_acts_fn, tokenized_dataset, cfg.batch_size, dict_size
        )
        tc_latents = select_alive_latents(
            firing_count, cfg.n_latents, cfg.dead_latent_threshold, cfg.random_seed
        )
        print(f"  Selected {len(tc_latents)} alive transcoder features")

        # Pass 2: collect activations for selected latents
        tc_acts = collect_transcoder_activations(
            base_model, transcoder, tokenized_dataset, cfg.batch_size, tc_latents
        )

        all_results["transcoder"] = run_method(
            cfg, "transcoder", tokenized_dataset, tc_acts, tc_latents, tokenizer_wrapper
        )

        # Free transcoder memory
        del transcoder, tc_acts
        torch.cuda.empty_cache()
    else:
        # Still need SPD model for the SPD evals
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
        if not cfg.eval_transcoder:
            # SPD model already loaded above
            pass

        # Get component counts
        n_components = {}
        for name in spd_modules_to_eval:
            n_components[name] = spd_model.components[name].C
            print(f"  {name}: {n_components[name]} components")

        # Pass 1: compute sparsity for both CI thresholds simultaneously
        print("Computing SPD CI sparsity (thresholds: {})...".format(ci_thresholds))
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

        # Select alive latents per module per threshold
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

        # Union of latents across thresholds for shared activation collection
        union_latents = {}
        for name in spd_modules_to_eval:
            all_lats = set()
            for thresh in ci_thresholds:
                all_lats.update(spd_latents[thresh][name])
            union_latents[name] = sorted(all_lats)
            print(f"  {name}: {len(union_latents[name])} unique latents across thresholds")

        # Pass 2: collect raw CI activations for union of latents
        spd_acts_raw = collect_spd_ci_activations(
            spd_model,
            tokenized_dataset,
            spd_modules_to_eval,
            cfg.batch_size,
            selected_latents=union_latents,
        )

        # Run autointerp for each threshold and module
        module_labels = {cfc_name: "spd_cfc", down_name: "spd_down_proj"}
        for thresh in ci_thresholds:
            for name in spd_modules_to_eval:
                base_label = module_labels[name]
                label = f"{base_label}_ci>{thresh}"

                # Map threshold-specific latents to indices in the union array
                union_list = union_latents[name]
                thresh_lats = spd_latents[thresh][name]
                local_indices = [union_list.index(lat) for lat in thresh_lats]

                # Slice and apply CI threshold (zero out values below threshold)
                acts = spd_acts_raw[name][:, :, local_indices]
                if thresh > 0:
                    acts = acts.clone()
                    acts[acts <= thresh] = 0.0

                all_results[label] = run_method(
                    cfg,
                    label,
                    tokenized_dataset,
                    acts,
                    thresh_lats,
                    tokenizer_wrapper,
                )

    # ---------- Summary ----------
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary_table = []
    for method, res in all_results.items():
        summary_table.append(
            [method, f"{res['autointerp_score']:.4f}", f"{res['std_dev']:.4f}", res["n_features"]]
        )
    print(
        tabulate(
            summary_table,
            headers=["Method", "Score", "Std Dev", "N Features"],
            tablefmt="simple_outline",
        )
    )
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    save_data = {
        "config": {
            "total_tokens": cfg.total_tokens,
            "seq_len": cfg.seq_len,
            "n_latents": cfg.n_latents,
            "dead_latent_threshold": cfg.dead_latent_threshold,
            "random_seed": cfg.random_seed,
            "openai_model": cfg.openai_model,
            "timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
    }
    os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)
    with open(cfg.save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Results saved to {cfg.save_path}")


if __name__ == "__main__":
    main()
