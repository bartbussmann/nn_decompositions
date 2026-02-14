"""Collect top-K activating examples for Transcoder features and SPD components.

Runs both models over OpenWebText and saves the top activating examples per
feature/component to disk. The cached results are loaded by feature_dashboard.py.

Usage:
    python analysis/collect_activations.py [--n_batches 50] [--top_k 20]
"""

import argparse
import heapq
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import BatchTopK, JumpReLUEncoder, TopK, Vanilla
from config import EncoderConfig

from spd.models.component_model import ComponentModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 8

ENCODER_CLASSES = {
    "vanilla": Vanilla,
    "topk": TopK,
    "batchtopk": BatchTopK,
    "jumprelu": JumpReLUEncoder,
}


def _make_example(
    token_ids: list[int], activation_values: list[float], max_position: int, max_value: float
) -> dict:
    return {
        "token_ids": token_ids,
        "activation_values": activation_values,
        "max_position": max_position,
        "max_value": max_value,
    }


def load_transcoder(checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict = json.load(f)
    dtype_str = cfg_dict.get("dtype", "torch.float32")
    cfg_dict["dtype"] = getattr(torch, dtype_str.replace("torch.", ""))
    cfg = EncoderConfig(**cfg_dict)
    encoder = ENCODER_CLASSES[cfg.encoder_type](cfg)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=DEVICE))
    encoder.eval()
    return encoder


def load_spd(checkpoint_path: str) -> ComponentModel:
    path = Path(checkpoint_path)
    if path.is_dir():
        pth_files = sorted(path.glob("model_*.pth"))
        assert pth_files, f"No model_*.pth files found in {path}"
        path = pth_files[-1]
    model = ComponentModel.from_pretrained(str(path))
    model.to(DEVICE)
    return model


def get_eval_batches(tokenizer, n_batches: int, batch_size: int, seq_len: int):
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)

    batches = []
    for _ in range(n_batches):
        texts = [next(data_iter)["text"] for _ in range(batch_size)]
        tokens = tokenizer(
            texts, truncation=True, max_length=seq_len, padding="max_length", return_tensors="pt"
        )
        batches.append((tokens["input_ids"].to(DEVICE), tokens["attention_mask"].to(DEVICE)))
    return batches


def _push_to_heap(heap: list, example: dict, max_size: int) -> None:
    """Push example onto a min-heap of size max_size, keyed by max_value."""
    val = example["max_value"]
    if len(heap) < max_size:
        heapq.heappush(heap, (val, id(example), example))
    elif val > heap[0][0]:
        heapq.heapreplace(heap, (val, id(example), example))


@torch.no_grad()
def collect_transcoder_activations(
    transcoder,
    gpt2_model: GPT2LMHeadModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    top_k: int,
) -> dict[int, list[dict]]:
    """Collect top-K activating examples per transcoder feature."""
    dict_size = transcoder.cfg.dict_size
    feature_heaps: dict[int, list] = {i: [] for i in range(dict_size)}
    ln2 = gpt2_model.transformer.h[LAYER].ln_2

    use_pre_enc_bias = transcoder.cfg.pre_enc_bias and transcoder.input_size == transcoder.output_size

    for input_ids, attention_mask in tqdm(batches, desc="Transcoder"):
        # Capture MLP input via hook
        captured = {}

        def _hook(_mod, _inp, out):
            captured["mlp_in"] = out.detach()

        handle = ln2.register_forward_hook(_hook)
        gpt2_model(input_ids, attention_mask=attention_mask)
        handle.remove()

        mlp_in = captured["mlp_in"]  # (B, S, 768)

        # Compute feature activations
        x_enc = mlp_in - transcoder.b_dec if use_pre_enc_bias else mlp_in
        if isinstance(transcoder, (TopK, BatchTopK)):
            acts = F.relu(x_enc @ transcoder.W_enc)  # (B, S, dict_size)
        else:
            acts = F.relu(x_enc @ transcoder.W_enc + transcoder.b_enc)

        # Vectorized: find max activation per feature across sequence
        max_vals, max_positions = acts.max(dim=1)  # (B, dict_size)

        # Only process features that are active
        batch_indices, feat_indices = torch.where(max_vals > 0)

        for b_idx, f_idx in zip(batch_indices.tolist(), feat_indices.tolist()):
            max_val = max_vals[b_idx, f_idx].item()
            heap = feature_heaps[f_idx]
            if len(heap) < top_k or max_val > heap[0][0]:
                example = _make_example(
                    token_ids=input_ids[b_idx].tolist(),
                    activation_values=acts[b_idx, :, f_idx].tolist(),
                    max_position=max_positions[b_idx, f_idx].item(),
                    max_value=max_val,
                )
                _push_to_heap(heap, example, top_k)

    # Extract sorted examples from heaps
    return {
        idx: [ex for _, _, ex in sorted(heap, reverse=True)]
        for idx, heap in feature_heaps.items()
        if heap
    }


@torch.no_grad()
def collect_spd_activations(
    spd_model: ComponentModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    module_name: str,
    top_k: int,
) -> dict[int, list[dict]]:
    """Collect top-K activating examples per SPD component by pre-sigmoid CI."""
    n_components = spd_model.module_to_c[module_name]
    component_heaps: dict[int, list] = {i: [] for i in range(n_components)}

    for input_ids, attention_mask in tqdm(batches, desc=f"SPD {module_name}"):
        out = spd_model(input_ids, attention_mask=attention_mask, cache_type="input")
        ci = spd_model.calc_causal_importances(
            pre_weight_acts=out.cache, sampling="none", detach_inputs=False
        )
        ci_pre = ci.pre_sigmoid[module_name]  # (B, S, C)
        ci_post = ci.lower_leaky[module_name]  # (B, S, C)

        # Sort/rank by pre-sigmoid, but store both
        max_vals, max_positions = ci_pre.max(dim=1)  # (B, C)
        batch_indices, comp_indices = torch.where(max_vals > 0)

        for b_idx, c_idx in zip(batch_indices.tolist(), comp_indices.tolist()):
            max_val = max_vals[b_idx, c_idx].item()
            heap = component_heaps[c_idx]
            if len(heap) < top_k or max_val > heap[0][0]:
                example = _make_example(
                    token_ids=input_ids[b_idx].tolist(),
                    activation_values=ci_post[b_idx, :, c_idx].tolist(),
                    max_position=max_positions[b_idx, c_idx].item(),
                    max_value=max_val,
                )
                _push_to_heap(heap, example, top_k)

    return {
        idx: [ex for _, _, ex in sorted(heap, reverse=True)]
        for idx, heap in component_heaps.items()
        if heap
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--transcoder_path", type=str, default="checkpoints/6144_batchtopk_k32_0.0003_final"
    )
    parser.add_argument("--spd_path", type=str, default="checkpoints/spd_s-694dd066")
    parser.add_argument(
        "--cache_dir", type=str, default=str(Path(__file__).parent / "activation_cache")
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer and GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(DEVICE).eval()

    print(f"Loading eval batches ({args.n_batches} batches)...")
    batches = get_eval_batches(tokenizer, args.n_batches, args.batch_size, args.seq_len)
    print(f"  Total tokens: {args.n_batches * args.batch_size * args.seq_len:,}")

    # Transcoder
    print("Loading transcoder...")
    transcoder = load_transcoder(args.transcoder_path)
    transcoder.to(DEVICE)
    print("Collecting transcoder activations...")
    tc_examples = collect_transcoder_activations(transcoder, gpt2_model, batches, args.top_k)
    print(f"  Features with examples: {len(tc_examples)}/{transcoder.cfg.dict_size}")
    del transcoder
    torch.cuda.empty_cache()

    # SPD
    print("Loading SPD model...")
    spd_model = load_spd(args.spd_path)

    module_names = list(spd_model.module_to_c.keys())
    cfc_name = [m for m in module_names if "c_fc" in m][0]
    cproj_name = [m for m in module_names if "c_proj" in m][0]

    print(f"Collecting SPD c_fc activations ({cfc_name})...")
    spd_cfc_examples = collect_spd_activations(spd_model, batches, cfc_name, args.top_k)
    print(f"  Components with examples: {len(spd_cfc_examples)}/{spd_model.module_to_c[cfc_name]}")

    print(f"Collecting SPD c_proj activations ({cproj_name})...")
    spd_cproj_examples = collect_spd_activations(spd_model, batches, cproj_name, args.top_k)
    print(
        f"  Components with examples: {len(spd_cproj_examples)}/{spd_model.module_to_c[cproj_name]}"
    )

    # Save
    cache_path = cache_dir / "activation_examples.pt"
    print(f"Saving cache to {cache_path}...")
    torch.save(
        {
            "transcoder": tc_examples,
            "spd_cfc": spd_cfc_examples,
            "spd_cproj": spd_cproj_examples,
            "config": {
                "n_batches": args.n_batches,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "top_k": args.top_k,
                "transcoder_path": args.transcoder_path,
                "spd_path": args.spd_path,
            },
        },
        cache_path,
    )
    print("Done!")


if __name__ == "__main__":
    main()
