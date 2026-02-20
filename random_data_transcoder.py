# %%
"""Train a transcoder on random data to see if it learns real model mechanisms.

Two modes:
  random_tokens       — uniformly sampled tokens (BOS prefix), run through the
                        full model to get activations. The model still produces
                        structured activations, but can't memorize datapoints.
  random_activations  — random vectors in activation space (per-dimension
                        Gaussian calibrated to real stats), passed directly
                        through the MLP. The transcoder sees inputs that would
                        never arise from natural text.

Both modes evaluate on real text (OpenWebText) to measure how well the
transcoder generalises.

Usage:
    python random_data_transcoder.py                    # defaults to random_tokens
    python random_data_transcoder.py random_tokens
    python random_data_transcoder.py random_activations
"""

import sys

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

from activation_store import (
    ActivationsStore,
    RandomActivationStore,
    RandomTokenActivationStore,
    DataConfig,
)
from transcoder import BatchTopKTranscoder
from config import EncoderConfig
from training import train_encoder

mode = sys.argv[1] if len(sys.argv) > 1 else "random_tokens"
assert mode in ("random_tokens", "random_activations"), (
    f"Unknown mode '{mode}'. Choose 'random_tokens' or 'random_activations'."
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load GPT-2 small
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

layer = 8
input_size = model.config.n_embd  # 768
output_size = model.config.n_embd  # 768

cfg = EncoderConfig(
    input_size=input_size,
    output_size=output_size,
    dict_size=768 * 8,
    encoder_type="batchtopk",
    top_k=32,
    l1_coeff=0.0,
    batch_size=4096,
    num_tokens=int(5e8),  # 500M tokens
    lr=3e-4,
    wandb_project="random_data_transcoder",
    device=device,
)

input_module = model.transformer.h[layer].ln_2
output_module = model.transformer.h[layer].mlp

shared_data_kwargs = dict(
    tokenizer=tokenizer,
    seq_len=256,
    model_batch_size=32,
    train_batch_size=cfg.batch_size,
    num_batches_in_buffer=10,
    device=device,
)

# Real data store — used for eval (and for calibration in random_activations mode)
real_data_config = DataConfig(dataset_name="Skylion007/openwebtext", **shared_data_kwargs)
real_store = ActivationsStore(
    model=model,
    input_module=input_module,
    output_module=output_module,
    data_config=real_data_config,
    input_size=input_size,
    output_size=output_size,
)

if mode == "random_tokens":
    random_data_config = DataConfig(dataset_name="unused", **shared_data_kwargs)
    train_store = RandomTokenActivationStore(
        model=model,
        input_module=input_module,
        output_module=output_module,
        data_config=random_data_config,
        input_size=input_size,
        output_size=output_size,
    )
elif mode == "random_activations":
    train_store = RandomActivationStore(
        output_module=output_module,
        input_size=input_size,
        output_size=output_size,
        batch_size=cfg.batch_size,
        device=device,
    )
    print("Calibrating random activation distribution from real data...")
    train_store.calibrate(real_store, n_batches=10)

transcoder = BatchTopKTranscoder(cfg)

print(f"\nRandom Data Transcoder Experiment — mode: {mode}")
print(f"  Layer: {layer}")
print(f"  Dict size: {cfg.dict_size}, Top-k: {cfg.top_k}")
print(f"  Training on: {mode}")
print(f"  Evaluating on: real text (OpenWebText)")

train_encoder(
    transcoder,
    train_store,
    cfg,
    eval_stores={"real_data": real_store},
)
