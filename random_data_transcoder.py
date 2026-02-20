# %%
"""Train a transcoder on random (uniformly sampled) tokens.

Hypothesis: since the model can't memorize random data, any features the
transcoder learns must capture real model mechanisms (the MLP's learned
computations) rather than dataset-specific patterns.

Training data: uniformly sampled tokens (with BOS prefix).
Evaluation: performance on both random data and real text (OpenWebText).
"""

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

from activation_store import ActivationsStore, RandomTokenActivationStore, DataConfig
from transcoder import BatchTopKTranscoder
from config import EncoderConfig
from training import train_encoder

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

# Hook into MLP input (layer norm before MLP) and MLP output
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

# Training store: random tokens
random_data_config = DataConfig(dataset_name="unused", **shared_data_kwargs)
random_store = RandomTokenActivationStore(
    model=model,
    input_module=input_module,
    output_module=output_module,
    data_config=random_data_config,
    input_size=input_size,
    output_size=output_size,
)

# Eval store: real text data
real_data_config = DataConfig(dataset_name="Skylion007/openwebtext", **shared_data_kwargs)
real_store = ActivationsStore(
    model=model,
    input_module=input_module,
    output_module=output_module,
    data_config=real_data_config,
    input_size=input_size,
    output_size=output_size,
)

transcoder = BatchTopKTranscoder(cfg)

print("Random Data Transcoder Experiment")
print(f"  Layer: {layer}")
print(f"  Dict size: {cfg.dict_size}, Top-k: {cfg.top_k}")
print(f"  Training on: random tokens (uniform, BOS prefix)")
print(f"  Evaluating on: random tokens + real text (OpenWebText)")

train_encoder(
    transcoder,
    random_store,
    cfg,
    eval_stores={"real_data": real_store},
)
