# %%
"""Train TopK and BatchTopK transcoders in parallel on MLP layer 8 of GPT-2 small."""

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

from activation_store import ActivationsStore, DataConfig
from base import BatchTopK, TopK
from config import EncoderConfig
from training import train_encoder_group

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load GPT-2 small
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

layer = 8
input_size = model.config.n_embd  # 768
output_size = model.config.n_embd  # 768

# Shared config values
shared = dict(
    input_size=input_size,
    output_size=output_size,
    dict_size=768,
    top_k=32,
    l1_coeff=0.0,
    batch_size=64,
    num_tokens=int(1e8),  # 100M tokens
    lr=3e-4,
    wandb_project="gpt2_transcoder",
    device=device,
)

topk_cfg = EncoderConfig(encoder_type="topk", **shared)
batchtopk_cfg = EncoderConfig(encoder_type="batchtopk", **shared)

# Hook into MLP input (layer norm before MLP) and MLP output
input_module = model.transformer.h[layer].ln_2
output_module = model.transformer.h[layer].mlp

data_config = DataConfig(
    dataset_name="Skylion007/openwebtext",
    tokenizer=tokenizer,
    text_column="text",
    seq_len=1024,
    model_batch_size=16,
    train_batch_size=shared["batch_size"],
    num_batches_in_buffer=10,
    device=device,
)

activation_store = ActivationsStore(
    model=model,
    input_module=input_module,
    output_module=output_module,
    data_config=data_config,
    input_size=input_size,
    output_size=output_size,
)

topk_transcoder = TopK(topk_cfg)
batchtopk_transcoder = BatchTopK(batchtopk_cfg)

print("Training transcoders:")
print(f"  [0] TopK: {topk_cfg.name}")
print(f"  [1] BatchTopK: {batchtopk_cfg.name}")
print(f"  Layer: {layer}")
print(f"  Dict size: {topk_cfg.dict_size}, Top-k: {topk_cfg.top_k}")

train_encoder_group(
    [topk_transcoder, batchtopk_transcoder],
    activation_store,
    [topk_cfg, batchtopk_cfg],
)
