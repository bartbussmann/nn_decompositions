"""Train a transcoder on the SimpleStories 4-layer LlamaSimple model (layer 3 MLP).

This script trains a TopK transcoder to decompose the MLP at layer 3 of the
canonical SimpleStories model used in SPD experiments.

Model: wandb:goodfire/spd/runs/erq48r3w (LlamaSimple 4-layer, 1.25M params)
Architecture: d_model=128, d_mlp=341, 4 layers, SwiGLU MLP
Target: Layer 3 MLP (input: resid_mid, output: mlp_out)
"""

import sys
from pathlib import Path

import torch
from simple_stories_train.models.llama_simple import LlamaSimple
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from activation_store import ActivationsStore, DataConfig
from transcoder import BatchTopK, TopK
from config import EncoderConfig
from training import train_encoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model_path = "wandb:goodfire/spd/runs/erq48r3w"
print(f"Loading model from {model_path}...")
model = LlamaSimple.from_pretrained(model_path)
model.to(torch.float32).to(device)
model.eval()

layer = 3
input_size = model.config.n_embd   # 128
output_size = model.config.n_embd  # 128

print(f"Model config: d_model={model.config.n_embd}, d_mlp={model.config.n_intermediate}")
print(f"Training transcoder on layer {layer} MLP")

cfg = EncoderConfig(
    input_size=input_size,
    output_size=output_size,
    dict_size=2048,
    encoder_type="topk",
    top_k=32,
    batch_size=512,
    num_tokens=int(1e8),
    lr=3e-4,
    wandb_project="ss_transcoder",
    perf_log_freq=500,
    device=device,
)

tokenizer = AutoTokenizer.from_pretrained("SimpleStories/test-SimpleStories-gpt2-1.25M")

data_config = DataConfig(
    dataset_name="lennart-finke/SimpleStories",
    tokenizer=tokenizer,
    text_column="story",
    seq_len=512,
    model_batch_size=64,
    train_batch_size=cfg.batch_size,
    num_batches_in_buffer=10,
    device=device,
    lowercase=True,
)

# Hook after RMSNorm (MLP input) and after MLP (MLP output)
input_module = model.h[layer].rms_2
output_module = model.h[layer].mlp

activation_store = ActivationsStore(
    model=model,
    input_module=input_module,
    output_module=output_module,
    data_config=data_config,
    input_size=input_size,
    output_size=output_size,
)

print(f"  Input/Output size: {input_size}")
print(f"  Dict size: {cfg.dict_size}")
print(f"  Top-k: {cfg.top_k}")

transcoder = TopK(cfg)

train_encoder(transcoder, activation_store, cfg)
