# %%
"""Train TopK and BatchTopK transcoders in parallel on MLP layer 8 of GPT-2 small."""

from activation_store import TranscoderActivationsStore
from base import BatchTopK, TopK
from config import TranscoderConfig
from training import train_encoder_group
from transformer_lens import HookedTransformer

# Shared config values (input_size/output_size auto-detected from model)
shared = dict(
    dict_size=768,
    model_name="gpt2-small",
    input_site="resid_mid",  # Input to MLP
    output_site="mlp_out",   # Output of MLP
    input_layer=8,
    output_layer=8,
    top_k=32,
    l1_coeff=0.0,
    batch_size=64,
    num_tokens=int(1e8),  # 100M tokens
    lr=3e-4,
    wandb_project="gpt2_transcoder",
    device="cuda",
)

# Create configs for each encoder type
topk_cfg = TranscoderConfig(encoder_type="topk", **shared)
batchtopk_cfg = TranscoderConfig(encoder_type="batchtopk", **shared)

# Load model
model = HookedTransformer.from_pretrained(topk_cfg.model_name).to(topk_cfg.dtype).to(topk_cfg.device)

# Create transcoders and activation store
topk_transcoder = TopK(topk_cfg)
batchtopk_transcoder = BatchTopK(batchtopk_cfg)
activation_store = TranscoderActivationsStore(model, topk_cfg)

# Train both in parallel
print("Training transcoders:")
print(f"  [0] TopK: {topk_cfg.name}")
print(f"  [1] BatchTopK: {batchtopk_cfg.name}")
print(f"  Input: {topk_cfg.input_hook_point}")
print(f"  Output: {topk_cfg.output_hook_point}")
print(f"  Dict size: {topk_cfg.dict_size}, Top-k: {topk_cfg.top_k}")

train_encoder_group(
    [topk_transcoder, batchtopk_transcoder],
    activation_store,
    model,
    [topk_cfg, batchtopk_cfg],
)
