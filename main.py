# %%
"""Train a TopK transcoder on MLP layer 8 of GPT-2 small."""

from activation_store import TranscoderActivationsStore
from base import TopK
from config import TranscoderConfig
from training import train_encoder
from transformer_lens import HookedTransformer

# Configure transcoder for MLP layer 8
# Input: residual stream before MLP (resid_mid)
# Output: MLP output (mlp_out)
cfg = TranscoderConfig(
    encoder_type="topk",
    input_size=768,
    output_size=768,
    dict_size=768 * 8,  # 6144 features
    model_name="gpt2-small",
    input_site="resid_mid",  # Input to MLP
    output_site="mlp_out",   # Output of MLP
    input_layer=8,
    output_layer=8,
    top_k=32,
    l1_coeff=0.0,
    batch_size=4096,
    num_tokens=int(1e8),  # 100M tokens (smaller for testing)
    lr=3e-4,
    wandb_project="gpt2_transcoder",
    device="cuda",
)

# Load model
model = HookedTransformer.from_pretrained(cfg.model_name).to(cfg.dtype).to(cfg.device)

# Create transcoder and activation store
transcoder = TopK(cfg)
activation_store = TranscoderActivationsStore(model, cfg)

# Train
print(f"Training transcoder: {cfg.name}")
print(f"  Input: {cfg.input_hook_point}")
print(f"  Output: {cfg.output_hook_point}")
print(f"  Dict size: {cfg.dict_size}, Top-k: {cfg.top_k}")
train_encoder(transcoder, activation_store, model, cfg)
