"""Train a BatchTopK transcoder on LlamaSimpleMLP layer 3 MLP.

Matches the SPD decomposition run s-275c8f21 as closely as possible:
- Same base model (wandb:goodfire/spd/t-32d1bb3b)
- Same dataset (danbraunai/pile-uncopyrighted-tok, pre-tokenized, streaming)
- Same sequence length (512) and batch size (64 sequences)
- Same number of steps (400k)

Usage:
    python transcoder_canonical_pile.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# SPD codebase for model loading
sys.path.insert(0, str(Path("/workspace/spd")))

from transformers import AutoTokenizer

from activation_store import ActivationsStore, DataConfig
from base import BatchTopK
from config import EncoderConfig
from training import train_encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 3
WANDB_MODEL_PATH = "wandb:goodfire/spd/t-32d1bb3b"


def load_model():
    """Load the LlamaSimpleMLP base model from wandb."""
    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

    print(f"Loading LlamaSimpleMLP from {WANDB_MODEL_PATH}...")
    model = LlamaSimpleMLP.from_pretrained(WANDB_MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return model


def compute_loss_llama(model, tokenizer, input_ids, attention_mask):
    """Compute CE loss for LlamaSimpleMLP (next-token prediction)."""
    targets = input_ids[:, 1:].contiguous()
    logits, _ = model(input_ids)
    logits = logits[:, :-1].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).item()


def main():
    model = load_model()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    input_size = model.config.n_embd   # 768
    output_size = model.config.n_embd  # 768

    # Hook into RMSNorm before MLP (input) and MLP (output) at layer 3
    input_module = model.h[LAYER].rms_2
    output_module = model.h[LAYER].mlp

    cfg = EncoderConfig(
        input_size=input_size,
        output_size=output_size,
        dict_size=4096,
        encoder_type="batchtopk",
        top_k=24,
        l1_coeff=0.0,
        batch_size=4096,
        num_tokens=400_000 * 4096,  # 400k steps * 4096 tokens/step
        lr=3e-4,
        wandb_project="pile_transcoder",
        device=DEVICE,
    )

    data_config = DataConfig(
        dataset_name="danbraunai/pile-uncopyrighted-tok",
        tokenizer=tokenizer,
        is_tokenized=True,
        token_column="input_ids",
        seq_len=512,
        model_batch_size=64,
        train_batch_size=cfg.batch_size,
        num_batches_in_buffer=10,
        device=DEVICE,
    )

    activation_store = ActivationsStore(
        model=model,
        input_module=input_module,
        output_module=output_module,
        data_config=data_config,
        input_size=input_size,
        output_size=output_size,
    )

    transcoder = BatchTopK(cfg)

    print(f"Training transcoder: {cfg.name}")
    print(f"  Model: LlamaSimpleMLP (t-32d1bb3b)")
    print(f"  Layer: {LAYER}")
    print(f"  Dict size: {cfg.dict_size}, Top-k: {cfg.top_k}")
    print(f"  Steps: {cfg.num_tokens // cfg.batch_size:,}")
    print(f"  Dataset: danbraunai/pile-uncopyrighted-tok")

    train_encoder(transcoder, activation_store, cfg, compute_loss_fn=compute_loss_llama)


if __name__ == "__main__":
    main()
