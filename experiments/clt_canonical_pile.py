"""Train a single Cross-Layer Transcoder (CLT) on LlamaSimpleMLP layers 0–3.

Instead of training 4 separate transcoders (one per layer), this trains a single
CLT that encodes from layer 0's pre-MLP residual stream and decodes to all 4
layers' MLP outputs simultaneously.

Matches transcoder_canonical_pile.py setup:
- Same base model (wandb:goodfire/spd/t-32d1bb3b)
- Same dataset (danbraunai/pile-uncopyrighted-tok, pre-tokenized, streaming)
- Same sequence length (512) and model batch size (64 sequences)

Usage:
    python experiments/clt_canonical_pile.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from transformers import AutoTokenizer

from activation_store import ActivationsStore, DataConfig
from base import BatchTopK
from config import EncoderConfig
from training import train_encoder

WANDB_MODEL_PATH = "wandb:goodfire/spd/t-32d1bb3b"
LAYERS = [0, 1, 2, 3]
TOP_K = 64
DICT_SIZE = 4096


def compute_loss_llama(model, tokenizer, input_ids, attention_mask):
    """Compute CE loss for LlamaSimpleMLP (next-token prediction)."""
    targets = input_ids[:, 1:].contiguous()
    logits, _ = model(input_ids)
    logits = logits[:, :-1].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).item()


def main():
    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

    device = f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else "cpu"

    print(f"Loading LlamaSimpleMLP from {WANDB_MODEL_PATH}...")
    model = LlamaSimpleMLP.from_pretrained(WANDB_MODEL_PATH)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    input_size = model.config.n_embd
    output_size = model.config.n_embd

    # Encoder reads from layer 0's pre-MLP residual stream
    input_module = model.h[LAYERS[0]].rms_2
    # Decoder targets all layers' MLP outputs
    output_modules = [model.h[layer].mlp for layer in LAYERS]

    cfg = EncoderConfig(
        input_size=input_size,
        output_size=output_size,
        dict_size=DICT_SIZE,
        encoder_type="batchtopk",
        top_k=TOP_K,
        l1_coeff=0.0,
        batch_size=4096,
        num_tokens=int(5e8),
        lr=3e-4,
        num_output_layers=len(LAYERS),
        wandb_project="pile_clt",
        device=device,
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
        buffer_on_cpu=False,
        device=device,
    )

    activation_store = ActivationsStore(
        model=model,
        input_module=input_module,
        output_module=output_modules,
        data_config=data_config,
        input_size=input_size,
        output_size=output_size,
    )

    clt = BatchTopK(cfg)

    print(f"Training CLT: {cfg.name}")
    print(f"  Model: LlamaSimpleMLP (t-32d1bb3b)")
    print(f"  Encoder input: layer {LAYERS[0]} rms_2")
    print(f"  Decoder targets: layers {LAYERS} MLP outputs")
    print(f"  Dict size: {DICT_SIZE}, Top-k: {TOP_K}")
    print(f"  Num output layers: {len(LAYERS)}")
    print(f"  Steps: {cfg.num_tokens // cfg.batch_size:,}")
    print(f"  Dataset: danbraunai/pile-uncopyrighted-tok")

    train_encoder(clt, activation_store, cfg, compute_loss_fn=compute_loss_llama)
    print("Training complete.")


if __name__ == "__main__":
    main()
