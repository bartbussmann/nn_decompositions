"""Train independent per-layer transcoders with cascading e2e on LlamaSimpleMLP layers 0–3.

Each layer's reconstruction modifies the residual stream before the next layer's
encoder runs. All 4 transcoders are trained jointly via a single KL loss.

Default model_batch_size is 16 (down from 64) to fit in ~8 GB GPU memory.
Use --model_batch_size to adjust for your GPU.

Usage:
    python experiments/exp_017_train_transcoder_cascading_e2e/transcoder_cascading_e2e_pile.py
    python experiments/exp_017_train_transcoder_cascading_e2e/transcoder_cascading_e2e_pile.py --model_batch_size 32
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from transformers import AutoTokenizer

from nn_decompositions.activation_store import MultiLayerActivationsStore, DataConfig
from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.utils import get_free_gpu
from nn_decompositions.config import EncoderConfig
from nn_decompositions.training import train_encoder_cascading

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


def get_logits_llama(model, input_ids, attention_mask):
    """Extract logits from LlamaSimpleMLP forward pass."""
    logits, _ = model(input_ids)
    return logits


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train transcoders cascading e2e")
    parser.add_argument("--model_batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--dict_size", type=int, default=DICT_SIZE)
    args = parser.parse_args()

    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

    device = get_free_gpu()

    print(f"Loading LlamaSimpleMLP from {WANDB_MODEL_PATH}...")
    model = LlamaSimpleMLP.from_pretrained(WANDB_MODEL_PATH)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    d_model = model.config.n_embd
    seq_len = 512

    input_modules = [model.h[layer].rms_2 for layer in LAYERS]
    output_modules = [model.h[layer].mlp for layer in LAYERS]

    cfgs = []
    encoders = []
    for layer in LAYERS:
        cfg = EncoderConfig(
            input_size=d_model,
            output_size=d_model,
            dict_size=args.dict_size,
            encoder_type="batchtopk",
            top_k=args.top_k,
            l1_coeff=0.0,
            batch_size=4096,
            num_tokens=int(5e8),
            lr=3e-4,
            wandb_project="pile_transcoder_cascading_e2e",
            device=device,
            e2e=True,
            run_name=f"cascading_L{LAYERS[0]}-{LAYERS[-1]}_k{args.top_k}_{args.dict_size}",
        )
        cfgs.append(cfg)
        encoders.append(BatchTopKTranscoder(cfg))

    data_config = DataConfig(
        dataset_name="danbraunai/pile-uncopyrighted-tok-shuffled",
        tokenizer=tokenizer,
        is_tokenized=True,
        token_column="input_ids",
        seq_len=seq_len,
        model_batch_size=args.model_batch_size,
        train_batch_size=cfgs[0].batch_size,
        num_batches_in_buffer=10,
        buffer_on_cpu=False,
        device=device,
    )

    activation_store = MultiLayerActivationsStore(
        model=model,
        input_modules=input_modules,
        output_modules=output_modules,
        data_config=data_config,
        input_size=d_model,
        output_size=d_model,
    )

    num_steps = cfgs[0].num_tokens // (args.model_batch_size * seq_len)
    print(f"Training transcoders (cascading e2e): {cfgs[0].name}")
    print(f"  Model: LlamaSimpleMLP (t-32d1bb3b)")
    print(f"  Layers: {LAYERS}")
    print(f"  Dict size: {args.dict_size}, Top-k: {args.top_k}")
    print(f"  Steps: {num_steps:,}")
    print(f"  model_batch_size: {args.model_batch_size}")
    print(f"  Dataset: danbraunai/pile-uncopyrighted-tok-shuffled")

    train_encoder_cascading(
        encoders, activation_store, cfgs,
        compute_loss_fn=compute_loss_llama,
        get_logits_fn=get_logits_llama,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
