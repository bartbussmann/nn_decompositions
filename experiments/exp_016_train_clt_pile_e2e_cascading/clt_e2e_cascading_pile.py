"""Train a Cross-Layer Transcoder (CLT) end-to-end with cascading on LlamaSimpleMLP layers 0–3.

Same setup as exp_003 but trained with e2e KL divergence on logits, with cascading
enabled: each layer's reconstruction affects the residual stream before the next
layer's encoder runs.

Default model_batch_size is 16 (down from 64 in exp_003) to fit in ~8 GB GPU memory.
Use --model_batch_size to adjust for your GPU.

Usage:
    python experiments/exp_016_train_clt_pile_e2e_cascading/clt_e2e_cascading_pile.py
    python experiments/exp_016_train_clt_pile_e2e_cascading/clt_e2e_cascading_pile.py --model_batch_size 32
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
from nn_decompositions.clt import CrossLayerTranscoder
from nn_decompositions.config import CLTConfig
from nn_decompositions.training import train_encoder

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
    parser = argparse.ArgumentParser(description="Train CLT e2e cascading")
    parser.add_argument("--model_batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--dict_size", type=int, default=DICT_SIZE)
    args = parser.parse_args()

    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

    device = f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else "cpu"

    print(f"Loading LlamaSimpleMLP from {WANDB_MODEL_PATH}...")
    model = LlamaSimpleMLP.from_pretrained(WANDB_MODEL_PATH)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    d_model = model.config.n_embd
    seq_len = 512

    input_modules = [model.h[layer].rms_2 for layer in LAYERS]
    output_modules = [model.h[layer].mlp for layer in LAYERS]

    cfg = CLTConfig(
        layers=LAYERS,
        input_size=d_model,
        output_size=d_model,
        dict_size=args.dict_size,
        encoder_type="batchtopk",
        top_k=args.top_k,
        l1_coeff=0.0,
        batch_size=4096,
        num_tokens=int(5e8),
        lr=3e-4,
        wandb_project="pile_clt_e2e_cascading",
        device=device,
        e2e=True,
        e2e_cascading=True,
    )

    data_config = DataConfig(
        dataset_name="danbraunai/pile-uncopyrighted-tok",
        tokenizer=tokenizer,
        is_tokenized=True,
        token_column="input_ids",
        seq_len=seq_len,
        model_batch_size=args.model_batch_size,
        train_batch_size=cfg.batch_size,
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

    clt = CrossLayerTranscoder(cfg)

    num_steps = cfg.num_tokens // (args.model_batch_size * seq_len)
    print(f"Training CLT (e2e cascading): {cfg.name}")
    print(f"  Model: LlamaSimpleMLP (t-32d1bb3b)")
    print(f"  Layers: {LAYERS}")
    print(f"  Dict size: {args.dict_size}, Top-k: {args.top_k}")
    print(f"  Steps: {num_steps:,}")
    print(f"  model_batch_size: {args.model_batch_size}")
    print(f"  Dataset: danbraunai/pile-uncopyrighted-tok")

    train_encoder(
        clt, activation_store, cfg,
        compute_loss_fn=compute_loss_llama,
        get_logits_fn=get_logits_llama,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
