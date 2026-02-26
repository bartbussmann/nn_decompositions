"""Train BatchTopKTranscoder transcoders on LlamaSimpleMLP MLPs for layers 0–3.

Sweep variant: all layers use the same top_k, swept over [8, 16, 32, 64].
Otherwise identical to exp_001 (same base model, dataset, hyperparameters).

Usage:
    python experiments/exp_010_train_transcoder_pile_sweep/transcoder_sweep_pile.py
    python experiments/exp_010_train_transcoder_pile_sweep/transcoder_sweep_pile.py --top_ks 8 16
"""

import os
import sys
import traceback
import uuid

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import multiprocessing as mp

import torch
import torch.nn.functional as F

# Add project root and SPD codebase to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from transformers import AutoTokenizer

from activation_store import ActivationsStore, DataConfig
from transcoder import BatchTopKTranscoder
from config import EncoderConfig
from training import train_encoder

LAYERS = [0, 1, 2, 3]
DEFAULT_TOP_KS = [8, 16, 32, 64]
WANDB_MODEL_PATH = "wandb:goodfire/spd/t-32d1bb3b"


def load_model(device):
    """Load the LlamaSimpleMLP base model from wandb onto the given device."""
    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

    print(f"[{device}] Loading LlamaSimpleMLP from {WANDB_MODEL_PATH}...")
    model = LlamaSimpleMLP.from_pretrained(WANDB_MODEL_PATH)
    model.to(device)
    model.eval()
    return model


def compute_loss_llama(model, tokenizer, input_ids, attention_mask):
    """Compute CE loss for LlamaSimpleMLP (next-token prediction)."""
    targets = input_ids[:, 1:].contiguous()
    logits, _ = model(input_ids)
    logits = logits[:, :-1].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).item()


def train_one_layer(layer, top_k, device):
    """Train a single transcoder (for one layer + top_k combo). Runs in a separate process."""
    try:
        wandb_dir = Path(f"wandb_layer{layer}_k{top_k}").resolve()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_dir)

        model = load_model(device)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        input_size = model.config.n_embd
        output_size = model.config.n_embd

        spd_batch_size = 64
        seq_len = 512

        input_module = model.h[layer].rms_2
        output_module = model.h[layer].mlp

        cfg = EncoderConfig(
            input_size=input_size,
            output_size=output_size,
            dict_size=4096,
            encoder_type="batchtopk",
            top_k=top_k,
            l1_coeff=0.0,
            batch_size=4096,
            num_tokens=int(5e8),
            lr=3e-4,
            wandb_project="pile_transcoder_sweep3",
            device=device,
            checkpoint_freq="final"
        )

        data_config = DataConfig(
            dataset_name="danbraunai/pile-uncopyrighted-tok",
            tokenizer=tokenizer,
            is_tokenized=True,
            token_column="input_ids",
            seq_len=seq_len,
            model_batch_size=spd_batch_size,
            train_batch_size=cfg.batch_size,
            num_batches_in_buffer=10,
            buffer_on_cpu=False,
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

        # Include layer and short random suffix so wandb runs/artifacts don't collide
        suffix = uuid.uuid4().hex[:6]
        cfg.run_name = f"{cfg.dict_size}_{cfg.encoder_type}_k{top_k}_{cfg.lr}_L{layer}_{suffix}"

        transcoder = BatchTopKTranscoder(cfg)

        print(f"[Layer {layer} | k={top_k} | {device}] Training transcoder: {cfg.run_name}")
        print(f"  Model: LlamaSimpleMLP (t-32d1bb3b)")
        print(f"  Layer: {layer}, top_k: {top_k}")
        print(f"  Dict size: {cfg.dict_size}")
        print(f"  Steps: {cfg.num_tokens // cfg.batch_size:,}")

        train_encoder(transcoder, activation_store, cfg, compute_loss_fn=compute_loss_llama)
        print(f"[Layer {layer} | k={top_k}] Training complete.")
    except Exception as e:
        print(f"[Layer {layer} | k={top_k}] FAILED: {e}")
        traceback.print_exc()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train transcoders: all layers × top_k sweep")
    parser.add_argument("--top_ks", type=int, nargs="+", default=DEFAULT_TOP_KS)
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    tasks = [(layer, k) for k in args.top_ks for layer in args.layers]
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"Training {len(tasks)} transcoders ({len(args.layers)} layers × {len(args.top_ks)} top_k values)")
    print(f"  Layers: {args.layers}")
    print(f"  top_k values: {args.top_ks}")
    print(f"  GPUs available: {n_gpus}")
    print(f"  Max parallel workers: {args.max_workers}")

    ctx = mp.get_context("spawn")

    # Process tasks in batches of max_workers.
    # Each batch spawns fresh processes that fully die between batches,
    # guaranteeing GPU memory is freed.
    for batch_start in range(0, len(tasks), args.max_workers):
        batch = tasks[batch_start : batch_start + args.max_workers]
        procs = []
        for i, (layer, k) in enumerate(batch):
            device = f"cuda:{i % max(1, n_gpus)}" if n_gpus else "cpu"
            p = ctx.Process(target=train_one_layer, args=(layer, k, device))
            p.start()
            procs.append((p, layer, k))

        for p, layer, k in procs:
            p.join()
            if p.exitcode == 0:
                print(f"[Layer {layer} | k={k}] Finished successfully.")
            else:
                print(f"[Layer {layer} | k={k}] FAILED (exit code {p.exitcode})")


if __name__ == "__main__":
    main()
