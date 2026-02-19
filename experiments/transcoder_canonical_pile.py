"""Train BatchTopK transcoders on LlamaSimpleMLP MLPs for layers 0–3.

Matches the SPD decomposition run s-275c8f21 as closely as possible:
- Same base model (wandb:goodfire/spd/t-32d1bb3b)
- Same dataset (danbraunai/pile-uncopyrighted-tok, pre-tokenized, streaming)
- Same sequence length (512) and batch size (64 sequences)

Each layer uses a different top_k: layer 0 → 21, layer 1 → 8, layer 2 → 13, layer 3 → 27.

Runs all four transcoder trainings in parallel (one process per layer). With 4+ GPUs
each uses a dedicated GPU; with fewer GPUs devices are shared (may OOM on a single GPU).

Usage:
    python experiments/transcoder_canonical_pile.py
"""

import os
import sys
import traceback

from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp

import torch
import torch.nn.functional as F

# Add project root and SPD codebase to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

from transformers import AutoTokenizer

from activation_store import ActivationsStore, DataConfig
from encoders import BatchTopK
from config import EncoderConfig
from training import train_encoder

# (layer_index, batchtopk / top_k) for each MLP
LAYER_CONFIGS = [(0, 21), (1, 8), (2, 13), (3, 27)]
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


def train_one_layer(layer_top_k_device):
    """Train a single transcoder (for one layer). Runs in a separate process."""
    layer, top_k, device = layer_top_k_device
    try:
        # Give each process its own wandb directory to avoid file-level races
        wandb_dir = Path(f"wandb_layer{layer}").resolve()
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
            wandb_project="pile_transcoder",
            device=device,
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

        transcoder = BatchTopK(cfg)

        print(f"[Layer {layer} | {device}] Training transcoder: {cfg.name}")
        print(f"  Model: LlamaSimpleMLP (t-32d1bb3b)")
        print(f"  Layer: {layer}, BatchTopK: {top_k}")
        print(f"  Dict size: {cfg.dict_size}, Top-k: {cfg.top_k}")
        print(f"  Steps: {cfg.num_tokens // cfg.batch_size:,}")
        print(f"  Dataset: danbraunai/pile-uncopyrighted-tok")

        train_encoder(transcoder, activation_store, cfg, compute_loss_fn=compute_loss_llama)
        print(f"[Layer {layer}] Training complete.")
        return layer
    except Exception as e:
        print(f"[Layer {layer}] FAILED: {e}")
        traceback.print_exc()
        raise


def main():
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_workers = len(LAYER_CONFIGS)
    # Assign device per job: cuda:0, cuda:1, ... (round-robin if fewer GPUs than jobs)
    devices = [
        f"cuda:{i % max(1, n_gpus)}" if n_gpus else "cpu"
        for i in range(n_workers)
    ]
    tasks = [
        (layer, top_k, device)
        for (layer, top_k), device in zip(LAYER_CONFIGS, devices)
    ]

    if n_gpus < n_workers:
        print(f"Only {n_gpus} GPU(s) available; {n_workers} jobs will share them (may OOM).")
    else:
        print(f"Running {n_workers} transcoder trainings in parallel on {n_gpus} GPU(s).")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as executor:
        futures = {
            executor.submit(train_one_layer, task): task[0]  # map future → layer
            for task in tasks
        }
        for future in as_completed(futures):
            layer = futures[future]
            try:
                future.result()
                print(f"[Layer {layer}] Finished successfully.")
            except Exception as e:
                print(f"[Layer {layer}] FAILED with exception: {e}")


if __name__ == "__main__":
    main()
