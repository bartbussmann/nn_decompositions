"""SAE sweep: train SAEs on MLP outputs and residual streams, 4 training modes, k=8/16/32/64.

Locations (8):
  - mlp_output: SAE reconstructs MLP output at each of 4 layers
  - resid_stream: SAE reconstructs residual stream (block output) at each of 4 layers

Training modes (4):
  - local: Standard MSE reconstruction loss (no e2e)
  - cascading: E2E KL loss, each layer sees modified stream from previous layers
  - parallel: E2E KL loss, all layers encode from clean activations
  - independent: E2E KL loss, per-layer (only that layer replaced)

Total: 2 locations x 4 modes x 4 top_k = 32 jobs, each producing 4 SAEs = 128 SAEs.

Usage:
    python experiments/exp_023_sae_sweep/sae_sweep.py
    python experiments/exp_023_sae_sweep/sae_sweep.py --top_ks 32
    python experiments/exp_023_sae_sweep/sae_sweep.py --locations mlp_output --modes local cascading
    python experiments/exp_023_sae_sweep/sae_sweep.py --min_free_gb 12
"""

import os
import sys
import time
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

# Jose model
JOSE_MODEL_CACHE = Path("experiments/exp_019_eval_e2e/jose_model_cache")
WANDB_PROJECT = "sae_sweep_jose3"
LAYERS = [0, 1, 2, 3]
DICT_SIZE = 4096
NUM_TOKENS = int(5e8)
LR = 3e-4
MODEL_BATCH_SIZE = 16
SEQ_LEN = 512
DATASET = "danbraunai/pile-uncopyrighted-tok-shuffled"

ALL_LOCATIONS = ["mlp_output", "resid_stream"]
ALL_MODES = ["local", "cascading", "parallel", "independent"]
ALL_TOP_KS = [8, 16, 32, 64]


@dataclass
class Job:
    name: str
    location: str  # "mlp_output" or "resid_stream"
    mode: str      # "local", "cascading", "parallel", "independent"
    top_k: int


def compute_loss_llama(model, tokenizer, input_ids, attention_mask):
    targets = input_ids[:, 1:].contiguous()
    logits, _ = model(input_ids)
    logits = logits[:, :-1].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).item()


def get_logits_llama(model, input_ids, attention_mask):
    logits, _ = model(input_ids)
    return logits


def train_one(job: Job, device: str, model_cache_path: str):
    """Train a single SAE sweep job. Runs in a subprocess."""
    try:
        wandb_dir = Path(f"wandb_{job.name}").resolve()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_dir)

        import json as json_mod
        from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig
        from transformers import AutoTokenizer

        from nn_decompositions.activation_store import MultiLayerActivationsStore, DataConfig
        from nn_decompositions.config import SAEConfig
        from nn_decompositions.sae import BatchTopKSAE
        from nn_decompositions.training import train_encoder_cascading, train_encoder_multilayer_local

        print(f"[{job.name}] Loading model on {device}...")
        with open(os.path.join(model_cache_path, "config.json")) as f:
            model_cfg = LlamaSimpleMLPConfig(**json_mod.load(f))
        model = LlamaSimpleMLP(model_cfg)
        state_dict = torch.load(
            os.path.join(model_cache_path, "state_dict.pt"),
            map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        d_model = model.config.n_embd

        # Hook modules based on location.
        # For SAEs: input_module == output_module (reconstruct same activation).
        if job.location == "mlp_output":
            hook_modules = [model.h[layer].mlp for layer in LAYERS]
        elif job.location == "resid_stream":
            hook_modules = [model.h[layer] for layer in LAYERS]
        else:
            raise ValueError(f"Unknown location: {job.location}")

        data_config = DataConfig(
            dataset_name=DATASET,
            tokenizer=tokenizer,
            is_tokenized=True,
            token_column="input_ids",
            seq_len=SEQ_LEN,
            model_batch_size=MODEL_BATCH_SIZE,
            train_batch_size=4096,
            num_batches_in_buffer=10,
            buffer_on_cpu=False,
            device=device,
        )

        activation_store = MultiLayerActivationsStore(
            model=model,
            input_modules=hook_modules,
            output_modules=hook_modules,
            data_config=data_config,
            input_size=d_model,
            output_size=d_model,
        )

        # Create 4 SAEs (one per layer)
        is_e2e = job.mode != "local"
        cfgs = []
        saes = []
        for layer in LAYERS:
            cfg = SAEConfig(
                input_size=d_model,
                output_size=d_model,
                dict_size=DICT_SIZE,
                encoder_type="batchtopk",
                top_k=job.top_k,
                l1_coeff=0.0,
                batch_size=4096,
                num_tokens=NUM_TOKENS,
                lr=LR,
                wandb_project=WANDB_PROJECT,
                device=device,
                e2e=is_e2e,
                run_name=job.name,
            )
            cfgs.append(cfg)
            saes.append(BatchTopKSAE(cfg))

        print(f"[{job.name}] Training 4 SAEs ({job.location}, {job.mode}, k={job.top_k})...")

        if job.mode == "local":
            # Local MSE training — each SAE trains on its own layer's activations
            train_encoder_multilayer_local(
                saes, activation_store, cfgs,
                compute_loss_fn=compute_loss_llama,
            )
        else:
            # E2E training — SAEs encode module output, not input
            train_encoder_cascading(
                saes, activation_store, cfgs,
                compute_loss_fn=compute_loss_llama,
                get_logits_fn=get_logits_llama,
                mode=job.mode,
                encode_output=True,
            )

        print(f"[{job.name}] DONE")

    except Exception as e:
        print(f"[{job.name}] FAILED: {e}")
        traceback.print_exc()


def get_free_gpus(min_free_bytes: float) -> list[int]:
    free_gpus = []
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        if free >= min_free_bytes:
            free_gpus.append(i)
    return free_gpus


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAE sweep with GPU queue")
    parser.add_argument("--top_ks", type=int, nargs="+", default=ALL_TOP_KS)
    parser.add_argument("--locations", type=str, nargs="+", default=ALL_LOCATIONS,
                        choices=ALL_LOCATIONS)
    parser.add_argument("--modes", type=str, nargs="+", default=ALL_MODES,
                        choices=ALL_MODES)
    parser.add_argument("--min_free_gb", type=float, default=12.0)
    args = parser.parse_args()

    jobs = [
        Job(
            name=f"sae_{loc}_{mode}_k{k}",
            location=loc, mode=mode, top_k=k,
        )
        for loc in args.locations
        for k in args.top_ks
        for mode in args.modes
    ]

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    min_free_bytes = args.min_free_gb * 1e9

    # Verify model cache exists
    model_cache_path = str(JOSE_MODEL_CACHE)
    assert Path(model_cache_path, "config.json").exists(), \
        f"Jose model cache not found at {model_cache_path}. Run exp_021 first."
    assert Path(model_cache_path, "state_dict.pt").exists(), \
        f"Jose model state_dict not found at {model_cache_path}."
    print(f"Using model cache at {model_cache_path}")

    print(f"\n=== SAE Sweep: {len(jobs)} jobs ({len(jobs) * 4} SAEs total) ===")
    for j in jobs:
        print(f"  {j.name}")
    print(f"GPUs available: {n_gpus}")
    print(f"Min free VRAM per job: {args.min_free_gb:.1f} GB")
    print()

    ctx = mp.get_context("spawn")
    pending = list(jobs)
    running: list[tuple[mp.Process, str, int]] = []
    busy_gpus: set[int] = set()

    while pending or running:
        still_running = []
        for p, name, gpu_id in running:
            if p.is_alive():
                still_running.append((p, name, gpu_id))
            else:
                status = "OK" if p.exitcode == 0 else f"FAILED (exit {p.exitcode})"
                print(f"[{name}] Finished: {status}")
                busy_gpus.discard(gpu_id)
        running = still_running

        if pending:
            free_gpus = [g for g in get_free_gpus(min_free_bytes) if g not in busy_gpus]
            for gpu_id in free_gpus:
                if not pending:
                    break
                job = pending.pop(0)
                device = f"cuda:{gpu_id}"
                print(f"[{job.name}] Launching on {device}")
                p = ctx.Process(target=train_one, args=(job, device, model_cache_path))
                p.start()
                running.append((p, job.name, gpu_id))
                busy_gpus.add(gpu_id)

        if pending or running:
            time.sleep(120)

    print("\n=== All jobs complete ===")


if __name__ == "__main__":
    main()
