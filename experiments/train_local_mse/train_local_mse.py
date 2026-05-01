"""Local MSE training sweep on the base LLM.

Per-layer MSE reconstruction (no end-to-end KL). Trains BatchTopK Transcoders
and CLTs at k = 8, 16, 32, 64. Polls GPU memory to find free devices and
launches jobs as capacity allows.

Use --dict_size 4096 (default) to log to `pile_local_sweep_jose`, or
--dict_size 32768 to log to `pile_local_sweep_jose_32k`.

Usage:
    python experiments/train_local_mse/train_local_mse.py
    python experiments/train_local_mse/train_local_mse.py --dict_size 32768
    python experiments/train_local_mse/train_local_mse.py --top_ks 32 64
    python experiments/train_local_mse/train_local_mse.py --types tc clt
"""

import os
import time
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch

from nn_decompositions.eval_utils import get_free_gpus
from experiments.paper_runs import (
    LLM_BASE_MODEL,
    LLM_BASE_MODEL_CACHE,
    LOCAL_PROJECT_BY_DICT_SIZE as PROJECT_BY_DICT_SIZE,
)

LAYERS = [0, 1, 2, 3]
NUM_TOKENS = int(5e8)
LR = 3e-4
MODEL_BATCH_SIZE = 16
SEQ_LEN = 512
DATASET = "danbraunai/pile-uncopyrighted-tok-shuffled"

ALL_TYPES = ["tc", "clt"]
ALL_TOP_KS = [8, 16, 32, 64]


@dataclass
class Job:
    name: str
    job_type: str
    top_k: int


def compute_loss_llama(model, tokenizer, input_ids, attention_mask):
    import torch.nn.functional as F
    targets = input_ids[:, 1:].contiguous()
    logits, _ = model(input_ids)
    logits = logits[:, :-1].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).item()


def train_one(job: Job, device: str, model_cache_path: str, dict_size: int, wandb_project: str):
    """Train a single job. Runs in a subprocess."""
    try:
        wandb_dir = Path(f"wandb_{job.name}").resolve()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_dir)

        import json as json_mod
        from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig
        from transformers import AutoTokenizer

        from nn_decompositions.activation_store import MultiLayerActivationsStore, DataConfig
        from nn_decompositions.config import CLTConfig, EncoderConfig
        from nn_decompositions.clt import CrossLayerTranscoder
        from nn_decompositions.transcoder import BatchTopKTranscoder
        from nn_decompositions.training import train_encoder, train_encoder_multilayer_local

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

        input_modules = [model.h[layer].rms_2 for layer in LAYERS]
        output_modules = [model.h[layer].mlp for layer in LAYERS]

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
            input_modules=input_modules,
            output_modules=output_modules,
            data_config=data_config,
            input_size=d_model,
            output_size=d_model,
        )

        if job.job_type == "tc":
            cfgs = []
            encoders = []
            for layer in LAYERS:
                cfg = EncoderConfig(
                    input_size=d_model,
                    output_size=d_model,
                    dict_size=dict_size,
                    encoder_type="batchtopk",
                    top_k=job.top_k,
                    l1_coeff=0.0,
                    batch_size=4096,
                    num_tokens=NUM_TOKENS,
                    lr=LR,
                    wandb_project=wandb_project,
                    device=device,
                    e2e=False,
                    run_name=job.name,
                )
                cfgs.append(cfg)
                encoders.append(BatchTopKTranscoder(cfg))

            print(f"[{job.name}] Training {len(encoders)} transcoders (local MSE)...")
            train_encoder_multilayer_local(
                encoders, activation_store, cfgs,
                compute_loss_fn=compute_loss_llama,
            )

        else:  # clt
            cfg = CLTConfig(
                layers=LAYERS,
                input_size=d_model,
                output_size=d_model,
                dict_size=dict_size,
                encoder_type="batchtopk",
                top_k=job.top_k,
                l1_coeff=0.0,
                batch_size=4096,
                num_tokens=NUM_TOKENS,
                lr=LR,
                wandb_project=wandb_project,
                run_name=job.name,
                device=device,
                e2e=False,
            )

            clt = CrossLayerTranscoder(cfg)

            print(f"[{job.name}] Training CLT (local MSE)...")
            train_encoder(
                clt, activation_store, cfg,
                compute_loss_fn=compute_loss_llama,
            )

        print(f"[{job.name}] DONE")
    except Exception as e:
        print(f"[{job.name}] FAILED: {e}")
        traceback.print_exc()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local MSE training sweep (base LLM) with GPU queue")
    parser.add_argument("--dict_size", type=int, default=4096, choices=[4096, 32768],
                        help="Dictionary size; selects which wandb project to log to")
    parser.add_argument("--top_ks", type=int, nargs="+", default=ALL_TOP_KS)
    parser.add_argument("--types", type=str, nargs="+", default=ALL_TYPES,
                        choices=ALL_TYPES)
    parser.add_argument("--min_free_gb", type=float, default=12.0,
                        help="Minimum free GPU memory (GB) to start a job")
    args = parser.parse_args()

    wandb_project = PROJECT_BY_DICT_SIZE[args.dict_size]

    jobs = [
        Job(name=f"{jtype}_k{k}", job_type=jtype, top_k=k)
        for k in args.top_ks
        for jtype in args.types
    ]

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    min_free_bytes = args.min_free_gb * 1e9

    # Shared base-model cache. Auto-downloads from wandb on first run.
    model_cache_path = str(Path(__file__).resolve().parent.parent.parent / LLM_BASE_MODEL_CACHE)
    os.makedirs(model_cache_path, exist_ok=True)

    if not (Path(model_cache_path) / "state_dict.pt").exists():
        print("Downloading base LLM from wandb...")
        from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
        model = LlamaSimpleMLP.from_pretrained(LLM_BASE_MODEL)
        torch.save(model.state_dict(), os.path.join(model_cache_path, "state_dict.pt"))
        import json
        with open(os.path.join(model_cache_path, "config.json"), "w") as f:
            json.dump(model.config.__dict__, f)
        del model

    print(f"Model cached at {model_cache_path}")
    print(f"Logging to wandb project: {wandb_project} (dict_size={args.dict_size})")

    print(f"\n=== Local MSE Sweep: {len(jobs)} jobs ===")
    for j in jobs:
        print(f"  {j.name}")
    print(f"GPUs available: {n_gpus}")
    print(f"Min free VRAM per job: {args.min_free_gb:.1f} GB")
    print()

    ctx = mp.get_context("spawn")
    pending = list(jobs)
    running: list[tuple[mp.Process, str]] = []
    busy_gpus: set[int] = set()

    while pending or running:
        # Reap finished processes
        still_running = []
        for p, name, gpu_id in running:
            if p.is_alive():
                still_running.append((p, name, gpu_id))
            else:
                status = "OK" if p.exitcode == 0 else f"FAILED (exit {p.exitcode})"
                print(f"[{name}] Finished: {status}")
                busy_gpus.discard(gpu_id)
        running = still_running

        # Launch pending jobs on free GPUs
        if pending:
            free_gpus = [g for g in get_free_gpus(min_free_bytes) if g not in busy_gpus]
            for gpu_id in free_gpus:
                if not pending:
                    break
                job = pending.pop(0)
                device = f"cuda:{gpu_id}"
                print(f"[{job.name}] Launching on {device}")
                p = ctx.Process(
                    target=train_one,
                    args=(job, device, model_cache_path, args.dict_size, wandb_project),
                )
                p.start()
                running.append((p, job.name, gpu_id))
                busy_gpus.add(gpu_id)

        if pending or running:
            time.sleep(120)

    print("\n=== All jobs complete ===")


if __name__ == "__main__":
    main()
