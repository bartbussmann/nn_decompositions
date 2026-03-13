"""Local MSE training sweep on jose's target model (t-9d2b8f02).

Same as exp_020 but using standard local MSE loss (per-layer reconstruction)
instead of end-to-end KL divergence on final token logits.

Transcoders and CLTs, k=8/16/32/64.

Polls GPU memory to find free devices and launches jobs as capacity allows.
All runs go to a single wandb project with descriptive run names.

Usage:
    python experiments/exp_032_local_sweep_jose/local_sweep_jose.py
    python experiments/exp_032_local_sweep_jose/local_sweep_jose.py --top_ks 32 64
    python experiments/exp_032_local_sweep_jose/local_sweep_jose.py --types tc clt
    python experiments/exp_032_local_sweep_jose/local_sweep_jose.py --min_free_gb 12
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

WANDB_MODEL_PATH = "goodfire/spd/runs/t-9d2b8f02"
WANDB_PROJECT = "pile_local_sweep_jose"
LAYERS = [0, 1, 2, 3]
DICT_SIZE = 4096
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


def train_one(job: Job, device: str, model_cache_path: str):
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
                    dict_size=DICT_SIZE,
                    encoder_type="batchtopk",
                    top_k=job.top_k,
                    l1_coeff=0.0,
                    batch_size=4096,
                    num_tokens=NUM_TOKENS,
                    lr=LR,
                    wandb_project=WANDB_PROJECT,
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
                dict_size=DICT_SIZE,
                encoder_type="batchtopk",
                top_k=job.top_k,
                l1_coeff=0.0,
                batch_size=4096,
                num_tokens=NUM_TOKENS,
                lr=LR,
                wandb_project=WANDB_PROJECT,
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


def get_free_gpus(min_free_bytes: float) -> list[int]:
    """Return GPU IDs with at least min_free_bytes of free VRAM."""
    free_gpus = []
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        if free >= min_free_bytes:
            free_gpus.append(i)
    return free_gpus


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local MSE training sweep (jose target model) with GPU queue")
    parser.add_argument("--top_ks", type=int, nargs="+", default=ALL_TOP_KS)
    parser.add_argument("--types", type=str, nargs="+", default=ALL_TYPES,
                        choices=ALL_TYPES)
    parser.add_argument("--min_free_gb", type=float, default=12.0,
                        help="Minimum free GPU memory (GB) to start a job")
    args = parser.parse_args()

    jobs = [
        Job(name=f"{jtype}_k{k}", job_type=jtype, top_k=k)
        for k in args.top_ks
        for jtype in args.types
    ]

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    min_free_bytes = args.min_free_gb * 1e9

    # Use jose's target model cache from exp_019 if available, otherwise download
    jose_cache = Path(__file__).resolve().parent.parent / "exp_019_eval_e2e" / "jose_model_cache"
    model_cache_path = str(Path(__file__).resolve().parent / "model_cache")
    os.makedirs(model_cache_path, exist_ok=True)

    if jose_cache.exists() and (jose_cache / "state_dict.pt").exists():
        print(f"Using cached jose target model from {jose_cache}")
        import shutil
        shutil.copy2(jose_cache / "state_dict.pt", os.path.join(model_cache_path, "state_dict.pt"))
        shutil.copy2(jose_cache / "config.json", os.path.join(model_cache_path, "config.json"))
    else:
        print("Downloading jose target model from wandb...")
        from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
        model = LlamaSimpleMLP.from_pretrained(WANDB_MODEL_PATH)
        torch.save(model.state_dict(), os.path.join(model_cache_path, "state_dict.pt"))
        import json
        with open(os.path.join(model_cache_path, "config.json"), "w") as f:
            json.dump(model.config.__dict__, f)
        del model

    print(f"Model cached at {model_cache_path}")

    print(f"\n=== Local MSE Sweep (jose): {len(jobs)} jobs ===")
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
                p = ctx.Process(target=train_one, args=(job, device, model_cache_path))
                p.start()
                running.append((p, job.name, gpu_id))
                busy_gpus.add(gpu_id)

        if pending or running:
            time.sleep(120)

    print("\n=== All jobs complete ===")


if __name__ == "__main__":
    main()
