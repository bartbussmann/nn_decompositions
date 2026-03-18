"""Train the missing tc_k64 (32k dict) on jose's target model.

Logs to the same wandb project (pile_local_sweep_jose_32k) so it appears
alongside the existing runs.

Usage:
    python experiments/exp_033_e2e_sweep_jose_32k/train_tc_k64_32k.py
    python experiments/exp_033_e2e_sweep_jose_32k/train_tc_k64_32k.py --device cuda:1
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path("/workspace/spd")))

WANDB_MODEL_PATH = "goodfire/spd/runs/t-9d2b8f02"
WANDB_PROJECT = "pile_local_sweep_jose_32k"
LAYERS = [0, 1, 2, 3]
DICT_SIZE = 32768
NUM_TOKENS = int(5e8)
LR = 3e-4
MODEL_BATCH_SIZE = 16
SEQ_LEN = 512
DATASET = "danbraunai/pile-uncopyrighted-tok-shuffled"
TOP_K = 64
RUN_NAME = "tc_k64"


def compute_loss_llama(model, tokenizer, input_ids, attention_mask):
    import torch.nn.functional as F
    targets = input_ids[:, 1:].contiguous()
    logits, _ = model(input_ids)
    logits = logits[:, :-1].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    device = args.device

    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig
    from transformers import AutoTokenizer

    from nn_decompositions.activation_store import MultiLayerActivationsStore, DataConfig
    from nn_decompositions.config import EncoderConfig
    from nn_decompositions.transcoder import BatchTopKTranscoder
    from nn_decompositions.training import train_encoder_multilayer_local

    # Load model (use cached if available)
    model_cache_path = Path(__file__).resolve().parent / "model_cache"
    model_cache_path.mkdir(exist_ok=True)

    if not (model_cache_path / "state_dict.pt").exists():
        jose_cache = Path(__file__).resolve().parent.parent / "exp_019_eval_e2e" / "jose_model_cache"
        if jose_cache.exists() and (jose_cache / "state_dict.pt").exists():
            import shutil
            shutil.copy2(jose_cache / "state_dict.pt", model_cache_path / "state_dict.pt")
            shutil.copy2(jose_cache / "config.json", model_cache_path / "config.json")
        else:
            print("Downloading jose target model from wandb...")
            model = LlamaSimpleMLP.from_pretrained(WANDB_MODEL_PATH)
            torch.save(model.state_dict(), model_cache_path / "state_dict.pt")
            with open(model_cache_path / "config.json", "w") as f:
                json.dump(model.config.__dict__, f)
            del model

    print(f"Loading model on {device}...")
    with open(model_cache_path / "config.json") as f:
        model_cfg = LlamaSimpleMLPConfig(**json.load(f))
    model = LlamaSimpleMLP(model_cfg)
    model.load_state_dict(torch.load(model_cache_path / "state_dict.pt", map_location="cpu", weights_only=True))
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

    cfgs = []
    encoders = []
    for layer in LAYERS:
        cfg = EncoderConfig(
            input_size=d_model,
            output_size=d_model,
            dict_size=DICT_SIZE,
            encoder_type="batchtopk",
            top_k=TOP_K,
            l1_coeff=0.0,
            batch_size=4096,
            num_tokens=NUM_TOKENS,
            lr=LR,
            wandb_project=WANDB_PROJECT,
            device=device,
            e2e=False,
            run_name=RUN_NAME,
        )
        cfgs.append(cfg)
        encoders.append(BatchTopKTranscoder(cfg))

    print(f"Training 4x BatchTopKTranscoder (dict={DICT_SIZE}, k={TOP_K})...")
    train_encoder_multilayer_local(
        encoders, activation_store, cfgs,
        compute_loss_fn=compute_loss_llama,
    )
    print("Done!")


if __name__ == "__main__":
    main()
