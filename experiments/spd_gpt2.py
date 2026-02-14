"""SPD decomposition of GPT-2 layer 8 MLP.

Decomposes the same MLP targeted by the transcoder experiments in main.py,
using Stochastic Parameter Decomposition from the SPD codebase.

Usage:
    python experiments/spd_gpt2.py
    python experiments/spd_gpt2.py --config_path experiments/spd_gpt2_config.yaml
"""

from pathlib import Path

import fire
import wandb
from transformers import GPT2LMHeadModel

from spd.configs import Config, LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.general_utils import save_pre_run_info, set_seed
from spd.utils.run_utils import setup_decomposition_run
from spd.utils.wandb_utils import init_wandb

DEFAULT_CONFIG = Path(__file__).parent / "spd_gpt2_config.yaml"


def main(config_path: str | Path = DEFAULT_CONFIG) -> None:
    config = Config.from_file(config_path)
    assert isinstance(config.task_config, LMTaskConfig)

    set_seed(config.seed)
    device = "cuda"

    _, run_id, tags = setup_decomposition_run(experiment_tag="lm")
    out_dir = Path("checkpoints") / f"spd_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.wandb_project:
        init_wandb(
            config=config,
            project=config.wandb_project,
            run_id=run_id,
            name=config.wandb_run_name,
            tags=tags,
        )
    logger.info(config)

    save_pre_run_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        spd_config=config,
        sweep_params=None,
        target_model=None,
        train_config=None,
        task_name=None,
    )

    logger.info("Loading GPT-2...")
    target_model = GPT2LMHeadModel.from_pretrained(config.pretrained_model_name)
    target_model.eval()

    logger.info("Loading dataset...")
    task = config.task_config
    train_data_config = DatasetConfig(
        name=task.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task.train_data_split,
        n_ctx=task.max_seq_len,
        is_tokenized=task.is_tokenized,
        streaming=task.streaming,
        column_name=task.column_name,
        shuffle_each_epoch=task.shuffle_each_epoch,
        seed=None,
    )
    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=config.microbatch_size,
        buffer_size=task.buffer_size,
        global_seed=config.seed,
        dist_state=None,
    )

    eval_data_config = DatasetConfig(
        name=task.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task.eval_data_split,
        n_ctx=task.max_seq_len,
        is_tokenized=task.is_tokenized,
        streaming=task.streaming,
        column_name=task.column_name,
        shuffle_each_epoch=task.shuffle_each_epoch,
        seed=None,
    )
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=config.eval_batch_size,
        buffer_size=task.buffer_size,
        global_seed=config.seed + 1,
        dist_state=None,
    )

    logger.info("Starting SPD optimization...")
    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
    )

    logger.info("Optimization finished.")
    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
