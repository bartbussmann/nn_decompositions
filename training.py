from queue import Empty

import torch
import torch.multiprocessing as mp
import tqdm
import wandb

from config import EncoderConfig
from logs import get_encoder_metrics, get_performance_metrics, save_checkpoint


def _wandb_process(cfg_dict: dict, log_queue: mp.Queue):
    """Separate process for wandb logging."""
    wandb.init(
        project=cfg_dict["wandb_project"],
        name=cfg_dict["name"],
        config=cfg_dict,
    )
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            wandb.log(log)
        except Empty:
            continue
    wandb.finish()


def train_encoder(encoder, activation_store, model, cfg: EncoderConfig):
    """Train any encoder (SAE or transcoder).

    Works with both ActivationsStore and TranscoderActivationsStore since
    both return (x_in, y_target) tuples from next_batch().
    """
    from logs import init_wandb, log_wandb, log_encoder_performance

    num_batches = cfg.num_tokens // cfg.batch_size
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)

    for i in pbar:
        x_in, y_target = activation_store.next_batch()
        output = encoder(x_in, y_target)

        log_wandb(output, i, wandb_run)

        if i % cfg.perf_log_freq == 0:
            log_encoder_performance(wandb_run, i, model, activation_store, encoder)

        if i % cfg.checkpoint_freq == 0:
            save_checkpoint(encoder, cfg, i)

        loss = output["loss"]
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "L0": f"{output['l0_norm']:.4f}",
            "L2": f"{output['l2_loss']:.4f}",
            "L1": f"{output.get('l1_loss', 0):.4f}",
        })
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.max_grad_norm)
        encoder.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    save_checkpoint(encoder, cfg, i)


def train_encoder_group(encoders, activation_store, model, cfgs: list[EncoderConfig]):
    """Train multiple encoders on the same activation stream with separate wandb runs."""
    from dataclasses import asdict

    num_batches = cfgs[0].num_tokens // cfgs[0].batch_size
    optimizers = [
        torch.optim.Adam(enc.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        for enc, cfg in zip(encoders, cfgs)
    ]
    pbar = tqdm.trange(num_batches)

    # Spawn separate wandb process for each encoder
    log_queues = []
    wandb_processes = []
    for cfg in cfgs:
        log_queue = mp.Queue()
        log_queues.append(log_queue)

        # Convert config to dict, handling non-serializable types
        cfg_dict = {
            k: str(v) if isinstance(v, torch.dtype) else v
            for k, v in asdict(cfg).items()
        }
        cfg_dict["name"] = cfg.name  # Ensure computed property is included

        process = mp.Process(target=_wandb_process, args=(cfg_dict, log_queue))
        process.start()
        wandb_processes.append(process)

    batch_tokens = activation_store.get_batch_tokens()

    for i in pbar:
        x_in, y_target = activation_store.next_batch()

        for idx, (encoder, cfg, optimizer, log_queue) in enumerate(
            zip(encoders, cfgs, optimizers, log_queues)
        ):
            output = encoder(x_in, y_target)
            loss = output["loss"]

            # Send metrics to wandb process via queue
            log_dict = get_encoder_metrics(output)
            log_dict["step"] = i
            log_queue.put(log_dict)

            if i % cfg.perf_log_freq == 0:
                perf_dict = get_performance_metrics(
                    model, activation_store, encoder, batch_tokens
                )
                perf_dict["step"] = i
                log_queue.put(perf_dict)

            if i % cfg.checkpoint_freq == 0:
                save_checkpoint(encoder, cfg, i)

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "L0": f"{output['l0_norm']:.4f}",
                "L2": f"{output['l2_loss']:.4f}",
                "L1": f"{output.get('l1_loss', 0):.4f}",
            })
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.max_grad_norm)
            encoder.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

    # Save final checkpoints and clean up wandb processes
    for encoder, cfg in zip(encoders, cfgs):
        save_checkpoint(encoder, cfg, i)

    for log_queue in log_queues:
        log_queue.put("DONE")
    for process in wandb_processes:
        process.join()


# Backwards compatibility aliases
train_sae = train_encoder
train_transcoder = train_encoder
