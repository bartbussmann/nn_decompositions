import torch
import tqdm

from config import EncoderConfig
from logs import (
    WandbLogger,
    init_wandb,
    log_encoder_performance,
    log_wandb,
    save_checkpoint,
)


def train_encoder(encoder, activation_store, model, cfg: EncoderConfig):
    """Train any encoder (SAE or transcoder).

    Works with both ActivationsStore and TranscoderActivationsStore since
    both return (x_in, y_target) tuples from next_batch().
    """
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
    num_batches = cfgs[0].num_tokens // cfgs[0].batch_size
    optimizers = [
        torch.optim.Adam(enc.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        for enc, cfg in zip(encoders, cfgs)
    ]
    pbar = tqdm.trange(num_batches)

    # Create threaded logger for each encoder
    loggers = [WandbLogger(cfg) for cfg in cfgs]
    batch_tokens = activation_store.get_batch_tokens()

    for i in pbar:
        x_in, y_target = activation_store.next_batch()

        for encoder, cfg, optimizer, logger in zip(encoders, cfgs, optimizers, loggers):
            output = encoder(x_in, y_target)
            loss = output["loss"]

            logger.log_encoder(output, i)

            if i % cfg.perf_log_freq == 0:
                logger.log_performance(model, activation_store, encoder, i, batch_tokens)

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

    # Save final checkpoints and finish loggers
    for encoder, cfg, logger in zip(encoders, cfgs, loggers):
        save_checkpoint(encoder, cfg, i)
        logger.finish()


# Backwards compatibility aliases
train_sae = train_encoder
train_transcoder = train_encoder
