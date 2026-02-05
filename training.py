import torch
import tqdm

from config import EncoderConfig
from logs import (
    init_wandb,
    log_encoder_performance,
    log_wandb,
    save_checkpoint,
)


def train_encoder(encoder, activation_store, cfg: EncoderConfig, model=None):
    """Train any encoder (SAE or transcoder).

    Works with both ActivationsStore and TranscoderActivationsStore since
    both return (x_in, y_target) tuples from next_batch().

    Args:
        encoder: The encoder to train
        activation_store: Store providing (x_in, y_target) batches
        cfg: Encoder configuration
        model: Optional HookedRootModule for performance logging. If None,
               performance metrics are skipped (useful for non-TransformerLens models).
    """
    num_batches = cfg.num_tokens // cfg.batch_size
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)

    for i in pbar:
        x_in, y_target = activation_store.next_batch()
        output = encoder(x_in, y_target)

        log_wandb(output, i, wandb_run)

        if model is not None and i % cfg.perf_log_freq == 0:
            log_encoder_performance(wandb_run, i, model, activation_store, encoder)

        if cfg.checkpoint_freq != "final" and i % cfg.checkpoint_freq == 0:
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

    save_checkpoint(encoder, cfg, "final")


def train_encoder_group(encoders, activation_store, model, cfgs: list[EncoderConfig]):
    """Train multiple encoders on the same activation stream.

    All encoders share a single wandb run with indexed metrics (loss_0, loss_1, etc).
    """
    num_batches = cfgs[0].num_tokens // cfgs[0].batch_size
    optimizers = [
        torch.optim.Adam(enc.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        for enc, cfg in zip(encoders, cfgs)
    ]
    pbar = tqdm.trange(num_batches)

    # Single wandb run for all encoders
    wandb_run = init_wandb(cfgs[0])

    for i in pbar:
        x_in, y_target = activation_store.next_batch()

        for encoder, cfg, optimizer in zip(encoders, cfgs, optimizers):
            output = encoder(x_in, y_target)

            # Log with encoder type suffix (loss_topk, loss_batchtopk, etc)
            log_wandb(output, i, wandb_run, suffix=cfg.encoder_type)

            if cfg.checkpoint_freq != "final" and i % cfg.checkpoint_freq == 0:
                save_checkpoint(encoder, cfg, i)

            loss = output["loss"]
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{output['l0_norm']:.4f}"})

            loss.backward()
            del output  # Free memory before next encoder

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.max_grad_norm)
            encoder.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

        del x_in, y_target  # Free batch memory

        # Performance logging
        if i % cfgs[0].perf_log_freq == 0:
            torch.cuda.empty_cache()
            batch_tokens = activation_store.get_batch_tokens()[: cfgs[0].n_eval_seqs]
            for encoder, cfg in zip(encoders, cfgs):
                log_encoder_performance(
                    wandb_run, i, model, activation_store, encoder,
                    suffix=cfg.encoder_type, batch_tokens=batch_tokens
                )
            del batch_tokens
            torch.cuda.empty_cache()

    # Save final checkpoints
    for encoder, cfg in zip(encoders, cfgs):
        save_checkpoint(encoder, cfg, "final")
