from contextlib import ExitStack
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from nn_decompositions.config import CLTConfig, EncoderConfig
from nn_decompositions.logs import (
    ComputeLossFn,
    init_wandb,
    log_clt_performance,
    log_encoder_performance,
    log_wandb,
    patched_forward,
    save_checkpoint,
)

GetLogitsFn = Callable[[nn.Module, torch.Tensor, torch.Tensor | None], torch.Tensor]


def _masked_kl(
    clean_logits: torch.Tensor,
    reconstr_logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """KL(clean || reconstr), masked to non-padding tokens."""
    clean_probs = F.softmax(clean_logits.detach(), dim=-1)
    reconstr_log_probs = F.log_softmax(reconstr_logits, dim=-1)
    if attention_mask is not None:
        mask = attention_mask.bool()
        clean_probs = clean_probs[mask]
        reconstr_log_probs = reconstr_log_probs[mask]
    return F.kl_div(reconstr_log_probs, clean_probs, reduction="batchmean")


def _e2e_step(encoder, activation_store, get_logits_fn: GetLogitsFn) -> dict:
    """Single end-to-end training step for transcoder/SAE."""
    cfg = encoder.cfg
    model = activation_store.model
    input_ids, attention_mask = activation_store.get_batch_tokens()

    # Clean forward: get logits, hooks capture intermediate activations
    with torch.no_grad():
        clean_logits = get_logits_fn(model, input_ids, attention_mask)
    input_acts = activation_store._input_acts
    output_acts = activation_store._output_acts

    # Encoder forward (with grad)
    input_flat = input_acts.reshape(-1, cfg.input_size)
    output_flat = output_acts.reshape(-1, cfg.output_size)
    output = encoder(input_flat, output_flat)
    reconstruction = output["output"].reshape(output_acts.shape)

    # Patched forward: model with encoder reconstruction
    with patched_forward(activation_store.output_module, lambda *a, **kw: reconstruction):
        reconstr_logits = get_logits_fn(model, input_ids, attention_mask)

    # Replace local MSE with KL, keep sparsity losses
    kl_loss = _masked_kl(clean_logits, reconstr_logits, attention_mask)
    sparsity_loss = output["loss"] - output["l2_loss"]
    output["kl_loss"] = kl_loss
    output["loss"] = kl_loss + sparsity_loss
    return output


def _e2e_step_clt(clt, activation_store, get_logits_fn: GetLogitsFn) -> dict:
    """Single end-to-end training step for CLT."""
    cfg = clt.cfg
    model = activation_store.model
    input_ids, attention_mask = activation_store.get_batch_tokens()

    with torch.no_grad():
        clean_logits = get_logits_fn(model, input_ids, attention_mask)
    input_acts_list = list(activation_store._input_acts)
    output_acts_list = list(activation_store._output_acts)

    seq_shape = output_acts_list[0].shape
    flat_inputs = [a.reshape(-1, cfg.input_size) for a in input_acts_list]
    flat_targets = [a.reshape(-1, cfg.output_size) for a in output_acts_list]
    output = clt(flat_inputs, flat_targets)
    reconstructions = [r.reshape(seq_shape) for r in output["output"]]

    def _make_const_fn(tensor):
        return lambda *a, **kw: tensor

    with ExitStack() as stack:
        for mod, recon in zip(activation_store.output_modules, reconstructions):
            stack.enter_context(patched_forward(mod, _make_const_fn(recon)))
        reconstr_logits = get_logits_fn(model, input_ids, attention_mask)

    kl_loss = _masked_kl(clean_logits, reconstr_logits, attention_mask)
    sparsity_loss = output["loss"] - output["l2_loss"]
    output["kl_loss"] = kl_loss
    output["loss"] = kl_loss + sparsity_loss
    return output


def train_encoder(
    encoder,
    activation_store,
    cfg: EncoderConfig | CLTConfig,
    compute_loss_fn: ComputeLossFn | None = None,
    get_logits_fn: GetLogitsFn | None = None,
):
    """Train any encoder (SAE, transcoder, or cross-layer transcoder)."""
    if cfg.e2e:
        assert get_logits_fn is not None, "get_logits_fn required for e2e training"
        dc = activation_store.data_config
        num_batches = cfg.num_tokens // (dc.model_batch_size * dc.seq_len)
    else:
        num_batches = cfg.num_tokens // cfg.batch_size

    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    pbar = tqdm.trange(num_batches)

    is_clt = isinstance(cfg, CLTConfig)
    log_perf = log_clt_performance if is_clt else log_encoder_performance

    wandb_run = init_wandb(cfg)

    for i in pbar:
        if cfg.e2e:
            output = (_e2e_step_clt if is_clt else _e2e_step)(
                encoder, activation_store, get_logits_fn
            )
        else:
            x_in, y_target = activation_store.next_batch()
            output = encoder(x_in, y_target)

        log_wandb(output, i, wandb_run)

        if i % cfg.perf_log_freq == 0:
            log_perf(wandb_run, i, activation_store, encoder,
                     compute_loss_fn=compute_loss_fn)

        if cfg.checkpoint_freq != "final" and i % cfg.checkpoint_freq == 0:
            save_checkpoint(encoder, cfg, i, wandb_run=wandb_run)

        loss = output["loss"]
        if cfg.e2e:
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "L0": f"{output['l0_norm']:.4f}",
                "KL": f"{output['kl_loss'].item():.4f}",
            })
        else:
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

    save_checkpoint(encoder, cfg, "final", wandb_run=wandb_run)


def train_encoder_group(
    encoders,
    activation_store,
    cfgs: list[EncoderConfig],
    compute_loss_fn: ComputeLossFn | None = None,
):
    """Train multiple encoders on the same activation stream."""
    num_batches = cfgs[0].num_tokens // cfgs[0].batch_size
    optimizers = [
        torch.optim.Adam(enc.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        for enc, cfg in zip(encoders, cfgs)
    ]
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfgs[0])

    for i in pbar:
        x_in, y_target = activation_store.next_batch()

        for encoder, cfg, optimizer in zip(encoders, cfgs, optimizers):
            output = encoder(x_in, y_target)

            log_wandb(output, i, wandb_run, suffix=cfg.encoder_type)

            if cfg.checkpoint_freq != "final" and i % cfg.checkpoint_freq == 0:
                save_checkpoint(encoder, cfg, i, wandb_run=wandb_run)

            loss = output["loss"]
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{output['l0_norm']:.4f}"})

            loss.backward()
            del output

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.max_grad_norm)
            encoder.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

        del x_in, y_target

        if i % cfgs[0].perf_log_freq == 0:
            torch.cuda.empty_cache()
            input_ids, attention_mask = activation_store.get_batch_tokens()
            batch_tokens = (
                input_ids[:cfgs[0].n_eval_seqs],
                attention_mask[:cfgs[0].n_eval_seqs] if attention_mask is not None else None,
            )
            for encoder, cfg in zip(encoders, cfgs):
                log_encoder_performance(
                    wandb_run, i, activation_store, encoder,
                    suffix=cfg.encoder_type, batch_tokens=batch_tokens,
                    compute_loss_fn=compute_loss_fn,
                )
            del batch_tokens
            torch.cuda.empty_cache()

    for encoder, cfg in zip(encoders, cfgs):
        save_checkpoint(encoder, cfg, "final", wandb_run=wandb_run)
