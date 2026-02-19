import torch
import torch.nn as nn

from config import EncoderConfig


class BaseEncoder(nn.Module):
    """Base class for encoder-decoder models (SAE, Transcoder, CLT).

    Subclasses implement encode() for their specific activation function.
    decode() and forward() are shared infrastructure.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        self.input_size = cfg.input_size
        self.output_size = cfg.output_size
        self.dict_size = cfg.dict_size
        self.num_input_layers = cfg.num_input_layers

        self.b_enc = nn.Parameter(torch.zeros(cfg.dict_size))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.input_size, cfg.dict_size))
        )

        if cfg.num_output_layers > 1:
            self.b_dec = nn.Parameter(torch.zeros(cfg.num_output_layers, cfg.output_size))
            self.W_dec = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(cfg.num_output_layers, cfg.dict_size, cfg.output_size)
                )
            )
        else:
            self.b_dec = nn.Parameter(torch.zeros(cfg.output_size))
            self.W_dec = nn.Parameter(
                torch.nn.init.kaiming_uniform_(torch.empty(cfg.dict_size, cfg.output_size))
            )
            # Initialize W_dec from W_enc only if input_size == output_size (SAE case)
            if cfg.input_size == cfg.output_size:
                self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        if cfg.enforce_layerwise_clt:
            assert cfg.num_input_layers > 1, "enforce_layerwise_clt requires num_input_layers > 1"
            assert cfg.num_output_layers > 1, "enforce_layerwise_clt requires num_output_layers > 1"
            assert cfg.num_input_layers == cfg.num_output_layers, (
                "enforce_layerwise_clt currently requires num_input_layers == num_output_layers"
            )
            assert cfg.input_size % cfg.num_input_layers == 0, (
                "input_size must be divisible by num_input_layers for layerwise CLT"
            )
            assert cfg.dict_size % cfg.num_input_layers == 0, (
                "dict_size must be divisible by num_input_layers for layerwise CLT"
            )
            self._init_layerwise_clt_masks()

        # Skip connection (per-layer for CLT)
        if cfg.skip_connection:
            if cfg.num_output_layers > 1:
                self.W_skip = nn.Parameter(
                    torch.zeros(cfg.num_output_layers, cfg.input_size, cfg.output_size)
                )
            else:
                self.W_skip = nn.Parameter(torch.zeros(cfg.input_size, cfg.output_size))
        else:
            self.W_skip = None

        # Post-encoder learnable per-feature scale
        if cfg.train_post_encoder:
            self.post_enc_scale = nn.Parameter(
                torch.full((cfg.dict_size,), cfg.post_encoder_scale)
            )
        else:
            self.post_enc_scale = None

        self.num_batches_not_active = torch.zeros((cfg.dict_size,)).to(cfg.device)

        self.to(cfg.dtype).to(cfg.device)

    # =========================================================================
    # Layerwise CLT masking
    # =========================================================================

    def _init_layerwise_clt_masks(self):
        """Build masks so features read from one input layer and write to downstream outputs only."""
        in_per_layer = self.input_size // self.cfg.num_input_layers
        feats_per_layer = self.dict_size // self.cfg.num_input_layers

        w_enc_mask = torch.zeros(self.input_size, self.dict_size)
        w_dec_mask = torch.zeros(self.cfg.num_output_layers, self.dict_size, self.output_size)

        for layer in range(self.cfg.num_input_layers):
            in_start = layer * in_per_layer
            in_end = (layer + 1) * in_per_layer
            feat_start = layer * feats_per_layer
            feat_end = (layer + 1) * feats_per_layer

            w_enc_mask[in_start:in_end, feat_start:feat_end] = 1.0
            # Feature group from source layer can only write to current + later output layers.
            w_dec_mask[layer:, feat_start:feat_end, :] = 1.0

        self.register_buffer("W_enc_mask", w_enc_mask)
        self.register_buffer("W_dec_mask", w_dec_mask)

    def _masked_W_enc(self) -> torch.Tensor:
        w_enc_mask = getattr(self, "W_enc_mask", None)
        if w_enc_mask is None:
            return self.W_enc
        return self.W_enc * w_enc_mask

    def _masked_W_dec(self) -> torch.Tensor:
        w_dec_mask = getattr(self, "W_dec_mask", None)
        if w_dec_mask is None:
            return self.W_dec
        return self.W_dec * w_dec_mask

    # =========================================================================
    # Preprocessing / postprocessing
    # =========================================================================

    def preprocess_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Normalize input if input_unit_norm is enabled."""
        if self.cfg.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        return x, None, None

    def postprocess_output(
        self, out: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None
    ) -> torch.Tensor:
        """Denormalize output if input_unit_norm is enabled."""
        if self.cfg.input_unit_norm and mean is not None:
            return out * std + mean
        return out

    # =========================================================================
    # Encode / decode
    # =========================================================================

    def _pre_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-encoder bias subtraction if applicable."""
        use_pre_enc_bias = (
            self.cfg.pre_enc_bias and self.input_size == self.output_size
            and self.cfg.num_output_layers == 1
        )
        return x - self.b_dec if use_pre_enc_bias else x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations. Override in subclasses."""
        raise NotImplementedError

    def decode(self, acts: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
        """Decode sparse activations to output(s). Includes skip connection."""
        W_dec = self._masked_W_dec()
        if self.cfg.num_output_layers > 1:
            y_pred = torch.einsum('bd,ldo->blo', acts, W_dec) + self.b_dec
        else:
            y_pred = acts @ W_dec + self.b_dec
        if self.W_skip is not None:
            if self.cfg.num_output_layers > 1:
                y_pred = y_pred + torch.einsum('bi,lio->blo', x_in, self.W_skip)
            else:
                y_pred = y_pred + x_in @ self.W_skip
        return y_pred

    def apply_post_encoder_scale(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """Apply learnable per-feature scale if train_post_encoder is enabled."""
        if self.post_enc_scale is not None:
            return pre_acts * self.post_enc_scale
        return pre_acts

    # =========================================================================
    # Training utilities
    # =========================================================================

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        w_dec_mask = getattr(self, "W_dec_mask", None)
        if w_dec_mask is not None:
            W_dec_normed = W_dec_normed * w_dec_mask
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        if w_dec_mask is not None:
            self.W_dec.grad *= w_dec_mask
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts: torch.Tensor):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    def _get_auxiliary_loss(
        self, y_target: torch.Tensor, y_pred: torch.Tensor, acts: torch.Tensor
    ) -> torch.Tensor:
        """Auxiliary loss to revive dead features (used by TopK variants)."""
        dead_features = self.num_batches_not_active >= self.cfg.n_batches_to_dead
        if dead_features.sum() > 0:
            residual = y_target.float() - y_pred.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg.top_k_aux, dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            if self.cfg.num_output_layers > 1:
                y_pred_aux = torch.einsum(
                    'bd,ldo->blo', acts_aux, self._masked_W_dec()[:, dead_features]
                )
            else:
                y_pred_aux = acts_aux @ self._masked_W_dec()[dead_features]
            return self.cfg.aux_penalty * (y_pred_aux.float() - residual.float()).pow(2).mean()
        return torch.tensor(0, dtype=y_target.dtype, device=y_target.device)

    def _build_loss_dict(
        self,
        y_target: torch.Tensor,
        y_pred: torch.Tensor,
        acts: torch.Tensor,
        y_pred_out: torch.Tensor,
        l0_norm: torch.Tensor,
        extra_losses: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        """Build standardized loss dict. l2_loss is computed here; callers pass
        additional loss terms (l1, aux, sparsity) via extra_losses."""
        l2_loss = (y_pred.float() - y_target.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        num_dead = (self.num_batches_not_active > self.cfg.n_batches_to_dead).sum()

        loss = l2_loss
        if extra_losses:
            loss = loss + sum(extra_losses.values())

        result = {
            "output": y_pred_out,
            "feature_acts": acts,
            "num_dead_features": num_dead,
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
        }
        if extra_losses:
            result.update(extra_losses)
        return result
