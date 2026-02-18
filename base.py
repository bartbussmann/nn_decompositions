import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from config import EncoderConfig


class SharedEncoder(nn.Module):
    """Base class for encoder-decoder models (SAE and Transcoder).

    Supports both SAE mode (input = target) and Transcoder mode (input != target).
    All subclasses use forward(x_in, y_target) signature.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        self.input_size = cfg.input_size
        self.output_size = cfg.output_size
        self.dict_size = cfg.dict_size
        self.num_input_layers = cfg.num_input_layers
        self.multi_layer = cfg.num_output_layers > 1
        self.true_clt = self.multi_layer and self.num_input_layers > 1

        if self.true_clt:
            assert self.input_size % self.num_input_layers == 0, (
                "For CLT with multiple input layers, input_size must be divisible by num_input_layers"
            )
            self.input_size_per_layer = self.input_size // self.num_input_layers
        else:
            self.input_size_per_layer = self.input_size

        if self.true_clt:
            self.b_enc = nn.Parameter(torch.zeros(self.num_input_layers, cfg.dict_size))
            self.W_enc = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(self.num_input_layers, self.input_size_per_layer, cfg.dict_size)
                )
            )
        else:
            self.b_enc = nn.Parameter(torch.zeros(cfg.dict_size))
            self.W_enc = nn.Parameter(
                torch.nn.init.kaiming_uniform_(torch.empty(cfg.input_size, cfg.dict_size))
            )

        if self.multi_layer:
            self.b_dec = nn.Parameter(torch.zeros(cfg.num_output_layers, cfg.output_size))
            if self.true_clt:
                self.W_dec = nn.Parameter(
                    torch.nn.init.kaiming_uniform_(
                        torch.empty(
                            self.num_input_layers,
                            cfg.num_output_layers,
                            cfg.dict_size,
                            cfg.output_size,
                        )
                    )
                )
            else:
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
        if self.true_clt:
            # Causal write mask: source layer s can only write to target layer t >= s.
            causal = torch.triu(
                torch.ones(self.num_input_layers, cfg.num_output_layers), diagonal=0
            )
            self.register_buffer("causal_mask", causal)
            self.W_dec.data.mul_(self.causal_mask[:, :, None, None])

        # Skip connection (per-layer for CLT)
        if cfg.skip_connection:
            if self.multi_layer:
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

    def decode(self, acts: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
        """Decode sparse activations to output(s). Includes skip connection."""
        if self.multi_layer:
            if self.true_clt:
                w_dec = self.W_dec * self.causal_mask[:, :, None, None]
                y_pred = torch.einsum('bsd,stdo->bto', acts, w_dec) + self.b_dec
            else:
                y_pred = torch.einsum('bd,ldo->blo', acts, self.W_dec) + self.b_dec
        else:
            y_pred = acts @ self.W_dec + self.b_dec
        if self.W_skip is not None:
            if self.multi_layer:
                y_pred = y_pred + torch.einsum('bi,lio->blo', x_in, self.W_skip)
            else:
                y_pred = y_pred + x_in @ self.W_skip
        return y_pred

    def encode_linear(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.true_clt:
            x_layers = x_enc.reshape(x_enc.shape[0], self.num_input_layers, self.input_size_per_layer)
            return torch.einsum('bsi,sid->bsd', x_layers, self.W_enc) + self.b_enc
        return x_enc @ self.W_enc + self.b_enc

    def apply_post_encoder_scale(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """Apply learnable per-feature scale if train_post_encoder is enabled."""
        if self.post_enc_scale is not None:
            return pre_acts * self.post_enc_scale
        return pre_acts

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        W_dec_normed = self.W_dec / norms
        if self.W_dec.grad is not None:
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
                -1, keepdim=True
            ) * W_dec_normed
            self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed
        if self.true_clt:
            self.W_dec.data.mul_(self.causal_mask[:, :, None, None])
            if self.W_dec.grad is not None:
                self.W_dec.grad.mul_(self.causal_mask[:, :, None, None])

    def update_inactive_features(self, acts: torch.Tensor):
        if acts.ndim == 3:
            active = acts.sum((0, 1)) > 0
        else:
            active = acts.sum(0) > 0
        self.num_batches_not_active += (~active).float()
        self.num_batches_not_active[active] = 0

    def _get_auxiliary_loss(
        self, y_target: torch.Tensor, y_pred: torch.Tensor, acts: torch.Tensor
    ) -> torch.Tensor:
        """Auxiliary loss to revive dead features (used by TopK variants)."""
        dead_features = self.num_batches_not_active >= self.cfg.n_batches_to_dead
        if dead_features.sum() > 0:
            residual = y_target.float() - y_pred.float()
            if acts.ndim == 3:
                acts_dead = acts[:, :, dead_features].reshape(-1, dead_features.sum())
                acts_topk_aux = torch.topk(
                    acts_dead,
                    min(self.cfg.top_k_aux, dead_features.sum()),
                    dim=-1,
                )
                acts_aux = torch.zeros_like(acts_dead).scatter(
                    -1, acts_topk_aux.indices, acts_topk_aux.values
                ).reshape(acts.shape[0], acts.shape[1], dead_features.sum())

                w_dec = (self.W_dec * self.causal_mask[:, :, None, None])[:, :, dead_features]
                y_pred_aux = torch.einsum('bsd,stdo->bto', acts_aux, w_dec)
            elif self.multi_layer:
                acts_topk_aux = torch.topk(
                    acts[:, dead_features],
                    min(self.cfg.top_k_aux, dead_features.sum()),
                    dim=-1,
                )
                acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                    -1, acts_topk_aux.indices, acts_topk_aux.values
                )
                y_pred_aux = torch.einsum('bd,ldo->blo', acts_aux, self.W_dec[:, dead_features])
            else:
                acts_topk_aux = torch.topk(
                    acts[:, dead_features],
                    min(self.cfg.top_k_aux, dead_features.sum()),
                    dim=-1,
                )
                acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                    -1, acts_topk_aux.indices, acts_topk_aux.values
                )
                y_pred_aux = acts_aux @ self.W_dec[dead_features]
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


class Vanilla(SharedEncoder):
    """Vanilla L1-regularized encoder-decoder."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__(cfg)

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        use_pre_enc_bias = (
            self.cfg.pre_enc_bias and self.input_size == self.output_size
            and not self.true_clt
        )
        x_enc = x_in - self.b_dec if use_pre_enc_bias else x_in
        acts = F.relu(self.apply_post_encoder_scale(self.encode_linear(x_enc)))
        y_pred = self.decode(acts, x_in)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts)

        l0_norm = (acts > 0).float().sum(-1).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * l1_norm
        return self._build_loss_dict(
            y_target, y_pred, acts, y_pred_out, l0_norm,
            extra_losses={"l1_loss": l1_loss},
        )


class TopK(SharedEncoder):
    """TopK sparse encoder-decoder with auxiliary loss."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__(cfg)

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        use_pre_enc_bias = (
            self.cfg.pre_enc_bias and self.input_size == self.output_size
            and not self.true_clt
        )
        x_enc = x_in - self.b_dec if use_pre_enc_bias else x_in
        acts = F.relu(self.apply_post_encoder_scale(self.encode_linear(x_enc)))
        acts_topk = torch.topk(acts, self.cfg.top_k, dim=-1)
        acts_sparse = torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)
        y_pred = self.decode(acts_sparse, x_in)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts_sparse)

        l0_norm = (acts_sparse > 0).float().sum(-1).mean()
        l1_norm = acts_sparse.float().abs().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * l1_norm
        aux_loss = self._get_auxiliary_loss(y_target, y_pred, acts)
        return self._build_loss_dict(
            y_target, y_pred, acts_sparse, y_pred_out, l0_norm,
            extra_losses={"l1_loss": l1_loss, "aux_loss": aux_loss},
        )


class BatchTopK(SharedEncoder):
    """BatchTopK - topk across flattened batch."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__(cfg)

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        use_pre_enc_bias = (
            self.cfg.pre_enc_bias and self.input_size == self.output_size
            and not self.true_clt
        )
        x_enc = x_in - self.b_dec if use_pre_enc_bias else x_in
        acts = F.relu(self.apply_post_encoder_scale(self.encode_linear(x_enc)))
        mult = x_in.shape[0] * (self.num_input_layers if self.true_clt else 1)
        acts_topk = torch.topk(acts.flatten(), self.cfg.top_k * mult, dim=-1)
        acts_sparse = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        y_pred = self.decode(acts_sparse, x_in)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts_sparse)

        l0_norm = (acts_sparse > 0).float().sum(-1).mean()
        l1_norm = acts_sparse.float().abs().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * l1_norm
        aux_loss = self._get_auxiliary_loss(y_target, y_pred, acts)
        return self._build_loss_dict(
            y_target, y_pred, acts_sparse, y_pred_out, l0_norm,
            extra_losses={"l1_loss": l1_loss, "aux_loss": aux_loss},
        )


# JumpReLU activation components
class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class JumpReLUActivation(nn.Module):
    def __init__(self, feature_size: int, bandwidth: float, device: str = "cpu"):
        super().__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class JumpReLUEncoder(SharedEncoder):
    """JumpReLU with learnable thresholds."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__(cfg)
        self.jumprelu = JumpReLUActivation(cfg.dict_size, cfg.bandwidth, cfg.device)

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        use_pre_enc_bias = (
            self.cfg.pre_enc_bias and self.input_size == self.output_size
            and not self.true_clt
        )
        x_enc = x_in - self.b_dec if use_pre_enc_bias else x_in
        pre_acts = F.relu(self.apply_post_encoder_scale(self.encode_linear(x_enc)))
        acts = self.jumprelu(pre_acts)
        y_pred = self.decode(acts, x_in)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts)

        l0_norm = (
            StepFunction.apply(pre_acts, self.jumprelu.log_threshold, self.cfg.bandwidth)
            .sum(dim=-1)
            .mean()
        )
        sparsity_loss = self.cfg.l1_coeff * l0_norm
        return self._build_loss_dict(
            y_target, y_pred, acts, y_pred_out, l0_norm,
            extra_losses={"sparsity_loss": sparsity_loss},
        )
