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

        self.b_dec = nn.Parameter(torch.zeros(cfg.output_size))
        self.b_enc = nn.Parameter(torch.zeros(cfg.dict_size))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.input_size, cfg.dict_size))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.dict_size, cfg.output_size))
        )
        # Initialize W_dec from W_enc only if input_size == output_size (SAE case)
        if cfg.input_size == cfg.output_size:
            self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
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

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts: torch.Tensor):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0


class Vanilla(SharedEncoder):
    """Vanilla L1-regularized encoder-decoder."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__(cfg)

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        # Preprocess both input and target
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        x_enc = x_in - self.b_dec if self.cfg.pre_enc_bias else x_in
        acts = F.relu(x_enc @ self.W_enc + self.b_enc)
        y_pred = acts @ self.W_dec + self.b_dec

        # Postprocess output (denormalize with target's stats for user)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts)
        return self._get_loss_dict(y_target, y_pred, acts, y_pred_out)

    def _get_loss_dict(
        self,
        y_target: torch.Tensor,
        y_pred: torch.Tensor,
        acts: torch.Tensor,
        y_pred_out: torch.Tensor,
    ) -> dict:
        l2_loss = (y_pred.float() - y_target.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg.n_batches_to_dead
        ).sum()
        return {
            "output": y_pred_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
        }


class TopK(SharedEncoder):
    """TopK sparse encoder-decoder with auxiliary loss."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__(cfg)

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        # pre_enc_bias only works when input_size == output_size
        use_pre_enc_bias = self.cfg.pre_enc_bias and self.input_size == self.output_size
        x_enc = x_in - self.b_dec if use_pre_enc_bias else x_in
        acts = F.relu(x_enc @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg.top_k, dim=-1)
        acts_sparse = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        y_pred = acts_sparse @ self.W_dec + self.b_dec

        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts_sparse)
        return self._get_loss_dict(y_target, y_pred, acts, y_pred_out, acts_sparse)

    def _get_loss_dict(
        self,
        y_target: torch.Tensor,
        y_pred: torch.Tensor,
        acts: torch.Tensor,
        y_pred_out: torch.Tensor,
        acts_sparse: torch.Tensor,
    ) -> dict:
        l2_loss = (y_pred.float() - y_target.float()).pow(2).mean()
        l1_norm = acts_sparse.float().abs().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * l1_norm
        l0_norm = (acts_sparse > 0).float().sum(-1).mean()
        aux_loss = self._get_auxiliary_loss(y_target, y_pred, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg.n_batches_to_dead
        ).sum()
        return {
            "output": y_pred_out,
            "feature_acts": acts_sparse,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }

    def _get_auxiliary_loss(
        self, y_target: torch.Tensor, y_pred: torch.Tensor, acts: torch.Tensor
    ) -> torch.Tensor:
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
            y_pred_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg.aux_penalty
                * (y_pred_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=y_target.dtype, device=y_target.device)


class BatchTopK(SharedEncoder):
    """BatchTopK - topk across flattened batch."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__(cfg)

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        # pre_enc_bias only works when input_size == output_size
        use_pre_enc_bias = self.cfg.pre_enc_bias and self.input_size == self.output_size
        x_enc = x_in - self.b_dec if use_pre_enc_bias else x_in
        acts = F.relu(x_enc @ self.W_enc)
        acts_topk = torch.topk(
            acts.flatten(), self.cfg.top_k * x_in.shape[0], dim=-1
        )
        acts_sparse = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        y_pred = acts_sparse @ self.W_dec + self.b_dec

        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts_sparse)
        return self._get_loss_dict(y_target, y_pred, acts, y_pred_out, acts_sparse)

    def _get_loss_dict(
        self,
        y_target: torch.Tensor,
        y_pred: torch.Tensor,
        acts: torch.Tensor,
        y_pred_out: torch.Tensor,
        acts_sparse: torch.Tensor,
    ) -> dict:
        l2_loss = (y_pred.float() - y_target.float()).pow(2).mean()
        l1_norm = acts_sparse.float().abs().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * l1_norm
        l0_norm = (acts_sparse > 0).float().sum(-1).mean()
        aux_loss = self._get_auxiliary_loss(y_target, y_pred, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg.n_batches_to_dead
        ).sum()
        return {
            "output": y_pred_out,
            "feature_acts": acts_sparse,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }

    def _get_auxiliary_loss(
        self, y_target: torch.Tensor, y_pred: torch.Tensor, acts: torch.Tensor
    ) -> torch.Tensor:
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
            y_pred_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg.aux_penalty
                * (y_pred_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=y_target.dtype, device=y_target.device)


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

        x_enc = x_in - self.b_dec if self.cfg.pre_enc_bias else x_in
        pre_acts = F.relu(x_enc @ self.W_enc + self.b_enc)
        acts = self.jumprelu(pre_acts)
        y_pred = acts @ self.W_dec + self.b_dec

        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts)
        return self._get_loss_dict(y_target, y_pred, acts, y_pred_out)

    def _get_loss_dict(
        self,
        y_target: torch.Tensor,
        y_pred: torch.Tensor,
        acts: torch.Tensor,
        y_pred_out: torch.Tensor,
    ) -> dict:
        l2_loss = (y_pred.float() - y_target.float()).pow(2).mean()

        l0 = (
            StepFunction.apply(acts, self.jumprelu.log_threshold, self.cfg.bandwidth)
            .sum(dim=-1)
            .mean()
        )
        sparsity_loss = self.cfg.l1_coeff * l0

        loss = l2_loss + sparsity_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg.n_batches_to_dead
        ).sum()

        return {
            "output": y_pred_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "sparsity_loss": sparsity_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0,
        }
