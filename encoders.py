"""Encoder variants: Vanilla, TopK, BatchTopK, JumpReLU.

Each implements encode() for its specific activation function and
forward() for training with loss computation.
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from base import BaseEncoder
from config import EncoderConfig


class Vanilla(BaseEncoder):
    """L1-regularized encoder-decoder."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pre_encode(x)
        return F.relu(self.apply_post_encoder_scale(x @ self._masked_W_enc() + self.b_enc))

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        acts = self.encode(x_in)
        y_pred = self.decode(acts, x_in)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts)

        l0_norm = (acts > 0).float().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * acts.float().abs().sum(-1).mean()
        return self._build_loss_dict(
            y_target, y_pred, acts, y_pred_out, l0_norm,
            extra_losses={"l1_loss": l1_loss},
        )


class TopK(BaseEncoder):
    """TopK sparse encoder-decoder with auxiliary loss."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pre_encode(x)
        pre_acts = F.relu(self.apply_post_encoder_scale(x @ self._masked_W_enc()))
        topk = torch.topk(pre_acts, self.cfg.top_k, dim=-1)
        return torch.zeros_like(pre_acts).scatter(-1, topk.indices, topk.values)

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        # Inline encoding to retain pre_acts for auxiliary loss
        x_enc = self._pre_encode(x_in)
        pre_acts = F.relu(self.apply_post_encoder_scale(x_enc @ self._masked_W_enc()))
        topk = torch.topk(pre_acts, self.cfg.top_k, dim=-1)
        acts = torch.zeros_like(pre_acts).scatter(-1, topk.indices, topk.values)

        y_pred = self.decode(acts, x_in)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts)

        l0_norm = (acts > 0).float().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * acts.float().abs().sum(-1).mean()
        aux_loss = self._get_auxiliary_loss(y_target, y_pred, pre_acts)
        return self._build_loss_dict(
            y_target, y_pred, acts, y_pred_out, l0_norm,
            extra_losses={"l1_loss": l1_loss, "aux_loss": aux_loss},
        )


class BatchTopK(BaseEncoder):
    """BatchTopK - topk across flattened batch."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pre_encode(x)
        pre_acts = F.relu(self.apply_post_encoder_scale(x @ self._masked_W_enc()))
        topk = torch.topk(pre_acts.flatten(), self.cfg.top_k * x.shape[0], dim=-1)
        return (
            torch.zeros_like(pre_acts.flatten())
            .scatter(-1, topk.indices, topk.values)
            .reshape(pre_acts.shape)
        )

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        # Inline encoding to retain pre_acts for auxiliary loss
        x_enc = self._pre_encode(x_in)
        pre_acts = F.relu(self.apply_post_encoder_scale(x_enc @ self._masked_W_enc()))
        topk = torch.topk(pre_acts.flatten(), self.cfg.top_k * x_in.shape[0], dim=-1)
        acts = (
            torch.zeros_like(pre_acts.flatten())
            .scatter(-1, topk.indices, topk.values)
            .reshape(pre_acts.shape)
        )

        y_pred = self.decode(acts, x_in)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts)

        l0_norm = (acts > 0).float().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * acts.float().abs().sum(-1).mean()
        aux_loss = self._get_auxiliary_loss(y_target, y_pred, pre_acts)
        return self._build_loss_dict(
            y_target, y_pred, acts, y_pred_out, l0_norm,
            extra_losses={"l1_loss": l1_loss, "aux_loss": aux_loss},
        )


# =============================================================================
# JumpReLU activation components
# =============================================================================

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


class JumpReLU(BaseEncoder):
    """JumpReLU with learnable thresholds."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__(cfg)
        self.jumprelu = JumpReLUActivation(cfg.dict_size, cfg.bandwidth, cfg.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pre_encode(x)
        pre_acts = F.relu(self.apply_post_encoder_scale(x @ self._masked_W_enc() + self.b_enc))
        return self.jumprelu(pre_acts)

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        # Inline encoding to retain pre_acts for StepFunction L0
        x_enc = self._pre_encode(x_in)
        pre_acts = F.relu(
            self.apply_post_encoder_scale(x_enc @ self._masked_W_enc() + self.b_enc)
        )
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
