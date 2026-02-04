import torch
import torch.autograd as autograd
import torch.nn as nn


class SharedEncoder(nn.Module):
    """Base class for SAEs and Transcoders with shared encoder/decoder logic."""

    def __init__(self, input_size: int, output_size: int, dict_size: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg["seed"])

        # Encoder: input_size -> dict_size
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(input_size, dict_size))
        )
        self.b_enc = nn.Parameter(torch.zeros(dict_size))

        # Decoder: dict_size -> output_size
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(dict_size, output_size))
        )
        self.b_dec = nn.Parameter(torch.zeros(output_size))

        # Initialize W_dec from W_enc transpose (if same size)
        if input_size == output_size:
            self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        # Dead feature tracking
        self.num_batches_not_active = torch.zeros(dict_size, device=cfg["device"])
        self.to(cfg["dtype"]).to(cfg["device"])

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """Normalize decoder weights and project gradients."""
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        """Track dead features."""
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0


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
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReLU(nn.Module):
    def __init__(self, feature_size, bandwidth, device="cpu"):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x):
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
        return x_grad, threshold_grad, None  # None for bandwidth
