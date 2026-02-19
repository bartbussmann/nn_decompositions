"""Cross-Layer Transcoder (CLT) implementation.

A CLT learns sparse features that read from the residual stream at one layer
and write to MLP outputs at that layer and all subsequent layers. This enables
single features to capture multi-step computations that span multiple MLP layers.

Architecture per layer i (of n total layers):
  - Encoder: W_enc[i] (input_size, dict_size), b_enc[i] (dict_size)
  - Decoders: W_dec[i] (n-i, dict_size, output_size) — one decoder vector per target layer j >= i
  - Decoder bias: b_dec[j] (output_size) — one shared bias per target layer

Reconstruction at target layer j:
  y_hat[j] = b_dec[j] + sum_{i <= j} acts[i] @ W_dec[i][j-i]
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from config import CLTConfig


# =============================================================================
# JumpReLU activation components (shared with transcoder.py)
# =============================================================================

class _RectangleFunction(autograd.Function):
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


class _JumpReLUFunction(autograd.Function):
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
            * _RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class _StepFunction(autograd.Function):
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
            * _RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


# =============================================================================
# Cross-Layer Transcoder
# =============================================================================

class CrossLayerTranscoder(nn.Module):
    """Cross-layer transcoder with per-layer encoders and triangular decoders.

    Features at source layer i read from residual stream at layer i and write
    to MLP outputs at layers i, i+1, ..., n-1 via separate decoder vectors.
    """

    def __init__(self, cfg: CLTConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        n = cfg.n_layers

        # Per-layer encoders
        self.W_enc = nn.ParameterList([
            nn.Parameter(nn.init.kaiming_uniform_(torch.empty(cfg.input_size, cfg.dict_size)))
            for _ in range(n)
        ])
        self.b_enc = nn.ParameterList([
            nn.Parameter(torch.zeros(cfg.dict_size))
            for _ in range(n)
        ])

        # Triangular decoders: W_dec[i] has shape (n-i, dict_size, output_size)
        # W_dec[i][j-i] is the decoder for features at layer i writing to layer j
        self.W_dec = nn.ParameterList([
            nn.Parameter(nn.init.kaiming_uniform_(torch.empty(n - i, cfg.dict_size, cfg.output_size)))
            for i in range(n)
        ])
        # Normalize each decoder vector to unit norm
        for p in self.W_dec:
            p.data[:] = p.data / p.data.norm(dim=-1, keepdim=True)

        # Per-target-layer decoder bias
        self.b_dec = nn.ParameterList([
            nn.Parameter(torch.zeros(cfg.output_size))
            for _ in range(n)
        ])

        # Dead feature tracking (per layer)
        self.num_batches_not_active = [
            torch.zeros(cfg.dict_size, device=cfg.device) for _ in range(n)
        ]

        # JumpReLU: per-layer learnable thresholds
        if cfg.encoder_type == "jumprelu":
            self.jumprelu_log_thresholds = nn.ParameterList([
                nn.Parameter(torch.zeros(cfg.dict_size, device=cfg.device))
                for _ in range(n)
            ])

        self.to(cfg.dtype).to(cfg.device)

    # -----------------------------------------------------------------
    # Encoding
    # -----------------------------------------------------------------

    def encode_layer(
        self, x: torch.Tensor, layer_idx: int, return_dense: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode residual stream input at one layer into sparse activations."""
        pre_acts = F.relu(x @ self.W_enc[layer_idx] + self.b_enc[layer_idx])

        if self.cfg.encoder_type == "vanilla":
            return (pre_acts, pre_acts) if return_dense else pre_acts

        if self.cfg.encoder_type == "topk":
            topk = pre_acts.topk(self.cfg.top_k, dim=-1)
            acts = torch.zeros_like(pre_acts).scatter(-1, topk.indices, topk.values)
            return (acts, pre_acts) if return_dense else acts

        if self.cfg.encoder_type == "batchtopk":
            k = self.cfg.top_k * x.shape[0]
            topk = pre_acts.flatten().topk(k, dim=-1)
            acts = (
                torch.zeros_like(pre_acts.flatten())
                .scatter(-1, topk.indices, topk.values)
                .reshape(pre_acts.shape)
            )
            return (acts, pre_acts) if return_dense else acts

        if self.cfg.encoder_type == "jumprelu":
            log_thresh = self.jumprelu_log_thresholds[layer_idx]
            acts = _JumpReLUFunction.apply(pre_acts, log_thresh, self.cfg.bandwidth)
            return (acts, pre_acts) if return_dense else acts

        raise ValueError(f"Unknown encoder_type: {self.cfg.encoder_type}")

    # -----------------------------------------------------------------
    # Decoding
    # -----------------------------------------------------------------

    def decode(self, all_acts: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode sparse activations into per-layer MLP output reconstructions.

        Args:
            all_acts: List of (batch, dict_size) tensors, one per source layer.

        Returns:
            List of (batch, output_size) reconstructions, one per target layer.
        """
        n = self.cfg.n_layers
        reconstructions = []
        for j in range(n):
            recon = self.b_dec[j].unsqueeze(0).expand(all_acts[0].shape[0], -1).clone()
            for i in range(j + 1):
                # W_dec[i][j-i] shape: (dict_size, output_size)
                recon = recon + all_acts[i] @ self.W_dec[i][j - i]
            reconstructions.append(recon)
        return reconstructions

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------

    def forward(
        self, inputs: list[torch.Tensor], targets: list[torch.Tensor],
    ) -> dict:
        """Full forward pass: encode all layers, decode, compute joint loss.

        Args:
            inputs: Per-layer encoder inputs (residual stream), each (batch, input_size).
            targets: Per-layer decoder targets (MLP outputs), each (batch, output_size).

        Returns:
            Dict with loss, metrics, and reconstructions.
        """
        n = self.cfg.n_layers

        # Encode all layers
        all_acts = []
        all_dense = []
        for i in range(n):
            acts, dense = self.encode_layer(inputs[i], i, return_dense=True)
            all_acts.append(acts)
            all_dense.append(dense)
            self._update_inactive(acts, i)

        # Decode
        reconstructions = self.decode(all_acts)

        # L2 reconstruction loss (mean across layers, batch, and features)
        per_layer_l2 = [
            (recon.float() - target.float()).pow(2).mean()
            for recon, target in zip(reconstructions, targets)
        ]
        l2_loss = torch.stack(per_layer_l2).mean()

        # Aggregate feature activations for sparsity metrics
        acts_cat = torch.cat(all_acts, dim=-1)
        l0_norm = (acts_cat != 0).float().sum(-1).mean()
        l1_norm = acts_cat.float().abs().sum(-1).mean()

        # Sparsity and auxiliary losses
        extra_losses = self._compute_sparsity_losses(
            all_acts, all_dense, targets, reconstructions,
        )

        # Total loss
        loss = l2_loss + sum(extra_losses.values())

        # Dead feature count
        num_dead = sum(
            (tracker > self.cfg.n_batches_to_dead).sum()
            for tracker in self.num_batches_not_active
        )

        result = {
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "num_dead_features": num_dead,
            "feature_acts": acts_cat,
            "output": reconstructions,
            "per_layer_l2": per_layer_l2,
        }
        result.update(extra_losses)
        return result

    # -----------------------------------------------------------------
    # Loss components
    # -----------------------------------------------------------------

    def _compute_sparsity_losses(
        self,
        all_acts: list[torch.Tensor],
        all_dense: list[torch.Tensor],
        targets: list[torch.Tensor],
        reconstructions: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute encoder-type-specific loss terms."""
        acts_cat = torch.cat(all_acts, dim=-1)
        device = acts_cat.device

        if self.cfg.encoder_type == "vanilla":
            l1_loss = self.cfg.l1_coeff * acts_cat.float().abs().sum(-1).mean()
            return {"l1_loss": l1_loss}

        if self.cfg.encoder_type in ("topk", "batchtopk"):
            l1_loss = self.cfg.l1_coeff * acts_cat.float().abs().sum(-1).mean()
            aux_loss = self._get_auxiliary_loss(all_dense, targets, reconstructions)
            return {"l1_loss": l1_loss, "aux_loss": aux_loss}

        if self.cfg.encoder_type == "jumprelu":
            # Sparsity penalty via differentiable step function
            total_l0 = torch.tensor(0.0, device=device)
            for i in range(self.cfg.n_layers):
                log_thresh = self.jumprelu_log_thresholds[i]
                total_l0 = total_l0 + _StepFunction.apply(
                    all_dense[i], log_thresh, self.cfg.bandwidth,
                ).sum(-1).mean()
            sparsity_loss = self.cfg.l1_coeff * total_l0 / self.cfg.n_layers
            return {"sparsity_loss": sparsity_loss}

        raise ValueError(f"Unknown encoder_type: {self.cfg.encoder_type}")

    def _get_auxiliary_loss(
        self,
        all_dense: list[torch.Tensor],
        targets: list[torch.Tensor],
        reconstructions: list[torch.Tensor],
    ) -> torch.Tensor:
        """Auxiliary loss to revive dead features (TopK/BatchTopK).

        Per source layer: dead features try to reconstruct the residual at their
        own layer using their same-layer decoder vectors.
        """
        device = targets[0].device
        total_aux = torch.tensor(0.0, device=device)
        n_contributing = 0

        for i in range(self.cfg.n_layers):
            dead = self.num_batches_not_active[i] >= self.cfg.n_batches_to_dead
            if dead.sum() == 0:
                continue

            residual = targets[i].float() - reconstructions[i].float()
            dense_dead = all_dense[i][:, dead]
            k = min(self.cfg.top_k_aux, int(dead.sum()))
            topk = dense_dead.topk(k, dim=-1)
            acts_aux = torch.zeros_like(dense_dead).scatter(-1, topk.indices, topk.values)

            # Same-layer decoder for dead features: W_dec[i][0][dead]
            y_pred_aux = acts_aux @ self.W_dec[i][0][dead]
            total_aux = total_aux + (y_pred_aux.float() - residual).pow(2).mean()
            n_contributing += 1

        if n_contributing == 0:
            return torch.tensor(0.0, device=device)
        return self.cfg.aux_penalty * total_aux / n_contributing

    # -----------------------------------------------------------------
    # Maintenance
    # -----------------------------------------------------------------

    def _update_inactive(self, acts: torch.Tensor, layer_idx: int):
        tracker = self.num_batches_not_active[layer_idx]
        active_mask = acts.sum(0) > 0
        tracker += (~active_mask).float()
        tracker[active_mask] = 0

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """Normalize each decoder vector to unit norm and project gradients."""
        for p in self.W_dec:
            normed = p / p.norm(dim=-1, keepdim=True)
            if p.grad is not None:
                proj = (p.grad * normed).sum(-1, keepdim=True) * normed
                p.grad -= proj
            p.data = normed
