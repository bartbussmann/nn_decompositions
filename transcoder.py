import torch
import torch.nn.functional as F

from base import JumpReLU, SharedEncoder, StepFunction


class BaseTranscoder(SharedEncoder):
    """Transcoder base: maps input to different output."""

    def __init__(self, cfg):
        super().__init__(
            cfg["input_size"],
            cfg["output_size"],
            cfg["dict_size"],
            cfg,
        )

    def get_loss_dict(self, y_target, y_pred, acts, acts_sparse=None):
        """Compute loss dict for transcoder output.

        Args:
            y_target: Target activations
            y_pred: Predicted activations
            acts: Pre-sparsity activations
            acts_sparse: Post-sparsity activations (for topk variants), defaults to acts
        """
        if acts_sparse is None:
            acts_sparse = acts

        l2_loss = (y_pred.float() - y_target.float()).pow(2).mean()
        l1_norm = acts_sparse.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_sparse > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(y_target, y_pred, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        output = {
            "transcoder_out": y_pred,
            "feature_acts": acts_sparse,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, y_target, y_pred, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = y_target.float() - y_pred.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            y_pred_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (y_pred_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=y_target.dtype, device=y_target.device)


class VanillaTranscoder(BaseTranscoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x_in, y_target):
        acts = F.relu(x_in @ self.W_enc + self.b_enc)
        y_pred = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        return self.get_loss_dict(y_target, y_pred, acts)


class TopKTranscoder(BaseTranscoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x_in, y_target):
        acts = F.relu(x_in @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        y_pred = acts_topk @ self.W_dec + self.b_dec
        self.update_inactive_features(acts_topk)
        return self.get_loss_dict(y_target, y_pred, acts, acts_topk)


class BatchTopKTranscoder(BaseTranscoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x_in, y_target):
        acts = F.relu(x_in @ self.W_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"] * x_in.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        y_pred = acts_topk @ self.W_dec + self.b_dec
        self.update_inactive_features(acts_topk)
        return self.get_loss_dict(y_target, y_pred, acts, acts_topk)


class JumpReLUTranscoder(BaseTranscoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.jumprelu = JumpReLU(
            feature_size=cfg["dict_size"], bandwidth=cfg["bandwidth"], device=cfg["device"]
        )

    def forward(self, x_in, y_target):
        pre_activations = F.relu(x_in @ self.W_enc + self.b_enc)
        acts = self.jumprelu(pre_activations)
        y_pred = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        return self.get_jumprelu_loss_dict(y_target, y_pred, acts)

    def get_jumprelu_loss_dict(self, y_target, y_pred, acts):
        l2_loss = (y_pred.float() - y_target.float()).pow(2).mean()

        l0 = (
            StepFunction.apply(acts, self.jumprelu.log_threshold, self.cfg["bandwidth"])
            .sum(dim=-1)
            .mean()
        )
        sparsity_loss = self.cfg["l1_coeff"] * l0

        loss = l2_loss + sparsity_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        output = {
            "transcoder_out": y_pred,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "sparsity_loss": sparsity_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0,
        }
        return output
