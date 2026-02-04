"""SAE (Sparse Autoencoder) wrappers.

SAE is a special case of transcoder where input = target.
These thin wrappers set input_size = output_size = act_size and call forward(x, x).
"""

from base import BatchTopK, JumpReLUEncoder, TopK, Vanilla


def _make_sae_class(base_class):
    """Factory to create SAE wrapper classes from transcoder base classes."""

    class SAE(base_class):
        def __init__(self, cfg: dict):
            # SAE: input_size = output_size = act_size
            sae_cfg = {
                **cfg,
                "input_size": cfg["act_size"],
                "output_size": cfg["act_size"],
            }
            super().__init__(sae_cfg)

        def forward(self, x):
            return super().forward(x, x)

    SAE.__name__ = f"{base_class.__name__}SAE"
    SAE.__qualname__ = f"{base_class.__name__}SAE"
    return SAE


VanillaSAE = _make_sae_class(Vanilla)
TopKSAE = _make_sae_class(TopK)
BatchTopKSAE = _make_sae_class(BatchTopK)
JumpReLUSAE = _make_sae_class(JumpReLUEncoder)
