"""SAE (Sparse Autoencoder) wrappers.

SAE is a special case of transcoder where input = target.
These thin wrappers call forward(x, x).
"""

from encoders import BatchTopK, JumpReLU, TopK, Vanilla
from config import SAEConfig


def _make_sae_class(base_class):
    """Factory to create SAE wrapper classes from encoder base classes."""

    class SAE(base_class):
        def __init__(self, cfg: SAEConfig):
            super().__init__(cfg)

        def forward(self, x):
            return super().forward(x, x)

    SAE.__name__ = f"{base_class.__name__}SAE"
    SAE.__qualname__ = f"{base_class.__name__}SAE"
    return SAE


VanillaSAE = _make_sae_class(Vanilla)
TopKSAE = _make_sae_class(TopK)
BatchTopKSAE = _make_sae_class(BatchTopK)
JumpReLUSAE = _make_sae_class(JumpReLU)
