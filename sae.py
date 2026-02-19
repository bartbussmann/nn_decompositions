"""SAE (Sparse Autoencoder) wrappers.

SAE is a special case of transcoder where input = target.
These thin wrappers call forward(x, x).
"""

from transcoder import BatchTopKTranscoder, JumpReLUTranscoder, TopKTranscoder, VanillaTranscoder
from config import SAEConfig


def _make_sae_class(base_class):
    """Factory to create SAE wrapper classes from transcoder base classes."""

    class SAE(base_class):
        def __init__(self, cfg: SAEConfig):
            super().__init__(cfg)

        def forward(self, x_in, y_target=None):
            if y_target is None:
                y_target = x_in
            return super().forward(x_in, y_target)

    # Strip "Transcoder" suffix before adding "SAE" suffix
    base_name = base_class.__name__.removesuffix("Transcoder")
    SAE.__name__ = f"{base_name}SAE"
    SAE.__qualname__ = f"{base_name}SAE"
    return SAE


VanillaSAE = _make_sae_class(VanillaTranscoder)
TopKSAE = _make_sae_class(TopKTranscoder)
BatchTopKSAE = _make_sae_class(BatchTopKTranscoder)
JumpReLUSAE = _make_sae_class(JumpReLUTranscoder)
