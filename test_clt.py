"""Smoke tests for CLT (Cross-Layer Transcoder) support.

Tests both backward compatibility (single-layer) and new multi-layer functionality.
"""

import torch
import torch.nn as nn

from base import BatchTopK, JumpReLUEncoder, TopK, Vanilla
from config import EncoderConfig, SAEConfig
from sae import TopKSAE


def test_single_layer_backward_compat():
    """Existing single-layer transcoder should work identically."""
    cfg = EncoderConfig(
        input_size=64,
        output_size=64,
        dict_size=256,
        encoder_type="topk",
        top_k=8,
        device="cpu",
    )
    encoder = TopK(cfg)

    x_in = torch.randn(16, 64)
    y_target = torch.randn(16, 64)
    output = encoder(x_in, y_target)

    assert output["output"].shape == (16, 64)
    assert output["feature_acts"].shape == (16, 256)
    assert output["loss"].requires_grad
    output["loss"].backward()
    assert encoder.W_dec.shape == (256, 64)
    assert encoder.b_dec.shape == (64,)
    print("PASS: single-layer backward compat (TopK)")


def test_sae_backward_compat():
    """SAE wrapper should still work."""
    cfg = SAEConfig(input_size=64, output_size=64, dict_size=256, top_k=8, device="cpu")
    sae = TopKSAE(cfg)

    x = torch.randn(16, 64)
    output = sae(x)

    assert output["output"].shape == (16, 64)
    assert output["loss"].requires_grad
    output["loss"].backward()
    print("PASS: SAE backward compat")


def test_multi_layer_topk():
    """Multi-layer CLT with TopK encoder: per-layer encoders + causal decoder."""
    num_layers = 3
    d_model = 64
    dict_size = 256
    cfg = EncoderConfig(
        input_size=d_model * num_layers,
        output_size=d_model,
        dict_size=dict_size,
        encoder_type="topk",
        top_k=8,
        num_input_layers=num_layers,
        num_output_layers=num_layers,
        device="cpu",
    )
    encoder = TopK(cfg)

    # Per-layer encoder: (L, d_model, dict_size)
    assert encoder.W_enc.shape == (num_layers, d_model, dict_size), \
        f"W_enc shape: {encoder.W_enc.shape}"
    # Causal decoder: (L, L, dict_size, d_model)
    assert encoder.W_dec.shape == (num_layers, num_layers, dict_size, d_model), \
        f"W_dec shape: {encoder.W_dec.shape}"
    assert encoder.b_dec.shape == (num_layers, d_model), \
        f"b_dec shape: {encoder.b_dec.shape}"

    x_in = torch.randn(16, d_model * num_layers)
    y_target = torch.randn(16, num_layers, d_model)
    output = encoder(x_in, y_target)

    assert output["output"].shape == (16, num_layers, d_model), \
        f"output shape: {output['output'].shape}"
    # Total features = num_layers * dict_size
    assert output["feature_acts"].shape == (16, num_layers * dict_size), \
        f"feature_acts shape: {output['feature_acts'].shape}"
    assert output["loss"].requires_grad

    output["loss"].backward()
    assert encoder.W_dec.grad is not None
    assert encoder.W_enc.grad is not None
    print("PASS: multi-layer TopK CLT")


def test_multi_layer_vanilla():
    """Multi-layer CLT with Vanilla encoder."""
    num_layers = 3
    d_model = 64
    dict_size = 256
    cfg = EncoderConfig(
        input_size=d_model * num_layers,
        output_size=d_model,
        dict_size=dict_size,
        encoder_type="vanilla",
        num_input_layers=num_layers,
        num_output_layers=num_layers,
        device="cpu",
    )
    encoder = Vanilla(cfg)
    x_in = torch.randn(16, d_model * num_layers)
    y_target = torch.randn(16, num_layers, d_model)
    output = encoder(x_in, y_target)
    assert output["output"].shape == (16, num_layers, d_model)
    assert output["feature_acts"].shape == (16, num_layers * dict_size)
    output["loss"].backward()
    print("PASS: multi-layer Vanilla CLT")


def test_multi_layer_batchtopk():
    """Multi-layer CLT with BatchTopK encoder."""
    num_layers = 3
    d_model = 64
    dict_size = 256
    cfg = EncoderConfig(
        input_size=d_model * num_layers,
        output_size=d_model,
        dict_size=dict_size,
        encoder_type="batchtopk",
        top_k=8,
        num_input_layers=num_layers,
        num_output_layers=num_layers,
        device="cpu",
    )
    encoder = BatchTopK(cfg)
    x_in = torch.randn(16, d_model * num_layers)
    y_target = torch.randn(16, num_layers, d_model)
    output = encoder(x_in, y_target)
    assert output["output"].shape == (16, num_layers, d_model)
    assert output["feature_acts"].shape == (16, num_layers * dict_size)
    output["loss"].backward()
    print("PASS: multi-layer BatchTopK CLT")


def test_multi_layer_jumprelu():
    """Multi-layer CLT with JumpReLU encoder."""
    num_layers = 3
    d_model = 64
    dict_size = 256
    cfg = EncoderConfig(
        input_size=d_model * num_layers,
        output_size=d_model,
        dict_size=dict_size,
        encoder_type="jumprelu",
        num_input_layers=num_layers,
        num_output_layers=num_layers,
        device="cpu",
    )
    encoder = JumpReLUEncoder(cfg)
    x_in = torch.randn(16, d_model * num_layers)
    y_target = torch.randn(16, num_layers, d_model)
    output = encoder(x_in, y_target)
    assert output["output"].shape == (16, num_layers, d_model)
    assert output["feature_acts"].shape == (16, num_layers * dict_size)
    output["loss"].backward()
    print("PASS: multi-layer JumpReLU CLT")


def test_skip_connection_single_layer():
    """Skip connection with single layer."""
    cfg = EncoderConfig(
        input_size=64,
        output_size=64,
        dict_size=256,
        top_k=8,
        skip_connection=True,
        device="cpu",
    )
    encoder = TopK(cfg)
    assert encoder.W_skip is not None
    assert encoder.W_skip.shape == (64, 64)

    x_in = torch.randn(16, 64)
    y_target = torch.randn(16, 64)
    output = encoder(x_in, y_target)
    output["loss"].backward()
    assert encoder.W_skip.grad is not None
    print("PASS: skip connection single-layer")


def test_skip_connection_multi_layer():
    """Per-layer skip connection with CLT."""
    num_layers = 3
    d_model = 64
    cfg = EncoderConfig(
        input_size=d_model * num_layers,
        output_size=d_model,
        dict_size=256,
        top_k=8,
        skip_connection=True,
        num_input_layers=num_layers,
        num_output_layers=num_layers,
        device="cpu",
    )
    encoder = TopK(cfg)
    assert encoder.W_skip is not None
    # Per-layer skip: (L, d_in_per_layer, output_size)
    assert encoder.W_skip.shape == (num_layers, d_model, d_model), \
        f"W_skip shape: {encoder.W_skip.shape}"

    x_in = torch.randn(16, d_model * num_layers)
    y_target = torch.randn(16, num_layers, d_model)
    output = encoder(x_in, y_target)
    output["loss"].backward()
    assert encoder.W_skip.grad is not None
    print("PASS: skip connection multi-layer CLT")


def test_post_encoder_scale():
    """Post-encoder learnable scale."""
    cfg = EncoderConfig(
        input_size=64,
        output_size=64,
        dict_size=256,
        top_k=8,
        train_post_encoder=True,
        post_encoder_scale=0.5,
        device="cpu",
    )
    encoder = TopK(cfg)
    assert encoder.post_enc_scale is not None
    assert encoder.post_enc_scale.shape == (256,)
    assert torch.allclose(encoder.post_enc_scale.data, torch.full((256,), 0.5))

    x_in = torch.randn(16, 64)
    y_target = torch.randn(16, 64)
    output = encoder(x_in, y_target)
    output["loss"].backward()
    assert encoder.post_enc_scale.grad is not None
    print("PASS: post-encoder scale")


def test_post_encoder_scale_clt():
    """Post-encoder learnable scale with CLT has total_features entries."""
    num_layers = 3
    d_model = 64
    dict_size = 256
    cfg = EncoderConfig(
        input_size=d_model * num_layers,
        output_size=d_model,
        dict_size=dict_size,
        top_k=8,
        train_post_encoder=True,
        post_encoder_scale=0.5,
        num_input_layers=num_layers,
        num_output_layers=num_layers,
        device="cpu",
    )
    encoder = TopK(cfg)
    assert encoder.post_enc_scale is not None
    # Total features = num_layers * dict_size
    assert encoder.post_enc_scale.shape == (num_layers * dict_size,)

    x_in = torch.randn(16, d_model * num_layers)
    y_target = torch.randn(16, num_layers, d_model)
    output = encoder(x_in, y_target)
    output["loss"].backward()
    assert encoder.post_enc_scale.grad is not None
    print("PASS: post-encoder scale CLT")


def test_decoder_norm():
    """Decoder weight normalization works for both single and multi-layer."""
    # Single layer
    cfg1 = EncoderConfig(
        input_size=64, output_size=64, dict_size=256, top_k=8, device="cpu",
    )
    enc1 = TopK(cfg1)
    x_in = torch.randn(16, 64)
    y_target = torch.randn(16, 64)
    out1 = enc1(x_in, y_target)
    out1["loss"].backward()
    enc1.make_decoder_weights_and_grad_unit_norm()
    norms1 = enc1.W_dec.norm(dim=-1)
    assert torch.allclose(norms1, torch.ones_like(norms1), atol=1e-6)
    print("PASS: decoder norm single-layer")

    # Multi-layer CLT
    num_layers = 3
    d_model = 64
    dict_size = 256
    cfg2 = EncoderConfig(
        input_size=d_model * num_layers, output_size=d_model, dict_size=dict_size,
        top_k=8, num_input_layers=num_layers, num_output_layers=num_layers,
        device="cpu",
    )
    enc2 = TopK(cfg2)
    x_in2 = torch.randn(16, d_model * num_layers)
    y_target2 = torch.randn(16, num_layers, d_model)
    out2 = enc2(x_in2, y_target2)
    out2["loss"].backward()
    enc2.make_decoder_weights_and_grad_unit_norm()
    # Shape (L, L, D, d_model) → norms along dim=-1 → (L, L, D)
    norms2 = enc2.W_dec.norm(dim=-1)
    # Upper triangle (causal entries) should have unit norms, lower triangle should be zero
    causal_mask = torch.triu(torch.ones(num_layers, num_layers))
    causal_norms = norms2[causal_mask.bool()]
    non_causal_norms = norms2[~causal_mask.bool()]
    assert torch.allclose(causal_norms, torch.ones_like(causal_norms), atol=1e-5), \
        f"Causal entries norms not unit: {causal_norms}"
    assert torch.allclose(non_causal_norms, torch.zeros_like(non_causal_norms), atol=1e-6), \
        f"Non-causal entries norms not zero: {non_causal_norms}"
    print("PASS: decoder norm multi-layer CLT")


def test_auxiliary_loss_multi_layer():
    """Auxiliary loss with dead features works for CLT."""
    num_layers = 3
    d_model = 64
    dict_size = 256
    cfg = EncoderConfig(
        input_size=d_model * num_layers,
        output_size=d_model,
        dict_size=dict_size,
        top_k=8,
        num_input_layers=num_layers,
        num_output_layers=num_layers,
        n_batches_to_dead=0,  # Force all features to be "dead"
        device="cpu",
    )
    encoder = TopK(cfg)
    # Simulate all features being dead
    encoder.num_batches_not_active[:] = 1

    x_in = torch.randn(16, d_model * num_layers)
    y_target = torch.randn(16, num_layers, d_model)
    output = encoder(x_in, y_target)
    assert "aux_loss" in output
    assert output["aux_loss"].item() > 0
    output["loss"].backward()
    print("PASS: auxiliary loss multi-layer CLT")


def test_activation_store_multi_io():
    """ActivationsStore accepts lists of input and output modules."""
    from activation_store import ActivationsStore, DataConfig
    from unittest.mock import MagicMock

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Linear(32, 32)
            self.layer1 = nn.Linear(32, 32)
            self.layer2 = nn.Linear(32, 32)

        def forward(self, x, **kwargs):
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    model = SimpleModel()

    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0

    data_config = DataConfig(
        dataset_name="dummy",
        tokenizer=tokenizer,
        device="cpu",
    )

    # Test single input/output (backward compat)
    store1 = ActivationsStore.__new__(ActivationsStore)
    store1.model = model
    store1.input_modules = [model.layer0]
    store1.output_modules = [model.layer1]
    store1.num_input_layers = 1
    store1.num_output_layers = 1
    store1.data_config = data_config
    store1.input_size = 32
    store1.output_size = 32
    store1._input_acts_list = [None]
    store1._output_acts_list = [None]

    model.layer0.register_forward_hook(lambda m, i, o: store1._input_acts_list.__setitem__(0, o.detach()))
    model.layer1.register_forward_hook(lambda m, i, o: store1._output_acts_list.__setitem__(0, o.detach()))

    x = torch.randn(4, 32)
    model(x)
    assert store1._input_acts_list[0] is not None
    assert store1._output_acts_list[0] is not None
    assert store1.output_module == model.layer1
    print("PASS: activation store single input/output")

    # Test multiple inputs and outputs (CLT)
    store2 = ActivationsStore.__new__(ActivationsStore)
    store2.model = model
    store2.input_modules = [model.layer0, model.layer1]
    store2.output_modules = [model.layer1, model.layer2]
    store2.num_input_layers = 2
    store2.num_output_layers = 2
    store2.data_config = data_config
    store2.input_size = 64  # 2 * 32 (concatenated)
    store2.output_size = 32
    store2._input_acts_list = [None, None]
    store2._output_acts_list = [None, None]

    def make_hook(lst, idx):
        def hook(m, i, o):
            lst[idx] = o.detach()
        return hook

    model.layer0.register_forward_hook(make_hook(store2._input_acts_list, 0))
    model.layer1.register_forward_hook(make_hook(store2._input_acts_list, 1))
    model.layer1.register_forward_hook(make_hook(store2._output_acts_list, 0))
    model.layer2.register_forward_hook(make_hook(store2._output_acts_list, 1))

    model(x)
    assert all(a is not None for a in store2._input_acts_list)
    assert all(a is not None for a in store2._output_acts_list)

    # get_activations should concatenate inputs
    input_acts, output_acts = store2.get_activations(x)
    assert input_acts.shape[-1] == 64, f"Expected concatenated input dim 64, got {input_acts.shape[-1]}"
    assert output_acts.shape[-2] == 2, f"Expected 2 output layers, got {output_acts.shape}"
    print("PASS: activation store multi-input/output (CLT)")


def test_all_features_combined():
    """CLT with all features: multi-layer + skip + post-encoder scale."""
    num_layers = 3
    d_model = 64
    dict_size = 256
    cfg = EncoderConfig(
        input_size=d_model * num_layers,
        output_size=d_model,
        dict_size=dict_size,
        top_k=8,
        num_input_layers=num_layers,
        num_output_layers=num_layers,
        skip_connection=True,
        train_post_encoder=True,
        post_encoder_scale=0.5,
        device="cpu",
    )
    encoder = TopK(cfg)
    assert encoder.W_skip.shape == (num_layers, d_model, d_model)
    assert encoder.post_enc_scale.shape == (num_layers * dict_size,)

    x_in = torch.randn(16, d_model * num_layers)
    y_target = torch.randn(16, num_layers, d_model)
    output = encoder(x_in, y_target)
    assert output["output"].shape == (16, num_layers, d_model)
    output["loss"].backward()

    # All parameters have gradients
    assert encoder.W_enc.grad is not None
    assert encoder.W_dec.grad is not None
    assert encoder.W_skip.grad is not None
    assert encoder.post_enc_scale.grad is not None
    print("PASS: all features combined")


def test_clt_concat_input_multi_output():
    """Full CLT: concatenated multi-layer input → multi-layer output with causal decoder."""
    num_layers = 4
    d_model = 32
    dict_size = 256
    cfg = EncoderConfig(
        input_size=d_model * num_layers,
        output_size=d_model,
        dict_size=dict_size,
        top_k=8,
        num_input_layers=num_layers,
        num_output_layers=num_layers,
        device="cpu",
    )
    encoder = TopK(cfg)
    # Per-layer encoder
    assert encoder.W_enc.shape == (num_layers, d_model, dict_size)
    # Causal decoder
    assert encoder.W_dec.shape == (num_layers, num_layers, dict_size, d_model)

    x_in = torch.randn(16, d_model * num_layers)
    y_target = torch.randn(16, num_layers, d_model)
    output = encoder(x_in, y_target)

    assert output["output"].shape == (16, num_layers, d_model)
    assert output["feature_acts"].shape == (16, num_layers * dict_size)
    output["loss"].backward()
    assert encoder.W_enc.grad is not None
    assert encoder.W_dec.grad is not None
    print("PASS: CLT concat input → multi output")


def test_causal_structure():
    """Verify the causal decoder: features at layer s only write to layers >= s."""
    num_layers = 3
    d_model = 32
    dict_size = 64
    cfg = EncoderConfig(
        input_size=d_model * num_layers,
        output_size=d_model,
        dict_size=dict_size,
        top_k=dict_size * num_layers,  # keep all features active
        num_input_layers=num_layers,
        num_output_layers=num_layers,
        device="cpu",
    )
    encoder = TopK(cfg)

    # Verify non-causal entries of W_dec are zero (lower triangle, where src > tgt)
    causal_mask = torch.triu(torch.ones(num_layers, num_layers))
    non_causal = encoder.W_dec.data[~causal_mask.bool()]
    assert torch.allclose(non_causal, torch.zeros_like(non_causal)), \
        "Non-causal entries of W_dec (src > tgt) should be zero"

    # Verify that activating only layer 2 features doesn't affect layer 0 or 1 outputs
    x_in = torch.randn(4, d_model * num_layers)
    # Create activations where only layer 2 features are nonzero
    acts = torch.zeros(4, num_layers * dict_size)
    acts[:, 2 * dict_size:3 * dict_size] = torch.randn(4, dict_size).abs()

    y_pred = encoder.decode(acts, x_in)
    # Layer 2 features should NOT contribute to layers 0 and 1
    # (ignoring b_dec which is a bias term)
    acts_zero = torch.zeros_like(acts)
    y_pred_zero = encoder.decode(acts_zero, x_in)

    # Layers 0 and 1 should be identical (only differ by layer 2 features which are causal-masked)
    assert torch.allclose(y_pred[:, 0], y_pred_zero[:, 0], atol=1e-6), \
        "Layer 2 features should not affect layer 0 output"
    assert torch.allclose(y_pred[:, 1], y_pred_zero[:, 1], atol=1e-6), \
        "Layer 2 features should not affect layer 1 output"
    # Layer 2 SHOULD differ
    assert not torch.allclose(y_pred[:, 2], y_pred_zero[:, 2], atol=1e-3), \
        "Layer 2 features should affect layer 2 output"
    print("PASS: causal structure verified")


def test_per_layer_encoder_independence():
    """Verify that each layer's encoder only reads from its own layer's input."""
    num_layers = 3
    d_model = 32
    dict_size = 64
    cfg = EncoderConfig(
        input_size=d_model * num_layers,
        output_size=d_model,
        dict_size=dict_size,
        top_k=8,
        num_input_layers=num_layers,
        num_output_layers=num_layers,
        device="cpu",
    )
    encoder = TopK(cfg)

    # Create input where only layer 1 has nonzero values
    x_in = torch.zeros(4, d_model * num_layers)
    x_in[:, d_model:2*d_model] = torch.randn(4, d_model)

    pre_acts = encoder.encode(x_in, use_bias=False)
    pre_acts_per_layer = pre_acts.reshape(4, num_layers, dict_size)

    # Only layer 1's features should have nonzero pre-activations
    assert torch.allclose(pre_acts_per_layer[:, 0], torch.zeros(4, dict_size), atol=1e-6), \
        "Layer 0 features should be zero when only layer 1 input is nonzero"
    assert not torch.allclose(pre_acts_per_layer[:, 1], torch.zeros(4, dict_size), atol=1e-3), \
        "Layer 1 features should be nonzero when layer 1 input is nonzero"
    assert torch.allclose(pre_acts_per_layer[:, 2], torch.zeros(4, dict_size), atol=1e-6), \
        "Layer 2 features should be zero when only layer 1 input is nonzero"
    print("PASS: per-layer encoder independence verified")


if __name__ == "__main__":
    test_single_layer_backward_compat()
    test_sae_backward_compat()
    test_multi_layer_topk()
    test_multi_layer_vanilla()
    test_multi_layer_batchtopk()
    test_multi_layer_jumprelu()
    test_skip_connection_single_layer()
    test_skip_connection_multi_layer()
    test_post_encoder_scale()
    test_post_encoder_scale_clt()
    test_decoder_norm()
    test_auxiliary_loss_multi_layer()
    test_activation_store_multi_io()
    test_all_features_combined()
    test_clt_concat_input_multi_output()
    test_causal_structure()
    test_per_layer_encoder_independence()
    print("\nAll tests passed!")
