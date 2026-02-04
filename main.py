# %%
from activation_store import ActivationsStore
from config import SAEConfig
from sae import BatchTopKSAE, JumpReLUSAE, TopKSAE, VanillaSAE
from training import train_sae
from transformer_lens import HookedTransformer

# Example: Train JumpReLU with different l1_coeff values
for l1_coeff in [0.004, 0.0018, 0.0008]:
    cfg = SAEConfig(
        encoder_type="jumprelu",
        act_size=768,
        dict_size=768 * 16,
        model_name="gpt2-small",
        layer=8,
        site="resid_pre",
        dataset_path="Skylion007/openwebtext",
        input_unit_norm=True,
        l1_coeff=l1_coeff,
        wandb_project="batchtopk_comparison",
        device="cuda",
    )

    sae = JumpReLUSAE(cfg)

    model = HookedTransformer.from_pretrained(cfg.model_name).to(cfg.dtype).to(cfg.device)
    activations_store = ActivationsStore(model, cfg)
    train_sae(sae, activations_store, model, cfg)

# Example: Compare TopK vs BatchTopK with different k values
for encoder_type in ["topk", "batchtopk"]:
    for top_k in [16, 32, 64]:
        cfg = SAEConfig(
            encoder_type=encoder_type,
            act_size=768,
            dict_size=768 * 16,
            model_name="gpt2-small",
            layer=8,
            site="resid_pre",
            dataset_path="Skylion007/openwebtext",
            input_unit_norm=True,
            top_k=top_k,
            l1_coeff=0.0,
            wandb_project="batchtopk_comparison",
            device="cuda",
        )

        if encoder_type == "topk":
            sae = TopKSAE(cfg)
        else:
            sae = BatchTopKSAE(cfg)

        model = HookedTransformer.from_pretrained(cfg.model_name).to(cfg.dtype).to(cfg.device)
        activations_store = ActivationsStore(model, cfg)
        train_sae(sae, activations_store, model, cfg)

# Example: Compare different dictionary sizes
for encoder_type in ["topk", "batchtopk"]:
    for dict_size in [768 * 4, 768 * 8, 768 * 32]:
        cfg = SAEConfig(
            encoder_type=encoder_type,
            act_size=768,
            dict_size=dict_size,
            model_name="gpt2-small",
            layer=8,
            site="resid_pre",
            dataset_path="Skylion007/openwebtext",
            input_unit_norm=True,
            top_k=32,
            l1_coeff=0.0,
            wandb_project="batchtopk_comparison",
            device="cuda",
        )

        if encoder_type == "topk":
            sae = TopKSAE(cfg)
        else:
            sae = BatchTopKSAE(cfg)

        model = HookedTransformer.from_pretrained(cfg.model_name).to(cfg.dtype).to(cfg.device)
        activations_store = ActivationsStore(model, cfg)
        train_sae(sae, activations_store, model, cfg)
