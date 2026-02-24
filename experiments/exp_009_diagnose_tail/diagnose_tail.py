"""Diagnose why dropping the last ~1000 SPD components causes a big performance jump.

Investigates:
1. CI score distribution - are tail components truly "dead"?
2. Weight norms (V, U) per component sorted by CI rank
3. Cumulative contribution of tail components
4. Whether weight_delta (residual) plays a role
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 8


def get_eval_batch(tokenizer, batch_size=8, seq_len=256):
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    dataset = dataset.shuffle(seed=0, buffer_size=10000)
    data_iter = iter(dataset)
    texts = [next(data_iter)["text"] for _ in range(batch_size)]
    tokens = tokenizer(texts, truncation=True, max_length=seq_len, padding="max_length", return_tensors="pt")
    return tokens["input_ids"].to(DEVICE), tokens["attention_mask"].to(DEVICE)


@torch.no_grad()
def main():
    print("Loading models...")
    spd_path = Path("checkpoints/spd_s-37e6215e")
    pth_files = sorted(spd_path.glob("model_*.pth"))
    assert pth_files, f"No model_*.pth files found in {spd_path}"
    spd_model = ComponentModel.from_pretrained(str(pth_files[-1]))
    spd_model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    input_ids, attention_mask = get_eval_batch(tokenizer)

    cfc_name = f"transformer.h.{LAYER}.mlp.c_fc"
    cproj_name = f"transformer.h.{LAYER}.mlp.c_proj"

    # Get CI scores
    out = spd_model(input_ids, attention_mask=attention_mask, cache_type="input")
    ci = spd_model.calc_causal_importances(out.cache, sampling="continuous")

    for module_name, short_name in [(cfc_name, "c_fc"), (cproj_name, "c_proj")]:
        print(f"\n{'='*80}")
        print(f"Module: {short_name}")
        print(f"{'='*80}")

        ci_pre = ci.pre_sigmoid[module_name]  # (batch, seq, C)
        ci_post = ci.lower_leaky[module_name]

        # Average CI across batch and sequence
        mean_ci_pre = ci_pre.mean(dim=(0, 1))  # (C,)
        mean_ci_post = ci_post.mean(dim=(0, 1))

        # Sort by mean CI (descending)
        sorted_indices = mean_ci_pre.argsort(descending=True)
        sorted_ci_pre = mean_ci_pre[sorted_indices]
        sorted_ci_post = mean_ci_post[sorted_indices]

        print(f"\n--- CI Score Distribution ---")
        print(f"  Mean pre-sigmoid CI: {mean_ci_pre.mean():.4f}")
        print(f"  Std pre-sigmoid CI:  {mean_ci_pre.std():.4f}")
        print(f"  Min pre-sigmoid CI:  {mean_ci_pre.min():.4f}")
        print(f"  Max pre-sigmoid CI:  {mean_ci_pre.max():.4f}")

        # Histogram of CI scores
        for threshold in [-10, -5, -2, -1, 0, 1, 2, 5, 10]:
            count = (mean_ci_pre > threshold).sum().item()
            print(f"  Components with mean CI > {threshold:>3}: {count:>5} / {len(mean_ci_pre)}")

        print(f"\n--- CI by rank (top/bottom) ---")
        print(f"  {'Rank':>6} | {'Pre-sigmoid':>12} | {'Post-sigmoid':>12}")
        for i in [0, 1, 2, 5, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 5500, 5800, 6000, 6100, 6143]:
            if i < len(sorted_ci_pre):
                print(f"  {i:>6} | {sorted_ci_pre[i]:>12.4f} | {sorted_ci_post[i]:>12.6f}")

        # Weight norms per component
        key = module_name.replace(".", "-")
        component = spd_model._components[key]
        V = component.V  # (d_in, C)
        U = component.U  # (C, d_out)

        v_norms = V.norm(dim=0)  # (C,)
        u_norms = U.norm(dim=1)  # (C,)
        vu_product = v_norms * u_norms  # proxy for component "weight"

        sorted_v_norms = v_norms[sorted_indices]
        sorted_u_norms = u_norms[sorted_indices]
        sorted_vu = vu_product[sorted_indices]

        print(f"\n--- Weight norms by CI rank ---")
        print(f"  {'Rank':>6} | {'||V_i||':>10} | {'||U_i||':>10} | {'||V||*||U||':>12} | {'CI pre':>10}")
        for i in [0, 1, 2, 5, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 5500, 5800, 6000, 6100, 6143]:
            if i < len(sorted_v_norms):
                print(f"  {i:>6} | {sorted_v_norms[i]:>10.4f} | {sorted_u_norms[i]:>10.4f} | {sorted_vu[i]:>12.4f} | {sorted_ci_pre[i]:>10.4f}")

        # Cumulative weight contribution from tail
        print(f"\n--- Cumulative ||V||*||U|| from tail ---")
        cumsum_vu = sorted_vu.flip(0).cumsum(0).flip(0)
        total_vu = sorted_vu.sum().item()
        for rank in [0, 1000, 2000, 3000, 4000, 5000, 5500, 5800, 6000, 6100]:
            if rank < len(cumsum_vu):
                tail = cumsum_vu[rank].item()
                print(f"  Rank {rank:>5}+ : cumulative ||V||*||U|| = {tail:.4f} ({100*tail/total_vu:.1f}% of total)")

        # Check weight_delta
        print(f"\n--- Weight delta (residual) ---")
        original_weight = component.weight  # Reconstructed from V @ U
        # Get original module weight
        target_module = spd_model.target_model
        for part in module_name.split("."):
            target_module = getattr(target_module, part)
        original_w = target_module.weight.data  # (d_out, d_in) for Linear, or (d_in, d_out) for Conv1D

        # Conv1D stores weight as (d_in, d_out)
        recon_w = component.weight  # (d_out, d_in) or similar
        print(f"  Component reconstructed weight shape: {recon_w.shape}")
        print(f"  Original module weight shape: {original_w.shape}")

        # Match shapes - Conv1D is (d_in, d_out), component.weight might be transposed
        if recon_w.shape != original_w.shape:
            delta = original_w - recon_w.T
        else:
            delta = original_w - recon_w
        print(f"  ||weight_delta||_F = {delta.norm():.6f}")
        print(f"  ||original_weight||_F = {original_w.norm():.6f}")
        print(f"  ||component_weight||_F = {recon_w.norm():.6f}")
        print(f"  Relative delta = {delta.norm() / original_w.norm():.6f}")

        # Now: what's the actual contribution of components 5000-6144?
        # Compute output with all vs top-5000 on a batch
        print(f"\n--- Reconstruction test: top-k vs full ---")
        pre_weight_acts = out.cache

        # Full forward (no mask)
        full_logits = spd_model(input_ids, attention_mask=attention_mask)

        # Forward with top-5000 mask
        ci_scores = ci.pre_sigmoid[module_name]
        for k in [5000, 5500, 5800, 6000, 6100, 6144]:
            if k >= ci_scores.shape[-1]:
                # All components
                mask = torch.ones_like(ci_scores)
            else:
                topk_indices = torch.topk(ci_scores, k, dim=-1).indices
                mask = torch.zeros_like(ci_scores)
                mask.scatter_(-1, topk_indices, 1.0)

            mask_infos = make_mask_infos({module_name: mask})
            masked_logits = spd_model(input_ids, attention_mask=attention_mask, mask_infos=mask_infos)

            labels = input_ids.clone()
            labels[input_ids == tokenizer.pad_token_id] = -100
            shift_logits = masked_logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            ).item()
            print(f"  k={k:>5}: CE={ce:.4f}")

        # Investigate: are the "dead" components actually carrying signal through weight_delta?
        # Check if ComponentModel uses weight_delta in forward
        print(f"\n--- Component mask_info routing ---")
        # When masking, does it include a weight_delta contribution?
        # Let's check by looking at what make_mask_infos produces
        test_mask = torch.ones(1, 1, ci_scores.shape[-1], device=DEVICE)
        test_mask_info = make_mask_infos({module_name: test_mask})
        info = test_mask_info[module_name]
        print(f"  MaskInfo type: {type(info)}")
        print(f"  Has weight_delta_and_mask: {info.weight_delta_and_mask is not None}")
        if info.weight_delta_and_mask is not None:
            wd, wm = info.weight_delta_and_mask
            print(f"  weight_delta shape: {wd.shape}, norm: {wd.norm():.6f}")

        # Check with a partial mask
        test_mask2 = torch.ones(1, 1, ci_scores.shape[-1], device=DEVICE)
        test_mask2[..., 5000:] = 0.0
        test_mask_info2 = make_mask_infos({module_name: test_mask2})
        info2 = test_mask_info2[module_name]
        print(f"\n  Partial mask (top-5000 = 1):")
        print(f"  Has weight_delta_and_mask: {info2.weight_delta_and_mask is not None}")
        if info2.weight_delta_and_mask is not None:
            wd2, wm2 = info2.weight_delta_and_mask
            print(f"  weight_delta shape: {wd2.shape}, norm: {wd2.norm():.6f}")

        # How often do tail components exceed the dead threshold?
        print(f"\n--- Tail firing frequency (CI post-sigmoid > 0.1 per token) ---")
        ci_post_all = ci.lower_leaky[module_name]  # (batch, seq, C)
        n_tokens = ci_post_all.shape[0] * ci_post_all.shape[1]
        for rank_start in [0, 50, 100, 5000, 5200, 5400, 5500, 5600, 5800, 6000]:
            rank_end = min(rank_start + 200, len(sorted_indices))
            component_ids = sorted_indices[rank_start:rank_end]
            active_per_token = (ci_post_all[:, :, component_ids] > 0.1).float()
            # fraction of (token, component) pairs that are active
            frac = active_per_token.mean().item()
            # fraction of components that are active on at least one token
            any_active = active_per_token.any(dim=(0, 1)).float().mean().item()
            print(f"  Ranks {rank_start:>5}-{rank_end:>5}: "
                  f"{100*frac:.2f}% of (token,comp) pairs active, "
                  f"{100*any_active:.1f}% of components active on >= 1 token")


if __name__ == "__main__":
    main()
