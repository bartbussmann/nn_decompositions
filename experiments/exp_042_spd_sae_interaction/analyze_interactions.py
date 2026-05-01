"""Analyze SPD-SAE interaction patterns from collected dashboard data.

Produces:
1. Cross-layer heatmap: SPD layer × SAE layer match counts
2. Mapping sparsity: 1-to-1 vs 1-to-many histograms
3. c_fc vs down_proj comparison
4. Feature circuits: input vs output SAE layer for each SPD component

Usage:
    python experiments/exp_042_spd_sae_interaction/analyze_interactions.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("experiments/exp_042_spd_sae_interaction/output")
DATA_PATH = OUTPUT_DIR / "dashboard_data_v2.json"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LAYERS = [0, 1, 2, 3]
LIFT_THRESHOLD = 10.0  # minimum lift to count as a "match"


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


# =============================================================================
# 1. Cross-layer heatmap
# =============================================================================

def plot_cross_layer_heatmap(components: dict, save_path: Path):
    """Heatmap: % of SPD components at layer X that have a match at SAE layer Y."""
    # Count SPD components per layer per module type
    n_comps = np.zeros(len(LAYERS), dtype=int)
    n_comps_cfc = np.zeros(len(LAYERS), dtype=int)
    n_comps_down = np.zeros(len(LAYERS), dtype=int)

    # Count components with at least one match at each SAE layer
    has_match = np.zeros((len(LAYERS), len(LAYERS)), dtype=int)
    has_match_cfc = np.zeros((len(LAYERS), len(LAYERS)), dtype=int)
    has_match_down = np.zeros((len(LAYERS), len(LAYERS)), dtype=int)

    for comp_key, comp in components.items():
        spd_layer = comp["spd_layer"]
        mod_type = comp["spd_mod_type"]
        n_comps[spd_layer] += 1
        if mod_type == "c_fc":
            n_comps_cfc[spd_layer] += 1
        else:
            n_comps_down[spd_layer] += 1

        # Which SAE layers does this component match?
        matched_sae_layers = set()
        for match in comp["sae_matches"]:
            if match["combined_lift"] >= LIFT_THRESHOLD:
                matched_sae_layers.add(match["sae_layer"])

        for sae_layer in matched_sae_layers:
            has_match[spd_layer][sae_layer] += 1
            if mod_type == "c_fc":
                has_match_cfc[spd_layer][sae_layer] += 1
            else:
                has_match_down[spd_layer][sae_layer] += 1

    # Convert to percentages
    pct = 100 * has_match / n_comps[:, None].clip(min=1)
    pct_cfc = 100 * has_match_cfc / n_comps_cfc[:, None].clip(min=1)
    pct_down = 100 * has_match_down / n_comps_down[:, None].clip(min=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, data, title in [
        (axes[0], pct, "All MLP modules"),
        (axes[1], pct_cfc, "c_fc only"),
        (axes[2], pct_down, "down_proj only"),
    ]:
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(LAYERS)))
        ax.set_yticks(range(len(LAYERS)))
        ax.set_xticklabels([f"SAE L{l}" for l in LAYERS])
        ax.set_yticklabels([f"SPD L{l}" for l in LAYERS])
        ax.set_xlabel("SAE residual stream layer")
        ax.set_ylabel("SPD component layer")
        ax.set_title(title)
        for i in range(len(LAYERS)):
            for j in range(len(LAYERS)):
                ax.text(j, i, f"{data[i][j]:.0f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if data[i][j] > 50 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8, label="%")

    fig.suptitle(f"% of SPD components with ≥1 SAE match (lift ≥ {LIFT_THRESHOLD})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# 2. Mapping sparsity: 1-to-1 vs 1-to-many
# =============================================================================

def plot_mapping_sparsity(components: dict, save_path: Path):
    """How many SAE features does each SPD component map to? And vice versa."""
    # Forward: SPD -> how many SAE features
    spd_to_n_sae = []
    # Reverse: SAE feature -> how many SPD components
    sae_to_spd_count = Counter()

    for comp_key, comp in components.items():
        n_matches = sum(1 for m in comp["sae_matches"] if m["combined_lift"] >= LIFT_THRESHOLD)
        spd_to_n_sae.append(n_matches)
        for match in comp["sae_matches"]:
            if match["combined_lift"] >= LIFT_THRESHOLD:
                sae_key = f"L{match['sae_layer']}_F{match['sae_feature']}"
                sae_to_spd_count[sae_key] += 1

    sae_to_n_spd = list(sae_to_spd_count.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(spd_to_n_sae, bins=range(0, max(spd_to_n_sae) + 2),
                 color="#2b6cb0", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel(f"# SAE features matched (lift ≥ {LIFT_THRESHOLD})")
    axes[0].set_ylabel("# SPD components")
    axes[0].set_title("SPD → SAE mapping sparsity")
    mean_fwd = np.mean(spd_to_n_sae)
    axes[0].axvline(mean_fwd, color="#d62728", linestyle="--", label=f"mean={mean_fwd:.1f}")
    axes[0].legend()

    if sae_to_n_spd:
        axes[1].hist(sae_to_n_spd, bins=range(1, max(sae_to_n_spd) + 2),
                     color="#c05621", edgecolor="white", alpha=0.8)
        axes[1].set_xlabel(f"# SPD components matched (lift ≥ {LIFT_THRESHOLD})")
        axes[1].set_ylabel("# SAE features")
        axes[1].set_title("SAE → SPD mapping sparsity")
        mean_rev = np.mean(sae_to_n_spd)
        axes[1].axvline(mean_rev, color="#d62728", linestyle="--", label=f"mean={mean_rev:.1f}")
        axes[1].legend()

    fig.suptitle("Mapping sparsity between SPD components and SAE features", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# 3. c_fc vs down_proj comparison
# =============================================================================

def plot_cfc_vs_down(components: dict, save_path: Path):
    """Compare c_fc and down_proj: top lift, number of matches, which SAE layers."""
    cfc_lifts = []
    down_lifts = []
    cfc_n_matches = []
    down_n_matches = []

    for comp_key, comp in components.items():
        top_lift = comp["sae_matches"][0]["combined_lift"] if comp["sae_matches"] else 0
        n_matches = sum(1 for m in comp["sae_matches"] if m["combined_lift"] >= LIFT_THRESHOLD)
        if comp["spd_mod_type"] == "c_fc":
            cfc_lifts.append(top_lift)
            cfc_n_matches.append(n_matches)
        else:
            down_lifts.append(top_lift)
            down_n_matches.append(n_matches)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Top lift distribution
    bins = np.logspace(0, np.log10(max(max(cfc_lifts, default=1), max(down_lifts, default=1)) + 1), 30)
    axes[0].hist(cfc_lifts, bins=bins, alpha=0.6, label="c_fc", color="#2b6cb0")
    axes[0].hist(down_lifts, bins=bins, alpha=0.6, label="down_proj", color="#c05621")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Top combined lift")
    axes[0].set_ylabel("# components")
    axes[0].set_title("Top lift per component")
    axes[0].legend()

    # Number of matches
    max_n = max(max(cfc_n_matches, default=0), max(down_n_matches, default=0))
    axes[1].hist(cfc_n_matches, bins=range(0, max_n + 2), alpha=0.6, label="c_fc", color="#2b6cb0")
    axes[1].hist(down_n_matches, bins=range(0, max_n + 2), alpha=0.6, label="down_proj", color="#c05621")
    axes[1].set_xlabel(f"# SAE matches (lift ≥ {LIFT_THRESHOLD})")
    axes[1].set_ylabel("# components")
    axes[1].set_title("Matches per component")
    axes[1].legend()

    # SAE layer preference by module type
    cfc_sae_layers = []
    down_sae_layers = []
    for comp_key, comp in components.items():
        for match in comp["sae_matches"]:
            if match["combined_lift"] >= LIFT_THRESHOLD:
                if comp["spd_mod_type"] == "c_fc":
                    cfc_sae_layers.append(match["sae_layer"])
                else:
                    down_sae_layers.append(match["sae_layer"])

    # Normalize: show as fraction relative to SPD layer
    cfc_relative = []
    down_relative = []
    for comp_key, comp in components.items():
        spd_layer = comp["spd_layer"]
        for match in comp["sae_matches"]:
            if match["combined_lift"] >= LIFT_THRESHOLD:
                delta = match["sae_layer"] - spd_layer
                if comp["spd_mod_type"] == "c_fc":
                    cfc_relative.append(delta)
                else:
                    down_relative.append(delta)

    deltas = sorted(set(cfc_relative + down_relative))
    if deltas:
        x = np.arange(len(deltas))
        width = 0.35
        cfc_counts = [cfc_relative.count(d) for d in deltas]
        down_counts = [down_relative.count(d) for d in deltas]
        axes[2].bar(x - width/2, cfc_counts, width, label="c_fc", color="#2b6cb0", alpha=0.8)
        axes[2].bar(x + width/2, down_counts, width, label="down_proj", color="#c05621", alpha=0.8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"{d:+d}" for d in deltas])
        axes[2].set_xlabel("SAE layer - SPD layer")
        axes[2].set_ylabel("# matches")
        axes[2].set_title("SAE layer relative to SPD layer")
        axes[2].legend()

    fig.suptitle("c_fc vs down_proj comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# 4. Feature circuits: input vs output
# =============================================================================

def plot_feature_circuits(components: dict, save_path: Path):
    """For SPD components, show whether they match SAE features at input (L-1) vs output (L)."""
    # For each component at layer L, categorize its SAE matches as:
    # - "input" (SAE layer < SPD layer, i.e. reads from earlier residual stream)
    # - "same" (SAE layer == SPD layer)
    # - "output" (SAE layer > SPD layer, writes to later residual stream)

    categories = {"input": [], "same": [], "output": []}
    categories_by_layer = {l: {"input": 0, "same": 0, "output": 0} for l in LAYERS}

    for comp_key, comp in components.items():
        spd_layer = comp["spd_layer"]
        for match in comp["sae_matches"]:
            if match["combined_lift"] >= LIFT_THRESHOLD:
                sae_layer = match["sae_layer"]
                lift = match["combined_lift"]
                if sae_layer < spd_layer:
                    cat = "input"
                elif sae_layer == spd_layer:
                    cat = "same"
                else:
                    cat = "output"
                categories[cat].append(lift)
                categories_by_layer[spd_layer][cat] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall pie/bar
    cats = ["input", "same", "output"]
    cat_labels = ["Input (SAE < SPD)", "Same layer", "Output (SAE > SPD)"]
    cat_colors = ["#66c2a5", "#fc8d62", "#8da0cb"]
    counts = [len(categories[c]) for c in cats]
    axes[0].bar(cat_labels, counts, color=cat_colors, edgecolor="white")
    axes[0].set_ylabel("# matches")
    axes[0].set_title(f"Match direction (lift ≥ {LIFT_THRESHOLD})")
    for i, c in enumerate(counts):
        axes[0].text(i, c + 1, str(c), ha="center", fontweight="bold")

    # Per SPD layer breakdown
    x = np.arange(len(LAYERS))
    width = 0.25
    for i, (cat, label, color) in enumerate(zip(cats, cat_labels, cat_colors)):
        vals = [categories_by_layer[l][cat] for l in LAYERS]
        axes[1].bar(x + (i - 1) * width, vals, width, label=label, color=color, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"SPD L{l}" for l in LAYERS])
    axes[1].set_ylabel("# matches")
    axes[1].set_title("Match direction by SPD layer")
    axes[1].legend(fontsize=9)

    fig.suptitle("Feature circuits: do SPD components read or write SAE features?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# 5. Lift distribution by layer pair
# =============================================================================

def plot_lift_by_layer_pair(components: dict, save_path: Path):
    """Box plot of combined lift for each (SPD layer, SAE layer) pair."""
    lifts_by_pair = defaultdict(list)
    for comp_key, comp in components.items():
        spd_layer = comp["spd_layer"]
        for match in comp["sae_matches"]:
            if match["combined_lift"] >= 2:  # lower threshold for this plot
                pair = (spd_layer, match["sae_layer"])
                lifts_by_pair[pair].append(match["combined_lift"])

    fig, ax = plt.subplots(figsize=(10, 5))

    pairs = sorted(lifts_by_pair.keys())
    labels = [f"SPD L{s}→SAE L{a}" for s, a in pairs]
    data = [lifts_by_pair[p] for p in pairs]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, (s, a) in zip(bp["boxes"], pairs):
        if s == a:
            patch.set_facecolor("#fc8d62")
        elif a < s:
            patch.set_facecolor("#66c2a5")
        else:
            patch.set_facecolor("#8da0cb")
    ax.set_yscale("linear")
    ax.set_ylabel("Combined lift")
    ax.set_title("Lift distribution by SPD-SAE layer pair")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.15)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#66c2a5", label="Input (SAE < SPD)"),
        Patch(color="#fc8d62", label="Same layer"),
        Patch(color="#8da0cb", label="Output (SAE > SPD)"),
    ], fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    print(f"Loading data from {DATA_PATH}...")
    components = load_data()
    print(f"Loaded {len(components)} components")

    plot_cross_layer_heatmap(components, OUTPUT_DIR / "cross_layer_heatmap.png")
    plot_mapping_sparsity(components, OUTPUT_DIR / "mapping_sparsity.png")
    plot_cfc_vs_down(components, OUTPUT_DIR / "cfc_vs_down_proj.png")
    plot_feature_circuits(components, OUTPUT_DIR / "feature_circuits.png")
    plot_lift_by_layer_pair(components, OUTPUT_DIR / "lift_by_layer_pair.png")


if __name__ == "__main__":
    main()
