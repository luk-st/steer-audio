"""
Analyze AUSteer discriminative scores across layers.

Computes AUSteer scores (if not already computed) and produces a visualization
showing which layers contain the most discriminative frequency bins.

Usage:
    # Compute + plot
    python steering/caa/analyze_austeer_layers.py --concept piano

    # Plot from existing scores
    python steering/caa/analyze_austeer_layers.py \
        --concept piano \
        --sv_path steering_vectors/austeer/ace_piano_k256_all
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fire import Fire

PATH_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PATH_PROJECT)
sys.path.append(os.path.join(PATH_PROJECT, "src", "models", "ace_step", "ACE"))


def load_or_compute(concept, sv_path, k, seed):
    """Load existing scores or compute them."""
    if sv_path is not None:
        austeer_file = os.path.join(sv_path, "austeer.pkl")
        if not os.path.exists(austeer_file):
            raise FileNotFoundError(f"Not found: {austeer_file}")
        print(f"Loading existing scores from {sv_path}")
        with open(austeer_file, "rb") as f:
            return pickle.load(f)

    # Compute fresh
    default_path = f"steering_vectors/austeer/ace_{concept}_k{k}_all"
    austeer_file = os.path.join(default_path, "austeer.pkl")

    if os.path.exists(austeer_file):
        print(f"Found existing scores at {default_path}")
        with open(austeer_file, "rb") as f:
            return pickle.load(f)

    print(f"Computing AUSteer scores for {concept} (all layers, k={k})...")
    subprocess.run(
        [
            sys.executable,
            "steering/austeer/compute_sv_austeer.py",
            "--concept", concept,
            "--k", str(k),
            "--layers", "all",
            "--seed", str(seed),
        ],
        check=True,
    )
    with open(austeer_file, "rb") as f:
        return pickle.load(f)


def analyze_and_plot(
    austeer_vectors,
    concept,
    global_top_k=256,
    save_path=None,
):
    """
    Create a multi-panel figure analyzing discriminative scores across layers.

    Panel 1: Per-layer count of globally top-k freq bins (averaged over steps)
    Panel 2: Per-layer mean discriminative score (averaged over steps)
    Panel 3: Heatmap of mean score per (layer, step)
    """
    steps = sorted(austeer_vectors.keys())
    sample_step = steps[0]
    layers = sorted(austeer_vectors[sample_step].keys(), key=lambda x: int(x[2:]))
    n_layers = len(layers)
    n_steps = len(steps)
    n_freqs = len(austeer_vectors[sample_step][layers[0]]["scores"])

    # ── Collect per-layer, per-step stats ────────────────────────────────────
    # score_matrix[layer_idx, step_idx] = mean abs score for that layer/step
    score_matrix = np.zeros((n_layers, n_steps))
    # global_topk_counts[layer_idx, step_idx] = how many of global top-k fall in this layer
    global_topk_counts = np.zeros((n_layers, n_steps))
    # max_score_matrix
    max_score_matrix = np.zeros((n_layers, n_steps))

    for si, step in enumerate(steps):
        # Collect all (layer, freq, |score|) tuples for global ranking
        all_entries = []
        for li, layer in enumerate(layers):
            data = austeer_vectors[step][layer]
            scores = np.abs(data["scores"])
            score_matrix[li, si] = scores.mean()
            max_score_matrix[li, si] = scores.max()
            for fi, s in enumerate(scores):
                all_entries.append((s, li))

        # Global top-k: rank all (layer, freq) pairs across all layers
        all_entries.sort(key=lambda x: x[0], reverse=True)
        for s, li in all_entries[:global_top_k]:
            global_topk_counts[li, si] += 1

    # Average over steps
    mean_topk_per_layer = global_topk_counts.mean(axis=1)
    mean_score_per_layer = score_matrix.mean(axis=1)
    max_score_per_layer = max_score_matrix.mean(axis=1)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"AUSteer Layer Analysis — concept: {concept}  "
        f"(global top-{global_top_k} of {n_freqs} freqs, {n_steps} steps)",
        fontsize=14,
        fontweight="bold",
    )
    layer_labels = [l.replace("tf", "") for l in layers]
    x = np.arange(n_layers)

    # Panel 1: Global top-k count per layer
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn(mean_topk_per_layer / mean_topk_per_layer.max())
    ax1.bar(x, mean_topk_per_layer, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel(f"# in global top-{global_top_k}")
    ax1.set_title(f"Global top-{global_top_k} freq bins per layer (avg over steps)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_labels, fontsize=8)
    ax1.axhline(
        global_top_k / n_layers, color="red", linestyle="--", alpha=0.5,
        label=f"uniform = {global_top_k / n_layers:.1f}",
    )
    ax1.legend()

    # Panel 2: Mean discriminative score per layer
    ax2 = axes[0, 1]
    ax2.bar(x, mean_score_per_layer, color="steelblue", edgecolor="black", linewidth=0.5, label="mean |score|")
    ax2.bar(x, max_score_per_layer, color="steelblue", edgecolor="black", linewidth=0.5, alpha=0.3, label="max |score|")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Discriminative score")
    ax2.set_title("Mean & max |score| per layer (avg over steps)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels, fontsize=8)
    ax2.legend()

    # Panel 3: Heatmap — mean score per (layer, step)
    ax3 = axes[1, 0]
    im = ax3.imshow(score_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax3.set_xlabel("Diffusion step")
    ax3.set_ylabel("Layer")
    ax3.set_title("Mean |score| per layer × step")
    ax3.set_yticks(range(n_layers))
    ax3.set_yticklabels(layer_labels, fontsize=8)
    step_ticks = list(range(0, n_steps, max(1, n_steps // 10)))
    ax3.set_xticks(step_ticks)
    ax3.set_xticklabels([str(steps[i]) for i in step_ticks], fontsize=8)
    fig.colorbar(im, ax=ax3, shrink=0.8)

    # Panel 4: Heatmap — global top-k count per (layer, step)
    ax4 = axes[1, 1]
    im2 = ax4.imshow(global_topk_counts, aspect="auto", cmap="hot", interpolation="nearest")
    ax4.set_xlabel("Diffusion step")
    ax4.set_ylabel("Layer")
    ax4.set_title(f"Global top-{global_top_k} count per layer × step")
    ax4.set_yticks(range(n_layers))
    ax4.set_yticklabels(layer_labels, fontsize=8)
    ax4.set_xticks(step_ticks)
    ax4.set_xticklabels([str(steps[i]) for i in step_ticks], fontsize=8)
    fig.colorbar(im2, ax=ax4, shrink=0.8)

    plt.tight_layout()

    if save_path is None:
        save_path = f"steering/outputs/austeer_layer_analysis_{concept}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n{'Layer':>6}  {'Top-k count':>11}  {'Mean |score|':>12}  {'Max |score|':>11}")
    print("-" * 48)
    ranked = sorted(range(n_layers), key=lambda i: mean_topk_per_layer[i], reverse=True)
    for li in ranked:
        print(
            f"{layers[li]:>6}  {mean_topk_per_layer[li]:>11.1f}  "
            f"{mean_score_per_layer[li]:>12.4f}  {max_score_per_layer[li]:>11.4f}"
        )


def main(
    concept: str,
    sv_path: str | None = None,
    k: int = 256,
    global_top_k: int = 256,
    seed: int = 10,
    save_path: str | None = None,
):
    """
    Analyze AUSteer discriminative scores across layers.

    Args:
        concept: Concept name (e.g., 'piano')
        sv_path: Path to existing austeer scores. If None, computes or finds them.
        k: Top-k for computing scores (only used if computing fresh)
        global_top_k: How many top freq bins to track in the global ranking plot
        seed: Seed for computation
        save_path: Where to save the plot. Auto-generated if None.
    """
    austeer_vectors = load_or_compute(concept, sv_path, k, seed)
    analyze_and_plot(austeer_vectors, concept, global_top_k=global_top_k, save_path=save_path)


if __name__ == "__main__":
    Fire(main)
