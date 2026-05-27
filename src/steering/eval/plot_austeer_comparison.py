"""
Plot AUSteer comparison: alignment (MUQ-T / CLAP) vs preservation (LPAPS) curves.

Auto-discovers all runs under a concept directory matching the structure:
    {base_dir}/layers_{layers}/{mode}_{k}/range_{min}_{max}/

The cleaned `ace_step/` tree only keeps one canonical AUSteer run per (concept, layer);
all the k×mode sweep variants live in `austeer_sweep/` (date folders merged).

Usage:
    python steering/eval/plot_austeer_comparison.py \
        pwr-mount/steering/outputs/austeer_sweep/concept_piano
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire

# mode → linestyle + marker
MODE_STYLES = {
    "additive":      dict(linestyle="-",  marker="o"),
    "multiplicative": dict(linestyle="--", marker="s"),
}

# k → color (colorblind-friendly palette, extends automatically)
K_PALETTE = [
    "#3A86FF",  # blue
    "#E63946",  # red
    "#2EC4B6",  # teal
    "#FF9F1C",  # orange
    "#8338EC",  # purple
    "#06D6A0",  # green
    "#FB5607",  # deep orange
    "#FFBE0B",  # yellow
]

def _k_colors(k_values):
    """Map sorted k values to colors."""
    return {k: K_PALETTE[i % len(K_PALETTE)] for i, k in enumerate(sorted(k_values))}


def discover_runs(base_dir):
    """Discover all austeer runs under base_dir.

    Expected structure:
        base_dir/layers_{layers}/{mode}_{k}/range_{min}_{max}/protocol_results/

    Returns list of (run_dir, label, layers, mode, k) tuples.
    """
    base = Path(base_dir)
    runs = []

    for layers_dir in sorted(base.iterdir()):
        if not layers_dir.is_dir() or not layers_dir.name.startswith("layers_"):
            continue
        layers = layers_dir.name.replace("layers_", "")

        for mode_k_dir in sorted(layers_dir.iterdir()):
            if not mode_k_dir.is_dir():
                continue
            match = re.match(r"^(additive|multiplicative)_(\d+)$", mode_k_dir.name)
            if not match:
                continue
            mode = match.group(1)
            k = int(match.group(2))

            for range_dir in sorted(mode_k_dir.iterdir()):
                if not range_dir.is_dir() or not range_dir.name.startswith("range_"):
                    continue
                results_dir = range_dir / "protocol_results"
                if not (results_dir / "lpaps.csv").exists():
                    print(f"  Skipping {range_dir} (no protocol_results)")
                    continue

                if layers == "tf6tf7":
                    layers_label = "{6,7}"
                elif layers == "no_tf6tf7":
                    layers_label = "\\{6,7}"
                else:
                    layers_label = layers
                label = f"{mode} k={k} {layers_label}"
                runs.append((str(range_dir), label, layers, mode, k))

    return runs


def get_style(mode, k, k_colors):
    """Return plot style dict for a run.
    
    Color   → encodes k value
    Linestyle + marker → encodes mode (additive vs multiplicative)
    Fill    → encodes layers (localized=hollow, all=filled)
    """
    color = k_colors[k]
    ms = MODE_STYLES.get(mode, dict(linestyle="-", marker="o"))
    return dict(
        color=color,
        linestyle=ms["linestyle"],
        marker=ms["marker"],
    )


def load_run(run_dir, alignment_metric, alpha_range=None):
    results_dir = Path(run_dir) / "protocol_results"
    lpaps = pd.read_csv(results_dir / "lpaps.csv")
    alignment = pd.read_csv(results_dir / f"{alignment_metric}.csv")
    df = lpaps[["alpha", "mean", "std"]].rename(
        columns={"mean": "lpaps_mean", "std": "lpaps_std"}
    )
    align_cols = alignment[["alpha", "mean", "std"]].rename(
        columns={"mean": "align_mean", "std": "align_std"}
    )
    df = df.merge(align_cols, on="alpha")
    if alpha_range is not None:
        lo, hi = alpha_range
        if lo is not None:
            df = df[df["alpha"] >= lo]
        if hi is not None:
            df = df[df["alpha"] <= hi]
    return df.sort_values("alpha").reset_index(drop=True)


def plot_curves(runs, alignment_metric, alpha_range, title, output_path):
    # Build k→color map from all runs
    all_k = {k for _, _, _, _, k in runs}
    k_colors = _k_colors(all_k)

    fig, ax = plt.subplots(figsize=(15, 12))

    for run_dir, label, layers, mode, k in runs:
        try:
            df = load_run(run_dir, alignment_metric, alpha_range)
        except Exception as e:
            print(f"  Error loading {run_dir}: {e}")
            continue

        if df.empty:
            continue

        style = get_style(mode, k, k_colors)
        # all=solid filled, tf6tf7=dashed hollow, no_tf6tf7=dotted half-filled
        if layers == "tf6tf7":
            line_style = "--"
            mfc = "white"
        elif layers == "no_tf6tf7":
            line_style = ":"
            mfc = style["color"]
            mfc = (*plt.matplotlib.colors.to_rgb(style["color"]), 0.4)  # semi-transparent
        else:
            line_style = style["linestyle"]
            mfc = style["color"]
        print(f"  {label}: {len(df)} alphas, range [{df['alpha'].min()}, {df['alpha'].max()}]")

        ax.plot(
            df["align_mean"], df["lpaps_mean"],
            linestyle=line_style,
            marker=style["marker"],
            color=style["color"],
            markerfacecolor=mfc,
            markeredgecolor=style["color"],
            markeredgewidth=1.5,
            label=label, markersize=6, linewidth=2.5,
        )

        zero = df[df["alpha"] == 0]
        if not zero.empty:
            ax.plot(
                zero["align_mean"].values[0], zero["lpaps_mean"].values[0],
                "*", color="black", markersize=10, zorder=10,
            )

        for idx, row in df.iterrows():
            if int(idx) % 2 == 1:
                ax.annotate(
                    f"{row['alpha']:.0f}",
                    (row["align_mean"], row["lpaps_mean"]),
                    fontsize=8, alpha=0.8,
                    textcoords="offset points", xytext=(4, 4),
                    color="black",
                )

    # ── Legend: one entry per k (color) + one per mode (linestyle/marker) ──
    from matplotlib.lines import Line2D
    k_handles = [
        Line2D([0], [0], color=k_colors[k], linewidth=2.5, label=f"k={k}")
        for k in sorted(all_k)
    ]
    mode_handles = [
        Line2D([0], [0], color="grey", linestyle=s["linestyle"],
               marker=s["marker"], linewidth=2.5, label=mode)
        for mode, s in MODE_STYLES.items()
    ]
    layer_handles = [
        Line2D([0], [0], color="grey", marker="o", linestyle="-",
               markerfacecolor="grey", label="all layers"),
        Line2D([0], [0], color="grey", marker="o", linestyle="-",
               markerfacecolor="white", markeredgecolor="grey",
               markeredgewidth=1.5, label="localized {6,7}"),
    ]

    leg1 = ax.legend(handles=k_handles,     title="k",      loc="upper left",  fontsize=14, title_fontsize=14)
    leg2 = ax.legend(handles=mode_handles,  title="mode",   loc="upper right", fontsize=14, title_fontsize=14)
    leg3 = ax.legend(handles=layer_handles, title="layers", loc="lower right", fontsize=14, title_fontsize=14)
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    ax.set_xlabel(rf"{alignment_metric.upper()} $\uparrow$", fontsize=30)
    ax.set_ylabel(r"LPAPS $\downarrow$", fontsize=30)
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


LAYERS_FILTER_MAP = {
    "all":    lambda layers: layers != "tf6tf7",
    "tf6tf7": lambda layers: layers == "tf6tf7",
    "none":   lambda layers: True,   # keep everything
}

MODE_FILTER_MAP = {
    "add": lambda mode: mode == "additive",
    "mul": lambda mode: mode == "multiplicative",
    "none": lambda mode: True,       # keep everything
}


def main(base_dir: str, l: str = "none", m: str = "none"):
    """
    Auto-discover and plot all AUSteer runs under base_dir.

    Args:
        base_dir: Path like pwr-mount/steering/outputs/austeer_sweep/concept_piano
        l:        Layer filter — 'all' | 'tf6tf7' | 'none' (default: none = plot both)
        m:        Mode filter  — 'add' | 'mul'    | 'none' (default: none = plot all)
    """
    if l not in LAYERS_FILTER_MAP:
        raise ValueError(f"--l must be one of {list(LAYERS_FILTER_MAP)}; got '{l}'")
    if m not in MODE_FILTER_MAP:
        raise ValueError(f"--m must be one of {list(MODE_FILTER_MAP)}; got '{m}'")

    layers_ok = LAYERS_FILTER_MAP[l]
    mode_ok   = MODE_FILTER_MAP[m]

    base = Path(base_dir)
    concept = base.name.replace("concept_", "")
    output_dir = base / "plots"

    print(f"Discovering runs in {base_dir}...")
    runs = discover_runs(base_dir)

    if not runs:
        print("No runs found. Expected structure:")
        print("  {base_dir}/layers_{layers}/{mode}_{k}/range_{min}_{max}/protocol_results/")
        return

    runs = [(rd, lb, ly, mo, k) for rd, lb, ly, mo, k in runs
            if layers_ok(ly) and mode_ok(mo)]

    if not runs:
        print(f"No runs left after filtering (--l={l} --m={m}).")
        return

    print(f"Found {len(runs)} runs after filtering (--l={l} --m={m})")

    for metric in ("muqt", "clap"):
        for sign, alpha_range in (("pos", (0, None)), ("neg", (None, 0))):
            print(f"\n=== {concept.upper()} {sign.upper()} ({metric.upper()}) ===")
            plot_curves(
                runs, metric, alpha_range,
                f"AUSteer {concept.upper()} ({sign})",
                str(output_dir / f"austeer_lpaps_{metric}_{concept}_{sign}.png"),
            )


if __name__ == "__main__":
    Fire(main)