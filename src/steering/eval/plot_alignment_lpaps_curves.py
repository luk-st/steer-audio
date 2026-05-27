"""
Alignment-vs-LPAPS curves per (concept, sign) for all steering methods.
Reads from the cleaned ace_step/concept_<X>/<all|loc>/<method>/ tree.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Cluster path; use "pwr-mount" prefix instead when reading via the local mount.
# BASE = "/data/lstaniszewski/code/audio-interv/pwr-mount"
BASE = "/lustre/pd03/hpc-kamildeja-1773916679/lukasz/projects/audio-interpretability"
ROOT = f"{BASE}/steering/outputs/ace_step"

# ----------------------------------------------------------------------------
# Method registry
# ----------------------------------------------------------------------------
# short_key -> display label (without "{7,8}" localized suffix)
METHOD_LABELS = {
    "pci":           "PI",
    "te":            "TextEmb",
    "token_embeds":  "TokEmb",
    "freesliders":   "FreeSliders",
    "cs":            "ConceptSliders",
    "austeer":       "AUSteer",
    "caa":           "CAA",
    "sae":           "SAE",
}

# left-to-right legend order
METHOD_ORDER = ["pci", "te", "token_embeds", "freesliders", "cs", "austeer", "caa", "sae"]

METHOD_COLORS = {
    "PI":              "#E63946",  # vivid red
    "TextEmb":         "#F4A261",  # warm orange
    "TokEmb":          "#2A9D8F",  # teal
    "FreeSliders":     "#8338EC",  # purple
    "ConceptSliders":  "#FFBE0B",  # gold
    "CAA":             "#3A86FF",  # bright blue
    "SAE":             "#FB5607",  # deep orange-red
    "AUSteer":         "#06D6A0",
}

# ----------------------------------------------------------------------------
# Per (concept, method): (all_bounds, loc_bounds) where bounds=(neg_min, pos_max).
# Use None for combos that don't exist (only sae has no "all" runs).
# These are kept verbatim from the previous script to preserve the exact ranges
# that were on each plot (see e.g. vocal_gender/freesliders neg=-3.3 even though
# the run only goes to -3 — preserving the original asymmetric request).
# ----------------------------------------------------------------------------
ALPHA_BOUNDS = {
    "piano": {
        "pci":           ((-30, 30),       (-30, 30)),
        "te":            ((-0.18, 0.98),   (-0.3, 1.0)),
        "token_embeds":  ((-1.73, 1.4),    (-1.9, 1.34)),
        "freesliders":   ((-1.96, 2.40),   (-2.25, 2.65)),
        "cs":            ((-0.18, 0.15),   (-0.5, 0.42)),
        "austeer":       ((-21, 21),       (-4, 5.25)),
        "caa":           ((-65, 65),       (-125, 115)),
        "sae":           (None,            (-25, 23)),
    },
    "vocal_gender": {
        "pci":           ((-30, 30),       (-30, 30)),
        "te":            ((-0.43, 0.78),   (-0.6, 1.05)),
        "token_embeds":  ((-8, 5.3),       (-9.5, 5.05)),
        "freesliders":   ((-2.76, 3),      (-3.3, 3.05)),
        "cs":            ((-0.3, 0.3),     (-0.8, 0.7)),
        "austeer":       ((-4.2, 3.2),     (-5.5, 3.9)),
        "caa":           ((-48, 48),       (-120, 100)),
        "sae":           (None,            (-22, 23)),
    },
    "tempo": {
        "pci":           ((-30, 30),       (-30, 30)),
        "te":            ((-0.35, 0.35),   (-0.55, 0.55)),
        "token_embeds":  ((-1.02, 1.6),    (-1.2, 1.85)),
        "freesliders":   ((-1.43, 1.7),    (-2.02, 1.88)),
        "cs":            ((-0.4, 0.44),    (-1.15, 1.15)),
        "austeer":       ((-2.8, 3.4),     (-2.15, 2.5)),
        "caa":           ((-41, 46),       (-70, 80)),
        "sae":           (None,            (-13.6, 15.6)),
    },
    "mood": {
        "pci":           ((-30, 30),       (-30, 30)),
        "te":            ((-0.45, 0.35),   (-0.7, 0.48)),
        "token_embeds":  ((-1.1, 1.2),     (-1.21, 1.21)),
        "freesliders":   ((-3.2, 2),       (-3.85, 2.65)),
        "cs":            ((-0.3, 0.2),     (-1.0, 0.7)),
        "austeer":       ((-14, 9.8),      (-4.2, 2.6)),
        "caa":           ((-59, 33),       (-108, 75)),
        "sae":           (None,            (-24, 16)),
    },
}

# Per concept: alignment metric to plot on the x-axis.
CONCEPTS = [
    ("piano",         "muqt"),
    ("vocal_gender", "clap"),
    ("tempo",         "muqt"),
    ("mood",          "muqt"),
]

# Concept name as it appears in the plot title.
TITLE_CONCEPT = {"vocal_gender": "VOCAL"}


def run_dir(concept: str, method: str, layer: str) -> str:
    """Path to one run dir under the cleaned ace_step/ tree.

    layer is 'all' or 'loc'; method is one of METHOD_ORDER.
    """
    return f"{ROOT}/concept_{concept}/{layer}/{method}"


def build_runs(concept: str, sign: str) -> list[tuple[str, str, tuple[float, float]]]:
    """Return [(run_dir, label, (alpha_lo, alpha_hi)), ...] for this concept × sign."""
    assert sign in ("pos", "neg")
    runs = []
    for method in METHOD_ORDER:
        for layer_idx, layer in enumerate(("all", "loc")):
            bounds = ALPHA_BOUNDS[concept][method][layer_idx]
            if bounds is None:
                continue
            neg_min, pos_max = bounds
            arange = (0, pos_max) if sign == "pos" else (neg_min, 0)
            label = METHOD_LABELS[method] + (" {7,8}" if layer == "loc" else "")
            runs.append((run_dir(concept, method, layer), label, arange))
    return runs


# Build all 8 (concept × sign) configs.
RUNS_CONFIGS = []
for concept, metric in CONCEPTS:
    title_label = TITLE_CONCEPT.get(concept, concept.upper())
    for sign, sign_word in (("pos", "POSITIVE"), ("neg", "NEGATIVE")):
        RUNS_CONFIGS.append({
            "title": f"STEERING {title_label} ({sign_word})",
            "alignment_metric": metric,
            "output": f"steering/outputs/plots/lpaps_{metric}_{concept}_{sign}.png",
            "runs": build_runs(concept, sign),
        })


def get_style(label: str) -> dict:
    """Return matplotlib style dict for a label (open marker for {7,8} variants)."""
    is_localized = "{7,8}" in label
    base = label.replace(" {7,8}", "").strip()
    color = METHOD_COLORS[base]
    if is_localized:
        return dict(color=color, linestyle="--", marker="o",
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.5)
    return dict(color=color, linestyle="-", marker="o",
                markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=1.5)


def load_run(run_dir_path: str, alignment_metric: str, alpha_range=None) -> pd.DataFrame:
    results_dir = Path(run_dir_path) / "protocol_results"
    lpaps = pd.read_csv(results_dir / "lpaps.csv")
    alignment = pd.read_csv(results_dir / f"{alignment_metric}.csv")
    df = lpaps[["alpha", "mean", "std"]].rename(columns={"mean": "lpaps_mean", "std": "lpaps_std"})
    align_cols = alignment[["alpha", "mean", "std"]].rename(columns={"mean": "align_mean", "std": "align_std"})
    df = df.merge(align_cols, on="alpha")
    if alpha_range is not None:
        lo, hi = alpha_range
        df = df[(df["alpha"] >= lo) & (df["alpha"] <= hi)]
    return df.sort_values("alpha").reset_index(drop=True)


def main() -> None:
    for config in RUNS_CONFIGS:
        run_data = []
        alignment_metric = config["alignment_metric"]
        for run_path, label, alpha_range in config["runs"]:
            df = load_run(run_path, alignment_metric, alpha_range)
            df["label"] = label
            run_data.append((label, df))
            print(f"{label}: {len(df)} alphas, range [{df['alpha'].min()}, {df['alpha'].max()}]")

        fig, ax = plt.subplots(figsize=(15, 12))

        for label, df in run_data:
            style = get_style(label)
            ax.plot(
                df["align_mean"], df["lpaps_mean"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                color=style["color"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=style["markeredgewidth"],
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

        ax.set_xlabel(rf"{alignment_metric.upper()} $\uparrow$", fontsize=30)
        ax.set_ylabel(r"LPAPS $\downarrow$", fontsize=30)
        ax.set_title(config["title"], fontsize=20, fontweight="bold")
        ax.legend(fontsize=25)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        Path(config["output"]).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(config["output"])
        plt.close()
        print(f"Saved: {config['output']}")


if __name__ == "__main__":
    main()
