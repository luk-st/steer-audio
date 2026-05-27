"""
Preservation vs delta-alignment curves per (concept, sign), with AUC shaded.

Visualizes exactly how the AUC reported by `auc.py` is computed:
  - x = preservation = (LPAPS_cutoff - LPAPS), where the cutoff is the
    global per-direction min(LPAPS_max) across the methods on the plot.
  - y = sign-corrected delta alignment:
        pos:  val(alpha) - val(0)
        neg:  val(0) - val(alpha)
  - Shaded area between the curve and y=0 == AUC = trapezoid integral.

Reads from the cleaned ace_step/concept_<X>/<all|loc>/<method>/ tree.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from numpy import trapezoid as _trapezoid
except ImportError:
    from numpy import trapz as _trapezoid  # type: ignore

# Cluster path; use "pwr-mount" prefix instead when reading via the local mount.
# BASE = "/data/lstaniszewski/code/audio-interv/pwr-mount"
BASE = "/lustre/pd03/hpc-kamildeja-1773916679/lukasz/projects/audio-interpretability"
ROOT = f"{BASE}/steering/outputs/ace_step"

OUTPUT_DIR = "steering/outputs/plots"

# ----------------------------------------------------------------------------
# Method registry (kept identical to plot_alignment_lpaps_curves.py)
# ----------------------------------------------------------------------------
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

METHOD_ORDER = ["pci", "te", "token_embeds", "freesliders", "cs", "austeer", "caa", "sae"]

METHOD_COLORS = {
    "PI":              "#E63946",
    "TextEmb":         "#F4A261",
    "TokEmb":          "#2A9D8F",
    "FreeSliders":     "#8338EC",
    "ConceptSliders":  "#FFBE0B",
    "CAA":             "#3A86FF",
    "SAE":             "#FB5607",
    "AUSteer":         "#06D6A0",
}

# Which (method, layer) variants exist. Mirrors ALPHA_BOUNDS in the sister
# script: only sae has no "all" variant.
HAS_LAYER = {m: {"all": True, "loc": True} for m in METHOD_ORDER}
HAS_LAYER["sae"]["all"] = False

CONCEPTS = [
    ("piano",         "muqt"),
    ("vocal_gender", "clap"),
    ("tempo",         "muqt"),
    ("mood",          "muqt"),
]
TITLE_CONCEPT = {"vocal_gender": "VOCAL"}


# ----------------------------------------------------------------------------
# Data loading (mirrors auc.py conventions)
# ----------------------------------------------------------------------------

def run_dir(concept: str, method: str, layer: str) -> str:
    return f"{ROOT}/concept_{concept}/{layer}/{method}"


def load_direction(results_dir: Path, metric: str, direction: str) -> pd.DataFrame | None:
    """Load lpaps + alignment, filter to one direction. Returns None if missing."""
    lpaps_path = results_dir / "protocol_results" / "lpaps.csv"
    align_path = results_dir / "protocol_results" / f"{metric}.csv"
    if not lpaps_path.exists() or not align_path.exists():
        return None

    lpaps_df = pd.read_csv(lpaps_path)
    align_df = pd.read_csv(align_path)

    if direction == "pos":
        lpaps_df = lpaps_df[lpaps_df["alpha"] >= 0]
        align_df = align_df[align_df["alpha"] >= 0]
    else:
        lpaps_df = lpaps_df[lpaps_df["alpha"] <= 0]
        align_df = align_df[align_df["alpha"] <= 0]

    lpaps_df = lpaps_df[["alpha", "mean"]].rename(columns={"mean": "lpaps"})
    align_df = align_df[["alpha", "mean"]].rename(columns={"mean": "val"})
    lpaps_df["alpha"] = lpaps_df["alpha"].round(6)
    align_df["alpha"] = align_df["alpha"].round(6)
    merged = lpaps_df.merge(align_df, on="alpha").sort_values("alpha").reset_index(drop=True)
    return merged if not merged.empty else None


def collect_runs(concept: str, sign: str, metric: str):
    """Return [(label, layer, df_full), ...] including raw lpaps/val for cutoff calc."""
    direction = "pos" if sign == "pos" else "neg"
    runs = []
    for method in METHOD_ORDER:
        for layer in ("all", "loc"):
            if not HAS_LAYER[method][layer]:
                continue
            df = load_direction(Path(run_dir(concept, method, layer)), metric, direction)
            if df is None:
                continue
            label = METHOD_LABELS[method] + (" {7,8}" if layer == "loc" else "")
            runs.append((label, layer, df))
    return runs


def get_style(label: str) -> dict:
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


# ----------------------------------------------------------------------------
# Plot one (concept, sign)
# ----------------------------------------------------------------------------

def plot_one(concept: str, sign: str, metric: str, output_path: str) -> None:
    direction = "pos" if sign == "pos" else "neg"
    sign_mult = 1.0 if sign == "pos" else -1.0
    runs = collect_runs(concept, sign, metric)
    if not runs:
        print(f"[skip] no runs for {concept} {sign}")
        return

    # Global cutoff = min(max(lpaps)) across methods on this plot — same as auc.py.
    cutoff = float(min(df["lpaps"].max() for _, _, df in runs))

    fig, ax = plt.subplots(figsize=(15, 12))

    for label, _layer, df in runs:
        baseline = float(df.loc[df["alpha"].abs().idxmin(), "val"])
        mask = df["lpaps"].values <= cutoff + 1e-9
        lp = df["lpaps"].values[mask]
        val = df["val"].values[mask]
        if len(lp) < 2:
            continue
        x = cutoff - lp                  # preservation, x in [0, cutoff]
        y = sign_mult * (val - baseline)  # sign-corrected delta alignment
        order = np.argsort(x)
        x, y = x[order], y[order]

        style = get_style(label)

        ax.plot(x, y,
                linestyle=style["linestyle"], marker=style["marker"],
                color=style["color"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=style["markeredgewidth"],
                label=label, markersize=6, linewidth=2.5)
        ax.fill_between(x, 0, y, color=style["color"], alpha=0.10, linewidth=0)

    metric_display = "MUQ" if metric == "muqt" else metric.upper()
    title_concept = TITLE_CONCEPT.get(concept, concept.upper())
    sign_word = "POS" if sign == "pos" else "NEG"

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(r"Audio Preservation $\uparrow$", fontsize=26)
    ax.set_ylabel(rf"Concept Gain ({metric_display}) $\uparrow$", fontsize=26)
    ax.set_title(rf"PRESERVATION x $\Delta${metric_display}: {title_concept} ({sign_word})",
                 fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.legend(fontsize=18, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}  (cutoff={cutoff:.3f}, n_methods={len(runs)})")


def main() -> None:
    for concept, metric in CONCEPTS:
        for sign in ("pos", "neg"):
            out = f"{OUTPUT_DIR}/auc_preservation_{metric}_{concept}_{sign}.png"
            plot_one(concept, sign, metric, out)


if __name__ == "__main__":
    main()
