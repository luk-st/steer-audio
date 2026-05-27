"""
Render the three external-evaluator figures for the paper's
"External evaluators" subsection.

For each concept (mood / tempo / vocal_gender) we produce one figure with
three curves on a shared y-axis: MuQ-T, CLAP, and an external acoustic
metric, each min-max normalised to [0, 1] over the alpha grid so the y-axes
align.  External metric per concept:

  - mood    -> spectral_centroid.csv (mean centroid in Hz)
  - tempo   -> onset_rate.csv         (raw onset events / s, "raw_mean")
  - vocals  -> vocal_gender_sam.csv   (P(female | SAM-audio stem))

We also report the Pearson correlation coefficient between each text-based
metric (MuQ-T, CLAP) and the external metric inside the legend so the figure
caption can refer to a single rho per pair.

Outputs PDFs to tada-paper/figs/external_eval_{mood,tempo,vocals}.pdf and
a small CSV with the four Pearson values per concept (12 rows total).

Usage:
    python steering/eval/plot_external_eval.py \
        --mood_dir   steering/outputs/ace_step/concept_mood/loc/sae \
        --tempo_dir  steering/outputs/ace_step/concept_tempo/loc/sae \
        --vocals_dir steering/outputs/ace_step/concept_vocal_gender/loc/sae \
        --out_dir    ../tada-paper/figs
"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire
from scipy.stats import pearsonr

COLORS = {"muqt": "#FE4A49", "clap": "#009FB7", "ext": "#6A4C93"}
LABELS_BASE = {"muqt": "MuQ", "clap": "CLAP"}


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _load(steering_dir: str, external_file: str, external_col: str
          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pr = os.path.join(steering_dir, "protocol_results")
    clap = pd.read_csv(os.path.join(pr, "clap.csv"))[["alpha", "mean"]].sort_values("alpha")
    muqt = pd.read_csv(os.path.join(pr, "muqt.csv"))[["alpha", "mean"]].sort_values("alpha")
    ext = pd.read_csv(os.path.join(pr, external_file))[["alpha", external_col]].rename(
        columns={external_col: "mean"}
    ).sort_values("alpha")
    return clap, muqt, ext


def _plot_one(
    clap: pd.DataFrame,
    muqt: pd.DataFrame,
    ext: pd.DataFrame,
    ext_label: str,
    title: str,
    save_path: str,
) -> dict:
    alphas = clap["alpha"].values
    clap_n = _minmax(clap["mean"].values)
    muqt_n = _minmax(muqt["mean"].values)
    ext_n = _minmax(ext["mean"].values)

    r_clap, p_clap = pearsonr(clap["mean"].values, ext["mean"].values)
    r_muqt, p_muqt = pearsonr(muqt["mean"].values, ext["mean"].values)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(alphas, muqt_n, "-o", color=COLORS["muqt"], linewidth=2.4,
            markersize=5, label=f"MuQ  ($\\rho={r_muqt:.2f}$)")
    ax.plot(alphas, clap_n, "-s", color=COLORS["clap"], linewidth=2.4,
            markersize=5, label=f"CLAP ($\\rho={r_clap:.2f}$)")
    ax.plot(alphas, ext_n, "-^", color=COLORS["ext"], linewidth=2.4,
            markersize=5, label=ext_label)
    ax.set_xlabel(r"Steering coefficient ($\alpha$)", fontsize=15)
    ax.set_ylabel("Normalised score", fontsize=15)
    ax.set_ylim(-0.03, 1.03)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="best", fontsize=12, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")
    return {
        "pearson_clap": float(r_clap), "p_clap": float(p_clap),
        "pearson_muqt": float(r_muqt), "p_muqt": float(p_muqt),
    }


def main(
    mood_dir: str,
    tempo_dir: str,
    vocals_dir: str,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    rows = []

    clap, muqt, ext = _load(mood_dir, "spectral_centroid.csv", "mean")
    stats = _plot_one(
        clap, muqt, ext,
        ext_label="Spectral centroid",
        title="",
        save_path=os.path.join(out_dir, "external_eval_mood.pdf"),
    )
    rows.append({"concept": "mood", "external_metric": "spectral_centroid", **stats})

    clap, muqt, ext = _load(tempo_dir, "onset_rate.csv", "raw_mean")
    stats = _plot_one(
        clap, muqt, ext,
        ext_label="Onset event rate",
        title="",
        save_path=os.path.join(out_dir, "external_eval_tempo.pdf"),
    )
    rows.append({"concept": "tempo", "external_metric": "onset_rate_raw", **stats})

    clap, muqt, ext = _load(vocals_dir, "vocal_gender_sam.csv", "mean")
    stats = _plot_one(
        clap, muqt, ext,
        ext_label="SAM-audio + classifier",
        title="",
        save_path=os.path.join(out_dir, "external_eval_vocals.pdf"),
    )
    rows.append({"concept": "vocals", "external_metric": "sam_female_prob", **stats})

    summary = pd.DataFrame(rows)
    summary_csv = os.path.join(out_dir, "external_eval_correlations.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    Fire(main)
