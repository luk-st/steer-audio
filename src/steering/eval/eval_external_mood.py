"""
External (CLAP/MuQ-independent) validation metric for `mood` steering.

Uses the mean spectral centroid (Hz) as an acoustic proxy for valence/brightness:
"cheerful" tracks tend to sit higher up the spectrum than "sad" tracks.  This is
an audio-only signal that is independent of CLAP/MuQ text-similarity, so an
agreement between the two confirms CLAP/MuQ as a faithful steering evaluator.

Outputs (in <steering_dir>/protocol_results/):
  - spectral_centroid.csv         per-alpha mean/std/scores
  - external_validation_mood.csv  Pearson + Spearman of CLAP/MuQ means vs centroid means
  - external_validation_mood.png  alignment curves overlaid with centroid curve

Usage:
    python steering/eval/eval_external_mood.py \
        --steering_dir pwr-mount/steering/outputs/ace_step/concept_mood/loc/sae \
        --num_workers 16
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from fire import Fire
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def _find_alpha_dir(steering_dir: str, alpha: float) -> str:
    candidates = [f"alpha_{alpha}", f"alpha_{alpha:.2f}"]
    if alpha == int(alpha):
        candidates.append(f"alpha_{int(alpha)}")
    for c in candidates:
        p = os.path.join(steering_dir, c)
        if os.path.exists(p):
            return p
    return os.path.join(steering_dir, f"alpha_{alpha}")


def _get_alphas(steering_dir: str) -> List[float]:
    alphas = []
    for d in os.listdir(steering_dir):
        if d.startswith("alpha_"):
            try:
                alphas.append(float(d.replace("alpha_", "")))
            except ValueError:
                continue
    return sorted(alphas)


def _centroid_one(path: str) -> float:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(sc))


def compute_centroid(steering_dir: str, num_workers: int) -> pd.DataFrame:
    alphas = _get_alphas(steering_dir)
    rows = []
    for alpha in tqdm(alphas, desc="Spectral centroid"):
        wavs = sorted(Path(_find_alpha_dir(steering_dir, alpha)).glob("*.wav"))
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = {ex.submit(_centroid_one, str(w)): i for i, w in enumerate(wavs)}
                results_with_idx = []
                for fut in as_completed(futures):
                    results_with_idx.append((futures[fut], fut.result()))
                results_with_idx.sort()
                scores = [v for _, v in results_with_idx]
        else:
            scores = [_centroid_one(str(w)) for w in wavs]
        rows.append({
            "alpha": alpha,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores,
        })
    return pd.DataFrame(rows)


def _correlate(alignment_df: pd.DataFrame, ext_df: pd.DataFrame, name: str) -> dict:
    merged = alignment_df.merge(ext_df, on="alpha", suffixes=("_align", "_ext"))
    if len(merged) < 3:
        return {"metric": name, "n_alpha": len(merged)}
    pr = pearsonr(merged["mean_align"], merged["mean_ext"])
    sr = spearmanr(merged["mean_align"], merged["mean_ext"])
    return {
        "metric": name,
        "n_alpha": int(len(merged)),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def _plot(clap_df, muqt_df, centroid_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df, label, color in [
        (axes[0], clap_df, "CLAP (music)", "#009FB7"),
        (axes[1], muqt_df, "MuQ-T", "#FE4A49"),
    ]:
        ax.errorbar(df["alpha"], df["mean"], yerr=df["std"],
                    fmt="-o", capsize=4, color=color, label=label, linewidth=2)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(label, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(centroid_df["alpha"], centroid_df["mean"], "-s",
                 color="#264653", linewidth=2, label="Spectral centroid (Hz)")
        ax2.set_ylabel("Spectral centroid (Hz)", color="#264653")
        ax2.tick_params(axis="y", labelcolor="#264653")
    plt.suptitle("Mood: CLAP / MuQ-T vs external acoustic metric (spectral centroid)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def main(steering_dir: str, num_workers: int = 16):
    output_dir = os.path.join(steering_dir, "protocol_results")
    os.makedirs(output_dir, exist_ok=True)

    centroid_df = compute_centroid(steering_dir, num_workers)
    centroid_df.to_csv(os.path.join(output_dir, "spectral_centroid.csv"), index=False)
    print(f"Saved {os.path.join(output_dir, 'spectral_centroid.csv')}")

    clap_path = os.path.join(output_dir, "clap.csv")
    muqt_path = os.path.join(output_dir, "muqt.csv")
    if not (os.path.exists(clap_path) and os.path.exists(muqt_path)):
        raise FileNotFoundError(
            f"Need existing {clap_path} and {muqt_path}. Run eval_steering_protocol first."
        )
    clap_df = pd.read_csv(clap_path)
    muqt_df = pd.read_csv(muqt_path)

    rows = [
        _correlate(clap_df, centroid_df, "clap_vs_centroid"),
        _correlate(muqt_df, centroid_df, "muqt_vs_centroid"),
    ]
    val_df = pd.DataFrame(rows)
    val_df.to_csv(os.path.join(output_dir, "external_validation_mood.csv"), index=False)
    print("\nValidation correlations:")
    print(val_df.to_string(index=False))

    _plot(clap_df, muqt_df, centroid_df,
          os.path.join(output_dir, "external_validation_mood.png"))


if __name__ == "__main__":
    Fire(main)
