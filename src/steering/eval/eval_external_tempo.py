"""
External (CLAP/MuQ-independent) validation metrics for `tempo` steering.

Two librosa-only descriptors are computed per generated audio:

1. ``bpm``          - librosa.beat.beat_track tempo (raw BPM).
2. ``onset_rate_w`` - peak-weighted onset event rate (events/s):
                      sum of onset-strength values at detected peaks divided by
                      audio duration.  Matches the rhythmic-density proxy used
                      in the concept-slider paper, which the authors found more
                      reliable than raw BPM for steered generations.

Also reports ``onset_rate_raw`` (peak count / second) for completeness.

Outputs (in <steering_dir>/protocol_results/):
  - bpm.csv, onset_rate.csv
  - external_validation_tempo.csv  Pearson + Spearman vs CLAP/MuQ-T means
  - external_validation_tempo.png  alignment curves overlaid with the two rhythm metrics

Usage:
    python steering/eval/eval_external_tempo.py \
        --steering_dir pwr-mount/steering/outputs/ace_step/concept_tempo/loc/sae \
        --num_workers 16
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

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


def _rhythm_one(path: str) -> Tuple[float, float, float]:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    duration = max(len(y) / float(sr), 1e-6)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(
        oenv, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10
    )
    onset_rate_raw = float(len(peaks)) / duration
    onset_rate_w = float(np.sum(oenv[peaks])) / duration if len(peaks) else 0.0
    return bpm, onset_rate_raw, onset_rate_w


def compute_rhythm(steering_dir: str, num_workers: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    alphas = _get_alphas(steering_dir)
    bpm_rows, onset_rows = [], []
    for alpha in tqdm(alphas, desc="Tempo metrics"):
        wavs = sorted(Path(_find_alpha_dir(steering_dir, alpha)).glob("*.wav"))
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                idx_to_fut = {i: ex.submit(_rhythm_one, str(w)) for i, w in enumerate(wavs)}
                results = [idx_to_fut[i].result() for i in sorted(idx_to_fut)]
        else:
            results = [_rhythm_one(str(w)) for w in wavs]
        bpms = [r[0] for r in results]
        raws = [r[1] for r in results]
        weighteds = [r[2] for r in results]
        bpm_rows.append({
            "alpha": alpha, "mean": float(np.mean(bpms)),
            "std": float(np.std(bpms)), "scores": bpms,
        })
        onset_rows.append({
            "alpha": alpha,
            "mean": float(np.mean(weighteds)),
            "std": float(np.std(weighteds)),
            "scores": weighteds,
            "raw_mean": float(np.mean(raws)),
            "raw_std": float(np.std(raws)),
            "raw_scores": raws,
        })
    return pd.DataFrame(bpm_rows), pd.DataFrame(onset_rows)


def _correlate(alignment_df, ext_df, name, ext_col="mean"):
    ext = ext_df[["alpha", ext_col]].rename(columns={ext_col: "mean"})
    merged = alignment_df.merge(ext, on="alpha", suffixes=("_align", "_ext"))
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


def _plot(clap_df, muqt_df, bpm_df, onset_df, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        (axes[0, 0], clap_df, bpm_df, "mean", "CLAP (music)", "BPM", "#009FB7", "#264653"),
        (axes[0, 1], muqt_df, bpm_df, "mean", "MuQ-T", "BPM", "#FE4A49", "#264653"),
        (axes[1, 0], clap_df, onset_df, "mean", "CLAP (music)", "Onset rate (weighted)", "#009FB7", "#6A4C93"),
        (axes[1, 1], muqt_df, onset_df, "mean", "MuQ-T", "Onset rate (weighted)", "#FE4A49", "#6A4C93"),
    ]
    for ax, align_df, ext_df, ext_col, align_lbl, ext_lbl, ac, ec in panels:
        ax.errorbar(align_df["alpha"], align_df["mean"], yerr=align_df["std"],
                    fmt="-o", capsize=4, color=ac, linewidth=2, label=align_lbl)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(align_lbl, color=ac)
        ax.tick_params(axis="y", labelcolor=ac)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(ext_df["alpha"], ext_df[ext_col], "-s", color=ec, linewidth=2, label=ext_lbl)
        ax2.set_ylabel(ext_lbl, color=ec)
        ax2.tick_params(axis="y", labelcolor=ec)
    plt.suptitle("Tempo: CLAP / MuQ-T vs external rhythm metrics (BPM, weighted onset rate)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def main(steering_dir: str, num_workers: int = 16):
    output_dir = os.path.join(steering_dir, "protocol_results")
    os.makedirs(output_dir, exist_ok=True)

    bpm_df, onset_df = compute_rhythm(steering_dir, num_workers)
    bpm_df.to_csv(os.path.join(output_dir, "bpm.csv"), index=False)
    onset_df.to_csv(os.path.join(output_dir, "onset_rate.csv"), index=False)
    print(f"Saved {os.path.join(output_dir, 'bpm.csv')}")
    print(f"Saved {os.path.join(output_dir, 'onset_rate.csv')}")

    clap_path = os.path.join(output_dir, "clap.csv")
    muqt_path = os.path.join(output_dir, "muqt.csv")
    if not (os.path.exists(clap_path) and os.path.exists(muqt_path)):
        raise FileNotFoundError(
            f"Need existing {clap_path} and {muqt_path}. Run eval_steering_protocol first."
        )
    clap_df = pd.read_csv(clap_path)
    muqt_df = pd.read_csv(muqt_path)

    rows = [
        _correlate(clap_df, bpm_df, "clap_vs_bpm"),
        _correlate(muqt_df, bpm_df, "muqt_vs_bpm"),
        _correlate(clap_df, onset_df, "clap_vs_onset_rate_weighted"),
        _correlate(muqt_df, onset_df, "muqt_vs_onset_rate_weighted"),
        _correlate(clap_df, onset_df, "clap_vs_onset_rate_raw", ext_col="raw_mean"),
        _correlate(muqt_df, onset_df, "muqt_vs_onset_rate_raw", ext_col="raw_mean"),
    ]
    val_df = pd.DataFrame(rows)
    val_df.to_csv(os.path.join(output_dir, "external_validation_tempo.csv"), index=False)
    print("\nValidation correlations:")
    print(val_df.to_string(index=False))

    _plot(clap_df, muqt_df, bpm_df, onset_df,
          os.path.join(output_dir, "external_validation_tempo.png"))


if __name__ == "__main__":
    Fire(main)
