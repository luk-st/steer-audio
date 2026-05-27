"""
External (CLAP/MuQ-independent) validation metric for `vocal_gender` steering.

Pipeline per generated audio:
  1. Source separation via Demucs (htdemucs, 4-stem) -> vocal stem.
  2. Gender classification on the vocal stem using
     ``alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech``
     (HF model with id2label {0:'female', 1:'male'}).
  3. Aggregate per-clip:
     - ``female_prob``: softmax probability of the 'female' class on the vocal stem
     - ``vocal_rms``: RMS energy of the vocal stem relative to the mix (proxy
       for vocal presence; very low at strongly negative alphas where vocals
       are suppressed).

Why Demucs instead of SAM-audio: the available virtualenv runs Python 3.10,
while SAM-audio (and its ``perception_models`` dependency) requires Python
>=3.11.  Demucs is the de-facto music source separator and produces a clean
vocal stem from music, which is what the downstream gender classifier needs.

Outputs (in <steering_dir>/protocol_results/):
  - vocal_gender.csv             per-alpha mean/std of female_prob (+ vocal_rms columns)
  - external_validation_vocals.csv  Pearson + Spearman of CLAP/MuQ vs vocal_gender
  - external_validation_vocals.png  alignment curves overlaid with female_prob

Usage:
    python steering/eval/eval_external_vocals.py \
        --steering_dir pwr-mount/steering/outputs/ace_step/concept_vocal_gender/loc/sae
"""

import gc
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from fire import Fire
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from demucs.apply import apply_model
from demucs.pretrained import get_model
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


GENDER_MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
DEMUCS_MODEL = "htdemucs"
CLASSIFIER_SR = 16000


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


def _load_demucs(device: str):
    model = get_model(DEMUCS_MODEL)
    model.to(device).eval()
    return model


def _load_classifier(device: str):
    feat = AutoFeatureExtractor.from_pretrained(GENDER_MODEL)
    clf = AutoModelForAudioClassification.from_pretrained(GENDER_MODEL).to(device).eval()
    id2label = clf.config.id2label
    if "female" in {v.lower() for v in id2label.values()}:
        female_idx = [k for k, v in id2label.items() if v.lower() == "female"][0]
    else:
        raise ValueError(f"Could not find 'female' in id2label: {id2label}")
    return feat, clf, int(female_idx)


def _prep_for_demucs(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav


@torch.inference_mode()
def _separate_vocal_batch(model, batch: torch.Tensor, device: str) -> torch.Tensor:
    """batch: (B, 2, T) on CPU. Returns vocals (B, T_mono) on CPU."""
    batch = batch.to(device)
    out = apply_model(model, batch, split=True, overlap=0.1, progress=False)
    vocal_idx = list(model.sources).index("vocals")
    return out[:, vocal_idx].mean(dim=1).cpu()


@torch.inference_mode()
def _gender_prob(feat, clf, female_idx: int, vocal_mono: torch.Tensor, src_sr: int, device: str) -> float:
    if src_sr != CLASSIFIER_SR:
        vocal_mono = torchaudio.functional.resample(vocal_mono, src_sr, CLASSIFIER_SR)
    inputs = feat(
        vocal_mono.numpy(), sampling_rate=CLASSIFIER_SR, return_tensors="pt", padding=True
    )
    input_values = inputs["input_values"].to(device)
    logits = clf(input_values).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return float(probs[female_idx].item())


def _vocal_rms(vocal: torch.Tensor, mix: torch.Tensor) -> float:
    v = vocal.flatten().numpy().astype(np.float64)
    m = mix.flatten().numpy().astype(np.float64)
    vocal_e = np.sqrt(np.mean(v * v) + 1e-12)
    mix_e = np.sqrt(np.mean(m * m) + 1e-12)
    return float(vocal_e / mix_e)


def compute_vocals(steering_dir: str, device: str, batch_size: int = 8) -> pd.DataFrame:
    alphas = _get_alphas(steering_dir)
    demucs_model = _load_demucs(device)
    feat, clf, female_idx = _load_classifier(device)
    target_sr = demucs_model.samplerate

    rows = []
    for alpha in tqdm(alphas, desc="Vocal gender"):
        wavs = sorted(Path(_find_alpha_dir(steering_dir, alpha)).glob("*.wav"))

        prepped, mixes = [], []
        for w in wavs:
            arr, sr = sf.read(str(w), dtype="float32", always_2d=False)
            wav = torch.from_numpy(arr).unsqueeze(0) if arr.ndim == 1 else torch.from_numpy(arr.T)
            prepped.append(_prep_for_demucs(wav, sr, target_sr))
            mono = wav.mean(0, keepdim=True)
            if sr != target_sr:
                mono = torchaudio.functional.resample(mono, sr, target_sr)
            mixes.append(mono.squeeze(0))

        max_len = max(p.shape[-1] for p in prepped)
        prepped = [torch.nn.functional.pad(p, (0, max_len - p.shape[-1])) for p in prepped]
        mixes = [torch.nn.functional.pad(m, (0, max_len - m.shape[-1])) for m in mixes]

        female_probs, rmses = [], []
        for i in range(0, len(prepped), batch_size):
            batch = torch.stack(prepped[i:i + batch_size])
            vocals_batch = _separate_vocal_batch(demucs_model, batch, device)
            for k, vocal in enumerate(vocals_batch):
                mix_k = mixes[i + k]
                female_probs.append(_gender_prob(feat, clf, female_idx, vocal, target_sr, device))
                rmses.append(_vocal_rms(vocal, mix_k))
            del vocals_batch, batch
            if device == "cuda":
                torch.cuda.empty_cache()

        rows.append({
            "alpha": alpha,
            "mean": float(np.mean(female_probs)),
            "std": float(np.std(female_probs)),
            "scores": female_probs,
            "vocal_rms_mean": float(np.mean(rmses)),
            "vocal_rms_std": float(np.std(rmses)),
            "vocal_rms_scores": rmses,
        })
        gc.collect()
    return pd.DataFrame(rows)


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


def _plot(clap_df, muqt_df, vg_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df, label, color in [
        (axes[0], clap_df, "CLAP (music)", "#009FB7"),
        (axes[1], muqt_df, "MuQ-T", "#FE4A49"),
    ]:
        ax.errorbar(df["alpha"], df["mean"], yerr=df["std"],
                    fmt="-o", capsize=4, color=color, linewidth=2, label=label)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(label, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(vg_df["alpha"], vg_df["mean"], "-s",
                 color="#6A4C93", linewidth=2, label="P(female | vocal stem)")
        ax2.set_ylabel("P(female | vocal stem)", color="#6A4C93")
        ax2.tick_params(axis="y", labelcolor="#6A4C93")
        ax2.set_ylim(0.0, 1.0)
    plt.suptitle("Female vocals: CLAP / MuQ-T vs external metric (Demucs vocal -> wav2vec2 gender)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def main(steering_dir: str, device: str = "cuda", batch_size: int = 8):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    output_dir = os.path.join(steering_dir, "protocol_results")
    os.makedirs(output_dir, exist_ok=True)

    vg_df = compute_vocals(steering_dir, device, batch_size=batch_size)
    vg_df.to_csv(os.path.join(output_dir, "vocal_gender.csv"), index=False)
    print(f"Saved {os.path.join(output_dir, 'vocal_gender.csv')}")

    clap_path = os.path.join(output_dir, "clap.csv")
    muqt_path = os.path.join(output_dir, "muqt.csv")
    if not (os.path.exists(clap_path) and os.path.exists(muqt_path)):
        raise FileNotFoundError(
            f"Need existing {clap_path} and {muqt_path}. Run eval_steering_protocol first."
        )
    clap_df = pd.read_csv(clap_path)
    muqt_df = pd.read_csv(muqt_path)

    rows = [
        _correlate(clap_df, vg_df, "clap_vs_female_prob"),
        _correlate(muqt_df, vg_df, "muqt_vs_female_prob"),
        _correlate(clap_df, vg_df, "clap_vs_vocal_rms", ext_col="vocal_rms_mean"),
        _correlate(muqt_df, vg_df, "muqt_vs_vocal_rms", ext_col="vocal_rms_mean"),
    ]
    val_df = pd.DataFrame(rows)
    val_df.to_csv(os.path.join(output_dir, "external_validation_vocals.csv"), index=False)
    print("\nValidation correlations:")
    print(val_df.to_string(index=False))

    _plot(clap_df, muqt_df, vg_df,
          os.path.join(output_dir, "external_validation_vocals.png"))


if __name__ == "__main__":
    Fire(main)
