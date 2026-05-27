"""
SAM-audio variant of the vocal_gender external-validation metric.

Pipeline per generated audio:
  1. Source separation via SAM-Audio (facebook/sam-audio-small or -large) with
     text prompt "singing" -> ``out.target[0]`` is the isolated vocal waveform.
  2. Gender classification on the vocal stem using
     ``alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech``.
  3. Aggregate per-clip:
     - ``female_prob``  : softmax probability of the 'female' class
     - ``vocal_rms``    : RMS energy of the vocal stem relative to the mix

Why patching: this repo's venv_sam pins Python 3.11 (via uv), torch 2.6 +
torchcodec 0.3, transformers 4.49, huggingface_hub 0.26 to satisfy SAM-audio.
SAM-audio's ``model.__init__`` unconditionally constructs the visual/text
rerankers (which need ImageBind / laion-clap / Judge weights). With
``reranking_candidates=1`` we don't need any reranker, so we monkey-patch
``sam_audio.model.model.create_ranker = lambda c: None`` *before* the model is
imported. Torchcodec is stubbed for the same reason (we feed tensors, never
decode from disk).

This script is intentionally shardable: ``--alpha_start`` / ``--alpha_end``
slice the alpha grid so multiple GPUs can run in parallel.

Outputs (in <steering_dir>/protocol_results/):
  - vocal_gender_sam.csv             per-alpha rows (or per-alpha-slice when sharding)
  - external_validation_vocals_sam.csv  Pearson + Spearman vs CLAP/MuQ
  - external_validation_vocals_sam.png  alignment curves overlaid with P(female | SAM-audio stem)

Usage (single GPU):
    python steering/eval/eval_external_vocals_sam.py \
        --steering_dir pwr-mount/steering/outputs/ace_step/concept_vocal_gender/loc/sae

Sharding example (4 GPUs):
    CUDA_VISIBLE_DEVICES=0 python ... --alpha_start 0  --alpha_end 8  --shard_id 0 &
    CUDA_VISIBLE_DEVICES=1 python ... --alpha_start 8  --alpha_end 16 --shard_id 1 &
    CUDA_VISIBLE_DEVICES=2 python ... --alpha_start 16 --alpha_end 24 --shard_id 2 &
    CUDA_VISIBLE_DEVICES=3 python ... --alpha_start 24 --alpha_end 32 --shard_id 3 &
"""

import gc
import importlib
import os
import sys
import time
import types
from pathlib import Path
from typing import List, Optional


# -- Stubs MUST be installed before any sam_audio import ----------------------

def _install_stubs():
    """Patch torchcodec + imagebind + laion_clap so SAM-audio import works."""
    _tc_decoders = types.ModuleType("torchcodec.decoders")

    class _NoDecode:
        def __init__(self, *a, **k):
            raise RuntimeError("torchcodec stub: pass torch.Tensor, not file paths")
    _tc_decoders.AudioDecoder = _NoDecode
    _tc_decoders.VideoDecoder = _NoDecode
    _tc = types.ModuleType("torchcodec")
    _tc.decoders = _tc_decoders
    sys.modules["torchcodec"] = _tc
    sys.modules["torchcodec.decoders"] = _tc_decoders

    fakes = [
        ("imagebind", {"__path__": []}),
        ("imagebind.models", {}),
        (
            "imagebind.models.imagebind_model",
            {
                "imagebind_huge": lambda *a, **k: None,
                "ModalityType": type("M", (), {"AUDIO": "a", "VISION": "v", "TEXT": "t"}),
            },
        ),
        (
            "imagebind.data",
            {
                "load_and_transform_audio_data": lambda *a, **k: None,
                "load_and_transform_video_data": lambda *a, **k: None,
                "load_and_transform_text": lambda *a, **k: None,
            },
        ),
        ("laion_clap", {"CLAP_Module": type("C", (), {"__init__": lambda *a, **k: None})}),
    ]
    for name, attrs in fakes:
        mo = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mo, k, v)
        sys.modules[name] = mo

    smm = importlib.import_module("sam_audio.model.model")
    smm.create_ranker = lambda c: None


_install_stubs()

# -- Real imports -------------------------------------------------------------

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from fire import Fire  # noqa: E402
from sam_audio import SAMAudio, SAMAudioProcessor  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


GENDER_MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
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


def _load_sam(model_name: str, device: str):
    model = SAMAudio.from_pretrained(model_name).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(model_name)
    return model, processor


def _load_classifier(device: str):
    feat = AutoFeatureExtractor.from_pretrained(GENDER_MODEL)
    clf = AutoModelForAudioClassification.from_pretrained(GENDER_MODEL).to(device).eval()
    id2label = clf.config.id2label
    female_idx = [k for k, v in id2label.items() if str(v).lower() == "female"][0]
    return feat, clf, int(female_idx)


@torch.inference_mode()
def _separate_one(model, processor, wav: torch.Tensor, src_sr: int, device: str):
    target_sr = processor.audio_sampling_rate
    if src_sr != target_sr:
        wav = torchaudio.functional.resample(wav, src_sr, target_sr)
        src_sr = target_sr
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    batch = processor(audios=[wav], descriptions=["singing"]).to(device)
    out = model.separate(batch, predict_spans=False, reranking_candidates=1)
    vocal = out.target[0].cpu()
    if vocal.ndim == 2:
        vocal = vocal.mean(dim=0)
    return vocal, target_sr


@torch.inference_mode()
def _gender_prob(feat, clf, female_idx: int, vocal_mono: torch.Tensor, src_sr: int, device: str) -> float:
    if src_sr != CLASSIFIER_SR:
        vocal_mono = torchaudio.functional.resample(vocal_mono, src_sr, CLASSIFIER_SR)
    inputs = feat(vocal_mono.numpy(), sampling_rate=CLASSIFIER_SR, return_tensors="pt", padding=True)
    logits = clf(inputs["input_values"].to(device)).logits
    return float(torch.softmax(logits, dim=-1)[0, female_idx].item())


def _vocal_rms_ratio(vocal: torch.Tensor, mix: torch.Tensor) -> float:
    v = vocal.flatten().numpy().astype(np.float64)
    m = mix.flatten().numpy().astype(np.float64)
    return float(np.sqrt(np.mean(v * v) + 1e-12) / np.sqrt(np.mean(m * m) + 1e-12))


def compute_vocals(
    steering_dir: str,
    device: str,
    model_name: str,
    alpha_start: Optional[int],
    alpha_end: Optional[int],
) -> pd.DataFrame:
    alphas = _get_alphas(steering_dir)
    if alpha_start is not None or alpha_end is not None:
        alphas = alphas[alpha_start or 0 : alpha_end if alpha_end is not None else len(alphas)]
        print(f"sharded to {len(alphas)} alphas: {alphas}")

    sam_model, sam_processor = _load_sam(model_name, device)
    feat, clf, female_idx = _load_classifier(device)
    target_sr = sam_processor.audio_sampling_rate

    rows = []
    for alpha in tqdm(alphas, desc=f"SAM-audio ({device})"):
        wavs = sorted(Path(_find_alpha_dir(steering_dir, alpha)).glob("*.wav"))
        female_probs, rms_vals = [], []
        t0 = time.time()
        for w in wavs:
            arr, sr = sf.read(str(w), dtype="float32", always_2d=False)
            wav = torch.from_numpy(arr).unsqueeze(0) if arr.ndim == 1 else torch.from_numpy(arr.T)
            mono_mix = wav.mean(0, keepdim=True)
            if sr != target_sr:
                mono_mix = torchaudio.functional.resample(mono_mix, sr, target_sr)
            vocal, vsr = _separate_one(sam_model, sam_processor, wav, sr, device)
            female_probs.append(_gender_prob(feat, clf, female_idx, vocal, vsr, device))
            rms_vals.append(_vocal_rms_ratio(vocal, mono_mix.squeeze(0)))
            del vocal, mono_mix
            if device == "cuda":
                torch.cuda.empty_cache()
        rows.append({
            "alpha": alpha,
            "mean": float(np.mean(female_probs)),
            "std": float(np.std(female_probs)),
            "scores": female_probs,
            "vocal_rms_mean": float(np.mean(rms_vals)),
            "vocal_rms_std": float(np.std(rms_vals)),
            "vocal_rms_scores": rms_vals,
            "elapsed_s": round(time.time() - t0, 2),
        })
        gc.collect()
    return pd.DataFrame(rows)


def _correlate(alignment_df: pd.DataFrame, ext_df: pd.DataFrame, name: str, ext_col: str = "mean") -> dict:
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
                 color="#6A4C93", linewidth=2, label="P(female | SAM-audio stem)")
        ax2.set_ylabel("P(female | SAM-audio stem)", color="#6A4C93")
        ax2.tick_params(axis="y", labelcolor="#6A4C93")
        ax2.set_ylim(0.0, 1.0)
    plt.suptitle("Female vocals: CLAP / MuQ-T vs SAM-audio + wav2vec2 gender")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def main(
    steering_dir: str,
    device: str = "cuda",
    model_name: str = "facebook/sam-audio-small",
    alpha_start: Optional[int] = None,
    alpha_end: Optional[int] = None,
    shard_id: Optional[int] = None,
    skip_correlations: bool = False,
):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    output_dir = os.path.join(steering_dir, "protocol_results")
    os.makedirs(output_dir, exist_ok=True)

    vg_df = compute_vocals(steering_dir, device, model_name, alpha_start, alpha_end)

    suffix = f"_shard{shard_id}" if shard_id is not None else ""
    csv_path = os.path.join(output_dir, f"vocal_gender_sam{suffix}.csv")
    vg_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    if skip_correlations or shard_id is not None:
        return

    clap_path = os.path.join(output_dir, "clap.csv")
    muqt_path = os.path.join(output_dir, "muqt.csv")
    if not (os.path.exists(clap_path) and os.path.exists(muqt_path)):
        raise FileNotFoundError(
            f"Need existing {clap_path} and {muqt_path}. Run eval_steering_protocol first."
        )
    clap_df = pd.read_csv(clap_path)
    muqt_df = pd.read_csv(muqt_path)

    rows = [
        _correlate(clap_df, vg_df, "clap_vs_female_prob_sam"),
        _correlate(muqt_df, vg_df, "muqt_vs_female_prob_sam"),
        _correlate(clap_df, vg_df, "clap_vs_vocal_rms_sam", ext_col="vocal_rms_mean"),
        _correlate(muqt_df, vg_df, "muqt_vs_vocal_rms_sam", ext_col="vocal_rms_mean"),
    ]
    val_df = pd.DataFrame(rows)
    val_csv = os.path.join(output_dir, "external_validation_vocals_sam.csv")
    val_df.to_csv(val_csv, index=False)
    print("\nSAM-audio validation correlations:")
    print(val_df.to_string(index=False))

    _plot(clap_df, muqt_df, vg_df,
          os.path.join(output_dir, "external_validation_vocals_sam.png"))


if __name__ == "__main__":
    Fire(main)
