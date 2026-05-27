"""
Comprehensive Steering Evaluation Protocol for ACE-Step model.

This script evaluates steering vectors using multiple metrics:
- Concept Alignment: CLAP (music checkpoint), MUQ-T
- Audio Preservation: LPAPS, FAD (vs baseline)
- Audio Quality: Audiobox aesthetics
- Steering Metrics: Conceptual Range (CR), Semantic Preservation (SP), Conceptual Smoothness (CSM)

Usage:
    # Basic: per-metric prompts are looked up from CONCEPT_TO_EVAL_PROMPTS
    python eval_steering_protocol.py \
        --steering_dir /path/to/outputs \
        --concept piano

    # With negative prompt: use different prompt for alpha < 0
    python eval_steering_protocol.py \
        --steering_dir /path/to/outputs \
        --concept piano \
        --negative_eval_prompt "no piano"

    # Evaluate only positive alphas (alpha >= 0)
    python eval_steering_protocol.py \
        --steering_dir /path/to/outputs \
        --concept piano \
        --positive_alphas_only

    # Skip aesthetics computation
    python eval_steering_protocol.py \
        --steering_dir /path/to/outputs \
        --concept piano \
        --skip_aesthetics
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Literal
from src.metrics.metrics import calculate_fad, calculate_muqt
from editing.eval import get_lpaps, get_clap
from src.steering.methods.sae.lib.configs.eval import CONCEPT_TO_EVAL_PROMPTS

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from fire import Fire
from tqdm import tqdm


# Monkey-patch torch.load to handle PyTorch 2.6+ weights_only default change
# This is needed for external packages like laion_clap that haven't updated yet
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

def get_alphas_from_dir(steering_dir: str) -> List[float]:
    """Extract alpha values from directory structure."""
    alphas = []
    for d in os.listdir(steering_dir):
        if d.startswith("alpha_"):
            try:
                alpha = float(d.replace("alpha_", ""))
                alphas.append(alpha)
            except ValueError:
                continue
    return sorted(alphas)


def _find_alpha_dir(steering_dir: str, alpha: float) -> str:
    """Find the actual alpha directory, handling int/float naming variations.

    Directories may be named alpha_-10 (int), alpha_-10.0 (float), or alpha_-10.00 (2dp).
    """
    # Try exact float format first (e.g. alpha_-2.0)
    alpha_dir = os.path.join(steering_dir, f"alpha_{alpha}")
    if os.path.exists(alpha_dir):
        return alpha_dir
    # Try 2 decimal places format (e.g. alpha_-2.00)
    alpha_dir = os.path.join(steering_dir, f"alpha_{alpha:.2f}")
    if os.path.exists(alpha_dir):
        return alpha_dir
    # Try integer format if alpha is a whole number (e.g. alpha_-2)
    if alpha == int(alpha):
        alpha_dir = os.path.join(steering_dir, f"alpha_{int(alpha)}")
        if os.path.exists(alpha_dir):
            return alpha_dir
    return os.path.join(steering_dir, f"alpha_{alpha}")


def load_audios_for_alpha(
    steering_dir: str, alpha: float
) -> Tuple[List[torch.Tensor], Optional[int]]:
    """Load all audio files for a given alpha value."""
    alpha_dir = _find_alpha_dir(steering_dir, alpha)
    audio_files = sorted(Path(alpha_dir).glob("*.wav"))

    audios = []
    sample_rate = None
    for audio_file in audio_files:
        audio, sr = torchaudio.load(audio_file)
        audios.append(audio)
        if sample_rate is None:
            sample_rate = sr

    return audios, sample_rate


def compute_clap_alignment(
    steering_dir: str,
    alphas: List[float],
    eval_prompt: str,
    negative_eval_prompt: Optional[str] = None,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Compute CLAP (music checkpoint) alignment scores.

    Uses get_clap from editing.eval which processes each audio individually
    with windowed CLAP, ensuring deterministic results across runs.

    Args:
        steering_dir: Directory containing alpha_* subdirectories
        alphas: List of alpha values to evaluate
        eval_prompt: Primary prompt (used for alpha >= 0, or all if no negative_eval_prompt)
        negative_eval_prompt: Optional prompt for alpha < 0
        device: Device to run on

    Returns:
        DataFrame with columns: alpha, mean, std, scores, prompt_used
    """
    print("\n=== Computing CLAP (Music) Alignment ===")
    if negative_eval_prompt:
        print(f"  Positive prompt (alpha >= 0): {eval_prompt}")
        print(f"  Negative prompt (alpha < 0): {negative_eval_prompt}")
    else:
        print(f"  Prompt: {eval_prompt}")

    results = []
    for alpha in tqdm(alphas, desc="CLAP"):
        audios, sample_rate = load_audios_for_alpha(steering_dir, alpha)

        # Select prompt based on alpha sign
        if negative_eval_prompt and alpha < 0:
            prompt = negative_eval_prompt
        else:
            prompt = eval_prompt

        prompts = [prompt] * len(audios)
        srs = [sample_rate] * len(audios)

        clap_df = get_clap(
            target_prompts=prompts,
            edits=audios,
            srs_edit=srs,
            device=torch.device(device),
        )

        scores = clap_df["clap"].tolist()
        results.append(
            {
                "alpha": alpha,
                "mean": np.mean(scores),
                "std": np.std(scores),
                "scores": scores,
                "prompt_used": prompt,
            }
        )

    return pd.DataFrame(results)


def compute_muqt_alignment(
    steering_dir: str,
    alphas: List[float],
    eval_prompt: str,
    sample_rate: int,
    negative_eval_prompt: Optional[str] = None,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Compute MUQ-T alignment scores.

    Args:
        steering_dir: Directory containing alpha_* subdirectories
        alphas: List of alpha values to evaluate
        eval_prompt: Primary prompt (used for alpha >= 0, or all if no negative_eval_prompt)
        sample_rate: Audio sample rate
        negative_eval_prompt: Optional prompt for alpha < 0
        device: Device to run on

    Returns:
        DataFrame with columns: alpha, mean, std, scores, prompt_used
    """
    print("\n=== Computing MUQ-T Alignment ===")
    if negative_eval_prompt:
        print(f"  Positive prompt (alpha >= 0): {eval_prompt}")
        print(f"  Negative prompt (alpha < 0): {negative_eval_prompt}")
    else:
        print(f"  Prompt: {eval_prompt}")

    results = []
    for alpha in tqdm(alphas, desc="MUQ-T"):
        alpha_dir = _find_alpha_dir(steering_dir, alpha)

        # Select prompt based on alpha sign
        if negative_eval_prompt and alpha < 0:
            prompt = negative_eval_prompt
        else:
            prompt = eval_prompt

        summary, per_audio = calculate_muqt(
            audio_dir=alpha_dir,
            prompts=[prompt],
            device=device,
            sr=sample_rate,
            resample_to_24k=True,
        )

        # Extract per-audio scores
        scores = [per_audio[i]["muqt_sim_p0"] for i in range(len(per_audio))]
        results.append(
            {
                "alpha": alpha,
                "mean": np.mean(scores),
                "std": np.std(scores),
                "scores": scores,
                "prompt_used": prompt,
            }
        )

    return pd.DataFrame(results)


def compute_lpaps_preservation(
    steering_dir: str,
    alphas: List[float],
    device: str = "cuda",
) -> pd.DataFrame:
    """Compute LPAPS scores relative to alpha=0 baseline."""
    print("\n=== Computing LPAPS Preservation ===")

    # Load baseline (alpha=0) audios
    baseline_audios, baseline_sr = load_audios_for_alpha(steering_dir, 0.0)

    results = []
    for alpha in tqdm(alphas, desc="LPAPS"):
        if alpha == 0.0:
            results.append(
                {
                    "alpha": alpha,
                    "mean": 0.0,
                    "std": 0.0,
                }
            )
            continue

        target_audios, target_sr = load_audios_for_alpha(steering_dir, alpha)

        lpaps_df = get_lpaps(
            source_audios=baseline_audios,
            edits=target_audios,
            srs_src=[baseline_sr] * len(baseline_audios),
            srs_edit=[target_sr] * len(target_audios),
            device=device,
        )

        results.append(
            {
                "alpha": alpha,
                "mean": lpaps_df["lpaps"].mean(),
                "std": lpaps_df["lpaps"].std(),
            }
        )

    return pd.DataFrame(results)


# NOTE: PSNR/SSIM computation commented out - too slow and disk-intensive
# def resample_alpha_dir(
#     steering_dir: str,
#     alpha: float,
#     target_sr: int = 32000,
# ) -> str:
#     """Resample all audio files in an alpha directory to target sample rate."""
#     from torchaudio.transforms import Resample
#
#     alpha_dir = os.path.join(steering_dir, f"alpha_{alpha}")
#     resampled_dir = os.path.join(steering_dir, f"alpha_{alpha}_32k")
#
#     if os.path.exists(resampled_dir):
#         # Already resampled
#         return resampled_dir
#
#     os.makedirs(resampled_dir, exist_ok=True)
#
#     for audio_file in Path(alpha_dir).glob("*.wav"):
#         audio, sr = torchaudio.load(audio_file)
#         if sr != target_sr:
#             resampler = Resample(sr, target_sr)
#             audio = resampler(audio)
#         torchaudio.save(os.path.join(resampled_dir, audio_file.name), audio, target_sr)
#
#     return resampled_dir
#
#
# def compute_psnr_ssim_preservation(
#     steering_dir: str,
#     alphas: List[float],
#     sample_rate: int,
#     device: str = "cuda",
# ) -> pd.DataFrame:
#     """Compute PSNR and SSIM scores relative to alpha=0 baseline using MusicAlignmentEval."""
#     print("\n=== Computing PSNR/SSIM Preservation ===")
#
#     from src.metrics.alignment import MusicAlignmentEval
#
#     # Resample baseline to 32kHz for MusicAlignmentEval compatibility
#     baseline_dir_32k = resample_alpha_dir(steering_dir, 0.0, target_sr=32000)
#
#     evaluator = MusicAlignmentEval(sampling_rate=32000, device=torch.device(device))
#
#     results = []
#     for alpha in tqdm(alphas, desc="PSNR/SSIM"):
#         if alpha == 0.0:
#             results.append(
#                 {
#                     "alpha": alpha,
#                     "psnr": float("inf"),
#                     "ssim": 1.0,
#                 }
#             )
#             continue
#
#         # Resample alpha dir to 32kHz
#         alpha_dir_32k = resample_alpha_dir(steering_dir, alpha, target_sr=32000)
#
#         try:
#             metrics = evaluator.main(
#                 generate_files_path=alpha_dir_32k,
#                 groundtruth_path=baseline_dir_32k,
#                 limit_num=None,
#             )
#             results.append(
#                 {
#                     "alpha": alpha,
#                     "psnr": float(metrics.get("psnr", "nan")),
#                     "ssim": float(metrics.get("ssim", "nan")),
#                 }
#             )
#         except Exception as e:
#             print(f"Warning: PSNR/SSIM failed for alpha={alpha}: {e}")
#             results.append(
#                 {
#                     "alpha": alpha,
#                     "psnr": float("nan"),
#                     "ssim": float("nan"),
#                 }
#             )
#
#     return pd.DataFrame(results)


def compute_fad_to_baseline(
    steering_dir: str,
    alphas: List[float],
) -> pd.DataFrame:
    """Compute FAD scores relative to alpha=0 baseline."""
    print("\n=== Computing FAD to Baseline ===")

    baseline_dir = _find_alpha_dir(steering_dir, 0.0)

    results = []
    for alpha in tqdm(alphas, desc="FAD"):
        if alpha == 0.0:
            results.append(
                {
                    "alpha": alpha,
                    "fad": 0.0,
                }
            )
            continue

        alpha_dir = _find_alpha_dir(steering_dir, alpha)

        try:
            fad_score = calculate_fad(alpha_dir, baseline_dir)
            results.append(
                {
                    "alpha": alpha,
                    "fad": fad_score,
                }
            )
        except Exception as e:
            print(f"Warning: FAD failed for alpha={alpha}: {e}")
            results.append(
                {
                    "alpha": alpha,
                    "fad": float("nan"),
                }
            )

    return pd.DataFrame(results)


def compute_aesthetics(
    steering_dir: str,
    alphas: List[float],
    sample_rate: int,
    device: str = "cuda",
    batch_size: int = 10,
) -> pd.DataFrame:
    """Compute audiobox aesthetics scores with batched processing to avoid OOM."""
    print("\n=== Computing Audiobox Aesthetics ===")

    try:
        from audiobox_aesthetics.infer import AesPredictor
    except ImportError:
        print("Warning: audiobox_aesthetics not installed. Skipping aesthetics.")
        return pd.DataFrame()

    predictor = AesPredictor(checkpoint_pth=None, data_col="wav") # type: ignore

    results = []
    for alpha in tqdm(alphas, desc="Aesthetics"):
        audios, sr = load_audios_for_alpha(steering_dir, alpha)

        # Process in batches to avoid CUDA OOM
        all_preds = []
        try:
            for i in range(0, len(audios), batch_size):
                batch_audios = audios[i : i + batch_size]
                inputs = [{"wav": audio, "sample_rate": sr} for audio in batch_audios]
                batch_preds = predictor.forward(inputs)
                all_preds.extend(batch_preds)

                # Clear CUDA cache between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Extract mean scores
            ce_scores = [p["CE"] for p in all_preds]
            cu_scores = [p["CU"] for p in all_preds]
            pc_scores = [p["PC"] for p in all_preds]
            pq_scores = [p["PQ"] for p in all_preds]

            results.append(
                {
                    "alpha": alpha,
                    "content_enjoyment_mean": np.mean(ce_scores),
                    "content_enjoyment_std": np.std(ce_scores),
                    "content_usefulness_mean": np.mean(cu_scores),
                    "content_usefulness_std": np.std(cu_scores),
                    "production_complexity_mean": np.mean(pc_scores),
                    "production_complexity_std": np.std(pc_scores),
                    "production_quality_mean": np.mean(pq_scores),
                    "production_quality_std": np.std(pq_scores),
                }
            )
        except Exception as e:
            print(f"Warning: Aesthetics failed for alpha={alpha}: {e}")
            results.append(
                {
                    "alpha": alpha,
                    "content_enjoyment_mean": float("nan"),
                    "content_enjoyment_std": float("nan"),
                    "content_usefulness_mean": float("nan"),
                    "content_usefulness_std": float("nan"),
                    "production_complexity_mean": float("nan"),
                    "production_complexity_std": float("nan"),
                    "production_quality_mean": float("nan"),
                    "production_quality_std": float("nan"),
                }
            )

    return pd.DataFrame(results)


def compute_conceptual_range(
    alignment_df: pd.DataFrame,
    has_negative_prompt: bool = False,
) -> float:
    """
    Compute Conceptual Range (CR) metric.

    CR = max(scores for alpha >= 0) - min(scores for alpha <= 0)

    Both sets include alpha=0, so if positive steering hurts alignment the
    max stays at the baseline value, and CR >= 0 is guaranteed.

    If has_negative_prompt=True (different eval prompts for pos/neg alphas):
        CR_pos = max(scores for alpha >= 0) - score(alpha=0)
        CR_neg = score(alpha=0) - min(scores for alpha <= 0)
        CR = CR_pos + CR_neg
    """
    alphas = alignment_df["alpha"].values
    means = alignment_df["mean"].values

    pos_mask = alphas >= 0
    neg_mask = alphas <= 0
    pos_means = means[pos_mask]
    neg_means = means[neg_mask]

    if len(pos_means) == 0 or len(neg_means) == 0:
        return float(means.max() - means.min())

    if not has_negative_prompt:
        return float(pos_means.max() - neg_means.min())

    # With negative prompt: each side uses a different eval prompt,
    # so scores are not directly comparable across sides.
    # Measure each side's range relative to alpha=0.
    baseline_mask = np.abs(alphas) < 1e-9
    if baseline_mask.any():
        baseline = float(means[baseline_mask].mean())
    else:
        baseline = float(means[np.argmin(np.abs(alphas))])

    cr_pos = max(float(pos_means.max()) - baseline, 0.0)
    cr_neg = max(baseline - float(neg_means.min()), 0.0)
    return cr_pos + cr_neg


def compute_semantic_preservation(
    df: pd.DataFrame,
    value_col: str = "mean",
    exclude_zero: bool = True,
) -> float:
    """
    Compute Semantic Preservation (SP) metric for any preservation metric.

    SP = mean across alphas of (per-alpha metric value)

    Args:
        df: DataFrame with 'alpha' column and value column
        value_col: Column name containing the metric values (default: "mean")
        exclude_zero: Whether to exclude alpha=0 from computation

    Returns:
        Mean of the metric across all alphas (excluding alpha=0 if specified).
        Lower is better for distance metrics (LPAPS, FAD).
        Higher is better for similarity metrics (SSIM, PSNR).
    """
    if exclude_zero:
        filtered = df[df["alpha"] != 0.0]
    else:
        filtered = df

    # Handle inf values (e.g., PSNR at alpha=0)
    values = filtered[value_col].replace([np.inf, -np.inf], np.nan)
    return values.mean()


def compute_conceptual_smoothness(
    alignment_df: pd.DataFrame,
) -> float:
    """
    Compute Conceptual Smoothness (CSM) metric.

    CSM = std of consecutive alignment score differences
    Lower is better (smoother transitions).

    Note: When different prompts are used for pos/neg alphas, smoothness
    is computed separately for each side and then averaged.
    """
    df_sorted = alignment_df.sort_values("alpha")

    # Split into negative and positive alphas
    neg_df = df_sorted[df_sorted["alpha"] < 0].sort_values("alpha")
    pos_df = df_sorted[df_sorted["alpha"] > 0].sort_values("alpha")

    # Compute consecutive differences
    diffs = []

    # Negative side (ordered from most negative to 0)
    if len(neg_df) > 0:
        neg_vals = neg_df["mean"].values
        neg_diffs = np.diff(neg_vals)
        diffs.extend(neg_diffs[neg_diffs >= 0])

    # Positive side (ordered from 0 to most positive)
    if len(pos_df) > 0:
        pos_vals = pos_df["mean"].values
        pos_diffs = np.diff(pos_vals)
        diffs.extend(pos_diffs[pos_diffs >= 0])

    if len(diffs) == 0:
        return float("nan")

    return np.std(diffs)


def compute_overall_score(
    cr: float,
    sp: float,
    csm: float,
    epsilon: float = 1.0,
) -> float:
    """
    Compute Overall Score (OS).

    OS = CR / (epsilon + SP) + (1 - CSM)
    """
    return cr / (epsilon + sp) + (1 - csm)


def plot_alignment_curves(
    clap_df: pd.DataFrame,
    muqt_df: pd.DataFrame,
    eval_prompts: dict,
    negative_eval_prompt: Optional[str],
    save_path: str,
):
    """Plot alignment curves for CLAP and MUQ-T."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Build per-metric title suffixes
    def _title_suffix(prompt):
        if negative_eval_prompt:
            s = f' (pos: "{prompt}", neg: "{negative_eval_prompt}")'
        else:
            s = f': "{prompt}"'
        return s[:57] + '..."' if len(s) > 60 else s

    # CLAP plot
    axes[0].errorbar(
        clap_df["alpha"],
        clap_df["mean"],
        yerr=clap_df["std"],
        fmt="-o",
        capsize=5,
        color="#009FB7",
        linewidth=2,
        markersize=6,
    )
    axes[0].set_xlabel(r"$\alpha$", fontsize=12)
    axes[0].set_ylabel("CLAP (Music) Similarity", fontsize=12)
    axes[0].set_title(f"CLAP{_title_suffix(eval_prompts['clap'])}")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].axvline(0, color="gray", linestyle="--", alpha=0.5)

    # MUQ-T plot
    axes[1].errorbar(
        muqt_df["alpha"],
        muqt_df["mean"],
        yerr=muqt_df["std"],
        fmt="-o",
        capsize=5,
        color="#FE4A49",
        linewidth=2,
        markersize=6,
    )
    axes[1].set_xlabel(r"$\alpha$", fontsize=12)
    axes[1].set_ylabel("MUQ-T Similarity", fontsize=12)
    axes[1].set_title(f"MUQ-T{_title_suffix(eval_prompts['muqt'])}")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].axvline(0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved alignment plot to {save_path}")


def plot_preservation_curves(
    lpaps_df: pd.DataFrame,
    fad_df: pd.DataFrame,
    save_path: str,
):
    """Plot preservation metric curves (LPAPS and FAD only)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # LPAPS
    axes[0].plot(
        lpaps_df["alpha"],
        lpaps_df["mean"],
        "-o",
        color="#2E86AB",
        linewidth=2,
        markersize=6,
    )
    axes[0].fill_between(
        lpaps_df["alpha"],
        lpaps_df["mean"] - lpaps_df["std"],
        lpaps_df["mean"] + lpaps_df["std"],
        alpha=0.3,
        color="#2E86AB",
    )
    axes[0].set_xlabel(r"$\alpha$", fontsize=12)
    axes[0].set_ylabel("LPAPS (↓ better)", fontsize=12)
    axes[0].set_title("Audio Perceptual Distance (LPAPS)")
    axes[0].grid(True, alpha=0.3)

    # FAD to baseline
    axes[1].plot(
        fad_df["alpha"], fad_df["mean"], "-o", color="#C73E1D", linewidth=2, markersize=6
    )
    axes[1].set_xlabel(r"$\alpha$", fontsize=12)
    axes[0].set_ylabel("LPAPS (↓ better)", fontsize=12)
    axes[0].set_title("Audio Perceptual Distance (LPAPS)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved preservation plot to {save_path}")


def plot_quality_curves(
    aesthetics_df: pd.DataFrame,
    save_path: str,
):
    """Plot audio quality metric curves (Audiobox Aesthetics)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    if not aesthetics_df.empty:
        # Content Enjoyment
        axes[0, 0].errorbar(
            aesthetics_df["alpha"],
            aesthetics_df["content_enjoyment_mean"],
            yerr=aesthetics_df["content_enjoyment_std"],
            fmt="-o",
            capsize=5,
            color="#5C8001",
            linewidth=2,
            markersize=6,
        )
        axes[0, 0].set_xlabel(r"$\alpha$", fontsize=12)
        axes[0, 0].set_ylabel("Score", fontsize=12)
        axes[0, 0].set_title("Content Enjoyment (↑ better)")
        axes[0, 0].grid(True, alpha=0.3)

        # Content Usefulness
        axes[0, 1].errorbar(
            aesthetics_df["alpha"],
            aesthetics_df["content_usefulness_mean"],
            yerr=aesthetics_df["content_usefulness_std"],
            fmt="-o",
            capsize=5,
            color="#3D5A80",
            linewidth=2,
            markersize=6,
        )
        axes[0, 1].set_xlabel(r"$\alpha$", fontsize=12)
        axes[0, 1].set_ylabel("Score", fontsize=12)
        axes[0, 1].set_title("Content Usefulness (↑ better)")
        axes[0, 1].grid(True, alpha=0.3)

        # Production Quality
        axes[1, 0].errorbar(
            aesthetics_df["alpha"],
            aesthetics_df["production_quality_mean"],
            yerr=aesthetics_df["production_quality_std"],
            fmt="-o",
            capsize=5,
            color="#EE6C4D",
            linewidth=2,
            markersize=6,
        )
        axes[1, 0].set_xlabel(r"$\alpha$", fontsize=12)
        axes[1, 0].set_ylabel("Score", fontsize=12)
        axes[1, 0].set_title("Production Quality (↑ better)")
        axes[1, 0].grid(True, alpha=0.3)

        # Production Complexity
        axes[1, 1].errorbar(
            aesthetics_df["alpha"],
            aesthetics_df["production_complexity_mean"],
            yerr=aesthetics_df["production_complexity_std"],
            fmt="-o",
            capsize=5,
            color="#98C1D9",
            linewidth=2,
            markersize=6,
        )
        axes[1, 1].set_xlabel(r"$\alpha$", fontsize=12)
        axes[1, 1].set_ylabel("Score", fontsize=12)
        axes[1, 1].set_title("Production Complexity")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        for ax in axes.flat:
            ax.text(
                0.5,
                0.5,
                "Aesthetics N/A",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved quality plot to {save_path}")


def run_aesthetics_only(steering_dir: str, device: str = "cuda"):
    """
    Compute only aesthetics metrics and merge with existing results.

    This is useful when the initial evaluation failed on aesthetics (e.g., CUDA OOM)
    but other metrics were computed successfully.
    """
    print("=== Aesthetics-Only Evaluation Mode ===")
    print(f"Steering directory: {steering_dir}")

    output_dir = os.path.join(steering_dir, "protocol_results")

    # Check for existing results
    all_results_path = os.path.join(output_dir, "all_results.json")
    summary_path = os.path.join(output_dir, "summary.json")

    if not os.path.exists(all_results_path):
        raise FileNotFoundError(
            f"No existing results found at {all_results_path}. "
            "Run full evaluation first, then use --aesthetics_only to add quality metrics."
        )

    # Load existing results
    print(f"Loading existing results from {all_results_path}")
    with open(all_results_path, "r") as f:
        results = json.load(f)

    existing_summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            existing_summary = json.load(f)

    # Get alphas and sample rate
    alphas = get_alphas_from_dir(steering_dir)
    print(f"Found {len(alphas)} alpha values")

    # Use alphas from summary if available (respects positive_alphas_only setting)
    if "alphas" in existing_summary:
        alphas = existing_summary["alphas"]
        print(f"Using {len(alphas)} alphas from existing summary")

    # Get sample rate
    sample_rate = existing_summary.get("sample_rate")
    if sample_rate is None:
        _, sample_rate = load_audios_for_alpha(steering_dir, alphas[0])
    print(f"Sample rate: {sample_rate}")

    # Compute aesthetics
    print("\n" + "=" * 50)
    print("Computing Audio Quality Metrics (Aesthetics)")
    print("=" * 50)

    aesthetics_df = compute_aesthetics(steering_dir, alphas, sample_rate, device)

    if aesthetics_df.empty:
        print("WARNING: Aesthetics computation returned empty results")
        return results

    # Update results with aesthetics
    results["aesthetics"] = aesthetics_df.to_dict()

    # Compute mean quality metrics
    quality_metrics = {
        "mean_CE": aesthetics_df["content_enjoyment_mean"].mean(),
        "mean_CU": aesthetics_df["content_usefulness_mean"].mean(),
        "mean_PC": aesthetics_df["production_complexity_mean"].mean(),
        "mean_PQ": aesthetics_df["production_quality_mean"].mean(),
    }

    print("\nAudio Quality (mean across all alphas):")
    print(f"  Content Enjoyment (CE) ↑: {quality_metrics['mean_CE']:.4f}")
    print(f"  Content Usefulness (CU) ↑: {quality_metrics['mean_CU']:.4f}")
    print(f"  Production Complexity (PC): {quality_metrics['mean_PC']:.4f}")
    print(f"  Production Quality (PQ) ↑: {quality_metrics['mean_PQ']:.4f}")

    results["quality_metrics"] = quality_metrics

    # Save updated results
    print("\n" + "=" * 50)
    print("Saving Updated Results")
    print("=" * 50)

    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Updated: {all_results_path}")

    # Save aesthetics CSV
    aesthetics_df.to_csv(os.path.join(output_dir, "aesthetics.csv"), index=False)
    print(f"Saved: {os.path.join(output_dir, 'aesthetics.csv')}")

    # Update summary
    if existing_summary:
        existing_summary["quality_metrics"] = quality_metrics
        with open(summary_path, "w") as f:
            json.dump(existing_summary, f, indent=2)
        print(f"Updated: {summary_path}")

    # Save quality metrics CSV
    quality_summary_df = pd.DataFrame([quality_metrics])
    quality_summary_df.to_csv(
        os.path.join(output_dir, "quality_summary.csv"), index=False
    )
    print(f"Saved: {os.path.join(output_dir, 'quality_summary.csv')}")

    # Update quality plot
    plot_quality_curves(aesthetics_df, os.path.join(output_dir, "quality_curves.png"))

    print(f"\n=== Aesthetics Update Complete ===")
    print(f"Results updated in: {output_dir}")

    return results


def main(
    steering_dir: str,
    concept: Literal["piano", "mood", "tempo", "vocal_gender", "vocal_style", "guitar_electronic", "violin", "rock_genre", "electronic_music"],
    negative_eval_prompt: Optional[str] = None,
    positive_alphas_only: bool = False,
    device: str = "cuda",
    skip_aesthetics: bool = False,
    aesthetics_only: bool = False,
    only_clap: bool = False,
    only_muqt: bool = False,
):
    """
    Run comprehensive steering evaluation protocol.

    Args:
        steering_dir: Directory containing alpha_* subdirectories with generated audios
        concept: Concept name — per-metric prompts are looked up from CONCEPT_TO_EVAL_PROMPTS
        negative_eval_prompt: Optional prompt for alpha < 0 (contextual evaluation)
        positive_alphas_only: If True, only evaluate alphas >= 0
        device: Device to run on
        skip_aesthetics: Skip audiobox aesthetics computation
        aesthetics_only: Only compute aesthetics and merge with existing results
        only_clap: If True, only compute/update CLAP alignment (load existing MUQ-T from CSV)
        only_muqt: If True, only compute/update MUQ-T alignment (load existing CLAP from CSV)
    """
    if only_clap and only_muqt:
        raise ValueError("Cannot set both --only_clap and --only_muqt")
    # Handle aesthetics_only mode
    if aesthetics_only:
        return run_aesthetics_only(steering_dir, device)

    eval_prompts = CONCEPT_TO_EVAL_PROMPTS[concept]

    print("=== Comprehensive Steering Evaluation Protocol ===")
    print(f"Steering directory: {steering_dir}")
    print(f"Eval prompts: CLAP='{eval_prompts['clap']}', MUQ-T='{eval_prompts['muqt']}'")
    if negative_eval_prompt:
        print(f"Negative eval prompt (for alpha < 0): {negative_eval_prompt}")
    if positive_alphas_only:
        print("Mode: Positive alphas only (alpha >= 0)")

    # Get alphas and sample rate
    alphas = get_alphas_from_dir(steering_dir)
    print(f"Found {len(alphas)} alpha values: {alphas}")

    if 0.0 not in alphas:
        raise ValueError("Baseline alpha=0.0 not found in steering directory")

    # Filter to positive alphas only if requested
    if positive_alphas_only:
        alphas = [a for a in alphas if a >= 0]
        print(f"Filtered to {len(alphas)} positive alphas: {alphas}")

    # Get sample rate from first audio
    _, sample_rate = load_audios_for_alpha(steering_dir, alphas[0])
    print(f"Sample rate: {sample_rate}")

    # Create output directory
    output_dir = os.path.join(steering_dir, "protocol_results")
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    has_negative_prompt = negative_eval_prompt is not None

    # 1. Concept Alignment Metrics
    print("\n" + "=" * 50)
    print("PHASE 1: Concept Alignment Metrics")
    print("=" * 50)

    if not only_muqt:
        clap_df = compute_clap_alignment(
            steering_dir, alphas, eval_prompts["clap"], negative_eval_prompt, device
        )
    else:
        clap_csv = os.path.join(output_dir, "clap.csv")
        if not os.path.exists(clap_csv):
            raise FileNotFoundError(f"--only_muqt requires existing {clap_csv}")
        clap_df = pd.read_csv(clap_csv)
        print(f"Loaded existing CLAP results from {clap_csv}")

    if not only_clap:
        muqt_df = compute_muqt_alignment(
            steering_dir, alphas, eval_prompts["muqt"], sample_rate, negative_eval_prompt, device
        )
    else:
        muqt_csv = os.path.join(output_dir, "muqt.csv")
        if not os.path.exists(muqt_csv):
            raise FileNotFoundError(f"--only_clap requires existing {muqt_csv}")
        muqt_df = pd.read_csv(muqt_csv)
        print(f"Loaded existing MUQ-T results from {muqt_csv}")

    results["clap"] = clap_df.to_dict()
    results["muqt"] = muqt_df.to_dict()

    # 2. Audio Preservation Metrics
    print("\n" + "=" * 50)
    print("PHASE 2: Audio Preservation Metrics")
    print("=" * 50)

    run_single_metric = only_clap or only_muqt
    if run_single_metric:
        lpaps_csv = os.path.join(output_dir, "lpaps.csv")
        if not os.path.exists(lpaps_csv):
            raise FileNotFoundError(
                f"--only_clap/--only_muqt requires existing {lpaps_csv}"
            )
        lpaps_df = pd.read_csv(lpaps_csv)
        print(f"Loaded existing LPAPS results from {lpaps_csv}")
    else:
        lpaps_df = compute_lpaps_preservation(steering_dir, alphas, device)
    results["lpaps"] = lpaps_df.to_dict()

    # NOTE: PSNR/SSIM/FAD computation disabled - too slow and disk-intensive

    # fad_baseline_df = compute_fad_to_baseline(steering_dir, alphas)
    # results["fad_baseline"] = fad_baseline_df.to_dict()

    # 3. Audio Quality Metrics
    print("\n" + "=" * 50)
    print("PHASE 3: Audio Quality Metrics")
    print("=" * 50)

    aesthetics_df = pd.DataFrame()
    if run_single_metric:
        aes_csv = os.path.join(output_dir, "aesthetics.csv")
        if os.path.exists(aes_csv):
            aesthetics_df = pd.read_csv(aes_csv)
            print(f"Loaded existing aesthetics from {aes_csv}")
        else:
            print("No existing aesthetics.csv found, skipping.")
    elif not skip_aesthetics:
        aesthetics_df = compute_aesthetics(steering_dir, alphas, sample_rate, device)
        if not aesthetics_df.empty:
            results["aesthetics"] = aesthetics_df.to_dict()

    # Compute mean aesthetics across all alphas
    quality_metrics = {}
    if not aesthetics_df.empty:
        quality_metrics = {
            "mean_CE": aesthetics_df["content_enjoyment_mean"].mean(),
            "mean_CU": aesthetics_df["content_usefulness_mean"].mean(),
            "mean_PC": aesthetics_df["production_complexity_mean"].mean(),
            "mean_PQ": aesthetics_df["production_quality_mean"].mean(),
        }
        print("\nAudio Quality (mean across all alphas):")
        print(f"  Content Enjoyment (CE) ↑: {quality_metrics['mean_CE']:.4f}")
        print(f"  Content Usefulness (CU) ↑: {quality_metrics['mean_CU']:.4f}")
        print(f"  Production Complexity (PC): {quality_metrics['mean_PC']:.4f}")
        print(f"  Production Quality (PQ) ↑: {quality_metrics['mean_PQ']:.4f}")

    results["quality_metrics"] = quality_metrics

    # 4. Steering Metrics
    print("\n" + "=" * 50)
    print("PHASE 4: Steering Metrics (CR, SP, CSM)")
    print("=" * 50)

    # Compute Semantic Preservation for preservation metrics
    sp_lpaps = compute_semantic_preservation(lpaps_df, value_col="mean")
    # sp_fad = compute_semantic_preservation(fad_baseline_df, value_col="fad")

    preservation_metrics = {
        "SP_LPAPS": sp_lpaps,  # ↓ better (distance)
        # "SP_FAD": sp_fad,  # ↓ better (distance)
    }

    print("\nSemantic Preservation (SP) across all alphas:")
    print(f"  SP_LPAPS ↓: {sp_lpaps:.4f}")
    # print(f"  SP_FAD ↓: {sp_fad:.4f}")

    results["preservation_metrics"] = preservation_metrics

    # Compute CR, CSM, OS for both CLAP and MUQ-T
    steering_metrics = {}

    for metric_name, alignment_df in [("clap", clap_df), ("muqt", muqt_df)]:
        cr = compute_conceptual_range(alignment_df, has_negative_prompt)
        csm = compute_conceptual_smoothness(alignment_df)
        # Use LPAPS as default SP for overall score (as in original paper)
        os_score = compute_overall_score(cr, sp_lpaps, csm)

        steering_metrics[metric_name] = {
            "conceptual_range": cr,
            "conceptual_smoothness": csm,
            "overall_score": os_score,
        }

        print(f"\n{metric_name.upper()} Steering Metrics:")
        print(f"  Conceptual Range (CR) ↑: {cr:.4f}")
        print(f"  Conceptual Smoothness (CSM) ↓: {csm:.4f}")
        print(f"  Overall Score (OS) ↑: {os_score:.4f}")

    results["steering_metrics"] = steering_metrics

    # 5. Generate Plots
    print("\n" + "=" * 50)
    print("PHASE 5: Generating Plots")
    print("=" * 50)

    plot_alignment_curves(
        clap_df,
        muqt_df,
        eval_prompts,
        negative_eval_prompt,
        os.path.join(output_dir, "alignment_curves.png"),
    )
    plot_preservation_curves(
        lpaps_df,
        lpaps_df,
        os.path.join(output_dir, "preservation_curves.png"),
    )
    plot_quality_curves(aesthetics_df, os.path.join(output_dir, "quality_curves.png"))

    # 6. Save Results
    print("\n" + "=" * 50)
    print("PHASE 6: Saving Results")
    print("=" * 50)

    # Save all results to JSON
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save CSVs (only overwrite metrics that were recomputed)
    if not only_muqt:
        clap_df.to_csv(os.path.join(output_dir, "clap.csv"), index=False)
    if not only_clap:
        muqt_df.to_csv(os.path.join(output_dir, "muqt.csv"), index=False)
    if not run_single_metric:
        lpaps_df.to_csv(os.path.join(output_dir, "lpaps.csv"), index=False)
    # fad_baseline_df.to_csv(os.path.join(output_dir, "fad_baseline.csv"), index=False)

    if not aesthetics_df.empty:
        aesthetics_df.to_csv(os.path.join(output_dir, "aesthetics.csv"), index=False)

    # Save summary
    summary = {
        "steering_dir": steering_dir,
        "eval_prompts": eval_prompts,
        "negative_eval_prompt": negative_eval_prompt,
        "positive_alphas_only": positive_alphas_only,
        "alphas": alphas,
        "sample_rate": sample_rate,
        "preservation_metrics": preservation_metrics,
        "quality_metrics": quality_metrics,
        "steering_metrics": steering_metrics,
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save preservation metrics as CSV
    preservation_summary_df = pd.DataFrame([preservation_metrics])
    preservation_summary_df.to_csv(
        os.path.join(output_dir, "preservation_summary.csv"), index=False
    )

    # Save quality metrics as CSV
    if quality_metrics:
        quality_summary_df = pd.DataFrame([quality_metrics])
        quality_summary_df.to_csv(
            os.path.join(output_dir, "quality_summary.csv"), index=False
        )

    # Save steering metrics as CSV for easy comparison
    steering_summary_rows = []
    for metric_name, metrics in steering_metrics.items():
        steering_summary_rows.append(
            {
                "alignment_metric": metric_name,
                "CR": metrics["conceptual_range"],
                "CSM": metrics["conceptual_smoothness"],
                "OS": metrics["overall_score"],
            }
        )
    steering_summary_df = pd.DataFrame(steering_summary_rows)
    steering_summary_df.to_csv(
        os.path.join(output_dir, "steering_summary.csv"), index=False
    )

    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    Fire(main)
