"""
Evaluation utilities for steering vectors.

Contains functions for loading configs, validating parameters,
and evaluating generated audio.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio

PATH_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.append(PATH_PROJECT)

from editing.eval import get_mulan
from src.steering.methods.caa.utils.constants import (
    AUDIO_LENGTH_IN_S,
    GUIDANCE_SCALE,
    MULTIPLIERS,
    NUM_INFERENCE_STEPS,
)


# =============================================================================
# Config Loading and Validation
# =============================================================================


def load_sv_config(config_path):
    """Load steering vector config from json file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def validate_sv_config(
    config,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
):
    """
    Validate steering vector config against script parameters.

    Args:
        config: Loaded config dict
        num_inference_steps: Expected number of inference steps
        guidance_scale: Expected guidance scale

    Raises:
        ValueError: If critical parameters don't match
    """
    # Validate critical parameters - these MUST match
    config_num_steps = config.get("num_inference_steps")
    if config_num_steps is not None and config_num_steps != num_inference_steps:
        raise ValueError(
            f"CRITICAL: num_inference_steps mismatch! "
            f"Config has {config_num_steps}, but script expects {num_inference_steps}. "
            f"Steering vectors were computed with different diffusion steps."
        )

    config_guidance = config.get("guidance_scale")
    if config_guidance is not None:
        # Check for CFG vs no-CFG mismatch (one <=1.0 means no CFG, >1.0 means CFG)
        config_uses_cfg = config_guidance > 1.0
        script_uses_cfg = guidance_scale > 1.0
        if config_uses_cfg != script_uses_cfg:
            raise ValueError(
                f"CRITICAL: guidance_scale CFG mismatch! "
                f"Config has {config_guidance} ({'CFG' if config_uses_cfg else 'no CFG'}), "
                f"but script expects {guidance_scale} ({'CFG' if script_uses_cfg else 'no CFG'}). "
                f"This would produce fundamentally different generations."
            )


def warn_config_mismatches(
    config,
    audio_duration=AUDIO_LENGTH_IN_S,
    guidance_scale=GUIDANCE_SCALE,
):
    """
    Warn about non-critical config mismatches.

    Args:
        config: Loaded config dict
        audio_duration: Expected audio duration
        guidance_scale: Expected guidance scale
    """
    config_audio_duration = config.get("audio_duration")
    if config_audio_duration is not None and config_audio_duration != audio_duration:
        print(
            f"WARNING: audio_duration mismatch. Config has {config_audio_duration}s, "
            f"using script value {audio_duration}s"
        )

    config_guidance_value = config.get("guidance_scale")
    if config_guidance_value is not None and config_guidance_value != guidance_scale:
        print(
            f"WARNING: guidance_scale mismatch. Config has {config_guidance_value}, "
            f"using script value {guidance_scale}"
        )


# =============================================================================
# Audio Evaluation
# =============================================================================


def eval_audios(
    save_dir_before_alphas,
    test_prompts,
    sample_rate,
    eval_prompts,
    multipliers=None,
):
    """
    Evaluate generated audios using MUQ-T similarity.

    Args:
        save_dir_before_alphas: Directory containing alpha_* subdirectories
        test_prompts: List of test prompts used for generation
        sample_rate: Audio sample rate
        eval_prompts: Evaluation prompt(s) for MUQ-T similarity
        multipliers: List of alpha multipliers to evaluate (default: MULTIPLIERS)
    """
    if multipliers is None:
        multipliers = MULTIPLIERS

    if not isinstance(eval_prompts, list):
        eval_prompts = [eval_prompts]

    for idx_ep, eval_prompt in enumerate(eval_prompts):
        all_dfs = []
        for alpha in multipliers:
            audios = [
                torchaudio.load(
                    save_dir_before_alphas + f"/alpha_{alpha}" + f"/p{p_idx}.wav"
                )[0]
                for p_idx in range(len(test_prompts))
            ]
            prompts = [eval_prompt] * len(audios)
            srs = [sample_rate] * len(audios)
            muqt_df_alpha = get_mulan(
                prompts, audios, srs, torch.device("cuda"), verbose=False
            )
            muqt_df_alpha["alpha"] = [alpha] * len(test_prompts)
            muqt_df_alpha["p_idx"] = list(range(len(test_prompts)))
            all_dfs.append(muqt_df_alpha)
        all_dfs = pd.concat(all_dfs)
        summary = all_dfs.groupby("alpha")["muqt_sim_p0"].agg(["mean", "std"])

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"],
            fmt="-o",
            capsize=5,
            color="#009FB7",
            ecolor="#AAA",
            linewidth=2,
            markersize=7,
        )
        ax.set_xlabel(r"$\alpha$", fontsize=14)
        ax.set_ylabel("MUQ-T Similarity", fontsize=14)
        pth_splitted = save_dir_before_alphas.split("/")
        ax.set_title(
            f'Similarity to prompt "{eval_prompt}" ({pth_splitted[-2]}, {pth_splitted[-1]})',
            fontsize=16,
            fontweight="bold",
        )
        ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.xaxis.labelpad = -10

        plt.xticks(summary.index, rotation=45, ha="right", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_dir_before_alphas + f"/muqt_sim_p{idx_ep}.png", dpi=150)
        plt.close()
        summary.to_csv(save_dir_before_alphas + f"/summary_p{idx_ep}.csv", index=True)

        print(f"  Saved plot to {save_dir_before_alphas}/muqt_sim_p{idx_ep}.png")
