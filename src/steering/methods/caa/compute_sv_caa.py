"""
Compute steering vectors for ACE-Step model using SAE prompts config.

This script generates steering vectors by computing the difference between
activations from positive and negative prompt pairs defined in
steering.sae.lib.configs.steer_prompts.CONCEPT_TO_PROMPTS.

Usage:
    python steering/caa/compute_sv_caa.py \
        --concept piano \
        --audio_duration 30.0 \
        --num_inference_steps 30 \
        --guidance_scale 5.0 \
        --seed 10
"""

import json
import os
import pickle
import sys
from pathlib import Path

import torch
from fire import Fire

PATH_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PATH_PROJECT)
sys.path.append(os.path.join(PATH_PROJECT, "src", "models", "ace_step", "ACE"))

from src.models.ace_step.ace_steering.controller import compute_num_cfg_passes
from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline
from src.steering.methods.caa.utils import (
    DEFAULT_DEVICE,
    compute_standard_steering_vectors,
    generate_vectors_standard,
    get_prompts_pairs,
)

DEFAULT_SAVE_DIR = "steering_vectors/caa"


def main(
    concept: str,
    num_inference_steps: int = 30,
    audio_duration: float = 30.0,
    guidance_scale_text: float = 0.0,
    guidance_scale_lyric: float = 0.0,
    guidance_scale: float = 5.0,
    guidance_interval: float = 1.0,
    guidance_interval_decay: float = 0.0,
    seed: int = 10,
    device: str = DEFAULT_DEVICE,
    save_dir: str = DEFAULT_SAVE_DIR,
    save_all_cfg_passes: bool = True,
):
    """
    Compute steering vectors for a concept defined in CONCEPT_TO_PROMPTS.

    Args:
        concept: Concept name from CONCEPT_TO_PROMPTS
                 (e.g., 'piano', 'mood', 'tempo', 'vocal_gender', 'drums')
        num_inference_steps: Number of diffusion steps
        audio_duration: Audio duration in seconds
        guidance_scale_text: Text guidance scale
        guidance_scale_lyric: Lyric guidance scale
        guidance_scale: Overall guidance scale
        guidance_interval: Guidance interval
        guidance_interval_decay: Guidance interval decay
        seed: Random seed
        device: Device to run on
        save_dir: Directory to save steering vectors
        save_all_cfg_passes: Whether to save all CFG passes
    """
    # Initialize pipeline
    print("Loading ACE-Step pipeline...")
    pipe = SimpleACEStepPipeline(device=device)
    pipe.load()
    print("Pipeline loaded")

    # Get prompts
    prompts_pos, prompts_neg, lyrics = get_prompts_pairs(concept)

    # Collect activations
    num_cfg_passes = compute_num_cfg_passes(guidance_scale_text, guidance_scale_lyric)
    pos_vectors, neg_vectors = generate_vectors_standard(
        prompts_pos=prompts_pos,
        prompts_neg=prompts_neg,
        pipe=pipe,
        device=device,
        save_all_cfg_passes=save_all_cfg_passes,
        audio_duration=audio_duration,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
        guidance_scale=guidance_scale,
        guidance_interval=guidance_interval,
        guidance_interval_decay=guidance_interval_decay,
    )

    # Compute steering vectors
    steering_vectors = compute_standard_steering_vectors(pos_vectors, neg_vectors)

    # Save steering vectors
    save_directory = Path(save_dir).resolve()
    vector_directory = f"ace_{concept}_passes{num_cfg_passes}_all{save_all_cfg_passes}"
    save_directory = (save_directory / vector_directory).resolve()
    os.makedirs(save_directory, exist_ok=True)

    with open((save_directory / "sv.pkl"), "wb") as f:
        pickle.dump(steering_vectors, f)
    with open((save_directory / "pos_vectors.pkl"), "wb") as f:
        pickle.dump(pos_vectors, f)
    with open((save_directory / "neg_vectors.pkl"), "wb") as f:
        pickle.dump(neg_vectors, f)
    with open((save_directory / "config.json"), "w") as f:
        json.dump(
            {
                "method": "standard_caa",
                "concept": concept,
                "lyrics": lyrics,
                "num_cfg_passes": num_cfg_passes,
                "save_all_cfg_passes": save_all_cfg_passes,
                "audio_duration": audio_duration,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                "device": device,
                "save_dir": save_dir,
                "guidance_scale_text": guidance_scale_text,
                "guidance_scale_lyric": guidance_scale_lyric,
                "guidance_scale": guidance_scale,
                "guidance_interval": guidance_interval,
                "guidance_interval_decay": guidance_interval_decay,
            },
            f,
            indent=2,
        )

    print(f"\nSteering vectors saved to: {save_directory}")
    print("Files:")
    print("  - sv.pkl: Steering vectors")
    print("  - pos_vectors.pkl: Positive activations")
    print("  - neg_vectors.pkl: Negative activations")
    print("  - config.json: Configuration")
    print("\nCompatible steering modes:")
    if save_all_cfg_passes:
        print("  - 'separate': (cond steered with cond vectors, uncond with uncond)")
        print("  - 'cond_only': (only cond steered, uses cond vectors)")
        print("  - 'both_cond': (both steered with cond vectors)")
    else:
        print("  - 'separate': (need --save_all_cfg_passes)")
        print("  - 'cond_only': (only cond steered, uses cond vectors)")
        print("  - 'both_cond': (both steered with cond vectors)")


if __name__ == "__main__":
    Fire(main)
