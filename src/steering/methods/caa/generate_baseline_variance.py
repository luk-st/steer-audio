"""
Generate two sets of audios from the same prompts with different seeds (no steering),
then calculate mean LPAPS and FAD between the two sets.

This measures the natural variance of the generation process.

Usage:
    python steering/caa/generate_baseline_variance.py \
        --seed_a 2115 \
        --seed_b 42 \
        --save_dir steering/outputs/baseline_variance
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
from fire import Fire
from tqdm import tqdm

PATH_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PATH_PROJECT)
sys.path.append(os.path.join(PATH_PROJECT, "src", "models", "ace_step", "ACE"))

from editing.eval import get_lpaps
from src.metrics.metrics import calculate_fad
from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline
from src.steering.methods.caa.utils import (
    AUDIO_LENGTH_IN_S,
    GENERATION_SEED,
    GUIDANCE_SCALE,
    NUM_INFERENCE_STEPS,
)
from src.steering.eval.test_prompts import LYRICS, NO_LYRICS, TEST_PROMPTS


def generate_set(
    pipe,
    test_prompts,
    lyrics,
    seed,
    audio_duration,
    num_inference_steps,
    guidance_scale,
    save_dir,
    save_mono=True,
):
    """Generate a set of audios and save them.

    Args:
        save_mono: Whether to save audio as mono (default True).
    """
    os.makedirs(save_dir, exist_ok=True)

    latents = pipe.prepare_latents(
        batch_size=len(test_prompts),
        audio_duration=audio_duration,
        seed=seed,
    )

    audios = pipe.generate(
        prompt=test_prompts,
        lyrics=lyrics,
        audio_duration=audio_duration,
        infer_step=num_inference_steps,
        manual_seed=seed,
        return_type="audio",
        guidance_scale=guidance_scale,
        latents=latents,
    )

    for i, audio in enumerate(audios):
        if save_mono:
            audio = audio.mean(dim=0, keepdim=True)
        torchaudio.save(
            os.path.join(save_dir, f"p{i}.wav"),
            audio.cpu(),
            pipe.sample_rate,
        )

    return audios


def main(
    seed_a: int = GENERATION_SEED,
    seed_b: int = 42,
    save_dir: str | None = None,
    concept: str = "none",
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    audio_duration: float = AUDIO_LENGTH_IN_S,
    save_mono: bool = True,
):
    """
    Generate two sets of audios with different seeds and compute LPAPS and FAD.

    Args:
        seed_a: Seed for the first set of audios.
        seed_b: Seed for the second set of audios.
        save_dir: Directory to save outputs. If None, auto-generated.
        concept: Concept name (only affects lyrics for vocal concepts).
        num_inference_steps: Number of diffusion steps.
        guidance_scale: Classifier-free guidance scale.
        audio_duration: Duration of generated audio in seconds.
        save_mono: Whether to save audio as mono (default True).
    """
    if save_dir is None:
        save_dir = f"steering/outputs/baseline_variance/{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    dir_a = os.path.join(save_dir, f"seed_{seed_a}")
    dir_b = os.path.join(save_dir, f"seed_{seed_b}")

    lyrics = NO_LYRICS
    test_prompts = TEST_PROMPTS

    if "vocal" in concept:
        lyrics = LYRICS
        test_prompts = [f"{p}, with vocal" for p in TEST_PROMPTS]

    # Save run config
    run_config = {
        "seed_a": seed_a,
        "seed_b": seed_b,
        "concept": concept,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "audio_duration": audio_duration,
        "save_mono": save_mono,
        "num_prompts": len(test_prompts),
    }
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Initialize pipeline (no steering)
    print("Loading ACE-Step pipeline...")
    pipe = SimpleACEStepPipeline(device="cuda")
    pipe.load()
    print("Pipeline loaded")

    # Generate set A
    print(f"Generating set A with seed={seed_a}...")
    audios_a = generate_set(
        pipe, test_prompts, lyrics, seed_a,
        audio_duration, num_inference_steps, guidance_scale, dir_a,
        save_mono=save_mono,
    )

    # Generate set B
    print(f"Generating set B with seed={seed_b}...")
    audios_b = generate_set(
        pipe, test_prompts, lyrics, seed_b,
        audio_duration, num_inference_steps, guidance_scale, dir_b,
        save_mono=save_mono,
    )

    # Compute LPAPS between the two sets
    print("Computing LPAPS between set A and set B...")
    device = torch.device("cuda")
    sample_rate = pipe.sample_rate

    source_audios = [a.cpu() for a in audios_a]
    edit_audios = [a.cpu() for a in audios_b]
    srs_a = [sample_rate] * len(source_audios)
    srs_b = [sample_rate] * len(edit_audios)

    lpaps_df = get_lpaps(source_audios, edit_audios, srs_a, srs_b, device)
    mean_lpaps = lpaps_df["lpaps"].mean()
    std_lpaps = lpaps_df["lpaps"].std()
    print(f"LPAPS: mean={mean_lpaps:.4f}, std={std_lpaps:.4f}")

    # Compute FAD between the two sets
    print("Computing FAD between set A and set B...")
    fad_score = calculate_fad(dir_a, dir_b)
    print(f"FAD: {fad_score:.4f}")

    # Save results
    results = {
        "lpaps": {
            "mean": float(mean_lpaps),
            "std": float(std_lpaps),
        },
        "fad": float(fad_score),
        "seed_a": seed_a,
        "seed_b": seed_b,
        "num_prompts": len(test_prompts),
    }
    results_path = os.path.join(save_dir, "baseline_variance_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    lpaps_df.to_csv(os.path.join(save_dir, "lpaps_per_prompt.csv"), index=False)

    print(f"\nResults saved to {results_path}")
    print(f"  LPAPS mean: {mean_lpaps:.4f} (+/- {std_lpaps:.4f})")
    print(f"  FAD:        {fad_score:.4f}")


if __name__ == "__main__":
    Fire(main)
