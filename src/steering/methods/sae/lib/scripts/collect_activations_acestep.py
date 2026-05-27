"""
Collect activations from ACE Step model for a given hookpoint and save them to a file.

This script collects activations from the ACE Step transformer model during audio generation.
It's inspired by the steering vector computation approach used in steering/caa/compute_steering_vectors.py
but adapted for SAE training purposes.

Usage:
    python collect_activations_acestep.py
"""

import os
import sys

from simple_parsing import parse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..",".."))
from accelerate import Accelerator

from src.steering.methods.sae.lib.hooked_model.hooked_model_acestep import HookedACEStepModel
from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline
from src.steering.methods.sae.lib.sae.cache_activations_runner_ace import CacheActivationsRunner
from src.steering.methods.sae.lib.sae.config import CacheActivationsRunnerConfig


def run():
    """
    Main function to collect activations from ACE Step model.

    This sets up the ACE Step pipeline, wraps it with a hooked model,
    and runs the activation caching process.
    """
    config = CacheActivationsRunnerConfig(
        hook_names=[
            "transformer_blocks.6.cross_attn",
            "transformer_blocks.7.cross_attn",
        ],
        dataset_type="csv",
        dataset_name="../data/music_caps_slow32.csv",
        dataset_duplicate_rows=1,
        column="caption",
        negative_prompt="",
        model_name="ace-step",
        num_inference_steps=30,
        audio_length_in_s=10.0,
        num_waveforms_per_prompt=1,
        guidance_scale=5.0,
        cache_every_n_timesteps=6,
        batch_size=16,
        max_num_examples=None,
        seed=42,
        ace_step_guidance_interval=1.0,
        ace_step_guidance_interval_decay=0.0,
        ace_step_guidance_scale_text=0.0,
        ace_step_guidance_scale_lyric=0.0,
    )

    args = parse(config)
    accelerator = Accelerator()

    print("Initializing ACE Step pipeline...")
    pipeline = SimpleACEStepPipeline(
        device=accelerator.device,
        dtype="bfloat16",  # ACE Step uses bfloat16
        persistent_storage_path="../res/ace_step",
    )
    pipeline.load()

    print("Creating hooked model...")
    hooked_model = HookedACEStepModel(
        pipeline=pipeline,
        device=str(accelerator.device),
    )

    print("Starting activation collection...")
    CacheActivationsRunner(args, hooked_model, accelerator).run()
    print("Activation collection complete!")


if __name__ == "__main__":
    run()
