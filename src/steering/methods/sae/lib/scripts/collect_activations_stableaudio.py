"""
Collect activations from a diffusion model for a given hookpoint and save them to a file.
"""

import os
import sys

from simple_parsing import parse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from diffusers.models.transformers.stable_audio_transformer import StableAudioDiTModel

from src.steering.methods.sae.lib.hooked_model.hooked_model_stableaudio import HookedStableAudioModel
from src.steering.methods.sae.lib.hooked_model.utils import get_timesteps
from src.steering.methods.sae.lib.sae.cache_activations_runner_stableaudio import CacheActivationsRunner
from src.steering.methods.sae.lib.sae.config import CacheActivationsRunnerConfig

def run():
    config = CacheActivationsRunnerConfig(
        hook_names=[
            "transformer_blocks.11.attn2",
        ],
        dataset_type="csv",
        dataset_name="data/musiccaps_other.csv", # "data/musiccaps_voice.csv" # "data/musiccaps_public.csv" # "data/musiccaps_other.csv"
        dataset_duplicate_rows=4,
        column="caption",
        negative_prompt="Low quality, average quality.",
        model_name="stabilityai/stable-audio-open-1.0",
        num_inference_steps=100,
        audio_length_in_s=10.0,
        num_waveforms_per_prompt=1,
        guidance_scale=7.0,
        cache_every_n_timesteps=10,
        along_freqs=False
    )
    args = parse(config)
    accelerator = Accelerator()

    pipe = DiffusionPipeline.from_pretrained(args.model_name, torch_dtype=args.dtype, use_safetensors=True).to(
        accelerator.device
    )
    model = StableAudioDiTModel.from_pretrained(
        args.model_name,
        subfolder="transformer",
        torch_dtype=args.dtype,
        use_safetensors=True,
    ).to(accelerator.device)
    pipe.scheduler.order = 1
    pipe.transformer = model

    hooked_model = HookedStableAudioModel(
        model=model,
        scheduler=pipe.scheduler,
        encode_prompt=pipe.encode_prompt,
        pipeline=pipe,
        vae=pipe.vae,
        accelerator = accelerator
    )

    CacheActivationsRunner(args, hooked_model, accelerator).run()


if __name__ == "__main__":
    run()
