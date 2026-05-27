"""
Collect activations from a diffusion model for a given hookpoint and save them to a file.
"""

import os
import sys

from simple_parsing import parse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from accelerate import Accelerator
from diffusers import AudioLDM2UNet2DConditionModel, DiffusionPipeline

import src.steering.methods.sae.lib.hooked_model.scheduler
from src.steering.methods.sae.lib.hooked_model.hooked_model_audioldm2 import HookedAudioLDM2Model
from src.steering.methods.sae.lib.hooked_model.utils import get_timesteps
from src.steering.methods.sae.lib.sae.cache_activations_runner import CacheActivationsRunner
from src.steering.methods.sae.lib.sae.config import CacheActivationsRunnerConfig


def run():
    config = CacheActivationsRunnerConfig(
        hook_names=[
            "up_blocks.1.attentions.5.transformer_blocks.0",
            "up_blocks.1.attentions.10.transformer_blocks.0",
        ],
        flatten_act_freq=True,
        arbitrary_F_dims=[4, 4],
        dataset_type="csv",
        dataset_name="data/musiccaps_other.csv",
        dataset_duplicate_rows=2,
        negative_prompt="Low quality, average quality.",
        model_name="cvssp/audioldm2-large",
        num_inference_steps=200,
        audio_length_in_s=9.0,
        num_waveforms_per_prompt=1,
        guidance_scale=5.0,
        cache_every_n_timesteps=10,
    )
    args = parse(config)
    accelerator = Accelerator()

    pipe = DiffusionPipeline.from_pretrained(args.model_name, torch_dtype=args.dtype, use_safetensors=True).to(
        accelerator.device
    )
    model = AudioLDM2UNet2DConditionModel.from_pretrained(
        args.model_name,
        subfolder="unet",
        torch_dtype=args.dtype,
        use_safetensors=True,
    )
    scheduler = src.hooked_model.scheduler.DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    pipe.unet = model

    hooked_model = HookedAudioLDM2Model(
        model=model,
        scheduler=scheduler,
        encode_prompt=pipe.encode_prompt,
        get_timesteps=get_timesteps,
        pipeline=pipe,
        vae=pipe.vae,
    )

    CacheActivationsRunner(args, hooked_model, accelerator).run()


if __name__ == "__main__":
    run()
