"""
Collect activations from a diffusion model for a given hookpoint and save them to a file.
"""

import os
import sys

from simple_parsing import parse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from accelerate import Accelerator
from diffusers import DiffusionPipeline

from src.steering.methods.sae.lib.hooked_model.hooked_model_sd3 import HookedDiffusionModel
from src.steering.methods.sae.lib.sae.cache_activations_runner_sd3 import CacheActivationsRunner
from src.steering.methods.sae.lib.sae.config import CacheActivationsRunnerConfig


def run():
    args = parse(CacheActivationsRunnerConfig)
    accelerator = Accelerator()
    # define model
    pipe = DiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=args.dtype,
        text_encoder_3=None,
        tokenizer_3=None,
        vae=None,
    ).to(accelerator.device)
    model = pipe.transformer
    scheduler = pipe.scheduler
    hooked_model = HookedDiffusionModel(
        model=model,
        scheduler=scheduler,
        encode_prompt=pipe.encode_prompt,
        vae=pipe.vae,
    )

    CacheActivationsRunner(args, hooked_model, accelerator).run()


if __name__ == "__main__":
    run()
