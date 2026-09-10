"""
SteeredStableAudioPipeline: Stable Audio Open wrapper with CAA steering via hooks.
"""

from typing import Literal, Optional

import torch
from diffusers.pipelines.stable_audio.pipeline_stable_audio import StableAudioPipeline

from src.models.stable_audio.constants import STABLE_AUDIO_CROSS_ATTENTION_LAYERS
from src.models.stable_audio.stable_audio_steering import VectorStoreStableAudio, register_vector_control_stable_audio

DEFAULT_REPO_ID = "stabilityai/stable-audio-open-1.0"


class SteeredStableAudioPipeline:
    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        disable_progress_bar: bool = True,
    ):
        self.repo_id = repo_id
        self.device = device
        self.dtype = dtype
        self.disable_progress_bar = disable_progress_bar

        self.pipe: Optional[StableAudioPipeline] = None
        self.controller: Optional[VectorStoreStableAudio] = None
        self._hooks: list = []
        self._registered_layers: list[str] = []

    def load(self) -> "SteeredStableAudioPipeline":
        pipe = StableAudioPipeline.from_pretrained(self.repo_id, torch_dtype=self.dtype)
        pipe = pipe.to(self.device)  # type: ignore
        pipe.enable_vae_slicing()
        if self.disable_progress_bar:
            pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        return self

    @property
    def sample_rate(self) -> int:
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded; call load() first.")
        return self.pipe.vae.config.sampling_rate

    def setup_steering(
        self,
        steering_vectors: Optional[dict] = None,
        layers_to_steer: Optional[list[str]] = None,
        steer: bool = True,
        alpha: float = 0.0,
        beta: float = 2.0,
        steer_back: bool = False,
        steer_mode: Literal[
            "cond_only",
            "uncond_only",
            "uncond_for_cond",
            "separate",
            "both_cond",
            "both_uncond",
        ] = "cond_only",
        save_only_cond: bool = True,
        normalize_sv_at_apply: bool = False,
        renorm_after_steer: bool = False,
        verbose: bool = False,
    ) -> VectorStoreStableAudio:
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded; call load() first.")

        self.clear_steering_hooks()

        if layers_to_steer is None:
            layers_to_steer = list(STABLE_AUDIO_CROSS_ATTENTION_LAYERS)

        self.controller = VectorStoreStableAudio(
            steering_vectors=steering_vectors,
            steer=steer,
            alpha=alpha,
            beta=beta,
            steer_back=steer_back,
            steer_mode=steer_mode,
            device=self.device,
            save_only_cond=save_only_cond,
            normalize_sv_at_apply=normalize_sv_at_apply,
            renorm_after_steer=renorm_after_steer,
        )
        self._hooks = register_vector_control_stable_audio(
            self.pipe.transformer, self.controller, layers_to_steer, verbose=verbose
        )
        self._registered_layers = list(layers_to_steer)
        return self.controller

    def clear_steering_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def reset_controller(self) -> None:
        if self.controller is not None:
            self.controller.reset()

    def generate(
        self,
        prompt,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        audio_length_in_s: float = 10.0,
        seed: int = 42,
        negative_prompt=None,
        num_waveforms_per_prompt: int = 1,
    ):
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded; call load() first.")

        # for batch dimension slicing
        do_cfg = guidance_scale > 1.0
        if self.controller is not None:
            self.controller.do_cfg = do_cfg
            self.controller.reset()

        generator = torch.Generator(self.pipe.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            audio_end_in_s=audio_length_in_s,
            generator=generator,
            negative_prompt=negative_prompt,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
        )
        return out.audios  # type: ignore
