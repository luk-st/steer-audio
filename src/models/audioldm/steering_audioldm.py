"""
SteeredAudioLDMPipeline: AudioLDM2 wrapper with CAA steering via torch hooks.

Mirrors the structure of SteeredACEStepPipeline but targets AudioLDM2's batched-CFG
UNet by hooking ``attn2`` outputs in the UNet's BasicTransformerBlocks.
"""

from typing import Literal, Optional

import torch
from diffusers.pipelines.audioldm2.pipeline_audioldm2 import AudioLDM2Pipeline

from src.models.audioldm.audioldm_steering import VectorStoreAudioLDM, register_vector_control_audioldm
from src.models.audioldm.constants import AUDIOLDM2_CROSS_ATTENTION_LAYERS

DEFAULT_REPO_ID = "cvssp/audioldm2-large"


class SteeredAudioLDMPipeline:
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

        self.pipe: Optional[AudioLDM2Pipeline] = None
        self.controller: Optional[VectorStoreAudioLDM] = None
        self._hooks: list = []
        self._registered_layers: list[str] = []

    def load(self) -> "SteeredAudioLDMPipeline":
        pipe = AudioLDM2Pipeline.from_pretrained(self.repo_id, torch_dtype=self.dtype)
        pipe = pipe.to(self.device) # type: ignore
        if self.disable_progress_bar:
            pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        return self

    @property
    def sample_rate(self) -> int:
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded; call load() first.")
        return self.pipe.vocoder.config.sampling_rate

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
    ) -> VectorStoreAudioLDM:
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded; call load() first.")

        self.clear_steering_hooks()

        if layers_to_steer is None:
            layers_to_steer = list(AUDIOLDM2_CROSS_ATTENTION_LAYERS)

        self.controller = VectorStoreAudioLDM(
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
        self._hooks = register_vector_control_audioldm(
            self.pipe.unet, self.controller, layers_to_steer, verbose=verbose
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
        num_inference_steps: int = 30,
        guidance_scale: float = 3.5,
        audio_length_in_s: float = 10.0,
        seed: int = 42,
        negative_prompt=None,
        num_waveforms_per_prompt: int = 1,
    ):
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded; call load() first.")

        do_cfg = guidance_scale > 1.0
        if self.controller is not None:
            self.controller.do_cfg = do_cfg
            self.controller.reset()

        generator = torch.Generator(self.pipe.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            audio_length_in_s=audio_length_in_s,
            generator=generator,
            negative_prompt=negative_prompt,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
        )
        return out.audios  # type: ignore
