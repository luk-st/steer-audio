from typing import Callable, Dict, List, Optional, Union

import einops
import numpy as np
import torch
from diffusers import StableAudioPipeline
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.utils import is_torch_xla_available
from torch import Value

from src.steering.methods.sae.lib.hooked_model.utils import locate_block, retrieve
from accelerate import Accelerator

class HookedStableAudioModel:
    def __init__(
        self,
        model: torch.nn.Module,
        scheduler,
        encode_prompt: Callable,
        pipeline: StableAudioPipeline,
        vae: torch.nn.Module,
        accelerator: Accelerator,
    ):
        """
        Initialize a hooked diffusion model.

        Args:
            model (torch.nn.Module): The base diffusion model (UNet or Transformer)
            scheduler: The noise scheduler
            encode_prompt (Callable): Function to encode text prompts into embeddings
            vae (torch.nn.Module, optional): The VAE model for latent encoding/decoding
        """
        # Core components
        self.model = model
        self.scheduler = scheduler
        self.vae = vae
        self.encode_prompt = encode_prompt
        self.pipeline = pipeline
        self.accelerator = accelerator

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        audio_end_in_s: Optional[float] = None,
        audio_start_in_s: Optional[float] = 0.0,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        initial_audio_waveforms: Optional[torch.Tensor] = None,
        initial_audio_sampling_rate: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        output_type: Optional[str] = "pt",
        **kwargs,
    ):
        downsample_ratio = self.vae.hop_length

        max_audio_length_in_s = self.model.config.sample_size * downsample_ratio / self.vae.config.sampling_rate
        if audio_end_in_s is None:
            audio_end_in_s = max_audio_length_in_s
        waveform_start = int(audio_start_in_s * self.vae.config.sampling_rate)
        waveform_end = int(audio_end_in_s * self.vae.config.sampling_rate)
        waveform_length = int(self.model.config.sample_size)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if negative_prompt is not None:
            if isinstance(negative_prompt, str) and batch_size > 1:
                negative_prompt = [negative_prompt] * batch_size
            elif isinstance(negative_prompt, list) and len(negative_prompt) != batch_size:
                raise ValueError(
                    f"The number of negative prompts ({len(negative_prompt)}) must be the same as the number of prompts ({batch_size})"
                )

        device = self.accelerator.device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self.pipeline.encode_prompt(
            prompt,
            device,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )
        seconds_start_hidden_states, seconds_end_hidden_states = self.pipeline.encode_duration(
            audio_start_in_s,
            audio_end_in_s,
            device,
            do_classifier_free_guidance and (negative_prompt is not None or negative_prompt_embeds is not None),
            batch_size,
        )
        text_audio_duration_embeds = torch.cat(
            [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
        )
        audio_duration_embeds = torch.cat([seconds_start_hidden_states, seconds_end_hidden_states], dim=2)

        if do_classifier_free_guidance and negative_prompt_embeds is None and negative_prompt is None:
            negative_text_audio_duration_embeds = torch.zeros_like(
                text_audio_duration_embeds, device=text_audio_duration_embeds.device
            )
            text_audio_duration_embeds = torch.cat(
                [negative_text_audio_duration_embeds, text_audio_duration_embeds], dim=0
            )
            audio_duration_embeds = torch.cat([audio_duration_embeds, audio_duration_embeds], dim=0)
        bs_embed, seq_len, hidden_size = text_audio_duration_embeds.shape
        text_audio_duration_embeds = text_audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
        text_audio_duration_embeds = text_audio_duration_embeds.view(
            bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
        )
        audio_duration_embeds = audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
        audio_duration_embeds = audio_duration_embeds.view(
            bs_embed * num_waveforms_per_prompt, -1, audio_duration_embeds.shape[-1]
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_vae = self.model.config.in_channels
        latents = self._prepare_latents(
            batch_size=batch_size * num_waveforms_per_prompt,
            num_channels_vae=num_channels_vae,
            sample_size=waveform_length,
            dtype=text_audio_duration_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
            initial_audio_waveforms=initial_audio_waveforms,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            audio_channels=self.vae.config.audio_channels,
        )

        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(generator, eta)
        rotary_embedding = get_1d_rotary_pos_embed(
            self.pipeline.rotary_embed_dim,
            latents.shape[2] + audio_duration_embeds.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )
        latents = self._denoise_loop(
            timesteps=timesteps,
            latents=latents,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            do_classifier_free_guidance=do_classifier_free_guidance,
            text_audio_duration_embeds=text_audio_duration_embeds,
            audio_duration_embeds=audio_duration_embeds,
            rotary_embedding=rotary_embedding,
            callback=callback,
            callback_steps=callback_steps,
            **extra_step_kwargs,
        )
        audio = self._postprocess_latents(
            latents=latents,
            output_type=output_type,
            waveform_start=waveform_start,
            waveform_end=waveform_end,
        )
        return audio

    @torch.no_grad()
    def run_with_hooks(
        self,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        **kwargs,
    ):
        """
        Run the pipeline with hooks at specified positions.

        Args:
            position_hook_dict: Dictionary mapping model positions to hooks.
                Keys: Position strings indicating where to register hooks
                Values: Single hook function or list of hook functions
                Each hook should accept (module, input, output) arguments
            prompt: Text prompt(s) to condition the model
            num_images_per_prompt: Number of images to generate per prompt
            device: Device to run inference on
            guidance_scale: Scale factor for classifier-free guidance
            num_inference_steps: Number of denoising steps
            in_channels: Number of input channels for latents
            sample_size: Size of generated image
            generator: Random number generator
            latents: Optional pre-generated latent vectors
            output_type: Type of output to return ('pil', 'latent', etc)
            **kwargs: Additional arguments passed to base pipeline
        """
        hooks = []
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]

        try:
            audio = self(
                prompt=prompt,
                negative_prompt=negative_prompt,
                audio_end_in_s=audio_length_in_s,
                num_inference_steps=num_inference_steps,
                num_waveforms_per_prompt=num_waveforms_per_prompt,
                guidance_scale=guidance_scale,
                generator=generator,
                latents=latents,
                output_type=output_type,
                **kwargs,
            )
        finally:
            for hook in hooks:
                hook.remove()

        return audio

    @torch.no_grad()
    def run_with_cache(
        self,
        positions_to_cache: List[str],
        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        save_input: bool = False,
        save_output: bool = True,
        unconditional: bool = False,
        along_freqs:bool = False,
        **kwargs,
    ):
        """
        Run pipeline while caching intermediate values at specified positions.
        Compatible with both UNet and Transformer-based models.

        Returns both the final image and a dictionary of cached values.
        """
        cache_input, cache_output = (
            dict() if save_input else None,
            dict() if save_output else None,
        )

        hooks = [
            self._register_cache_hook(position, cache_input, cache_output, unconditional, along_freqs)
            for position in positions_to_cache
        ]
        hooks = [hook for hook in hooks if hook is not None]

        audio = self(
            prompt=prompt,
            negative_prompt=negative_prompt,
            audio_end_in_s=audio_length_in_s,
            num_inference_steps=num_inference_steps,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,
            output_type=output_type,
            **kwargs,
        )

        # Stack cached tensors along time dimension
        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict["input"] = cache_input

        if save_output:
            for position, block in cache_output.items():
                cache_output[position] = torch.stack(block, dim=1)
            cache_dict["output"] = cache_output

        for hook in hooks:
            hook.remove()

        return audio, cache_dict

    def _register_cache_hook(
        self,
        position: str,
        cache_input: Dict,
        cache_output: Dict,
        unconditional: bool = False,
        along_freqs:bool = False,
    ):
        block = locate_block(position, self.model)

        def hook(module, input, kwargs, output):
            if cache_input is not None:
                if position not in cache_input:
                    cache_input[position] = []
                input_to_cache = retrieve(input, unconditional)
                if len(input_to_cache.shape) == 3:
                    if along_freqs:
                        input_to_cache = einops.rearrange(input_to_cache, "b t f -> b f t")
                else:
                    raise ValueError(f"Input to cache has shape {input_to_cache.shape}, expected 3")
                cache_input[position].append(input_to_cache)

            if cache_output is not None:
                if position not in cache_output:
                    cache_output[position] = []
                output_to_cache = retrieve(output, unconditional)
                if len(output_to_cache.shape) == 3:
                    if along_freqs:
                        output_to_cache = einops.rearrange(output_to_cache, "b t f -> b f t")
                else:
                    raise ValueError(f"Output to cache has shape {output_to_cache.shape}, expected 3")
                cache_output[position].append(output_to_cache)

        return block.register_forward_hook(hook, with_kwargs=True)

    def _register_general_hook(self, position, hook):
        block = locate_block(position, self.model)
        return block.register_forward_hook(hook)

    def _denoise_loop(
        self,
        timesteps,
        latents,
        guidance_scale,
        num_inference_steps,
        do_classifier_free_guidance,
        text_audio_duration_embeds,
        audio_duration_embeds,
        rotary_embedding,
        callback,
        callback_steps,
        **kwargs,
    ):
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.model(
                    latent_model_input,
                    t.unsqueeze(0),
                    encoder_hidden_states=text_audio_duration_embeds,
                    global_hidden_states=audio_duration_embeds,
                    rotary_embedding=rotary_embedding,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        return latents

    def _postprocess_latents(self, latents, output_type, waveform_start, waveform_end):
        if not output_type == "latent":
            audio = self.vae.decode(latents).sample
            audio = audio[:, :, waveform_start:waveform_end]
        else:
            audio = latents

        audio = audio.cpu()
        if output_type == "np":
            audio = audio.float().numpy()
        return audio

    def _prepare_latents(
        self,
        batch_size,
        num_channels_vae,
        sample_size,
        dtype,
        device,
        generator,
        latents=None,
        initial_audio_waveforms=None,
        num_waveforms_per_prompt=None,
        audio_channels=None,
    ):
        return self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_vae=num_channels_vae,
            sample_size=sample_size,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
            initial_audio_waveforms=initial_audio_waveforms,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            audio_channels=audio_channels,
        )
