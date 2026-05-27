from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import AudioLDM2Pipeline
import einops

from src.steering.methods.sae.lib.hooked_model.utils import locate_block, retrieve


class HookedAudioLDM2Model:
    def __init__(
        self,
        model: torch.nn.Module,
        scheduler,
        encode_prompt: Callable,
        get_timesteps: Callable,
        pipeline: AudioLDM2Pipeline,
        vae: torch.nn.Module,
    ):
        """
        Initialize a hooked diffusion model.

        Args:
            model (torch.nn.Module): The base diffusion model (UNet or Transformer)
            scheduler: The noise scheduler
            encode_prompt (Callable): Function to encode text prompts into embeddings
            get_timesteps (Callable): Function to generate timesteps for inference
            vae (torch.nn.Module, optional): The VAE model for latent encoding/decoding
        """
        # Core components
        self.model = model
        self.scheduler = scheduler
        self.vae = vae
        self.encode_prompt = encode_prompt
        self.get_timesteps = get_timesteps
        self.pipeline = pipeline

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 200,
        num_waveforms_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        **kwargs,
    ):
        vocoder_upsample_factor = (
            np.prod(self.pipeline.vocoder.config.upsample_rates) / self.pipeline.vocoder.config.sampling_rate
        )
        if audio_length_in_s is None:
            audio_length_in_s = (
                self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor * vocoder_upsample_factor
            )

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.pipeline.vocoder.config.sampling_rate)
        if height % self.pipeline.vae_scale_factor != 0:
            height = int(np.ceil(height / self.pipeline.vae_scale_factor)) * self.pipeline.vae_scale_factor

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        if negative_prompt is not None:
            if isinstance(negative_prompt, str) and batch_size > 1:
                negative_prompt = [negative_prompt] * batch_size
            elif isinstance(negative_prompt, list) and len(negative_prompt) != batch_size:
                raise ValueError(
                    f"The number of negative prompts ({len(negative_prompt)}) must be the same as the number of prompts ({batch_size})"
                )

        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, attention_mask, generated_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            transcription=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            generated_prompt_embeds=None,
            negative_generated_prompt_embeds=None,
            attention_mask=None,
            negative_attention_mask=None,
            max_new_tokens=None,
        )

        # Initialize latent vectors
        num_channels_latents = self.pipeline.unet.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(generator, eta=0.0)
        # Run denoising process
        latents = self._denoise_loop(
            timesteps,
            latents,
            guidance_scale,
            prompt_embeds,
            attention_mask,
            generated_prompt_embeds,
            **extra_step_kwargs,
        )
        audio = self._postprocess_latents(latents, output_type, original_waveform_length)
        return audio

    @torch.no_grad()
    def run_with_hooks(
        self,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 200,
        num_waveforms_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 3.5,
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
                audio_length_in_s=audio_length_in_s,
                num_inference_steps=num_inference_steps,
                num_waveforms_per_prompt=num_waveforms_per_prompt,
                device=device,
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
        negative_prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 200,
        num_waveforms_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        save_input: bool = False,
        save_output: bool = True,
        unconditional: bool = False,
        flatten_act_freq: bool = False,
        arbitrary_F_dims: List[int] = None,
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

        if arbitrary_F_dims is not None:
            assert len(arbitrary_F_dims) == len(positions_to_cache)
        else:
            arbitrary_F_dims = [None] * len(positions_to_cache)

        hooks = [
            self._register_cache_hook(position, cache_input, cache_output, unconditional, flatten_act_freq, F_dim)
            for position, F_dim in zip(positions_to_cache, arbitrary_F_dims)
        ]
        hooks = [hook for hook in hooks if hook is not None]

        audio = self(
            prompt=prompt,
            negative_prompt=negative_prompt,
            audio_length_in_s=audio_length_in_s,
            num_inference_steps=num_inference_steps,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            device=device,
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
        flatten_act_freq: bool = False,
        freq_dim: int | None = None,
    ):
        block = locate_block(position, self.model)

        def hook(module, input, kwargs, output):
            if cache_input is not None:
                if position not in cache_input:
                    cache_input[position] = []
                input_to_cache = retrieve(input, unconditional)
                if len(input_to_cache.shape) == 4:
                    if flatten_act_freq:
                        input_to_cache = einops.rearrange(input_to_cache, 'b c t f -> b t (c f)')
                    else:
                        input_to_cache = einops.rearrange(input_to_cache, 'b c t f -> b (t f) c')
                elif len(input_to_cache.shape) == 3 and freq_dim is not None and flatten_act_freq is True:
                    input_to_cache = einops.rearrange(input_to_cache, 'b (t f) c -> b t (c f)', f=freq_dim)
                cache_input[position].append(input_to_cache)

            if cache_output is not None:
                if position not in cache_output:
                    cache_output[position] = []
                output_to_cache = retrieve(output, unconditional)
                if len(output_to_cache.shape) == 4:
                    if flatten_act_freq:
                        output_to_cache = einops.rearrange(output_to_cache, 'b c t f -> b t (c f)')
                    else:
                        output_to_cache = einops.rearrange(output_to_cache, 'b c t f -> b (t f) c')
                elif len(output_to_cache.shape) == 3 and freq_dim is not None and flatten_act_freq is True:
                    output_to_cache = einops.rearrange(output_to_cache, 'b (t f) c -> b t (c f)', f=freq_dim)
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
        prompt_embeds,
        attention_mask,
        generated_prompt_embeds,
        **kwargs,
    ):
        for i, t in enumerate(timesteps):
            # Double latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Get model prediction
            noise_pred = self.model(
                latent_model_input,
                t,
                encoder_hidden_states=generated_prompt_embeds,
                encoder_hidden_states_1=prompt_embeds,
                encoder_attention_mask_1=attention_mask,
                return_dict=False,
            )[0]

            # Apply classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Update latents using scheduler
            latents = self.scheduler.step(noise_pred, t, latents, **kwargs).prev_sample

        return latents

    def _postprocess_latents(self, latents, output_type, original_waveform_length):
        if not output_type == "latent":
            latents = latents / self.vae.config.scaling_factor
            mel_spectrogram = self.vae.decode(latents, return_dict=False)[0]
            audio = self.pipeline.mel_spectrogram_to_waveform(mel_spectrogram)
            audio = audio[:, :original_waveform_length]
        else:
            audio = latents

        audio = audio.cpu()
        if output_type == "np":
            audio = audio.numpy()
        return audio

    def _prepare_latents(
        self,
        batch_size,
        num_images_per_prompt,
        in_channels,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (batch_size * num_images_per_prompt, in_channels, height, width)
        if latents is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Incorrect latents shape. Got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        return latents
