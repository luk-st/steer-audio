from typing import Callable, Dict, List, Optional, Union

import torch

from src.steering.methods.sae.lib.hooked_model.utils import locate_block, postprocess_image, retrieve


class HookedDiffusionModel:
    def __init__(
        self,
        model: torch.nn.Module,
        scheduler,
        encode_prompt: Callable,
        get_timesteps: Callable,
        vae: Optional[torch.nn.Module] = None,
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

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        in_channels: int = 4,
        sample_size: int = 64,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):
        self.scheduler.num_inference_steps = num_inference_steps
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        do_classifier_free_guidance = guidance_scale > 1.0

        # Generate text embeddings from prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            None,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Get timesteps for the diffusion process
        timesteps = self.get_timesteps(
            num_inference_steps,
            self.scheduler.config.num_train_timesteps,
            self.scheduler.config.timestep_spacing,
            self.scheduler.config.steps_offset,
            device,
        )

        # Initialize latent vectors
        latents = self._prepare_latents(
            batch_size,
            num_images_per_prompt,
            in_channels,
            sample_size,
            sample_size,
            self.model.dtype,
            device,
            generator,
            latents,
        )

        # Run denoising process
        latents = self._denoise_loop(
            timesteps,
            latents,
            guidance_scale,
            prompt_embeds,
            **kwargs,
        )

        # Convert latents to final image
        image = self._postprocess_latents(latents, output_type, generator)
        return image

    @torch.no_grad()
    def run_with_hooks(
        self,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        in_channels: int = 4,
        sample_size: int = 64,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
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
            image = self(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                in_channels=in_channels,
                sample_size=sample_size,
                generator=generator,
                latents=latents,
                output_type=output_type,
                **kwargs,
            )
        finally:
            for hook in hooks:
                hook.remove()

        return image

    @torch.no_grad()
    def run_with_cache(
        self,
        positions_to_cache: List[str],
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        in_channels: int = 4,
        sample_size: int = 64,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        save_input: bool = False,
        save_output: bool = True,
        unconditional: bool = False,
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
            self._register_cache_hook(
                position, cache_input, cache_output, unconditional
            )
            for position in positions_to_cache
        ]
        hooks = [hook for hook in hooks if hook is not None]

        image = self(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            in_channels=in_channels,
            sample_size=sample_size,
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

        return image, cache_dict

    def _register_cache_hook(
        self,
        position: str,
        cache_input: Dict,
        cache_output: Dict,
        unconditional: bool = False,
    ):
        block = locate_block(position, self.model)

        def hook(module, input, kwargs, output):
            if cache_input is not None:
                if position not in cache_input:
                    cache_input[position] = []
                input_to_cache = retrieve(input, unconditional)
                if len(input_to_cache.shape) == 4:
                    input_to_cache = input_to_cache.view(
                        input_to_cache.shape[0], input_to_cache.shape[1], -1
                    ).permute(0, 2, 1)
                cache_input[position].append(input_to_cache)

            if cache_output is not None:
                if position not in cache_output:
                    cache_output[position] = []
                output_to_cache = retrieve(output, unconditional)
                if len(output_to_cache.shape) == 4:
                    output_to_cache = output_to_cache.view(
                        output_to_cache.shape[0], output_to_cache.shape[1], -1
                    ).permute(0, 2, 1)
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
        **kwargs,
    ):
        for i, t in enumerate(timesteps):
            # Double latents for classifier-free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Get model prediction
            noise_pred = self.model(
                latent_model_input,
                t,
                prompt_embeds,
                **kwargs,
            )[0]

            # Apply classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Update latents using scheduler
            latents = self.scheduler.step(
                noise_pred, t, latents, **kwargs, return_dict=False
            )[0]

        return latents

    def _postprocess_latents(self, latents, output_type, generator):
        if not output_type == "latent" and self.vae is not None:
            latents = latents / self.vae.config.scaling_factor
            if self.vae.config.shift_factor is not None:
                latents = latents + self.vae.config.shift_factor
            image = self.vae.decode(
                latents,
                return_dict=False,
                generator=generator,
            )[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]
        image = postprocess_image(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if output_type == "latent":
            image = image.cpu().numpy()
        return image

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
            latents = torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Incorrect latents shape. Got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        return latents
