from typing import Callable, Dict, List, Optional, Union

import torch

from src.steering.methods.sae.lib.hooked_model.utils import locate_block
from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline


def retrieve_acestep(io):
    """
    Retrieve activations for ACE Step.

    Unlike other models that batch conditional/unconditional together,
    ACE Step runs separate forward passes for CFG. This function returns
    activations as-is without chunking.

    Note: This collects activations from ALL CFG passes (conditional,
    unconditional, and optionally text_only). If you need only conditional
    activations, filter during SAE training using metadata about which
    passes correspond to which data.

    Args:
        io: Input/output tensor or tuple
    """
    if isinstance(io, tuple):
        if len(io) == 1:
            io = io[0]
        else:
            # Some layers might return multiple outputs, take the first
            io = io[0]

    if isinstance(io, torch.Tensor):
        # ACE Step runs separate passes, so return as-is
        return io.detach().cpu()
    else:
        raise ValueError("Input/Output must be a tensor or tuple")


class HookedACEStepModel:
    def __init__(
        self,
        pipeline: SimpleACEStepPipeline,
        device: str = "cuda",
    ):
        """
        Initialize a hooked ACE Step model.

        Args:
            pipeline: The ACE Step pipeline
            device: Device to run on
        """
        # Core components
        self.pipeline = pipeline
        self.device = device
        self.model = pipeline.ace_step_transformer

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        audio_duration: float = 10.0,
        lyrics: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        manual_seed: int = 42,
        latents: Optional[torch.Tensor] = None,
        return_type: str = "audio",
        **kwargs,
    ):
        """
        Generate audio using ACE Step pipeline.
        """
        output = self.pipeline.generate(
            prompt=prompt,
            audio_duration=audio_duration,
            lyrics=lyrics,
            infer_step=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            manual_seed=manual_seed,
            latents=latents,
            return_type=return_type,
            use_erg_lyric=False,
            **kwargs,
        )
        return output

    @torch.no_grad()
    def run_with_hooks(
        self,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        prompt: Union[str, List[str]],
        audio_duration: float = 10.0,
        lyrics: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        manual_seed: int = 42,
        latents: Optional[torch.Tensor] = None,
        return_type: str = "latent",
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
            audio_duration: Duration of audio to generate
            lyrics: Optional lyrics
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale factor for classifier-free guidance
            guidance_scale_text: Text guidance scale
            guidance_scale_lyric: Lyric guidance scale
            guidance_interval: Guidance interval
            guidance_interval_decay: Guidance interval decay
            manual_seed: Random seed
            latents: Optional pre-generated latent vectors
            return_type: Type of output to return ('audio', 'latent')
            **kwargs: Additional arguments passed to pipeline
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
            output = self(
                prompt=prompt,
                audio_duration=audio_duration,
                lyrics=lyrics,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_text=guidance_scale_text,
                guidance_scale_lyric=guidance_scale_lyric,
                guidance_interval=guidance_interval,
                guidance_interval_decay=guidance_interval_decay,
                manual_seed=manual_seed,
                latents=latents,
                return_type=return_type,
                **kwargs,
            )
        finally:
            for hook in hooks:
                hook.remove()

        return output

    @torch.no_grad()
    def run_with_cache(
        self,
        positions_to_cache: List[str],
        prompt: Union[str, List[str]],
        audio_duration: float = 10.0,
        lyrics: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        manual_seed: int = 42,
        latents: Optional[torch.Tensor] = None,
        return_type: str = "audio",
        save_input: bool = False,
        save_output: bool = True,
        unconditional: bool = False,
        **kwargs,
    ):
        """
        Run pipeline while caching intermediate values at specified positions.

        Note: Collects activations from ALL CFG passes. ACE Step runs 2-3 separate
        forward passes (conditional, unconditional, optional text_only) per denoising
        step. All passes are collected and stacked along batch dimension.

        Returns both the final audio and a dictionary of cached values.
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

        output = self(
            prompt=prompt,
            audio_duration=audio_duration,
            lyrics=lyrics,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            manual_seed=manual_seed,
            latents=latents,
            return_type=return_type,
            **kwargs,
        )

        # Determine number of CFG passes
        num_cfg_passes = self._compute_num_cfg_passes(
            guidance_scale_text, guidance_scale_lyric
        )

        # Stack cached tensors: CFG passes along batch dimension, timesteps along time dimension
        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = self._stack_cfg_activations(
                    block, num_cfg_passes
                )
            cache_dict["input"] = cache_input

        if save_output:
            for position, block in cache_output.items():
                cache_output[position] = self._stack_cfg_activations(
                    block, num_cfg_passes
                )
            cache_dict["output"] = cache_output

        for hook in hooks:
            hook.remove()

        return output, cache_dict

    def _compute_num_cfg_passes(
        self, guidance_scale_text: float = 0.0, guidance_scale_lyric: float = 0.0
    ) -> int:
        """Compute number of CFG passes based on guidance settings."""
        do_double_guidance = (
            guidance_scale_text is not None
            and guidance_scale_text > 1.0
            and guidance_scale_lyric is not None
            and guidance_scale_lyric > 1.0
        )
        return 3 if do_double_guidance else 2

    def _stack_cfg_activations(
        self, activations: List[torch.Tensor], num_cfg_passes: int
    ) -> torch.Tensor:
        """
        Stack activations properly handling CFG passes.

        Args:
            activations: List of activation tensors from hooks (length = num_timesteps * num_cfg_passes)
            num_cfg_passes: Number of CFG passes (2 or 3)

        Returns:
            Tensor of shape (batch_size * num_cfg_passes, num_timesteps, d_sample_size, d_in)
        """
        num_total_calls = len(activations)
        num_timesteps = num_total_calls // num_cfg_passes

        timestep_groups = []
        for t in range(num_timesteps):
            cfg_passes = activations[t * num_cfg_passes : (t + 1) * num_cfg_passes]
            stacked = torch.cat(
                cfg_passes, dim=0
            )
            timestep_groups.append(stacked)

        result = torch.stack(
            timestep_groups, dim=1
        )
        return result

    def _register_cache_hook(
        self,
        position: str,
        cache_input: Dict,
        cache_output: Dict,
        steer_unconditional: bool = False,
        sae_mode: str = "sequence", # "sequence" or "frequency"
    ):
        """
        Register a hook that caches activations at a specific position.

        Note: This collects from ALL CFG passes. ACE Step runs separate
        forward passes for conditional/unconditional, not batched together.
        """
        block = locate_block(position, self.model)

        def hook(module, input, kwargs, output):
            if cache_input is not None:
                if position not in cache_input:
                    cache_input[position] = []
                input_to_cache = retrieve_acestep(input)
                # Handle different input shapes
                if isinstance(input_to_cache, torch.Tensor):
                    if len(input_to_cache.shape) == 3:
                        # (batch, time, features)
                        cache_input[position].append(input_to_cache)
                    else:
                        raise ValueError(
                            f"Unexpected input shape: {input_to_cache.shape}"
                        )

            if cache_output is not None:
                if position not in cache_output:
                    cache_output[position] = []
                output_to_cache = retrieve_acestep(output)
                # Handle different output shapes
                if isinstance(output_to_cache, torch.Tensor):
                    if len(output_to_cache.shape) == 3:
                        # (batch, time, features)
                        cache_output[position].append(output_to_cache)
                    else:
                        raise ValueError(
                            f"Unexpected output shape: {output_to_cache.shape}"
                        )

        return block.register_forward_hook(hook, with_kwargs=True)

    def _register_general_hook(self, position, hook):
        """Register a general hook at a specific position."""
        block = locate_block(position, self.model)
        return block.register_forward_hook(hook)

    def get_timesteps(
        self,
        num_inference_steps,
        num_train_timesteps,
        timestep_spacing,
        steps_offset,
        device,
    ):
        """Get timesteps for the scheduler (compatibility method)."""
        # ACE Step uses its own scheduling, return a simple range for compatibility
        return torch.arange(num_inference_steps, device=device)
