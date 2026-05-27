from typing import Optional

import torch

from models import PipelineWrapper


def sdedit_forward_noise(model: PipelineWrapper, x0: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """
    Simple SDEdit forward process: directly add noise to x0 to get xt at given timestep.
    This is much simpler than the full inversion process.

    Args:
        model: The pipeline wrapper (StableAudWrapper)
        x0: Clean latent tensor
        timestep: Target timestep tensor

    Returns:
        xt: Noisy latent at timestep t
    """
    # For stable audio, we use sigma-based noise scheduling
    if hasattr(model, "model") and hasattr(model.model, "scheduler") and hasattr(model.model.scheduler, "sigmas"):
        # Stable Audio uses sigma-based scheduling
        sigmas = model.model.scheduler.sigmas
        timesteps = model.model.scheduler.timesteps.to(model.device)

        # Find the sigma corresponding to the timestep
        t_to_idx = {float(v): k for k, v in enumerate(timesteps)}
        if float(timestep) in t_to_idx:
            sigma_idx = t_to_idx[float(timestep)]
            sigma = sigmas[sigma_idx]
        else:
            # If exact timestep not found, find closest
            closest_t = min(timesteps, key=lambda x: abs(float(x) - float(timestep)))
            sigma_idx = t_to_idx[float(closest_t)]
            sigma = sigmas[sigma_idx]

        # Add noise: xt = x0 + noise * sigma
        noise = torch.randn_like(x0)
        xt = x0 + noise * sigma

        return xt
    else:
        # Fallback to alpha-based scheduling for other models
        alpha_bar = model.model.scheduler.alphas_cumprod
        sqrt_alpha_bar = alpha_bar[timestep] ** 0.5
        sqrt_one_minus_alpha_bar = (1 - alpha_bar[timestep]) ** 0.5

        noise = torch.randn_like(x0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

        return xt


def sdedit_denoise(
    model: PipelineWrapper,
    xt: torch.Tensor,
    target_prompts: list[str],
    neg_prompts: list[str],
    cfg_scale: float,
    start_timestep: torch.Tensor,
    num_inference_steps: int,
    duration: Optional[float] = None,
    prog_bar: bool = True,
) -> torch.Tensor:
    """
    SDEdit denoising process: denoise from xt using target prompts.

    Args:
        model: The pipeline wrapper
        xt: Noisy latent to start denoising from
        target_prompts: Target prompts for generation
        neg_prompts: Negative prompts
        cfg_scale: Classifier-free guidance scale
        start_timestep: Timestep to start denoising from
        num_inference_steps: Total number of diffusion steps
        duration: Audio duration for stable audio
        prog_bar: Whether to show progress bar

    Returns:
        x0: Denoised latent
    """
    from tqdm import tqdm

    # Get text embeddings
    text_embeddings_hidden_states, text_embeddings_class_labels, text_embeddings_boolean_prompt_mask = (
        model.encode_text(target_prompts)
    )
    uncond_embeddings_hidden_states, uncond_embeddings_class_labels, uncond_boolean_prompt_mask = model.encode_text(
        neg_prompts, negative=True
    )

    # Setup extra inputs for stable audio
    model.setup_extra_inputs(xt, init_timestep=start_timestep, audio_end_in_s=duration)

    # Get timesteps from start_timestep onwards
    timesteps = model.model.scheduler.timesteps.to(model.device)

    # Find the index of start_timestep
    t_to_idx = {float(v): k for k, v in enumerate(timesteps)}
    if float(start_timestep) in t_to_idx:
        start_idx = t_to_idx[float(start_timestep)]
    else:
        # Find closest timestep
        closest_t = min(timesteps, key=lambda x: abs(float(x) - float(start_timestep)))
        start_idx = t_to_idx[float(closest_t)]

    # Only use timesteps from start_idx onwards (denoising part)
    denoising_timesteps = timesteps[start_idx:]

    current_sample = xt
    op = tqdm(denoising_timesteps) if prog_bar else denoising_timesteps

    with torch.no_grad():
        for t in op:
            # Scale model input
            model_input = model.model.scheduler.scale_model_input(current_sample, t)

            # Unconditional prediction
            uncond_out, _, _ = model.unet_forward(
                model_input,
                timestep=t,
                encoder_hidden_states=uncond_embeddings_hidden_states,
                encoder_attention_mask=uncond_boolean_prompt_mask,
                class_labels=uncond_embeddings_class_labels,
            )

            # Conditional prediction
            cond_out, _, _ = model.unet_forward(
                model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings_hidden_states,
                encoder_attention_mask=text_embeddings_boolean_prompt_mask,
                class_labels=text_embeddings_class_labels,
            )

            # Classifier-free guidance
            noise_pred = uncond_out.sample + cfg_scale * (cond_out.sample - uncond_out.sample)

            # Denoise step
            current_sample = model.model.scheduler.step(noise_pred, t, current_sample).prev_sample

    return current_sample
