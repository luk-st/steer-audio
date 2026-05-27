import argparse
import calendar
import os
import time
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torchaudio
from ddm_inversion.ddim_inversion import ddim_inversion, text2image_ldm_stable
from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from sdedit_utils import sdedit_denoise, sdedit_forward_noise
from torch import inference_mode

from models import load_model, load_model_inversion
from utils import get_spec, load_audio, set_reproducability

HF_TOKEN = os.getenv("HF_TOKEN")  # Required for stable audio open
ENABLE_TQDM = False


def create_stable_audio_hook(target_encoder_hidden_states, hook_counter=None):
    """
    Create a hook function that replaces encoder_hidden_states with target prompt encoding
    for specific transformer layers, while preserving null/unconditional predictions for CFG.

    Args:
        target_encoder_hidden_states: The target prompt embeddings to use
        hook_counter: Dict to track hook applications (modified in-place)
    """

    def cross_attn_hook(module, args, kwargs):
        # Replace encoder_hidden_states with target prompt encoding
        if "encoder_hidden_states" in kwargs:
            current_hidden_states = kwargs["encoder_hidden_states"]

            if current_hidden_states is not None:
                is_all_zeros = torch.all(current_hidden_states == 0.0).item()

                if hook_counter is not None:
                    hook_counter["total_calls"] += 1
                    if is_all_zeros:
                        hook_counter["unconditional_preserved"] += 1
                    else:
                        hook_counter["conditional_replaced"] += 1

                # Only replace if it's NOT all zeros (i.e., it's a conditional prediction)
                # This preserves unconditional predictions in CFG
                if not is_all_zeros:
                    kwargs["encoder_hidden_states"] = target_encoder_hidden_states
                # Otherwise, keep the original zero embeddings for unconditional CFG

        return args, kwargs

    return cross_attn_hook


def apply_stable_audio_hooks(model, layers_to_hook, target_encoder_hidden_states, verbose=False):
    """
    Apply hooks to specific transformer layers in Stable Audio model.

    Args:
        model: The Stable Audio model
        layers_to_hook: List of layer names to hook (e.g., ["transformer.blocks.5", "transformer.blocks.10"])
        target_encoder_hidden_states: Target prompt encoding to use in hooked layers
        verbose: Whether to print hook application messages

    Returns:
        Tuple of (hook_handles, hook_counter) for cleanup and statistics
    """
    if not layers_to_hook or len(layers_to_hook) == 0:
        return [], {}

    # Initialize counter for tracking hook applications
    hook_counter = {"total_calls": 0, "conditional_replaced": 0, "unconditional_preserved": 0}

    hook_handles = []
    hook_function = create_stable_audio_hook(target_encoder_hidden_states, hook_counter)

    for layer_name in layers_to_hook:
        try:
            # Navigate to the specific layer in the model
            layer_parts = layer_name.split(".")
            target_layer = model

            for part in layer_parts:
                if part.isdigit():
                    target_layer = target_layer[int(part)]
                else:
                    target_layer = getattr(target_layer, part)

            # Apply hook to the layer - this assumes the layer has a forward method that accepts encoder_hidden_states
            hook_handle = target_layer.register_forward_pre_hook(hook_function, with_kwargs=True)
            hook_handles.append(hook_handle)

            if verbose:
                print(f"Applied hook to layer: {layer_name}")

        except (AttributeError, IndexError, TypeError) as e:
            print(f"Warning: Could not apply hook to layer {layer_name}: {e}")

    return hook_handles, hook_counter


def run_stable_audio_edit(
    init_aud: str,
    cfg_src: list[float],
    cfg_tar: list[float],
    num_diffusion_steps: int,
    target_prompt: list[str],
    source_prompt: list[str],
    target_neg_prompt: list[str],
    tstart: list[int],
    mode: str,
    results_path: str | None = None,
    fix_alpha: float = 0.1,
    cutoff_points: list[float] | None = None,
    verbose: bool = False,
    test_rand_gen: bool = False,
    numerical_fix: bool = True,
    save_edit_wav_path: Optional[str] = None,
    eta: float = 1.0,
    layers_to_hook: list[str] | None = None,
):
    assert mode in ["ddpm", "ddim", "sdedit"], "Invalid mode, must be one of ['ddpm', 'ddim', 'sdedit']"
    assert (
        results_path is not None or save_edit_wav_path is not None
    ), "Either results_path or save_edit_wav_path must be provided"

    model_id = "stabilityai/stable-audio-open-1.0"
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN is required for stable audio model")
    device = f"cuda:0"
    torch.cuda.set_device(0)

    cfg_scale_src = cfg_src
    cfg_scale_tar = cfg_tar

    current_GMT = time.gmtime()
    time_stamp_name = calendar.timegm(current_GMT)
    if mode == "ddpm":
        image_name_png = (
            f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_'
            + f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_'
            + f"skip_{int(num_diffusion_steps) - int(tstart[0])}_{time_stamp_name}"
        )
    else:
        if tstart != num_diffusion_steps:
            image_name_png = (
                f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_'
                + f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_'
                + f"skip_{int(num_diffusion_steps) - int(tstart[0])}_{time_stamp_name}"
            )
        else:
            image_name_png = (
                f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_'
                + f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_'
                + f"{num_diffusion_steps}timesteps_{time_stamp_name}"
            )

    if len(tstart) != len(target_prompt):
        if len(tstart) == 1:
            tstart *= len(target_prompt)
        else:
            raise ValueError("T-start amount and target prompt amount don't match.")
    tstart = torch.tensor(tstart, dtype=torch.int)
    skip = num_diffusion_steps - tstart

    # Load stable audio model
    ldm_stable = load_model(model_id, device, num_diffusion_steps, token=HF_TOKEN, edit_method=mode)

    # For DDIM with StableAudio, use DPMSolverMultistepScheduler for denoising and DPMSolverMultistepInverseScheduler for inversion
    if mode == "ddim":
        ldm_stable_inverse = load_model_inversion(
            model_id, device, num_diffusion_steps, token=HF_TOKEN, edit_method=mode
        )
    else:
        ldm_stable_inverse = ldm_stable

    # Load audio (stable-audio doesn't use STFT)
    x0, sr, duration = load_audio(
        init_aud, ldm_stable.get_fn_STFT(), device=device, stft=False, model_sr=ldm_stable.get_sr()
    )
    max_audio_length_in_s = (
        ldm_stable.model.transformer.config.sample_size * ldm_stable.model.vae.hop_length / ldm_stable.model.vae.config.sampling_rate
    )
    if duration > max_audio_length_in_s:
        duration = max_audio_length_in_s
        x0 = x0[..., :int(max_audio_length_in_s * sr)]
        print(f"Warning: Audio {init_aud} is too long and will be truncated to {max_audio_length_in_s} seconds...")

    torch.cuda.empty_cache()

    with inference_mode():
        w0 = ldm_stable.vae_encode(x0)

        # Choose processing method
        if mode == "sdedit":
            # SDEdit: Simple forward noise + denoising
            if verbose:
                print(f"Running SDEdit with {(tstart[0] / num_diffusion_steps) * 100:.1f}% noise level")

            # Calculate target timestep for SDEdit
            timesteps = ldm_stable.model.scheduler.timesteps.to(device)
            target_timestep_idx = len(timesteps) - tstart[0]
            if target_timestep_idx < 0:
                target_timestep_idx = 0
            elif target_timestep_idx >= len(timesteps):
                target_timestep_idx = len(timesteps) - 1
            target_timestep = timesteps[target_timestep_idx]

            # Forward noise: x0 -> xt
            xt = sdedit_forward_noise(ldm_stable, w0, target_timestep)

            hook_handles = []
            hook_counter = {}
            if layers_to_hook and len(layers_to_hook) > 0:
                # Get target text embeddings for hooking
                target_text_embeddings_hidden_states, _, _ = ldm_stable.encode_text(target_prompt)

                # Apply hooks to specified layers
                hook_handles, hook_counter = apply_stable_audio_hooks(
                    ldm_stable.model.transformer, layers_to_hook, target_text_embeddings_hidden_states, verbose=verbose
                )

                if verbose and hook_handles:
                    print(f"Applied {len(hook_handles)} hooks for SDEdit denoising")

            try:
                # Denoise: xt -> x0
                denoising_prompt = source_prompt if (layers_to_hook and len(layers_to_hook) > 0) else target_prompt
                w0 = sdedit_denoise(
                    model=ldm_stable,
                    xt=xt,
                    target_prompts=denoising_prompt,
                    neg_prompts=target_neg_prompt,
                    cfg_scale=cfg_scale_tar[0] if isinstance(cfg_scale_tar, list) else cfg_scale_tar,
                    start_timestep=target_timestep,
                    num_inference_steps=num_diffusion_steps,
                    duration=duration,
                    prog_bar=verbose,
                )
            finally:
                for handle in hook_handles:
                    handle.remove()
                if verbose and hook_handles:
                    print(f"Removed {len(hook_handles)} hooks after SDEdit denoising")
                    if hook_counter:
                        print(
                            f"Hook statistics - Total calls: {hook_counter['total_calls']}, "
                            f"Conditional replaced: {hook_counter['conditional_replaced']}, "
                            f"Unconditional preserved: {hook_counter['unconditional_preserved']}"
                        )
        elif mode == "ddim":
            # DDIM inversion
            if len(cfg_scale_src) > 1:
                raise ValueError("DDIM only supports one cfg_scale_src value")
            wT = ddim_inversion(
                ldm_stable_inverse,
                w0,
                source_prompt,
                cfg_scale_src[0],
                num_inference_steps=num_diffusion_steps,
                skip=skip[0],
                duration=duration,
            )

            # if skip != 0:
            #     warnings.warn(
            #         "Plain DDIM Inversion should be run with t_start == num_diffusion_steps. "
            #         "You are now running partial DDIM inversion.",
            #         RuntimeWarning,
            #     )
            if len(cfg_scale_tar) > 1:
                raise ValueError("DDIM only supports one cfg_scale_tar value")
            if len(source_prompt) > 1:
                raise ValueError("DDIM only supports one source_prompt value")
            if len(target_prompt) > 1:
                raise ValueError("DDIM only supports one target_prompt value")

            hook_handles = []
            hook_counter = {}
            if layers_to_hook and len(layers_to_hook) > 0:
                target_text_embeddings_hidden_states, _, _ = ldm_stable.encode_text(target_prompt)

                hook_handles, hook_counter = apply_stable_audio_hooks(
                    ldm_stable.model.transformer, layers_to_hook, target_text_embeddings_hidden_states, verbose=verbose
                )

                if verbose and hook_handles:
                    print(f"Applied {len(hook_handles)} hooks for DDIM denoising")

            try:
                denoising_prompt = source_prompt if (layers_to_hook and len(layers_to_hook) > 0) else target_prompt
                w0 = text2image_ldm_stable(
                    ldm_stable,
                    denoising_prompt,
                    num_diffusion_steps,
                    cfg_scale_tar[0],
                    wT,
                    skip=skip,
                    duration=duration,
                )
            finally:
                for handle in hook_handles:
                    handle.remove()
                if verbose and hook_handles:
                    print(f"Removed {len(hook_handles)} hooks after DDIM denoising")
                    if hook_counter:
                        print(
                            f"Hook statistics - Total calls: {hook_counter['total_calls']}, "
                            f"Conditional replaced: {hook_counter['conditional_replaced']}, "
                            f"Unconditional preserved: {hook_counter['unconditional_preserved']}"
                        )

        else:  # ddpm
            wt, zs, wts, extra_info = inversion_forward_process(
                ldm_stable,
                w0,
                etas=eta,
                prompts=source_prompt,
                cfg_scales=cfg_scale_src,
                prog_bar=ENABLE_TQDM,
                num_inference_steps=num_diffusion_steps,
                cutoff_points=cutoff_points,
                numerical_fix=numerical_fix,
                duration=duration,
            )

            hook_handles = []
            hook_counter = {}
            if layers_to_hook and len(layers_to_hook) > 0:
                target_text_embeddings_hidden_states, _, _ = ldm_stable.encode_text(target_prompt)

                hook_handles, hook_counter = apply_stable_audio_hooks(
                    ldm_stable.model.transformer, layers_to_hook, target_text_embeddings_hidden_states, verbose=verbose
                )

                if verbose and hook_handles:
                    print(f"Applied {len(hook_handles)} hooks for DDPM reverse process")

            try:
                denoising_prompt = source_prompt if (layers_to_hook and len(layers_to_hook) > 0) else target_prompt
                w0, _ = inversion_reverse_process(
                    ldm_stable,
                    xT=wts if not test_rand_gen else torch.randn_like(wts),
                    tstart=tstart,
                    fix_alpha=fix_alpha,
                    etas=eta,
                    prompts=denoising_prompt,
                    neg_prompts=target_neg_prompt,
                    cfg_scales=cfg_scale_tar,
                    prog_bar=ENABLE_TQDM,
                    zs=(
                        zs[: int(num_diffusion_steps - min(skip))]
                        if not test_rand_gen
                        else torch.randn_like(zs[: int(num_diffusion_steps - min(skip))])
                    ),
                    cutoff_points=cutoff_points,
                    duration=duration,
                    extra_info=extra_info,
                )
            finally:
                for handle in hook_handles:
                    handle.remove()
                if verbose and hook_handles:
                    print(f"Removed {len(hook_handles)} hooks after DDPM reverse process")
                    if hook_counter:
                        print(
                            f"Hook statistics - Total calls: {hook_counter['total_calls']}, "
                            f"Conditional replaced: {hook_counter['conditional_replaced']}, "
                            f"Unconditional preserved: {hook_counter['unconditional_preserved']}"
                        )

        # Create save path based on mode
        if results_path is not None:
            if mode == "sdedit":
                save_path = os.path.join(
                    f"./{results_path}/",
                    "stable-audio-open-1.0",
                    os.path.basename(init_aud).split(".")[0],
                    f"sdedit_pmt_{target_prompt[0].replace(' ', '_')}_neg_{target_neg_prompt[0].replace(' ', '_')}",
                )
            else:
                save_path = os.path.join(
                    f"./{results_path}/",
                    "stable-audio-open-1.0",
                    os.path.basename(init_aud).split(".")[0],
                    "src_" + "__".join([x.replace(" ", "_") for x in source_prompt]),
                    "dec_"
                    + "__".join([x.replace(" ", "_") for x in target_prompt])
                    + "__neg__"
                    + "__".join([x.replace(" ", "_") for x in target_neg_prompt]),
                )
            os.makedirs(save_path, exist_ok=True)

    # VAE decode audio
    with inference_mode():
        x0_dec = ldm_stable.vae_decode(w0)

        # For stable-audio, audio is directly decoded
        audio = x0_dec.detach().clone().cpu().squeeze(0)
        orig_audio = x0.detach().clone().cpu()

        # Get spectrograms for visualization
        x0_dec_spec = get_spec(x0_dec, ldm_stable.get_fn_STFT())
        x0_spec = get_spec(x0.unsqueeze(0), ldm_stable.get_fn_STFT())

        if x0_dec_spec.dim() < 4:
            x0_dec_spec = x0_dec_spec[None, :, :, :]
            x0_spec = x0_spec[None, :, :, :]

    # Generate final timestamp for output naming
    current_GMT = time.gmtime()
    time_stamp_name = calendar.timegm(current_GMT)

    if mode == "sdedit":
        noise_percentage = (tstart[0] / num_diffusion_steps) * 100
        cfg_val = cfg_scale_tar[0] if isinstance(cfg_scale_tar, list) else cfg_scale_tar
        image_name_png = f"sdedit_cfg{cfg_val}_t{tstart[0]}_noise{noise_percentage:.1f}pct_{time_stamp_name}"
    elif mode == "ddpm":
        image_name_png = (
            f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_'
            + f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_'
            + f'skip_{"-".join([str(x) for x in skip.numpy()])}_{time_stamp_name}'
        )
    else:  # ddim
        if skip != 0:
            image_name_png = (
                f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_'
                + f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_'
                + f'skip_{"-".join([str(x) for x in skip.numpy()])}_{time_stamp_name}'
            )
        else:
            image_name_png = (
                f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_'
                + f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_'
                + f"{num_diffusion_steps}timesteps_{time_stamp_name}"
            )

    if save_edit_wav_path is not None:
        torchaudio.save(save_edit_wav_path, audio, sample_rate=sr)
    else:
        save_full_path_spec = os.path.join(save_path, image_name_png + ".png")
        save_full_path_wave = os.path.join(save_path, image_name_png + ".wav")
        save_full_path_origwave = os.path.join(save_path, "orig.wav")

        if x0_dec_spec.shape[2] > x0_dec_spec.shape[3]:
            x0_dec_spec_np = x0_dec_spec[0, 0].T.cpu().detach().numpy()
            x0_spec_np = x0_spec[0, 0].T.cpu().detach().numpy()
        else:
            x0_dec_spec_np = x0_dec_spec[0, 0].cpu().detach().numpy()
            x0_spec_np = x0_spec[0, 0].cpu().detach().numpy()

        plt.imsave(save_full_path_spec, x0_dec_spec_np)
        torchaudio.save(save_full_path_wave, audio, sample_rate=sr)
        torchaudio.save(save_full_path_origwave, orig_audio, sample_rate=sr)

    if verbose:
        print(f"\n=== {mode.upper()} Results ===")
        if mode == "sdedit":
            print(f"Noise level: {(tstart[0] / num_diffusion_steps) * 100:.1f}% (t={tstart[0]}/{num_diffusion_steps})")
            print(f"Target prompt: '{target_prompt[0]}'")
            print(f"Negative prompt: '{target_neg_prompt[0]}'")
            cfg_val = cfg_scale_tar[0] if isinstance(cfg_scale_tar, list) else cfg_scale_tar
            print(f"CFG scale: {cfg_val}")

        if save_edit_wav_path is not None:
            print(f"Generated audio saved to: {save_edit_wav_path}")
        else:
            print(f"Results saved to: {save_path}")
            print(f"Generated audio: {save_full_path_wave}")
            print(f"Original audio: {save_full_path_origwave}")
            print(f"Spectrogram: {save_full_path_spec}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text-based audio editing with Stable Audio.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--init_aud", type=str, required=True, help="Audio to invert and extract PCs from")
    parser.add_argument(
        "--cfg_src", type=float, nargs="+", default=[1.0], help="Classifier-free guidance strength for forward process"
    )
    parser.add_argument(
        "--cfg_tar", type=float, nargs="+", default=[3.5], help="Classifier-free guidance strength for reverse process"
    )
    parser.add_argument(
        "--num_diffusion_steps", type=int, default=200, help="Number of diffusion steps for Stable Audio"
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        nargs="+",
        default=[""],
        required=True,
        help="Prompt to accompany the reverse process. Should describe the wanted edited audio.",
    )
    parser.add_argument(
        "--source_prompt",
        type=str,
        nargs="+",
        default=[""],
        help="Prompt to accompany the forward process. Should describe the original audio.",
    )
    parser.add_argument(
        "--target_neg_prompt",
        type=str,
        nargs="+",
        default=[""],
        help="Negative prompt to accompany the inversion and generation process",
    )
    parser.add_argument(
        "--tstart",
        type=int,
        nargs="+",
        default=[100],
        help="Diffusion timestep to start the reverse process from. Controls editing strength.",
    )
    parser.add_argument("--results_path", type=str, default="results", help="path to dump results")

    parser.add_argument("--cutoff_points", type=float, nargs="*", default=None)
    parser.add_argument(
        "--mode",
        default="ddpm",
        choices=["ddpm", "ddim", "sdedit"],
        help="Run ddpm editing, DDIM inversion, or SDEdit.",
    )
    parser.add_argument("--fix_alpha", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument(
        "--layers_to_hook",
        type=str,
        nargs="*",
        default=None,
        help="Transformer layers to hook for cross-attention replacement (e.g., transformer.blocks.5 transformer.blocks.10)",
    )

    args = parser.parse_args()
    args.eta = 1.0
    args.numerical_fix = True
    args.test_rand_gen = False

    set_reproducability(args.seed, extreme=False)

    run_stable_audio_edit(
        init_aud=args.init_aud,
        cfg_src=args.cfg_src,
        cfg_tar=args.cfg_tar,
        num_diffusion_steps=args.num_diffusion_steps,
        target_prompt=args.target_prompt,
        source_prompt=args.source_prompt,
        target_neg_prompt=args.target_neg_prompt,
        tstart=args.tstart,
        results_path=args.results_path,
        cutoff_points=args.cutoff_points,
        mode=args.mode,
        fix_alpha=args.fix_alpha,
        verbose=args.verbose,
        layers_to_hook=args.layers_to_hook,
    )
