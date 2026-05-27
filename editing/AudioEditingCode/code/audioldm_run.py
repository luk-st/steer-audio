import argparse
import base64
import calendar
import os
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torchaudio
from ddm_inversion.ddim_inversion import ddim_inversion, text2image_ldm_stable
from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from sdedit_utils import sdedit_denoise, sdedit_forward_noise
from torch import inference_mode

from models import load_model
from utils import load_audio, set_reproducability
from env import ALDM2_TEMP_DIR

ENABLE_TQDM = True

def create_truncated_audio(audio_path: str, max_duration: float = 60.0) -> str:
    """AudioLDM works too slow for long audios, we truncate them to 60s.
    """
    waveform, sr = torchaudio.load(audio_path)
    max_samples = int(sr * max_duration)

    if waveform.shape[1] > max_samples:
        waveform = waveform[..., :max_samples]

    temp_dir = Path(ALDM2_TEMP_DIR)
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)

    curr_date_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
    encode_str = f"{audio_path}_{curr_date_str}_{os.getpid()}"

    filename = str(base64.b64encode(encode_str.encode("ascii")).decode("ascii"))
    new_filename = filename + ".wav"
    temp_file_path = temp_dir / new_filename

    waveform = waveform.to(torch.float32)
    if torch.max(torch.abs(waveform)) > 0:
        waveform = waveform / torch.max(torch.abs(waveform))

    torchaudio.save(temp_file_path, waveform, sr, format="wav", encoding="PCM_S", bits_per_sample=16)
    time.sleep(5)

    return str(temp_file_path)


def create_audioldm_hook(
    target_encoder_hidden_states,
    target_class_labels,
    target_prompt_mask,
    hook_counter=None,
    n_layers_registered: int = 0,
):
    """
    Create a hook function that replaces encoder_hidden_states and encoder_attention_mask with target prompt encoding
    for specific cross-attention layers in AudioLDM, while preserving null/unconditional predictions for CFG.

    Args:
        target_encoder_hidden_states: The target prompt embeddings to use
        target_class_labels: The target class labels to use
        target_prompt_mask: The target prompt mask to use
        hook_counter: Dict to track hook applications (modified in-place)
        n_layers_hooked: number of layers in DM registered to this hook
    """

    def cross_attn_hook(module, args, kwargs):
        # Replace encoder_hidden_states with target prompt encoding

        is_unconditional = (n_layers_registered == 0) or (
            (hook_counter["total_calls"] // n_layers_registered) % 2 == 0
        )
        hook_counter["total_calls"] += 1
        if is_unconditional:
            hook_counter["unconds"] += 1
        else:
            hook_counter["conds"] += 1

        if is_unconditional is False:
            if kwargs.get("attention_mask") is not None:
                kwargs["encoder_hidden_states"] = target_class_labels
                attention_mask = target_prompt_mask.clone()
                attention_mask = (1 - attention_mask.to(args[0].dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)
                kwargs["attention_mask"] = attention_mask
            else:
                kwargs["encoder_hidden_states"] = target_encoder_hidden_states
        return args, kwargs

    return cross_attn_hook


def apply_audioldm_hooks(
    model, layers_to_hook, target_encoder_hidden_states, target_class_labels, target_prompt_mask, verbose=False
):
    """
    Apply hooks to specific cross-attention layers in AudioLDM model.

    Args:
        model: The AudioLDM model
        layers_to_hook: List of layer names to hook (e.g., ["up_blocks.1.attentions.5.transformer_blocks.0.attn2"])
        target_encoder_hidden_states: Target prompt encoding to use in hooked layers
        target_class_labels: Target class labels to use in hooked layers
        target_prompt_mask: Target prompt mask to use in hooked layers
        verbose: Whether to print hook application messages

    Returns:
        Tuple of (hook_handles, hook_counter) for cleanup and statistics
    """
    if not layers_to_hook or len(layers_to_hook) == 0:
        return [], {}

    # Initialize counter for tracking hook applications
    hook_counter = {"total_calls": 0, "conds": 0, "unconds": 0}

    hook_handles = []
    hook_function = create_audioldm_hook(
        target_encoder_hidden_states=target_encoder_hidden_states,
        target_class_labels=target_class_labels,
        target_prompt_mask=target_prompt_mask,
        hook_counter=hook_counter,
        n_layers_registered=len(layers_to_hook),
    )

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

            # Apply hook to the cross-attention layer
            hook_handle = target_layer.register_forward_pre_hook(hook_function, with_kwargs=True)
            hook_handles.append(hook_handle)

            if verbose:
                print(f"Applied hook to layer: {layer_name}")

        except (AttributeError, IndexError, TypeError) as e:
            raise ValueError(f"Warning: Could not apply hook to layer {layer_name}: {e}")

    return hook_handles, hook_counter


def run_audioldm_edit(
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
    model_id: str = "cvssp/audioldm2-large",
):
    """
    Run text-based audio editing with AudioLDM2 models with optional cross-attention hooks.

    Args:
        init_aud: Path to input audio file
        cfg_src: CFG scales for forward process (inversion)
        cfg_tar: CFG scales for reverse process (generation)
        num_diffusion_steps: Number of diffusion steps
        target_prompt: Target prompts for editing
        source_prompt: Source prompts describing original audio
        target_neg_prompt: Negative prompts
        tstart: Start timesteps for reverse process
        mode: Editing mode ('ddpm', 'ddim', 'sdedit')
        results_path: Path to save results
        fix_alpha: Alpha parameter for DDPM
        cutoff_points: Cutoff points for DDPM
        verbose: Enable verbose output
        test_rand_gen: Use random generation for testing
        numerical_fix: Enable numerical fix
        save_edit_wav_path: Direct path to save edited audio
        eta: Eta parameter for DDPM
        layers_to_hook: List of layer names to hook for cross-attention replacement
        model_id: AudioLDM model ID to use
    """
    assert mode in ["ddpm", "ddim", "sdedit"], "Invalid mode, must be one of ['ddpm', 'ddim', 'sdedit']"
    assert (
        results_path is not None or save_edit_wav_path is not None
    ), "Either results_path or save_edit_wav_path must be provided"

    # Validate model_id is AudioLDM
    valid_models = [
        "cvssp/audioldm-s-full-v2",
        "cvssp/audioldm-l-full",
        "cvssp/audioldm2",
        "cvssp/audioldm2-large",
        "cvssp/audioldm2-music",
    ]
    assert model_id in valid_models, f"model_id must be one of {valid_models}"

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

    # Load AudioLDM model
    ldm_stable = load_model(model_id, device, num_diffusion_steps, edit_method=mode)
    ldm_stable_inverse = ldm_stable

    audio_path_truncated: str = create_truncated_audio(init_aud, max_duration=60.0)
    x0, sr, duration = load_audio(
        audio_path_truncated, ldm_stable.get_fn_STFT(), device=device, stft=True, model_sr=ldm_stable.get_sr()
    )
    if os.path.exists(audio_path_truncated):
        os.remove(audio_path_truncated)

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
                (
                    target_text_embeddings_hidden_states,
                    target_text_embeddings_class_labels,
                    target_text_embeddings_boolean_prompt_mask,
                ) = ldm_stable.encode_text(target_prompt)
                hook_handles, hook_counter = apply_audioldm_hooks(
                    model=ldm_stable.model.unet,
                    layers_to_hook=layers_to_hook,
                    target_encoder_hidden_states=target_text_embeddings_hidden_states,
                    target_class_labels=target_text_embeddings_class_labels,
                    target_prompt_mask=target_text_embeddings_boolean_prompt_mask,
                    verbose=verbose,
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
                            f"Conditional replaced: {hook_counter['conds']}, "
                            f"Unconditional preserved: {hook_counter['unconds']}"
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
                (
                    target_text_embeddings_hidden_states,
                    target_text_embeddings_class_labels,
                    target_text_embeddings_boolean_prompt_mask,
                ) = ldm_stable.encode_text(target_prompt)
                hook_handles, hook_counter = apply_audioldm_hooks(
                    model=ldm_stable.model.unet,
                    layers_to_hook=layers_to_hook,
                    target_encoder_hidden_states=target_text_embeddings_hidden_states,
                    target_class_labels=target_text_embeddings_class_labels,
                    target_prompt_mask=target_text_embeddings_boolean_prompt_mask,
                    verbose=verbose,
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
                            f"Conditional replaced: {hook_counter['conds']}, "
                            f"Unconditional preserved: {hook_counter['unconds']}"
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
                (
                    target_text_embeddings_hidden_states,
                    target_text_embeddings_class_labels,
                    target_text_embeddings_boolean_prompt_mask,
                ) = ldm_stable.encode_text(target_prompt)
                hook_handles, hook_counter = apply_audioldm_hooks(
                    model=ldm_stable.model.unet,
                    layers_to_hook=layers_to_hook,
                    target_encoder_hidden_states=target_text_embeddings_hidden_states,
                    target_class_labels=target_text_embeddings_class_labels,
                    target_prompt_mask=target_text_embeddings_boolean_prompt_mask,
                    verbose=verbose,
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
                            f"Conditional replaced: {hook_counter['conds']}, "
                            f"Unconditional preserved: {hook_counter['unconds']}"
                        )

        # Create save path based on mode
        if results_path is not None and save_edit_wav_path is None:
            save_path = os.path.join(
                f"{results_path}/",
                model_id.split("/")[1] + "_" + mode + ("_hooks" if layers_to_hook else ""),
            )
            os.makedirs(save_path, exist_ok=True)

    # VAE decode audio
    with inference_mode():
        x0_dec = ldm_stable.vae_decode(w0)

        # For AudioLDM models, decode to mel spectrogram then to audio
        if x0_dec.dim() < 4:
            x0_dec = x0_dec[None, :, :, :]

        with torch.no_grad():
            audio = ldm_stable.decode_to_mel(x0_dec)
            orig_audio = ldm_stable.decode_to_mel(x0)

        # Get spectrograms for visualization
        x0_dec_spec = x0_dec
        x0_spec = x0

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

        # Handle spectrogram saving for AudioLDM
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

        print(f"Model: {model_id}")
        if layers_to_hook:
            print(f"Hooked layers: {layers_to_hook}")
        if save_edit_wav_path is not None:
            print(f"Generated audio saved to: {save_edit_wav_path}")
        else:
            print(f"Results saved to: {save_path}")
            print(f"Generated audio: {save_full_path_wave}")
            print(f"Original audio: {save_full_path_origwave}")
            print(f"Spectrogram: {save_full_path_spec}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text-based audio editing with AudioLDM models.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--init_aud", type=str, required=True, help="Audio to invert and extract PCs from")
    parser.add_argument(
        "--cfg_src", type=float, nargs="+", default=[3.0], help="Classifier-free guidance strength for forward process"
    )
    parser.add_argument(
        "--cfg_tar",
        type=float,
        nargs="+",
        default=[12.0],
        help="Classifier-free guidance strength for reverse process",
    )
    parser.add_argument("--num_diffusion_steps", type=int, default=200, help="Number of diffusion steps for AudioLDM")
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
        help="Cross-attention layers to hook for replacement (e.g., down_blocks.0.attentions.0.transformer_blocks.0.attn2)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        choices=[
            "cvssp/audioldm-s-full-v2",
            "cvssp/audioldm-l-full",
            "cvssp/audioldm2",
            "cvssp/audioldm2-large",
            "cvssp/audioldm2-music",
        ],
        default="cvssp/audioldm2-large",
        help="AudioLDM model to use",
    )

    args = parser.parse_args()
    args.eta = 1.0
    args.numerical_fix = True
    args.test_rand_gen = False

    set_reproducability(args.seed, extreme=False)

    run_audioldm_edit(
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
        model_id=args.model_id,
    )
