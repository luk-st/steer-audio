"""
Train Concept Sliders (textual LoRA) for ACE-Step audio diffusion model.

Adapts Gandikota et al. 2023 "Concept Sliders" to learn semantic steering
directions via LoRA fine-tuning. The LoRA encodes a concept direction by
training to match CFG-guided velocity predictions.

Training loss (enhance mode):
    MSE(v_lora(x_t, c_pos), v_base(x_t, c_neut) + η * (v_base(x_t, c_pos) - v_base(x_t, c_uncond)))

Where:
    v_lora  = velocity prediction with LoRA ON
    v_base  = velocity prediction with LoRA OFF (frozen base model)
    c_pos   = positive concept prompt (e.g., "a song, with piano")
    c_neut  = neutral prompt (e.g., "a song")
    c_uncond = unconditional (empty text)
    η       = guidance scale for concept direction

With --with_denoising (default), each iteration partially denoises from noise
to a random intermediate state before computing the loss. This produces more
realistic training inputs matching what the model sees during inference.

Usage:
    python steering/cs/train_concept_slider.py \
        --concept piano \
        --iterations 1000 \
        --lr 1e-4 \
        --output_dir steering_vectors/concept_slider/ace_piano_r4

    # Without partial denoising (faster, less accurate)
    python steering/cs/train_concept_slider.py \
        --concept piano \
        --with_denoising False
"""

import gc
import json
import math
import os
import random
import re
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
from fire import Fire
from peft import LoraConfig
from tqdm import tqdm

PATH_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PATH_PROJECT)
sys.path.append(os.path.join(PATH_PROJECT, "src", "models", "ace_step", "ACE"))

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from src.steering.methods.sae.lib.configs.steer_prompts import CONCEPT_TO_PROMPTS
from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline


def flush():
    torch.cuda.empty_cache()
    gc.collect()


@torch.no_grad()
def pre_encode_prompts(pipe, positive_prompts, neutral_prompts, lyrics, device, dtype):
    """Pre-encode all prompt variants via T5 + transformer.encode().

    All three prompt types (positive, neutral, unconditional) use the same
    lyrics so the guidance direction captures only the text concept difference.

    Returns:
        encoded_pairs: List of dicts with 'pos' and 'neutral' keys,
            each a tuple of (encoder_hidden_states, encoder_hidden_mask).
        uncond: Tuple of (encoder_hidden_states, encoder_hidden_mask).
    """
    transformer = pipe.ace_step_transformer

    # Common conditioning: zero speaker (no speaker conditioning in released model)
    speaker_embeds = torch.zeros(1, 512, device=device, dtype=dtype)

    # Prepare lyrics (shared across all prompts for clean concept isolation)
    if lyrics and len(lyrics.strip()) > 0:
        lyric_tokens = pipe.tokenize_lyrics(lyrics, debug=False)
        lyric_mask = [1] * len(lyric_tokens)
    else:
        lyric_tokens = [0]
        lyric_mask = [0]
    lyric_idx = torch.tensor(lyric_tokens, device=device, dtype=torch.long).unsqueeze(0)
    lyric_mask_t = torch.tensor(lyric_mask, device=device, dtype=torch.long).unsqueeze(0)

    # Encode unconditional (empty text, same lyrics)
    text_hs_uncond, text_mask_uncond = pipe.get_text_embeddings([""])
    enc_uncond, mask_uncond = transformer.encode(
        text_hs_uncond, text_mask_uncond, speaker_embeds, lyric_idx, lyric_mask_t,
    )
    uncond = (enc_uncond.detach(), mask_uncond.detach())

    # Encode each positive/neutral pair
    encoded_pairs = []
    for pos_text, neut_text in zip(positive_prompts, neutral_prompts):
        text_hs_pos, text_mask_pos = pipe.get_text_embeddings([pos_text])
        enc_pos, mask_pos = transformer.encode(
            text_hs_pos, text_mask_pos, speaker_embeds, lyric_idx, lyric_mask_t,
        )

        text_hs_neut, text_mask_neut = pipe.get_text_embeddings([neut_text])
        enc_neut, mask_neut = transformer.encode(
            text_hs_neut, text_mask_neut, speaker_embeds, lyric_idx, lyric_mask_t,
        )

        encoded_pairs.append({
            "pos": (enc_pos.detach(), mask_pos.detach()),
            "neutral": (enc_neut.detach(), mask_neut.detach()),
        })

    return encoded_pairs, uncond


@torch.no_grad()
def partial_denoise(
    transformer,
    noise,
    attn_mask,
    enc_cond,
    mask_cond,
    enc_uncond,
    mask_uncond,
    scheduler,
    timesteps,
    steps_to,
    cfg_scale,
    frame_length,
):
    """Partially denoise from pure noise using CFG, returning intermediate latent.

    Runs `steps_to` denoising steps with LoRA ON (no gradients), producing a
    realistic intermediate state at the corresponding timestep.

    Args:
        transformer: ACEStepTransformer2DModel (with LoRA enabled)
        noise: Initial pure noise latent [1, 8, 16, frame_length]
        attn_mask: Attention mask [1, frame_length]
        enc_cond: Conditional encoder hidden states (target prompt)
        mask_cond: Conditional encoder mask
        enc_uncond: Unconditional encoder hidden states
        mask_uncond: Unconditional encoder mask
        scheduler: FlowMatchEulerDiscreteScheduler (already has timesteps set)
        timesteps: Scheduler timesteps tensor
        steps_to: Number of steps to denoise (1..len(timesteps))
        cfg_scale: Guidance scale for CFG during denoising
        frame_length: Output length for decode

    Returns:
        denoised_latents: Intermediate latent at the target timestep
        current_timestep: The timestep value at the intermediate state
    """
    latents = noise.clone()
    scheduler._step_index = None  # Reset scheduler state for this iteration

    for step_idx in range(steps_to):
        t = timesteps[step_idx]
        t_expanded = t.unsqueeze(0)

        # Conditional prediction
        v_cond = transformer.decode(
            hidden_states=latents,
            attention_mask=attn_mask,
            encoder_hidden_states=enc_cond,
            encoder_hidden_mask=mask_cond,
            timestep=t_expanded,
            output_length=frame_length,
        ).sample

        # Unconditional prediction
        v_uncond = transformer.decode(
            hidden_states=latents,
            attention_mask=attn_mask,
            encoder_hidden_states=enc_uncond,
            encoder_hidden_mask=mask_uncond,
            timestep=t_expanded,
            output_length=frame_length,
        ).sample

        # CFG combination
        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Scheduler step
        latents = scheduler.step(
            model_output=v_guided,
            timestep=t,
            sample=latents,
            return_dict=False,
        )[0]

        del v_cond, v_uncond, v_guided

    # Map to the 1000-step schedule to get the current timestep
    # After steps_to steps out of max_steps, we're at this fraction of denoising
    current_t = int(timesteps[steps_to - 1].item()) if steps_to < len(timesteps) else 0
    # Use the next timestep (where we'd predict from)
    if steps_to < len(timesteps):
        current_t = int(timesteps[steps_to].item())
    else:
        current_t = 1  # nearly clean

    return latents.detach(), torch.tensor([current_t], device=latents.device, dtype=latents.dtype)


def main(
    concept: str,
    output_dir: str | None = None,
    lora_config_path: str | None = None,
    iterations: int = 1000,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    eta: float = 1.0,
    audio_duration: float = 10.0,
    save_every: int = 200,
    seed: int = 42,
    max_grad_norm: float = 1.0,
    gradient_checkpointing: bool = True,
    with_denoising: bool = True,
    max_denoising_steps: int = 50,
    denoise_cfg_scale: float = 5.0,  # CFG scale during partial denoising
    layers: str = "all",
    device: str = "cuda",
):
    """
    Train a Concept Slider (LoRA) for the ACE-Step audio diffusion model.

    Args:
        concept: Concept to train. Options: piano, mood, tempo, vocal_gender, drums
        output_dir: Where to save the trained LoRA. Auto-generated if None.
        lora_config_path: Path to LoRA config JSON. Default: steering/cs/lora_config.json
        iterations: Number of training iterations.
        lr: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW.
        eta: Guidance scale in the concept sliders loss (controls concept strength).
        audio_duration: Audio duration in seconds for training latents.
            10s recommended (saves ~3x memory vs 30s; direction generalizes).
        save_every: Save checkpoint every N iterations.
        seed: Random seed for reproducibility.
        max_grad_norm: Max gradient norm for clipping (0 to disable).
        gradient_checkpointing: Enable gradient checkpointing to save memory.
        with_denoising: If True, partially denoise from noise before computing loss.
            Produces more realistic training inputs. Default: True.
        max_denoising_steps: Number of steps in the partial denoising schedule.
            Each iteration randomly samples 1..max_denoising_steps steps to run.
        denoise_cfg_scale: CFG guidance scale during partial denoising.
        layers: Which transformer blocks to apply LoRA to.
            'all' = all 24 blocks (default), 'tf6tf7' = blocks 6 and 7 only.
        device: Device to train on.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    dtype = torch.bfloat16

    # Resolve paths
    cs_dir = os.path.dirname(os.path.abspath(__file__))
    if lora_config_path is None:
        lora_config_path = os.path.join(cs_dir, "lora_config.json")
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f"steering_vectors/concept_slider/ace_{concept}_r4_{timestamp}"

    # Validate concept
    if concept not in CONCEPT_TO_PROMPTS:
        raise ValueError(
            f"Unknown concept: {concept}. "
            f"Available: {list(CONCEPT_TO_PROMPTS.keys())}"
        )

    # Load LoRA config
    with open(lora_config_path) as f:
        lora_cfg = json.load(f)

    # Resolve layer selection
    LAYER_CONFIGS = {
        "all": list(range(24)),
        "tf6tf7": [6, 7],
        "tf6": [6],
        "tf7": [7],
    }
    if layers not in LAYER_CONFIGS:
        raise ValueError(f"Unknown layers: {layers}. Available: {list(LAYER_CONFIGS.keys())}")
    layers_to_transform = LAYER_CONFIGS[layers]

    # =========================================================================
    # Load pipeline and setup LoRA
    # =========================================================================
    print("Loading ACE-Step pipeline...")
    pipe = SimpleACEStepPipeline(device=device, dtype="bfloat16")
    pipe.load()
    print("Pipeline loaded")

    transformer = pipe.ace_step_transformer

    # Add LoRA adapter via PEFT (with optional layer filtering)
    lora_kwargs = dict(lora_cfg)
    if layers != "all":
        # Build regex pattern for target_modules restricted to specific blocks.
        # PEFT supports regex when target_modules is a single string.
        block_pattern = "|".join(str(i) for i in layers_to_transform)
        base_targets = lora_kwargs["target_modules"]
        target_pattern = "|".join(re.escape(t) for t in base_targets)
        lora_kwargs["target_modules"] = (
            rf"transformer_blocks\.({block_pattern})\..*\.({target_pattern})$"
        )
    peft_config = LoraConfig(**lora_kwargs)
    transformer.add_adapter(adapter_config=peft_config, adapter_name="concept_slider")

    if gradient_checkpointing:
        transformer.gradient_checkpointing = True

    # Freeze base model, only LoRA params get gradients
    transformer.requires_grad_(False)
    lora_params = []
    for name, param in transformer.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            lora_params.append(param)
    num_lora_params = sum(p.numel() for p in lora_params)

    # =========================================================================
    # Pre-encode all prompts (frozen, LoRA-independent)
    # =========================================================================
    get_prompts = CONCEPT_TO_PROMPTS[concept]
    neutral_prompts, positive_prompts, lyrics = get_prompts()

    print("Pre-encoding prompts...")
    transformer.disable_adapters()
    encoded_pairs, uncond = pre_encode_prompts(
        pipe, positive_prompts, neutral_prompts, lyrics, device, dtype
    )
    print(f"  Encoded {len(encoded_pairs)} prompt pairs + 1 unconditional")

    # Free text encoder / DCAE memory (not needed during training)
    pipe.text_encoder_model.cpu()
    if hasattr(pipe, "music_dcae") and pipe.music_dcae is not None:
        pipe.music_dcae.cpu()
    flush()

    # =========================================================================
    # Setup scheduler for partial denoising
    # =========================================================================
    denoise_scheduler = None
    denoise_timesteps = None
    if with_denoising:
        denoise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )
        denoise_timesteps, _ = retrieve_timesteps(
            denoise_scheduler,
            num_inference_steps=max_denoising_steps,
            device=device,
        )

    # =========================================================================
    # Setup optimizer and training config
    # =========================================================================
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

    frame_length = math.ceil(audio_duration * 44100 / 512 / 8)
    num_prompts = len(encoded_pairs)

    print(f"\nTraining concept slider for '{concept}'")
    print(f"  Prompt pairs:   {num_prompts}")
    print(f"  Iterations:     {iterations}")
    print(f"  LoRA rank:      {lora_cfg['r']}, alpha: {lora_cfg['lora_alpha']}")
    print(f"  LoRA params:    {num_lora_params:,}")
    print(f"  Frame length:   {frame_length} ({audio_duration}s audio)")
    print(f"  Guidance eta:   {eta}")
    print(f"  Layers:         {layers} ({len(layers_to_transform)} blocks)")
    print(f"  With denoising: {with_denoising}")
    if with_denoising:
        print(f"  Denoise steps:  1..{max_denoising_steps}")
        print(f"  Denoise CFG:    {denoise_cfg_scale}")
    print(f"  Output:         {output_dir}\n")

    # =========================================================================
    # Training loop
    # =========================================================================
    losses = []
    pbar = tqdm(range(iterations), desc="Training")

    for i in pbar:
        optimizer.zero_grad()

        # --- Sample random prompt pair ---
        idx = random.randint(0, num_prompts - 1)
        enc_pos, mask_pos = encoded_pairs[idx]["pos"]
        enc_neut, mask_neut = encoded_pairs[idx]["neutral"]
        enc_uncond, mask_uncond = uncond

        # Random noise latents: [1, 8, 16, frame_length]
        noise = torch.randn(1, 8, 16, frame_length, device=device, dtype=dtype)
        attn_mask = torch.ones(1, frame_length, device=device, dtype=torch.long)

        if with_denoising:
            # --- Partial denoise with LoRA ON to get realistic intermediate ---
            steps_to = random.randint(1, max_denoising_steps)
            transformer.enable_adapters()
            transformer.eval()
            denoised_latents, timestep = partial_denoise(
                transformer=transformer,
                noise=noise,
                attn_mask=attn_mask,
                enc_cond=enc_neut,
                mask_cond=mask_neut,
                enc_uncond=enc_uncond,
                mask_uncond=mask_uncond,
                scheduler=denoise_scheduler,
                timesteps=denoise_timesteps,
                steps_to=steps_to,
                cfg_scale=denoise_cfg_scale,
                frame_length=frame_length,
            )
            input_latents = denoised_latents
            del noise
        else:
            # --- No denoising: use pure noise with random timestep ---
            t = random.randint(1, 999)
            timestep = torch.tensor([t], device=device, dtype=dtype)
            input_latents = noise

        # --- LoRA OFF: velocity for positive, neutral, unconditional ---
        transformer.disable_adapters()
        transformer.eval()
        with torch.no_grad():
            v_pos = transformer.decode(
                hidden_states=input_latents,
                attention_mask=attn_mask,
                encoder_hidden_states=enc_pos,
                encoder_hidden_mask=mask_pos,
                timestep=timestep,
                output_length=frame_length,
            ).sample

            v_neut = transformer.decode(
                hidden_states=input_latents,
                attention_mask=attn_mask,
                encoder_hidden_states=enc_neut,
                encoder_hidden_mask=mask_neut,
                timestep=timestep,
                output_length=frame_length,
            ).sample

            v_uncond = transformer.decode(
                hidden_states=input_latents,
                attention_mask=attn_mask,
                encoder_hidden_states=enc_uncond,
                encoder_hidden_mask=mask_uncond,
                timestep=timestep,
                output_length=frame_length,
            ).sample

        # Guidance target: v_neutral + η * (v_positive - v_unconditional)
        guidance_target = v_neut + eta * (v_pos - v_uncond)
        del v_pos, v_neut, v_uncond

        # --- LoRA ON: velocity for target (positive prompt, gradients flow) ---
        transformer.enable_adapters()
        transformer.train()
        v_target = transformer.decode(
            hidden_states=input_latents,
            attention_mask=attn_mask,
            encoder_hidden_states=enc_pos,
            encoder_hidden_mask=mask_pos,
            timestep=timestep,
            output_length=frame_length,
        ).sample

        # --- Loss and update ---
        loss = F.mse_loss(v_target, guidance_target)
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)

        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        avg_recent = sum(losses[-50:]) / len(losses[-50:])
        pbar.set_description(f"Loss: {loss_val:.6f} (avg50: {avg_recent:.6f})")

        del v_target, guidance_target, input_latents, loss
        if (i + 1) % 10 == 0:
            flush()

        # --- Save checkpoint ---
        if (i + 1) % save_every == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint_{i + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            transformer.save_lora_adapter(ckpt_dir, adapter_name="concept_slider")
            tqdm.write(f"  Checkpoint saved at iteration {i + 1}")

    # =========================================================================
    # Save final weights and config
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)
    transformer.save_lora_adapter(output_dir, adapter_name="concept_slider")

    train_config = {
        "concept": concept,
        "lora_config": lora_cfg,
        "lora_config_path": lora_config_path,
        "layers": layers,
        "layers_to_transform": layers_to_transform,
        "iterations": iterations,
        "lr": lr,
        "weight_decay": weight_decay,
        "eta": eta,
        "audio_duration": audio_duration,
        "seed": seed,
        "max_grad_norm": max_grad_norm,
        "gradient_checkpointing": gradient_checkpointing,
        "with_denoising": with_denoising,
        "max_denoising_steps": max_denoising_steps,
        "denoise_cfg_scale": denoise_cfg_scale,
        "num_lora_params": num_lora_params,
        "frame_length": frame_length,
        "num_prompts": num_prompts,
        "final_loss": losses[-1] if losses else None,
        "avg_loss_last_50": sum(losses[-50:]) / len(losses[-50:]) if losses else None,
    }
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)

    print(f"\nTraining complete! LoRA saved to {output_dir}")
    if losses:
        print(f"  Final loss:      {losses[-1]:.6f}")
        print(f"  Avg loss (last 50): {avg_recent:.6f}")


if __name__ == "__main__":
    Fire(main)
