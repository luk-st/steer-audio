"""FreeSliders library functions — extracted from the original ``run_freesliders.py``.

Pure library: no CLI, no Fire, no script-level side effects. Both
``FreeSlidersSteeringController`` (the new unified interface) and any direct callers
import from here.

The algorithm: at each diffusion step past ``split_step`` the noise prediction is
modified as ``noise_neutral + alpha * (noise_pos - noise_neg)``. ``noise_neutral``
is the standard CFG-guided prediction from the neutral prompt; ``noise_pos`` /
``noise_neg`` are single conditional forward passes for the positive / negative
prompts. Stage 1 (steps before ``split_step``) is plain neutral CFG.
"""

from __future__ import annotations

import math
import os
import sys

import torch

# The acestep package lives under src/models/ace_step/ACE/. Make it importable
# without forcing every consumer to set sys.path themselves.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "src", "models", "ace_step", "ACE"),
):
    if _p not in sys.path:
        sys.path.append(_p)

from acestep.apg_guidance import (
    MomentumBuffer,
    apg_forward,
    cfg_double_condition_forward,
    cfg_forward,
    cfg_zero_star,
)
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from src.models.ace_step.ACE.acestep.cpu_offload import cpu_offload
from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline
from src.steering.methods.freesliders.layer_hooks import is_all_layers, layer_specific_decode


# ---------------------------------------------------------------------------
# Prompt triple construction
# ---------------------------------------------------------------------------


def build_prompt_triple(test_prompt: str, concept: str) -> tuple[str, str, str]:
    """Return ``(neutral, positive, negative)`` for a given concept.

    Neutral is the raw test prompt; positive / negative receive concept-specific
    modifiers. Subset of concepts is intentional — extend ``configs`` to add new
    concepts, or pass ``positive_prompt_suffix`` / ``negative_prompt_suffix``
    directly to :class:`FreeSlidersSteeringController` to bypass this table.
    """
    p = test_prompt
    configs = {
        "piano": {"pos": f"{p} with piano", "neg": f"{p}"},
        "drums": {"pos": f"{p} with drums", "neg": f"{p}"},
        "guitar": {"pos": f"{p} with guitar", "neg": f"{p}"},
        "saxophone": {"pos": f"{p} with saxophone", "neg": f"{p}"},
        "mood": {"pos": f"a happy {p}", "neg": f"a sad {p}"},
        "tempo": {"pos": f"fast {p}", "neg": f"slow {p}"},
        "vocal_gender": {
            "pos": f"{p} with female vocal",
            "neg": f"{p} with male vocal",
        },
        "vocal_style": {
            "pos": f"{p}, with rap vocal",
            "neg": f"{p}, with sing vocal",
        },
        "guitar_electronic": {"pos": f"{p}, with acoustic guitar", "neg": f"{p}, with electric guitar"},
        "violin": {"pos": f"{p}, with violin", "neg": f"{p}"},
        "rock_genre": {"pos": f"jazz song, {p}", "neg": f"rock song, {p}"},
        "electronic_music": {"pos": f"classical song, {p}", "neg": f"electronic song, {p}"},
    }
    if concept not in configs:
        raise ValueError(f"Unknown concept: {concept}. Available: {list(configs.keys())}")
    c = configs[concept]
    return p, c["pos"], c["neg"]


# ---------------------------------------------------------------------------
# Batched T5 embedding computation
# ---------------------------------------------------------------------------


@cpu_offload("text_encoder_model")
def compute_batched_fs_embeddings(
    pipe: SimpleACEStepPipeline,
    neutral_prompts: list[str],
    positive_prompts: list[str],
    negative_prompts: list[str],
    use_erg_tag: bool = True,
    tau: float = 0.01,
    l_min: int = 8,
    l_max: int = 10,
) -> dict:
    """Tokenise and embed all prompt triples with shared padding (T5 level).

    Neutral embeddings run alone so they're identical regardless of batch
    composition (bfloat16 numerics). Pos+neg run together for efficiency.
    Returns batched tensors of shape ``[N, max_seq, 1024]``.
    """
    N = len(neutral_prompts)
    assert len(positive_prompts) == N and len(negative_prompts) == N

    all_prompts: list[str] = []
    for n, p, neg in zip(neutral_prompts, positive_prompts, negative_prompts):
        all_prompts.extend([n, p, neg])

    inputs = pipe.text_tokenizer(
        all_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(pipe.device) for k, v in inputs.items()}

    if pipe.text_encoder_model.device != pipe.device:
        pipe.text_encoder_model.to(pipe.device)

    neutral_ids = inputs["input_ids"][0::3]
    neutral_attn = inputs["attention_mask"][0::3]
    pos_ids = inputs["input_ids"][1::3]
    pos_attn = inputs["attention_mask"][1::3]
    neg_ids = inputs["input_ids"][2::3]
    neg_attn = inputs["attention_mask"][2::3]

    with torch.no_grad():
        neutral_hs = pipe.text_encoder_model(
            input_ids=neutral_ids,
            attention_mask=neutral_attn,
        ).last_hidden_state

    pn_ids = torch.cat([pos_ids, neg_ids], dim=0)
    pn_attn = torch.cat([pos_attn, neg_attn], dim=0)
    with torch.no_grad():
        pn_hs = pipe.text_encoder_model(
            input_ids=pn_ids,
            attention_mask=pn_attn,
        ).last_hidden_state
    positive_hs = pn_hs[:N]
    negative_hs = pn_hs[N:]

    erg_neutral_hs = None
    if use_erg_tag:
        handlers = []

        def hook(module, input, output):
            output[:] *= tau
            return output

        for i in range(l_min, l_max):
            h = pipe.text_encoder_model.encoder.block[i].layer[0].SelfAttention.q.register_forward_hook(hook)
            handlers.append(h)

        with torch.no_grad():
            erg_neutral_hs = pipe.text_encoder_model(
                input_ids=neutral_ids,
                attention_mask=neutral_attn,
            ).last_hidden_state

        for h in handlers:
            h.remove()

    result = {
        "neutral_hs": neutral_hs,
        "positive_hs": positive_hs,
        "negative_hs": negative_hs,
        "neutral_masks": neutral_attn,
        "positive_masks": pos_attn,
        "negative_masks": neg_attn,
    }
    if erg_neutral_hs is not None:
        result["erg_neutral_hs"] = erg_neutral_hs

    return result


# ---------------------------------------------------------------------------
# Lyrics helper
# ---------------------------------------------------------------------------


def prepare_lyrics(pipe, lyrics_list):
    """Tokenise lyrics into batched tensors."""
    lyric_token_ids, lyric_masks = [], []
    for lyric in lyrics_list:
        if len(lyric) > 0:
            tok = pipe.tokenize_lyrics(lyric, debug=False)
            lm = [1] * len(tok)
        else:
            tok = [0]
            lm = [0]
        lyric_token_ids.append(tok)
        lyric_masks.append(lm)
    max_len = max(len(s) for s in lyric_token_ids)
    padded_ids = [ids + [0] * (max_len - len(ids)) for ids in lyric_token_ids]
    padded_masks = [m + [0] * (max_len - len(m)) for m in lyric_masks]
    return (
        torch.tensor(padded_ids, device=pipe.device, dtype=torch.long),
        torch.tensor(padded_masks, device=pipe.device, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# ERG helpers (replicated from text2music_diffusion_process)
# ---------------------------------------------------------------------------


def _encode_with_erg_lyric(pipe, inputs, tau=0.01, l_min=4, l_max=6):
    """Encode with temperature scaling on lyric encoder (ERG lyric)."""
    handlers = []

    def hook(module, input, output):
        output[:] *= tau
        return output

    for i in range(l_min, l_max):
        h = pipe.ace_step_transformer.lyric_encoder.encoders[i].self_attn.linear_q.register_forward_hook(hook)
        handlers.append(h)

    enc_hs, enc_mask = pipe.ace_step_transformer.encode(**inputs)

    for h in handlers:
        h.remove()
    return enc_hs, enc_mask


def _decode_with_erg_diffusion(pipe, hidden_states, timestep, inputs, tau=0.01, l_min=15, l_max=20):
    """Decode with temperature scaling on diffusion decoder (ERG diffusion)."""
    handlers = []

    def hook(module, input, output):
        output[:] *= tau
        return output

    for i in range(l_min, l_max):
        h = pipe.ace_step_transformer.transformer_blocks[i].attn.to_q.register_forward_hook(hook)
        handlers.append(h)
        h = pipe.ace_step_transformer.transformer_blocks[i].cross_attn.to_q.register_forward_hook(hook)
        handlers.append(h)

    sample = pipe.ace_step_transformer.decode(hidden_states=hidden_states, timestep=timestep, **inputs).sample

    for h in handlers:
        h.remove()
    return sample


# ---------------------------------------------------------------------------
# Diffusion loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def freesliders_diffusion(
    pipe,
    # T5 text embeddings  [B, seq, 1024]
    neutral_text_hs,
    positive_text_hs,
    negative_text_hs,
    # T5 attention masks  [B, seq]
    neutral_text_mask,
    positive_text_mask,
    negative_text_mask,
    # ERG null T5 embeddings (or None)
    null_text_hs,
    # Shared conditioning
    speaker_embds,
    lyric_token_ids,
    lyric_mask,
    # Latents & generators
    latents,
    random_generators,
    # FreeSliders params
    split_step: int,
    alpha: float,
    target_layers: list[str] | None = None,
    # Generation params
    audio_duration: float = 30.0,
    infer_steps: int = 30,
    guidance_scale: float = 5.0,
    omega_scale: float = 10.0,
    scheduler_type: str = "euler",
    cfg_type: str = "apg",
    guidance_interval: float = 1.0,
    guidance_interval_decay: float = 0.0,
    min_guidance_scale: float = 0.0,
    use_erg_lyric: bool = True,
    use_erg_diffusion: bool = True,
    guidance_scale_text: float = 0.0,
    guidance_scale_lyric: float = 0.0,
):
    """Run the FreeSliders diffusion loop and return the final latents.

    Replicates the neutral-prompt CFG handling from
    ``text2music_diffusion_process`` and adds the pos/neg passes + FreeSliders
    combination at every step past ``split_step``.
    """
    device = pipe.device
    dtype = pipe.dtype
    transformer = pipe.ace_step_transformer

    if next(transformer.parameters()).device != torch.device(device):
        transformer.to(device)

    bsz = latents.shape[0]

    do_cfg = guidance_scale > 1.0
    do_double_cfg = (
        guidance_scale_text is not None
        and guidance_scale_text > 1.0
        and guidance_scale_lyric is not None
        and guidance_scale_lyric > 1.0
    )

    if scheduler_type == "euler":
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    elif scheduler_type == "heun":
        from acestep.schedulers.scheduling_flow_match_heun_discrete import (
            FlowMatchHeunDiscreteScheduler,
        )

        scheduler = FlowMatchHeunDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    elif scheduler_type == "pingpong":
        from acestep.schedulers.scheduling_flow_match_pingpong import (
            FlowMatchPingPongScheduler,
        )

        scheduler = FlowMatchPingPongScheduler(num_train_timesteps=1000, shift=3.0)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    frame_length = math.ceil(audio_duration * 44100 / 512 / 8)
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps=infer_steps, device=device
    )

    target_latents = latents.to(device)
    attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)

    enc_hs_neutral, enc_mask = transformer.encode(
        neutral_text_hs,
        neutral_text_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
    )

    enc_hs_positive, _ = transformer.encode(
        positive_text_hs,
        positive_text_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
    )

    enc_hs_negative, _ = transformer.encode(
        negative_text_hs,
        negative_text_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
    )

    if use_erg_lyric:
        enc_hs_null, _ = _encode_with_erg_lyric(
            pipe,
            inputs={
                "encoder_text_hidden_states": (
                    null_text_hs if null_text_hs is not None else torch.zeros_like(neutral_text_hs)
                ),
                "text_attention_mask": neutral_text_mask,
                "speaker_embeds": torch.zeros_like(speaker_embds),
                "lyric_token_idx": lyric_token_ids,
                "lyric_mask": lyric_mask,
            },
        )
    else:
        enc_hs_null, _ = transformer.encode(
            torch.zeros_like(neutral_text_hs),
            neutral_text_mask,
            torch.zeros_like(speaker_embds),
            torch.zeros_like(lyric_token_ids),
            lyric_mask,
        )

    enc_hs_no_lyric = None
    if do_double_cfg:
        if use_erg_lyric:
            enc_hs_no_lyric, _ = _encode_with_erg_lyric(
                pipe,
                inputs={
                    "encoder_text_hidden_states": neutral_text_hs,
                    "text_attention_mask": neutral_text_mask,
                    "speaker_embeds": torch.zeros_like(speaker_embds),
                    "lyric_token_idx": lyric_token_ids,
                    "lyric_mask": lyric_mask,
                },
            )
        else:
            enc_hs_no_lyric, _ = transformer.encode(
                neutral_text_hs,
                neutral_text_mask,
                torch.zeros_like(speaker_embds),
                torch.zeros_like(lyric_token_ids),
                lyric_mask,
            )

    start_idx = int(num_inference_steps * ((1 - guidance_interval) / 2))
    end_idx = int(num_inference_steps * (guidance_interval / 2 + 0.5))
    momentum_buffer = MomentumBuffer()

    for i, t in enumerate(timesteps):
        latent_input = target_latents
        timestep = t.expand(bsz)
        output_length = latent_input.shape[-1]

        is_guided = start_idx <= i < end_idx and do_cfg
        is_stage2 = i >= split_step

        if is_guided:
            if guidance_interval_decay > 0 and (end_idx - start_idx) > 1:
                progress = (i - start_idx) / (end_idx - start_idx - 1)
                current_gs = guidance_scale - (
                    (guidance_scale - min_guidance_scale) * progress * guidance_interval_decay
                )
            else:
                current_gs = guidance_scale

            noise_cond = transformer.decode(
                hidden_states=latent_input,
                attention_mask=attention_mask,
                encoder_hidden_states=enc_hs_neutral,
                encoder_hidden_mask=enc_mask,
                output_length=output_length,
                timestep=timestep,
            ).sample

            noise_text_only = None
            if do_double_cfg and enc_hs_no_lyric is not None:
                noise_text_only = transformer.decode(
                    hidden_states=latent_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=enc_hs_no_lyric,
                    encoder_hidden_mask=enc_mask,
                    output_length=output_length,
                    timestep=timestep,
                ).sample

            if use_erg_diffusion:
                noise_uncond = _decode_with_erg_diffusion(
                    pipe,
                    hidden_states=latent_input,
                    timestep=timestep,
                    inputs={
                        "encoder_hidden_states": enc_hs_null,
                        "encoder_hidden_mask": enc_mask,
                        "output_length": output_length,
                        "attention_mask": attention_mask,
                    },
                )
            else:
                noise_uncond = transformer.decode(
                    hidden_states=latent_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=enc_hs_null,
                    encoder_hidden_mask=enc_mask,
                    output_length=output_length,
                    timestep=timestep,
                ).sample

            if do_double_cfg and noise_text_only is not None:
                noise_neutral = cfg_double_condition_forward(
                    cond_output=noise_cond,
                    uncond_output=noise_uncond,
                    only_text_cond_output=noise_text_only,
                    guidance_scale_text=guidance_scale_text,
                    guidance_scale_lyric=guidance_scale_lyric,
                )
            elif cfg_type == "apg":
                noise_neutral = apg_forward(
                    pred_cond=noise_cond,
                    pred_uncond=noise_uncond,
                    guidance_scale=current_gs,
                    momentum_buffer=momentum_buffer,
                )
            elif cfg_type == "cfg":
                noise_neutral = cfg_forward(
                    cond_output=noise_cond,
                    uncond_output=noise_uncond,
                    cfg_strength=current_gs,
                )
            elif cfg_type == "cfg_star":
                noise_neutral = cfg_zero_star(
                    noise_pred_with_cond=noise_cond,
                    noise_pred_uncond=noise_uncond,
                    guidance_scale=current_gs,
                    i=i,
                    zero_steps=1,
                    use_zero_init=True,
                )
            else:
                raise ValueError(f"Unknown cfg_type: {cfg_type}")
        else:
            noise_neutral = transformer.decode(
                hidden_states=latent_input,
                attention_mask=attention_mask,
                encoder_hidden_states=enc_hs_neutral,
                encoder_hidden_mask=enc_mask,
                output_length=output_length,
                timestep=timestep,
            ).sample

        if is_stage2 and alpha != 0.0:
            decode_kwargs = dict(
                hidden_states=latent_input,
                attention_mask=attention_mask,
                encoder_hidden_mask=enc_mask,
                output_length=output_length,
                timestep=timestep,
            )

            if target_layers is not None and not is_all_layers(target_layers):
                with layer_specific_decode(transformer, target_layers, enc_hs_positive, enc_mask):
                    noise_pos = transformer.decode(
                        encoder_hidden_states=enc_hs_neutral, **decode_kwargs
                    ).sample
                with layer_specific_decode(transformer, target_layers, enc_hs_negative, enc_mask):
                    noise_neg = transformer.decode(
                        encoder_hidden_states=enc_hs_neutral, **decode_kwargs
                    ).sample
            else:
                noise_pos = transformer.decode(
                    encoder_hidden_states=enc_hs_positive, **decode_kwargs
                ).sample
                noise_neg = transformer.decode(
                    encoder_hidden_states=enc_hs_negative, **decode_kwargs
                ).sample

            noise_pred = noise_neutral + alpha * (noise_pos - noise_neg)
        else:
            noise_pred = noise_neutral

        target_latents = scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=target_latents,
            return_dict=False,
            omega=omega_scale,
            generator=(random_generators[0] if scheduler_type == "heun" else None),
        )[0]

    return target_latents
