"""Prompt-Injection (PI / PCI) steering as a unified :class:`Controller`.

Training-free; ``mount()`` is a no-op. Generation switches the prompt
embedding from neutral to positive (or negative) for the last ``|alpha|``
steps of diffusion. ``alpha`` is the **switch length** in steps (signed):

* ``alpha > 0``: neutral for the first ``N - alpha`` steps, positive after.
* ``alpha < 0``: neutral for the first ``N - |alpha|`` steps, negative after.
* ``alpha = 0``: neutral throughout (baseline).
"""

from __future__ import annotations

from typing import Any

import torch

from src.steering import Controller
from src.steering.registry import register_method
from src.steering.methods.freesliders.layer_hooks import (
    encode_null_conditioning,
    is_all_layers,
    layer_specific_replace_embeds,
    parse_layers,
)


@register_method("pci")
class PCISteeringController(Controller):
    """Prompt-injection / hard-switch steering."""

    def __init__(
        self,
        concept: str,
        alpha: int = 5,
        target_layers: "str | list[str]" = "all",
        *,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        neutral_prompt: str | None = None,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        cfg_type: str = "apg",
        scheduler_type: str = "euler",
        omega_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.concept = concept
        self.alpha = int(alpha)
        self.target_layers = target_layers
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.neutral_prompt = neutral_prompt
        self.use_erg_tag = use_erg_tag
        self.use_erg_lyric = use_erg_lyric
        self.use_erg_diffusion = use_erg_diffusion
        self.cfg_type = cfg_type
        self.scheduler_type = scheduler_type
        self.omega_scale = omega_scale

    @classmethod
    def from_pretrained(cls, repo_id: str = "", alpha: float = 5.0, **kwargs: Any) -> "PCISteeringController":
        """PI is training-free — ``repo_id`` is accepted-but-ignored."""
        return cls(alpha=int(alpha), **kwargs)

    def mount(self, model: Any) -> None:
        self._is_mounted = True

    def set_alpha(self, alpha: float) -> None:
        self.alpha = int(alpha)

    def generate(self, model: Any, **kwargs: Any) -> Any:
        from src.steering.methods.pci.core import build_prompt_triple, compute_batched_pi_embeddings
        from src.steering.methods.freesliders.core import prepare_lyrics

        pipe = model.pipeline

        prompt = kwargs.get("prompt")
        if prompt is None:
            raise ValueError("PCI generate requires a ``prompt`` kwarg.")
        prompts: list[str] = [prompt] if isinstance(prompt, str) else list(prompt)

        lyrics = kwargs.get("lyrics", "[inst]")
        lyrics_list: list[str] = [lyrics] * len(prompts) if isinstance(lyrics, str) else list(lyrics)
        if len(lyrics_list) == 1 and len(prompts) > 1:
            lyrics_list = lyrics_list * len(prompts)

        audio_duration = float(kwargs.get("audio_duration", 30.0))
        infer_step = int(kwargs.get("infer_step", 30))
        seed = int(kwargs.get("manual_seed", 42))
        guidance_scale = float(kwargs.get("guidance_scale", 5.0))

        neutral_prompts, positive_prompts, negative_prompts = [], [], []
        for p in prompts:
            if self.positive_prompt is not None and self.negative_prompt is not None and self.neutral_prompt is not None:
                neutral_prompts.append(self.neutral_prompt.format(p=p))
                positive_prompts.append(self.positive_prompt.format(p=p))
                negative_prompts.append(self.negative_prompt.format(p=p))
            else:
                n, pos, neg = build_prompt_triple(p, self.concept)
                neutral_prompts.append(n)
                positive_prompts.append(pos)
                negative_prompts.append(neg)

        emb = compute_batched_pi_embeddings(
            pipe, neutral_prompts, positive_prompts, negative_prompts,
            use_erg_tag=self.use_erg_tag,
        )
        lyric_token_ids, lyric_masks = prepare_lyrics(pipe, lyrics_list)
        latents = pipe.prepare_latents(len(prompts), audio_duration, seed)
        speaker_embds = torch.zeros(len(prompts), 512, device=pipe.device, dtype=pipe.dtype)
        random_generators, _ = pipe.set_seeds(len(prompts), seed)

        target_layers_list = parse_layers(self.target_layers)
        use_layer_hooks = not is_all_layers(target_layers_list)
        null_hs = emb.get("erg_neutral_hs")

        # Direction & switch step from signed alpha.
        if self.alpha > 0:
            target_hs = emb["positive_hs"]
            target_mask = emb["positive_masks"]
            switch_step = infer_step - self.alpha
        elif self.alpha < 0:
            target_hs = emb["negative_hs"]
            target_mask = emb["negative_masks"]
            switch_step = infer_step - abs(self.alpha)
        else:
            target_hs = None
            target_mask = None
            switch_step = None

        call_kwargs = {
            "encoder_text_hidden_states": emb["neutral_hs"],
            "text_attention_mask": emb["neutral_masks"],
            "speaker_embeds": speaker_embds,
            "lyric_token_idx": lyric_token_ids,
            "lyric_mask": lyric_masks,
            "random_generators": random_generators,
            "latents": latents,
            "encoder_text_hidden_states_null": null_hs,
            "retake_random_generators": None,
            "audio_duration": audio_duration,
            "oss_steps": [],
            "infer_step": infer_step,
            "guidance_scale": guidance_scale,
            "scheduler_type": self.scheduler_type,
            "cfg_type": self.cfg_type,
            "omega_scale": self.omega_scale,
            "guidance_interval": kwargs.get("guidance_interval", 1.0),
            "guidance_interval_decay": 0.0,
            "min_guidance_scale": 0.0,
            "use_erg_lyric": self.use_erg_lyric,
            "use_erg_diffusion": self.use_erg_diffusion,
            "guidance_scale_text": 0.0,
            "guidance_scale_lyric": 0.0,
        }
        if self.alpha != 0:
            call_kwargs["replace_embeds"] = True
            call_kwargs["replace_embeds_params"] = {
                "timestep_start": switch_step,
                "replace_text_embeds": target_hs,
                "replace_text_mask": target_mask,
                "replace_text_embeds_null": null_hs,
            }

        transformer = pipe.ace_step_transformer
        if use_layer_hooks and self.alpha != 0:
            with torch.no_grad():
                neutral_enc_hs, neutral_enc_mask = transformer.encode(
                    emb["neutral_hs"], emb["neutral_masks"],
                    speaker_embds, lyric_token_ids, lyric_masks,
                )
                null_enc_hs, null_enc_mask = encode_null_conditioning(
                    pipe, emb["neutral_masks"], speaker_embds,
                    lyric_token_ids, lyric_masks,
                    null_text_hs=null_hs, use_erg_lyric=self.use_erg_lyric,
                )
            with layer_specific_replace_embeds(
                transformer, target_layers_list,
                neutral_enc_hs, neutral_enc_mask,
                null_enc_hs, null_enc_mask,
            ):
                target_latents = pipe.task_text2music(**call_kwargs)
        else:
            target_latents = pipe.task_text2music(**call_kwargs)

        if kwargs.get("return_type", "audio") == "latent":
            return target_latents
        return pipe.decode_latents_to_audios(
            latents=target_latents,
            target_wav_duration_second=audio_duration,
        )
