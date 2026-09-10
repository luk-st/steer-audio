"""FreeSliders steering as a unified :class:`Controller`.

Unlike CAA / SAE / AUSteer / ConceptSlider (which all intervene at the
activation level), FreeSliders rewrites the **noise-prediction loop**: at every
diffusion step past ``split_step`` it blends three forward passes
(``neutral_CFG + alpha * (positive - negative)``). That can't be expressed as a
plain forward hook, so :meth:`mount` is a no-op and :meth:`generate` overrides
the default pipeline call.

The heavy lifting (T5 embedding, 3-pass diffusion loop, VAE decode) reuses the
helpers already factored out in ``run_freesliders.py``. No reimplementation —
both the legacy runner and this Controller share the same core functions.
"""

from __future__ import annotations

from typing import Any

import torch

from src.steering.methods.freesliders.layer_hooks import parse_layers
from src.steering import Controller
from src.steering.registry import register_method


@register_method("freesliders")
class FreeSlidersSteeringController(Controller):
    """FreeSliders 3-pass noise-blending steering."""

    def __init__(
        self,
        concept: str,
        alpha: float = 1.0,
        target_layers: "str | list[str]" = "all",
        split_step: int = 5,
        *,
        positive_prompt_suffix: str | None = None,
        negative_prompt_suffix: str | None = None,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        cfg_type: str = "apg",
        scheduler_type: str = "euler",
        omega_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.concept = concept
        self.alpha = alpha
        self.target_layers = target_layers  # e.g. "all" / "tf6tf7"
        self.split_step = split_step
        self.positive_prompt_suffix = positive_prompt_suffix
        self.negative_prompt_suffix = negative_prompt_suffix
        self.use_erg_tag = use_erg_tag
        self.use_erg_lyric = use_erg_lyric
        self.use_erg_diffusion = use_erg_diffusion
        self.cfg_type = cfg_type
        self.scheduler_type = scheduler_type
        self.omega_scale = omega_scale

    @classmethod
    def from_pretrained(cls, repo_id: str = "", alpha: float = 1.0, **kwargs: Any) -> "FreeSlidersSteeringController":
        """FreeSliders is training-free — no Hub artifacts to download.

        ``repo_id`` is ignored. The "config" is just the concept name plus
        optional prompt suffixes; pass them via ``kwargs``.
        """
        return cls(alpha=alpha, **kwargs)

    def mount(self, model: Any) -> None:
        # No persistent hooks; layer overrides are installed transiently inside
        # the diffusion loop by ``layer_specific_decode``.
        self._is_mounted = True

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Generation override
    # ------------------------------------------------------------------

    def generate(self, model: Any, **kwargs: Any) -> Any:
        """Run a FreeSliders generation.

        Accepts the same kwargs as ``pipe.generate``: ``prompt`` (str or list),
        ``lyrics``, ``audio_duration``, ``infer_step``, ``manual_seed``,
        ``guidance_scale``, and (optionally) ``return_type``. Builds prompt
        triples from ``self.concept`` and dispatches to the existing core.
        """
        # Late import — pulls in heavy ACE-Step deps lazily.
        from src.steering.methods.freesliders.core import (
            build_prompt_triple,
            compute_batched_fs_embeddings,
            freesliders_diffusion,
            prepare_lyrics,
        )

        pipe = model.pipeline

        prompt = kwargs.get("prompt")
        if prompt is None:
            raise ValueError("FreeSliders generate requires a ``prompt`` kwarg.")
        prompts: list[str] = [prompt] if isinstance(prompt, str) else list(prompt)

        lyrics = kwargs.get("lyrics", "[inst]")
        lyrics_list: list[str] = [lyrics] * len(prompts) if isinstance(lyrics, str) else list(lyrics)
        if len(lyrics_list) == 1 and len(prompts) > 1:
            lyrics_list = lyrics_list * len(prompts)

        audio_duration = float(kwargs.get("audio_duration", 30.0))
        infer_step = int(kwargs.get("infer_step", 30))
        seed = int(kwargs.get("manual_seed", 42))
        guidance_scale = float(kwargs.get("guidance_scale", 5.0))
        return_type = kwargs.get("return_type", "audio")

        # Build (neutral, positive, negative) per prompt.
        triples = [self._build_triple(p, build_prompt_triple) for p in prompts]
        neutral_prompts = [t[0] for t in triples]
        positive_prompts = [t[1] for t in triples]
        negative_prompts = [t[2] for t in triples]

        # T5 embeddings, lyrics, latents, speakers.
        emb = compute_batched_fs_embeddings(
            pipe,
            neutral_prompts,
            positive_prompts,
            negative_prompts,
            use_erg_tag=self.use_erg_tag,
        )
        lyric_token_ids, lyric_masks = prepare_lyrics(pipe, lyrics_list)
        latents = pipe.prepare_latents(len(prompts), audio_duration, seed)
        speaker_embds = torch.zeros(len(prompts), 512, device=pipe.device, dtype=pipe.dtype)
        random_generators, _ = pipe.set_seeds(len(prompts), seed)

        # The diffusion loop. Returns final latents.
        target_layers = parse_layers(self.target_layers)
        target_latents = freesliders_diffusion(
            pipe,
            neutral_text_hs=emb["neutral_hs"],
            positive_text_hs=emb["positive_hs"],
            negative_text_hs=emb["negative_hs"],
            neutral_text_mask=emb["neutral_masks"],
            positive_text_mask=emb["positive_masks"],
            negative_text_mask=emb["negative_masks"],
            null_text_hs=emb.get("erg_neutral_hs"),
            speaker_embds=speaker_embds,
            lyric_token_ids=lyric_token_ids,
            lyric_mask=lyric_masks,
            latents=latents,
            random_generators=random_generators,
            split_step=self.split_step,
            alpha=self.alpha,
            target_layers=target_layers,
            audio_duration=audio_duration,
            infer_steps=infer_step,
            guidance_scale=guidance_scale,
            omega_scale=self.omega_scale,
            scheduler_type=self.scheduler_type,
            cfg_type=self.cfg_type,
            use_erg_lyric=self.use_erg_lyric,
            use_erg_diffusion=self.use_erg_diffusion,
        )

        if return_type == "latent":
            return target_latents
        return pipe.decode_latents_to_audios(
            latents=target_latents,
            target_wav_duration_second=audio_duration,
        )

    # ------------------------------------------------------------------

    def _build_triple(self, raw_prompt: str, builder) -> tuple[str, str, str]:
        """Return ``(neutral, positive, negative)`` for ``raw_prompt``.

        If both ``positive_prompt_suffix`` and ``negative_prompt_suffix`` are
        set, use them directly. Otherwise fall back to the per-concept
        ``build_prompt_triple`` table, overriding the neutral with the
        concept-specific ``CONCEPT_TO_NEUTRAL_ADDON`` (matches the paper:
        positive/negative are built from the raw prompt, neutral is the addon).
        """
        if self.positive_prompt_suffix is not None and self.negative_prompt_suffix is not None:
            return (
                raw_prompt,
                f"{raw_prompt}{self.positive_prompt_suffix}",
                f"{raw_prompt}{self.negative_prompt_suffix}",
            )
        from src.steering.methods.caa.utils.constants import CONCEPT_TO_NEUTRAL_ADDON

        _neutral, pos, neg = builder(raw_prompt, self.concept)
        neutral = CONCEPT_TO_NEUTRAL_ADDON.get(self.concept, "{p}").format(p=raw_prompt)
        return neutral, pos, neg
