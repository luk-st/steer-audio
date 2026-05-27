"""Token-Embedding (TokE) steering as a unified :class:`Controller`.

Loads a per-concept direction vector (``direction.pt`` produced by
:class:`TokEmbScorer`) and, at generation, adds ``alpha * direction`` to the
concept-token's hidden state in the neutral prompt embedding.
"""

from __future__ import annotations

from pathlib import Path
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
from src.steering.methods.tokemb.core import CONCEPT_NEUTRAL_CONFIG


@register_method("tokemb")
class TokEmbSteeringController(Controller):
    """Add a precomputed direction at the concept-token position.

    ``mount()`` is a no-op (no persistent activation hooks); the intervention
    happens inside :meth:`generate` via ``pipe.task_text2music(replace_embeds=True)``.
    """

    def __init__(
        self,
        concept: str,
        direction: torch.Tensor,
        alpha: float = 0.5,
        target_layers: "str | list[str]" = "all",
        te_split_step: int = 0,
        *,
        neutral_addon_template: str | None = None,
        target_token: str | None = None,
        target_token_occurrence: "str | None" = None,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        cfg_type: str = "apg",
        scheduler_type: str = "euler",
        omega_scale: float = 10.0,
    ) -> None:
        """Steering knobs:
            ``concept``: per-concept lookup key. Sets defaults for
                ``neutral_addon_template``, ``target_token``, and
                ``target_token_occurrence``.
            ``alpha``: scale applied to ``direction``.
            ``target_layers``: which transformer blocks see the modified
                embedding ("all" / "tf6tf7" / ["tf6", "tf7"]).
            ``te_split_step``: diffusion step at which to switch from the
                un-modified to the modified neutral embedding.

        Advanced overrides:
            ``neutral_addon_template``: format string with the literal
                ``{p}`` slot for the user prompt (e.g.
                ``"{p}, with grand piano"``). Defaults to
                ``CONCEPT_TO_NEUTRAL_ADDON[concept]``.
            ``target_token``: the word whose hidden-state position receives
                ``alpha * direction``. Defaults to
                ``CONCEPT_NEUTRAL_CONFIG[concept]["concept_word"]``. Must
                tokenise to a single token and must appear in
                ``neutral_addon_template.format(p=<prompt>)``.
            ``target_token_occurrence``: ``"first"`` or ``"last"`` — which match
                to take when the target word appears multiple times in the
                wrapped prompt. Defaults to
                ``CONCEPT_NEUTRAL_CONFIG[concept]["occurrence"]``.
        """
        super().__init__()
        self.concept = concept
        self.direction = direction
        self.alpha = alpha
        self.target_layers = target_layers
        self.te_split_step = te_split_step
        self.neutral_addon_template = neutral_addon_template
        self.target_token = target_token
        self.target_token_occurrence = target_token_occurrence
        self.use_erg_tag = use_erg_tag
        self.use_erg_lyric = use_erg_lyric
        self.use_erg_diffusion = use_erg_diffusion
        self.cfg_type = cfg_type
        self.scheduler_type = scheduler_type
        self.omega_scale = omega_scale

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        alpha: float = 0.5,
        concept: str | None = None,
        **kwargs: Any,
    ) -> "TokEmbSteeringController":
        """Load a direction tensor from ``<repo_id>/<concept>_direction.pt`` or
        from a direct ``.pt`` path. ``concept`` defaults to the value stored
        inside the artifact."""
        path = Path(repo_id)
        if path.is_file():
            direction_path = path
        elif path.is_dir():
            if concept is None:
                raise ValueError(
                    'For directory artifacts, pass --method-kwargs \'{"concept":"piano"}\''
                )
            direction_path = path / f"{concept}_direction.pt"
            if not direction_path.exists():
                raise FileNotFoundError(direction_path)
        else:
            from huggingface_hub import snapshot_download

            local = Path(
                snapshot_download(
                    repo_id=repo_id, allow_patterns=["*.pt", "*.json", "*.md"]
                )
            )
            if concept is None:
                pts = list(local.glob("*_direction.pt"))
                if len(pts) != 1:
                    raise ValueError(
                        f"{repo_id} holds {len(pts)} direction files; pass --method-kwargs "
                        f'\'{{"concept":"..."}}\' to disambiguate.'
                    )
                direction_path = pts[0]
            else:
                direction_path = local / f"{concept}_direction.pt"

        data = torch.load(direction_path, map_location="cpu")
        direction = data["direction"]
        resolved_concept = concept or data.get("concept")
        if resolved_concept is None:
            raise ValueError(
                f"Direction file {direction_path} has no 'concept' field; pass --method-kwargs "
                f'\'{{"concept":"..."}}\'.'
            )
        return cls(
            concept=resolved_concept,
            direction=direction,
            alpha=alpha,
            **kwargs,
        )

    def mount(self, model: Any) -> None:
        self._is_mounted = True

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha

    def generate(self, model: Any, **kwargs: Any) -> Any:
        from src.steering.methods.caa.utils.constants import CONCEPT_TO_NEUTRAL_ADDON
        from src.steering.methods.freesliders.core import prepare_lyrics
        from src.steering.methods.tokemb.core import compute_batched_neutral_embeddings

        pipe = model.pipeline

        prompt = kwargs.get("prompt")
        if prompt is None:
            raise ValueError("TokEmb generate requires a ``prompt`` kwarg.")
        prompts: list[str] = [prompt] if isinstance(prompt, str) else list(prompt)

        lyrics = kwargs.get("lyrics", "[inst]")
        lyrics_list: list[str] = (
            [lyrics] * len(prompts) if isinstance(lyrics, str) else list(lyrics)
        )
        if len(lyrics_list) == 1 and len(prompts) > 1:
            lyrics_list = lyrics_list * len(prompts)

        audio_duration = float(kwargs.get("audio_duration", 30.0))
        infer_step = int(kwargs.get("infer_step", 30))
        seed = int(kwargs.get("manual_seed", 42))
        guidance_scale = float(kwargs.get("guidance_scale", 5.0))

        # Augment each prompt with the concept-specific neutral addon.
        addon = self.neutral_addon_template or CONCEPT_TO_NEUTRAL_ADDON[self.concept]
        neutral_prompts = [addon.format(p=p) for p in prompts]

        import warnings as _warnings
        from src.steering import PromptRewriteWarning

        _warnings.warn(
            f"TokEmbSteeringController augments the input prompt with a concept-specific "
            f"neutral template for concept={self.concept!r}. "
            f"Preview for prompt[0]={prompts[0]!r}:\n"
            f"  neutral → {neutral_prompts[0]!r}\n"
            f"Override via neutral_addon_template on the controller.",
            category=PromptRewriteWarning,
            stacklevel=2,
        )

        cfg = CONCEPT_NEUTRAL_CONFIG.get(self.concept, {})
        target_token = self.target_token or cfg.get("concept_word")
        target_token_occurrence = self.target_token_occurrence or cfg.get(
            "occurrence", "last"
        )
        if target_token is None:
            raise ValueError(
                f"TokEmb has no default target_token for concept={self.concept!r}. "
                f"Pass target_token=<word> (must tokenise to a single token and appear "
                f"in the wrapped neutral prompt)."
            )
        emb = compute_batched_neutral_embeddings(
            pipe,
            neutral_prompts,
            concept_word=target_token,
            occurrence=target_token_occurrence,
            use_erg_tag=self.use_erg_tag,
        )
        lyric_token_ids, lyric_masks = prepare_lyrics(pipe, lyrics_list)
        latents = pipe.prepare_latents(len(prompts), audio_duration, seed)
        speaker_embds = torch.zeros(
            len(prompts), 512, device=pipe.device, dtype=pipe.dtype
        )
        random_generators, _ = pipe.set_seeds(len(prompts), seed)

        direction = self.direction.to(pipe.device, dtype=emb["neutral_hs"].dtype)
        target_layers_list = parse_layers(self.target_layers)
        use_layer_hooks = not is_all_layers(target_layers_list)
        null_hs = emb.get("erg_neutral_hs")

        # Add alpha * direction at the concept-token position.
        mod_hs = emb["neutral_hs"].clone()
        batch_idx = torch.arange(len(prompts), device=pipe.device)
        mod_hs[batch_idx, emb["seq_ids_in_prompts"], :] += self.alpha * direction
        mod_mask = emb["neutral_masks"]

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
        if self.alpha != 0.0:
            call_kwargs["replace_embeds"] = True
            call_kwargs["replace_embeds_params"] = {
                "timestep_start": self.te_split_step,
                "replace_text_embeds": mod_hs,
                "replace_text_mask": mod_mask,
                "replace_text_embeds_null": null_hs,
            }

        transformer = pipe.ace_step_transformer
        if use_layer_hooks and self.alpha != 0.0:
            with torch.no_grad():
                neutral_enc_hs, neutral_enc_mask = transformer.encode(
                    emb["neutral_hs"],
                    emb["neutral_masks"],
                    speaker_embds,
                    lyric_token_ids,
                    lyric_masks,
                )
                null_enc_hs, null_enc_mask = encode_null_conditioning(
                    pipe,
                    emb["neutral_masks"],
                    speaker_embds,
                    lyric_token_ids,
                    lyric_masks,
                    null_text_hs=null_hs,
                    use_erg_lyric=self.use_erg_lyric,
                )
            with layer_specific_replace_embeds(
                transformer,
                target_layers_list,
                neutral_enc_hs,
                neutral_enc_mask,
                null_enc_hs,
                null_enc_mask,
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
