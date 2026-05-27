"""AudioLDM2 CAA steering as a unified :class:`Controller`.

Wraps the existing ``VectorStoreAudioLDM`` (forward-hook collector/applier on
``attn2`` outputs in the AudioLDM2 UNet) so AudioLDM2 CAA composes with
:class:`SteerableAudioLDMModel` through the same surface as every ACE-Step method.

Key difference from CAA-on-ACE-Step: AudioLDM2 does CFG by batching the
conditional/unconditional passes together (``[uncond..., cond...]``), so the
controller slices the batch dimension instead of tracking a CFG-pass counter.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

from src.steering import Controller
from src.steering.registry import register_method
from src.steering.hub import SteeringVectorArtifact

from src.models.audioldm.audioldm_steering.controller import (
    VectorStoreAudioLDM,
    register_vector_control_audioldm,
    resolve_audioldm_layers,
)

SteerMode = Literal[
    "cond_only",
    "uncond_only",
    "uncond_for_cond",
    "separate",
    "both_cond",
    "both_uncond",
]


@register_method("audioldm_caa")
class AudioLDMCAASteeringController(Controller):
    """CAA steering on AudioLDM2 ``attn2`` cross-attention outputs."""

    def __init__(
        self,
        steering_vectors: dict[Any, Any],
        alpha: float = 0.0,
        beta: float = 2.0,
        *,
        steer_back: bool = False,
        steer_mode: SteerMode = "cond_only",
        save_only_cond: bool = True,
        normalize_sv_at_apply: bool = False,
        renorm_after_steer: bool = False,
        target_layers: "str | list[str]" = "all",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.steering_vectors = steering_vectors
        self.alpha = alpha
        self.beta = beta
        self.steer_back = steer_back
        self.steer_mode = steer_mode
        self.save_only_cond = save_only_cond
        self.normalize_sv_at_apply = normalize_sv_at_apply
        self.renorm_after_steer = renorm_after_steer
        # Accept either an AudioLDM2 layer preset name or an explicit list of
        # module paths. resolve_audioldm_layers() handles both.
        self.target_layers = target_layers
        self.device = device
        self._vector_store: VectorStoreAudioLDM | None = None

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        alpha: float = 0.0,
        **kwargs: Any,
    ) -> "AudioLDMCAASteeringController":
        """Load AudioLDM2 CAA steering vectors from disk or the Hub.

        Recognised layout matches ``compute_sv_caa_audioldm.py``::

            <repo>/
              config.json   — generation config + ``layers_preset`` + ``layers_to_steer``
              sv.pkl        — the steering-vector dict
        """
        artifact = SteeringVectorArtifact.from_pretrained(repo_id)
        with (artifact.root / "sv.pkl").open("rb") as f:
            sv = pickle.load(f)
        config = artifact.config

        # Default to whatever layer preset was used at compute time.
        target_layers = kwargs.pop("target_layers", config.get("layers_preset", "all"))
        return cls(
            steering_vectors=sv, alpha=alpha, target_layers=target_layers, **kwargs
        )

    def mount(self, model: Any) -> None:
        """``model`` is a UNet — i.e. ``SteerableAudioLDMModel.transformer``."""
        layer_names = resolve_audioldm_layers(self.target_layers)
        self._vector_store = VectorStoreAudioLDM(
            steering_vectors=self.steering_vectors,
            steer=True,
            alpha=self.alpha,
            beta=self.beta,
            steer_back=self.steer_back,
            steer_mode=self.steer_mode,
            device=self.device,
            save_only_cond=self.save_only_cond,
            normalize_sv_at_apply=self.normalize_sv_at_apply,
            renorm_after_steer=self.renorm_after_steer,
        )
        handles = register_vector_control_audioldm(
            model,
            self._vector_store,
            layer_names,
            verbose=False,
        )
        self._handles.extend(handles)
        self._is_mounted = True

    def reset(self) -> None:
        if self._vector_store is not None:
            self._vector_store.reset()

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        if self._vector_store is not None:
            self._vector_store.alpha = alpha

    def generate(self, model: Any, **kwargs: Any) -> Any:
        """Run a single AudioLDM2 generation.

        AudioLDM2 CFG batches uncond+cond together; the controller needs to know
        whether CFG is active so it slices the batch correctly.
        """
        import torch

        if self._vector_store is None:
            raise RuntimeError(
                "AudioLDMCAASteeringController.generate called before mount."
            )

        guidance_scale = float(kwargs.get("guidance_scale", 3.5))
        self._vector_store.do_cfg = guidance_scale > 1.0
        self._vector_store.reset()

        seed = int(kwargs.pop("seed", kwargs.pop("manual_seed", 42)))
        generator = torch.Generator(model.device).manual_seed(seed)

        call_kwargs = {
            "prompt": kwargs["prompt"],
            "num_inference_steps": int(
                kwargs.get("num_inference_steps", kwargs.get("infer_step", 30))
            ),
            "guidance_scale": guidance_scale,
            "audio_length_in_s": float(
                kwargs.get("audio_length_in_s", kwargs.get("audio_duration", 10.0))
            ),
            "generator": generator,
        }
        for opt in ("negative_prompt", "num_waveforms_per_prompt"):
            if opt in kwargs:
                call_kwargs[opt] = kwargs[opt]
        return torch.as_tensor(model.pipeline(**call_kwargs).audios)
