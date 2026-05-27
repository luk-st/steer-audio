"""AUSteer steering as a unified :class:`Controller`.

Wraps the existing ``AUSteerSteeringController`` (a ``VectorControl`` subclass) so it
mounts on the model through the same surface as CAA / SAE.
"""

from __future__ import annotations

import pickle
from typing import Any, Literal

from src.steering import Controller
from src.steering.registry import register_method
from src.steering.hub import SteeringVectorArtifact

from src.models.ace_step.ace_steering.controller import compute_num_cfg_passes
from src.steering.methods.caa.method import mount_vector_control
from src.steering.methods.caa.utils.controllers import (
    AUSteerSteeringController as _AUSteerVC,
)


@register_method("austeer")
class AUSteerSteeringController(Controller):
    """AUSteer-style sparse steering on ACE-Step cross-attention."""

    def __init__(
        self,
        austeer_vectors: dict[Any, Any],
        k: int = 256,
        alpha: float = 10.0,
        active_layers: list[str] | None = None,
        mode: Literal["additive", "multiplicative"] = "additive",
        *,
        steer_mode: Literal["cond_only", "both_cond"] = "cond_only",
        num_cfg_passes: int = 2,
        renorm: bool = False,
        start_step: int = 0,
        target_layers: "str | list[str]" = "all",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        from src.steering.methods.freesliders.layer_hooks import parse_layers

        self.austeer_vectors = austeer_vectors
        self.k = k
        self.alpha = alpha
        # `active_layers` is artifact metadata (which layers were collected
        # during compute), set by from_pretrained — distinct from
        # `target_layers` which is the user's inference-time choice.
        self.active_layers = active_layers
        self.mode = mode
        self.steer_mode = steer_mode
        self.num_cfg_passes = num_cfg_passes
        self.renorm = renorm
        self.start_step = start_step
        self.target_layers = parse_layers(target_layers)
        self.device = device
        self._vc: _AUSteerVC | None = None

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        alpha: float = 10.0,
        k: int = 256,
        **kwargs: Any,
    ) -> "AUSteerSteeringController":
        artifact = SteeringVectorArtifact.from_pretrained(repo_id)
        with (artifact.root / "austeer.pkl").open("rb") as f:
            austeer = pickle.load(f)
        config = artifact.config
        ncfg = config.get("num_cfg_passes") or compute_num_cfg_passes(
            guidance_scale_text=config.get("guidance_scale_text", 0.0),
            guidance_scale_lyric=config.get("guidance_scale_lyric", 0.0),
        )
        return cls(
            austeer_vectors=austeer,
            alpha=alpha,
            k=k,
            num_cfg_passes=ncfg,
            active_layers=config.get("layers_collected"),
            **kwargs,
        )

    def mount(self, model: Any) -> None:
        self._vc = _AUSteerVC(
            austeer_vectors=self.austeer_vectors,
            k=self.k,
            alpha=self.alpha,
            active_layers=self.active_layers,
            mode=self.mode,
            device=self.device,
            steer=True,
            steer_mode=self.steer_mode,
            num_cfg_passes=self.num_cfg_passes,
            renorm=self.renorm,
            start_step=self.start_step,
        )
        mount_vector_control(self, model, self._vc, explicit_layers=self.target_layers)
        self._is_mounted = True

    def reset(self) -> None:
        if self._vc is not None:
            self._vc.reset()

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        if self._vc is not None:
            self._vc.alpha = alpha
