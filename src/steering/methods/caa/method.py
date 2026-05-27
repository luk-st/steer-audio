"""CAA (Contrastive Activation Addition) as a unified :class:`Controller`.

Thin adapter — the underlying intervention logic still lives in
``src/models/ace_step/ace_steering/controller.py`` (``VectorStore``). This file
glues that into the universal interface so CAA composes with every other method::

    from src.steering import SteerableACEModel, CAASteeringController

    model = SteerableACEModel(device="cuda")
    ctrl = CAASteeringController.from_pretrained("user/ace-piano-caa", alpha=20)
    with model.steer(ctrl):
        audio = model.generate(prompt="instrumental music", seed=0)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

from src.steering import Controller
from src.steering.registry import register_method
from src.steering.hub import SteeringVectorArtifact

# Reuse the existing CAA implementation verbatim — no duplication.
from src.models.ace_step.ace_steering.controller import (
    VectorStore,
    compute_num_cfg_passes,
    register_vector_control,
)

SteerMode = Literal[
    "cond_only",
    "uncond_only",
    "uncond_for_cond",
    "separate",
    "both_cond",
    "both_uncond",
]


@register_method("caa")
class CAASteeringController(Controller):
    """Contrastive activation addition steering on ACE-Step cross-attention.

    Wraps :class:`VectorStore` and the existing block-forward patcher. The
    universal :meth:`mount` / :meth:`unmount` lifecycle takes care of restoring
    every patched ``LinearTransformerBlock.forward`` afterwards.

    """

    def __init__(
        self,
        steering_vectors: dict[Any, Any],
        alpha: float = 10.0,
        beta: float = 2.0,
        *,
        steer_back: bool = False,
        steer_mode: SteerMode = "cond_only",
        save_only_cond: bool = True,
        num_cfg_passes: int | None = None,
        renorm_after_steer: bool = False,
        start_step: int = 0,
        target_layers: "str | list[str]" = "all",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        from src.steering.methods.freesliders.layer_hooks import parse_layers

        self.steering_vectors = steering_vectors
        self.alpha = alpha
        self.beta = beta
        self.steer_back = steer_back
        self.steer_mode = steer_mode
        self.save_only_cond = save_only_cond
        self.num_cfg_passes = num_cfg_passes
        self.renorm_after_steer = renorm_after_steer
        self.start_step = start_step
        # Accept either a preset string ("all", "tf6tf7", ...) or an explicit
        # list of block names (e.g. ["tf6", "tf7"]). Resolve to a list once.
        self.target_layers = parse_layers(target_layers)
        self.device = device
        self._vector_store: VectorStore | None = None

    # --- factory ------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        alpha: float = 10.0,
        renorm: bool | None = None,
        **kwargs: Any,
    ) -> "CAASteeringController":
        """Load steering vectors from the Hub (or local path) and build a controller.

        Recognised layout (matches ``steering/caa/compute_sv_caa.py``)::

            <repo>/
              config.json   — generation config the SVs were computed under
              sv.pkl        — the actual steering-vector dict

        ``renorm`` is an alias for ``renorm_after_steer``.
        """
        if renorm is not None:
            kwargs.setdefault("renorm_after_steer", renorm)
        artifact = SteeringVectorArtifact.from_pretrained(repo_id)
        sv_path = artifact.root / "sv.pkl"
        with sv_path.open("rb") as f:
            sv = pickle.load(f)
        config = artifact.config

        # Derive num_cfg_passes from the config if available.
        ncfg = config.get("num_cfg_passes")
        if ncfg is None:
            ncfg = compute_num_cfg_passes(
                guidance_scale_text=config.get("guidance_scale_text", 0.0),
                guidance_scale_lyric=config.get("guidance_scale_lyric", 0.0),
            )

        return cls(
            steering_vectors=sv,
            alpha=alpha,
            num_cfg_passes=ncfg,
            **kwargs,
        )

    # --- lifecycle ----------------------------------------------------------

    def mount(self, model: Any) -> None:
        """Patch every ``LinearTransformerBlock.forward`` to call our VectorStore."""
        self._vector_store = VectorStore(
            steering_vectors=self.steering_vectors,
            steer=True,
            alpha=self.alpha,
            beta=self.beta,
            steer_back=self.steer_back,
            device=self.device,
            save_only_cond=self.save_only_cond,
            steer_mode=self.steer_mode,
            num_cfg_passes=self.num_cfg_passes,
            renorm_after_steer=self.renorm_after_steer,
            start_step=self.start_step,
        )

        mount_vector_control(
            self,
            model,
            self._vector_store,
            explicit_layers=self.target_layers,
        )
        self._is_mounted = True

    def reset(self) -> None:
        if self._vector_store is not None:
            self._vector_store.reset()

    def set_alpha(self, alpha: float) -> None:
        """Update steering strength. Effective for the next generation."""
        self.alpha = alpha
        if self._vector_store is not None:
            self._vector_store.alpha = alpha


def _iter_linear_transformer_blocks(net):
    """Yield every ``LinearTransformerBlock`` reachable under ``net``."""
    if type(net).__name__ == "LinearTransformerBlock":
        yield net
        return
    if hasattr(net, "children"):
        for child in net.children():
            yield from _iter_linear_transformer_blocks(child)


def mount_vector_control(
    controller: Controller,
    transformer: Any,
    vector_control: Any,
    explicit_layers: list[str] | None = None,
) -> None:
    """Snapshot every block's ``forward`` then run ``register_vector_control``.

    Shared helper so every method built on top of ``VectorControl`` (CAA,
    AUSteer, ...) gets the same correct mount/unmount behaviour without
    duplicating the traversal logic.

    Records ``_CLASS_FALLBACK`` for blocks whose ``forward`` was the class
    method (the common case): on unmount we ``delattr`` the instance override
    rather than restoring a stale bound method.
    """
    for _, block in transformer.transformer_blocks.named_children():
        for sub in _iter_linear_transformer_blocks(block):
            if "forward" in sub.__dict__:
                controller._patches.append((sub, "forward", sub.__dict__["forward"]))
            else:
                controller._patches.append((sub, "forward", Controller._CLASS_FALLBACK))
    register_vector_control(
        transformer,
        vector_control,
        verbose=False,
        explicit_layers=explicit_layers,
    )
