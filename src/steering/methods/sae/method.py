"""SAE-based steering as a unified :class:`Controller`.

Each hookpoint gets its own SAE (per-layer activations are *not* shared across
hookpoints — see ``memory/feedback_sae_per_layer_acts.md``). The wrapper accepts
either a single ``Sae`` + single hookpoint, or a dict ``{hookpoint: Sae}`` with a
matching dict of feature/multiplier configs.

Under the hood we reuse the existing forward hooks
(``ACEStepTimestepInterventionHook`` and ``ACEStepPrecomputedSteeringHook`` from
``src/steering/methods/sae/lib/hooked_model/acestep_hooks.py``) so the intervention math is
identical to the prior implementation — only the mount / cleanup interface is new.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from src.steering import Controller
from src.steering.registry import register_method

from src.steering.methods.sae.lib.hooked_model.acestep_hooks import (
    ACEStepPrecomputedSteeringHook,
    ACEStepTimestepInterventionHook,
)
from src.steering.methods.sae.lib.hooked_model.utils import locate_block
from src.steering.methods.sae.lib.sae.sae import Sae


InterventionMode = Literal["pre_topk", "post_topk", "inject", "steering_vector"]
SaeMode = Literal["sequence", "frequency"]


@dataclass
class LayerSpec:
    """SAE intervention config for one hookpoint.

    Use exactly one of (a) ``features_per_timestep`` for live SAE encode/decode
    interventions, or (b) ``steering_vectors`` for precomputed vector addition.
    """

    sae: Sae
    sae_mode: SaeMode = "sequence"
    # (a) live intervention
    features_per_timestep: dict[int, list[int]] | None = None
    multiplier: float | dict[int, float] = 1.0
    intervention_mode: InterventionMode = "pre_topk"
    add_error: bool = False
    # (b) precomputed steering vectors
    steering_vectors: dict[int, torch.Tensor] | None = None
    # shared
    uncond_preds: bool = False
    negate_for_uncond: bool = False
    start_step: int = 0

    def __post_init__(self) -> None:
        live = self.features_per_timestep is not None
        precomp = self.steering_vectors is not None
        if live == precomp:
            raise ValueError(
                "LayerSpec needs exactly one of features_per_timestep or "
                "steering_vectors set."
            )


def load_features_from_score_cache(
    scores_path: str,
    *,
    score_filename: str,
    selection_method: str = "tfidf",
    top_k: int = 20,
) -> dict[int, list[int]]:
    """Load a per-timestep score pickle and return top-k features per timestep.

    ``scores_path`` is either a local ``.pkl`` path or an HF dataset / model
    repo id whose root contains ``score_filename`` (e.g. ``"tf6_scores.pkl"``
    or ``"tf7_scores.pkl"`` — always explicit, so the per-hookpoint choice
    is visible at the call site). The pickle holds a dict keyed by
    selection-method name; the value is either a tensor of shape
    ``[num_timesteps, num_features]`` (the modern format) or a legacy nested
    ``{"scores": tensor}`` dict.
    """
    import pickle
    from pathlib import Path as _Path

    path = _Path(scores_path)
    if path.is_dir():
        path = path / score_filename
    if not path.exists():
        # Try HF Hub: download <score_filename> from the repo root.
        from huggingface_hub import hf_hub_download

        path = _Path(hf_hub_download(scores_path, score_filename))

    with path.open("rb") as f:
        cached = pickle.load(f)

    if selection_method not in cached:
        raise KeyError(
            f"selection_method={selection_method!r} not in score cache at {path}. "
            f"Available keys: {sorted(k for k in cached if k != 'concept')}."
        )
    value = cached[selection_method]
    # Legacy nested format: {"tfidf": {"scores": tensor}, ...}
    if isinstance(value, dict) and "scores" in value:
        value = value["scores"]
    if not hasattr(value, "shape"):
        raise TypeError(
            f"Score cache entry {selection_method!r} is not a tensor: {type(value)}"
        )

    # value: tensor of shape [num_timesteps, num_features]
    out: dict[int, list[int]] = {}
    for t in range(value.shape[0]):
        out[t] = torch.argsort(value[t], descending=True)[:top_k].tolist()
    return out


@register_method("sae")
class SAESteeringController(Controller):
    """SAE feature / steering-vector interventions across one or more hookpoints."""

    def __init__(self, specs: Mapping[str, LayerSpec]) -> None:
        super().__init__()
        if not specs:
            raise ValueError("SAESteeringController requires at least one LayerSpec.")
        self.specs: dict[str, LayerSpec] = dict(specs)
        self._hooks: list[Any] = []

    # --- factory ------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        hookpoint: str,
        sae_mode: SaeMode = "sequence",
        features_per_timestep: dict[int, list[int]] | None = None,
        multiplier: float | dict[int, float] | None = None,
        alpha: float | None = None,  # synonym for multiplier — for the unified runner
        intervention_mode: InterventionMode = "pre_topk",
        steering_vectors: dict[int, torch.Tensor] | None = None,
        # Pick top-k features per timestep from a score table.
        scores_cache_path: str | None = None,
        score_filename: str | None = None,
        selection_method: str = "tfidf",
        top_k: int = 20,
        **layer_kwargs: Any,
    ) -> "SAESteeringController":
        """Load a single SAE (from a local path or a Hub repo) and wrap it.

        ``repo_id`` resolution:
        * If a local directory containing ``cfg.json`` exists at that exact
          path, it's loaded directly.
        * If a local directory exists with a ``hookpoint``-named subdir, that
          subdir is loaded.
        * Otherwise it's treated as a Hub repo id (``<org>/<name>``).

        Feature selection (mutually exclusive — pick one):
        * ``features_per_timestep``: explicit ``{step: [feat_idx, ...]}`` dict.
        * ``scores_cache_path``: path or HF repo id pointing at a per-timestep
          score table pickle. The top ``top_k`` features per timestep are
          selected by ``selection_method`` (e.g. ``"tfidf"``, ``"diff"``).
        * ``steering_vectors``: precomputed ``{step: tensor}`` (advanced).

        For multi-layer setups (typical for ``tf6tf7``-style configs), build
        the ``specs`` dict directly and call ``SAESteeringController(specs)``.
        """
        if multiplier is None and alpha is None:
            multiplier = 1.0
        elif multiplier is None:
            multiplier = alpha
        # else: explicit multiplier wins (alpha ignored).

        # If a score cache was provided AND no explicit feature list, pick
        # top-k per timestep from it.
        if (
            features_per_timestep is None
            and steering_vectors is None
            and scores_cache_path is not None
        ):
            # A direct ``.pkl`` path carries its own filename; a dir/repo id
            # needs an explicit ``score_filename``.
            sp = Path(scores_cache_path)
            if sp.is_file():
                resolved_filename = sp.name
                scores_dir_or_repo: str = (
                    str(sp.parent) if sp.parent != Path("") else "."
                )
            else:
                if score_filename is None:
                    raise ValueError(
                        "score_filename must be specified explicitly when using "
                        "scores_cache_path (e.g. 'tf7_scores.pkl' or 'tf6_scores.pkl')."
                    )
                resolved_filename = score_filename
                scores_dir_or_repo = scores_cache_path
            features_per_timestep = load_features_from_score_cache(
                scores_dir_or_repo,
                score_filename=resolved_filename,
                selection_method=selection_method,
                top_k=top_k,
            )

        local = Path(repo_id)
        if (local / "cfg.json").exists():
            sae = Sae.load_from_disk(local)
        elif (local / hookpoint / "cfg.json").exists():
            sae = Sae.load_from_disk(local / hookpoint)
        else:
            sae = Sae.load_from_hub(repo_id, hookpoint=hookpoint)
        spec = LayerSpec(
            sae=sae,
            sae_mode=sae_mode,
            features_per_timestep=features_per_timestep,
            multiplier=multiplier,
            intervention_mode=intervention_mode,
            steering_vectors=steering_vectors,
            **layer_kwargs,
        )
        return cls({hookpoint: spec})

    # --- lifecycle ----------------------------------------------------------

    def mount(self, model: Any) -> None:
        self._hooks = []
        for hookpoint, spec in self.specs.items():
            block = locate_block(hookpoint, model)
            hook = self._build_hook(spec)
            self._add_forward_hook(block, hook)
            self._hooks.append(hook)
        self._is_mounted = True

    def reset(self) -> None:
        for h in self._hooks:
            h.counter = -1

    def set_alpha(self, alpha: float) -> None:
        """Set the multiplier on every layer spec (and every live hook).

        Useful for alpha sweeps with a single-feature config. For multi-spec
        configs with per-layer multipliers, edit ``self.specs`` directly.
        """
        for spec in self.specs.values():
            spec.multiplier = alpha
        for h in self._hooks:
            h.multiplier = alpha

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _build_hook(spec: LayerSpec):
        if spec.steering_vectors is not None:
            mult = spec.multiplier if isinstance(spec.multiplier, float) else 1.0
            return ACEStepPrecomputedSteeringHook(
                steering_vectors=spec.steering_vectors,
                multiplier=mult,
                sae_mode=spec.sae_mode,
                uncond_preds=spec.uncond_preds,
                negate_for_uncond=spec.negate_for_uncond,
                start_step=spec.start_step,
            )
        return ACEStepTimestepInterventionHook(
            sae=spec.sae,
            features_per_timestep=spec.features_per_timestep,
            multiplier=spec.multiplier,
            sae_mode=spec.sae_mode,
            uncond_preds=spec.uncond_preds,
            add_error=spec.add_error,
            negate_for_uncond=spec.negate_for_uncond,
            intervention_mode=spec.intervention_mode,
            start_step=spec.start_step,
        )
