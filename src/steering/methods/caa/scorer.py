"""CAA scorer — computes steering vectors for one concept."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.steering import Scorer
from src.steering.scorer import register_scorer
from src.steering.model import SteerableACEModel


@register_scorer("caa")
class CAAScorer(Scorer):
    """Wraps ``steering/caa/compute_sv_caa.main`` so the unified runner can
    use it via :class:`CAAScorer`.

    Artifact layout (consumable by ``CAASteeringController.from_pretrained``)::

        <output_dir>/<auto-named-subdir>/
          config.json
          sv.pkl
          pos_vectors.pkl
          neg_vectors.pkl
    """

    def compute(
        self,
        model: SteerableACEModel,
        output_dir: str | Path,
        *,
        concept: str,
        num_inference_steps: int = 30,
        audio_duration: float = 30.0,
        guidance_scale: float = 5.0,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        seed: int = 10,
        save_all_cfg_passes: bool = True,
        **_: Any,
    ) -> Path:
        # The existing compute script builds its own pipeline; we ignore the
        # passed-in ``model`` to avoid resource conflicts (it would load
        # weights twice). This keeps the prototype contained.
        from src.steering.methods.caa.compute_sv_caa import main as _compute_caa

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _compute_caa(
            concept=concept,
            num_inference_steps=num_inference_steps,
            audio_duration=audio_duration,
            guidance_scale=guidance_scale,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            seed=seed,
            save_dir=str(out),
            save_all_cfg_passes=save_all_cfg_passes,
        )
        return out
