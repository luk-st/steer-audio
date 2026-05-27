"""AUSteer scorer — computes activation-momentum scores for one concept."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.steering import Scorer
from src.steering.scorer import register_scorer
from src.steering.model import SteerableACEModel


@register_scorer("austeer")
class AuSteerScorer(Scorer):
    """Wraps ``steering/austeer/compute_sv_austeer.main``.

    Artifact layout (consumable by ``AUSteerSteeringController.from_pretrained``)::

        <output_dir>/<auto-named-subdir>/
          config.json
          austeer.pkl
    """

    def compute(
        self,
        model: SteerableACEModel,
        output_dir: str | Path,
        *,
        concept: str,
        layers: str = "all",
        num_inference_steps: int = 30,
        audio_duration: float = 30.0,
        guidance_scale: float = 5.0,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        seed: int = 10,
        **_: Any,
    ) -> Path:
        from src.steering.methods.austeer.compute_sv_austeer import main as _compute_austeer

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _compute_austeer(
            concept=concept,
            layers=layers,
            num_inference_steps=num_inference_steps,
            audio_duration=audio_duration,
            guidance_scale=guidance_scale,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            seed=seed,
            save_dir=str(out),
        )
        return out
