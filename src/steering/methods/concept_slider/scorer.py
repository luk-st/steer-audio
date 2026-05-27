"""Concept Slider scorer — trains the LoRA adapter for one concept."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.steering import Scorer
from src.steering.scorer import register_scorer
from src.steering.model import SteerableACEModel


@register_scorer("concept_slider")
class ConceptSliderTrainScorer(Scorer):
    """Wraps ``steering/cs/train_concept_slider.main``.

    Artifact layout (consumable by ``ConceptSlidersSteeringController.from_pretrained``)::

        <output_dir>/
          pytorch_lora_weights.safetensors
          train_config.json
    """

    def compute(
        self,
        model: SteerableACEModel,
        output_dir: str | Path,
        *,
        concept: str,
        iterations: int = 1000,
        lr: float = 1e-4,
        eta: float = 1.0,
        audio_duration: float = 10.0,
        save_every: int = 200,
        seed: int = 42,
        layers: str = "all",
        lora_config_path: str | None = None,
        with_denoising: bool = True,
        max_denoising_steps: int = 50,
        denoise_cfg_scale: float = 5.0,
        gradient_checkpointing: bool = True,
        **_: Any,
    ) -> Path:
        from src.steering.methods.concept_slider.train_concept_slider import main as _train_cs

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _train_cs(
            concept=concept,
            output_dir=str(out),
            lora_config_path=lora_config_path,
            iterations=iterations,
            lr=lr,
            eta=eta,
            audio_duration=audio_duration,
            save_every=save_every,
            seed=seed,
            layers=layers,
            with_denoising=with_denoising,
            max_denoising_steps=max_denoising_steps,
            denoise_cfg_scale=denoise_cfg_scale,
            gradient_checkpointing=gradient_checkpointing,
        )
        return out
