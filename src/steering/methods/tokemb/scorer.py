"""TokE scorer — computes and saves the per-concept direction vector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.steering import Scorer
from src.steering.scorer import register_scorer
from src.steering.model import SteerableACEModel


@register_scorer("tokemb")
class TokEmbScorer(Scorer):
    """Compute a TokE direction vector and write ``<concept>_direction.pt``.

    Artifact layout (consumable by ``TokEmbSteeringController.from_pretrained``)::

        <output_dir>/
          <concept>_direction.pt   # {"direction": Tensor[hidden_dim], "concept": str, ...}
    """

    def compute(
        self,
        model: SteerableACEModel,
        output_dir: str | Path,
        *,
        concept: str,
        **_: Any,
    ) -> Path:
        from src.steering.methods.tokemb.core import compute_direction

        if model is None:
            model = SteerableACEModel(device="cuda")
        if not getattr(model.pipeline, "loaded", False):
            model.pipeline.load()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        direction = compute_direction(model.pipeline, concept)
        save_path = out / f"{concept}_direction.pt"
        torch.save(
            {
                "direction": direction.cpu(),
                "concept": concept,
                "hidden_dim": int(direction.shape[0]),
            },
            save_path,
        )
        print(f"Saved -> {save_path}  (shape {tuple(direction.shape)})")
        return out
