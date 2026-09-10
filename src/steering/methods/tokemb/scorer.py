"""TokE scorer — computes and saves the per-concept direction vector."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from src.steering import Scorer
from src.steering.scorer import register_scorer
from src.steering.model import SteerableACEModel


def _git_sha() -> str:
    """Short git SHA of the working tree, or 'unknown' outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[4], text=True,
        ).strip()
    except Exception:
        return "unknown"


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
        # Provenance sidecar: the .pt itself records no compute settings, so the
        # artifact cannot otherwise be checked against the config that made it.
        (out / f"{concept}_config.json").write_text(
            json.dumps(
                {
                    "method": "tokemb",
                    "concept": concept,
                    "hidden_dim": int(direction.shape[0]),
                    "git_sha": _git_sha(),
                },
                indent=2,
            )
        )
        print(f"Saved -> {save_path}  (shape {tuple(direction.shape)})")
        return out
