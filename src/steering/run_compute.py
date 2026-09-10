"""Universal compute / train runner.

One CLI that drives every registered :class:`Scorer` — the offline counterpart
to ``src/steering/run_eval.py``. After the artifact directory is produced, point
``run_eval.py --artifact <dir>`` at it.

Usage::

    # CAA: compute steering vectors for one concept
    python src/steering/run_compute.py \\
        --scorer caa --concept piano \\
        --output steering_vectors/caa/ace_piano

    # AUSteer: compute momentum-based scores
    python src/steering/run_compute.py \\
        --scorer austeer --concept piano --layers tf6tf7 \\
        --output steering_vectors/austeer/ace_piano_tf6tf7

    # Concept Slider: train LoRA
    python src/steering/run_compute.py \\
        --scorer concept_slider --concept piano \\
        --output steering_vectors/concept_slider/ace_piano_lora \\
        --scorer-kwargs '{"iterations": 1000, "layers": "all"}'

    # SAE activations cache (preparation for SAE training)
    python src/steering/run_compute.py \\
        --scorer sae-activations \\
        --output activations/ace_piano \\
        --scorer-kwargs '{"hook_names": ["transformer_blocks.7.cross_attn"],
                          "dataset_name": "../data/music_caps_slow32.csv"}'

    # SAE training: deferred to the dedicated CLI (see scorer's error).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)

from src.steering.cli import apply_yaml_config  # noqa: E402


def _import_methods() -> None:
    """Trigger every method package's import so both Controllers and Scorers
    register themselves before we look them up."""
    # Single import populates both registries.
    import src.steering.methods  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML experiment config. Fields mirror the CLI flags below.",
    )
    parser.add_argument(
        "--scorer",
        default=None,
        choices=[
            None,
            "caa",
            "austeer",
            "concept_slider",
            "sae-activations",
            "sae-scores",
            "sae-train",
            "tokemb",
            "audioldm_caa",
            "stable_audio_caa",
        ],
        help="Which registered Scorer to run.",
    )
    parser.add_argument(
        "--concept",
        default=None,
        help="Concept name (required for caa / austeer / concept_slider).",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output directory for the produced artifact.",
    )
    parser.add_argument(
        "--scorer-kwargs",
        type=json.loads,
        default={},
        help="JSON dict of scorer-specific kwargs forwarded to compute().",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip loading SteerableACEModel (scorers that build their own pipeline "
        "internally can run without one — e.g. CAA, AUSteer, ConceptSlider).",
    )
    args = parser.parse_args()

    if args.config is not None:
        args = apply_yaml_config(parser, args, path_keys=("output",))

    # Validate now that YAML merge is done.
    for required in ("scorer", "output"):
        if getattr(args, required) is None:
            parser.error(f"--{required} is required (set it on the CLI or in --config).")

    _import_methods()

    from src.steering import SteerableACEModel
    from src.steering.scorer import get_scorer

    cls = get_scorer(args.scorer)
    scorer = cls()

    kwargs: dict[str, Any] = dict(args.scorer_kwargs)
    if args.concept is not None and "concept" not in kwargs:
        kwargs["concept"] = args.concept

    model = None
    if not args.no_model:
        print(f"Loading SteerableACEModel on {args.device}...")
        model = SteerableACEModel(device=args.device)
        # Only sae-activations needs the pipeline loaded here; other scorers
        # load it themselves when required.
        if args.scorer == "sae-activations":
            model.pipeline.load()

    print(f"Running {args.scorer} -> {args.output}")
    out = scorer.compute(model, args.output, **kwargs) # type: ignore
    print(f"Done. Artifact directory: {out}")


if __name__ == "__main__":
    main()
