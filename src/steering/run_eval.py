"""Universal alpha-sweep eval runner.

One CLI that drives every registered steering method through the same interface
(``SteerableACEModel`` + ``Controller``).

Output layout is compatible with ``eval_steering_protocol.py``::

    <save_dir>/
      run_config.json
      alpha_<value>/p<i>.wav
      ...

Usage::

    # CAA
    python src/steering/run_eval.py \\
        --method caa \\
        --artifact steering_vectors/caa/ace_piano \\
        --concept piano \\
        --alphas -40,-20,0,20,40

    # SAE — features picked at one hookpoint
    python src/steering/run_eval.py \\
        --method sae \\
        --artifact lukasz-staniszewski/ace-step-sae-tf7-cross-attn \\
        --concept piano \\
        --alphas -30,-10,0,10,30 \\
        --method-kwargs '{"hookpoint": "transformer_blocks.7.cross_attn",
                          "features_per_timestep": {"0": [42]},
                          "intervention_mode": "steering_vector"}'

    # Concept Slider (LoRA)
    python src/steering/run_eval.py \\
        --method concept_slider \\
        --artifact steering_vectors/concept_slider/ace_piano_r8_eta7_500_tf6tf7 \\
        --concept piano \\
        --alphas -2,-1,0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)


def _parse_alphas(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",")]


def _load_prompts_file(path: Path, column: str = "prompt") -> list[str]:
    """Read eval prompts from a TXT (one per line) or CSV (named column, default `prompt`)."""
    if path.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        if column not in df.columns:
            raise SystemExit(
                f"CSV {path} has columns {list(df.columns)} — none named {column!r}. "
                f"Set --prompts-column to pick a different one."
            )
        return df[column].astype(str).tolist()
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _alphas_from_range(
    min_range: float, max_range: float, steps_per_side: int
) -> list[float]:
    """Symmetric sweep: ``steps_per_side`` neg + 0 + ``steps_per_side`` pos."""
    neg = [
        round(min_range * i / steps_per_side, 4) for i in range(steps_per_side, 0, -1)
    ]
    pos = [
        round(max_range * i / steps_per_side, 4) for i in range(1, steps_per_side + 1)
    ]
    return neg + [0.0] + pos


def _eval_prompts(concept: str) -> list[str]:
    """Return the default list of generation prompts.

    The paper-canonical eval set is the 100-prompt :data:`TEST_PROMPTS`
    (``src/steering/eval/test_prompts.py``) — diverse music descriptions sampled
    across genres so the steering alpha's effect can be measured against a
    realistic distribution. ``concept`` is unused in the default path; it's
    kept for backward compatibility and for callers that want to override this
    function to pick concept-specific prompts.

    Override via the CLI/YAML flags ``--prompts`` (inline list) or
    ``--prompts-file`` (TXT / CSV); see ``configs/steering/`` for examples.
    """
    try:
        from src.steering.eval.test_prompts import TEST_PROMPTS

        return list(TEST_PROMPTS)
    except ImportError:
        return [f"a song about {concept}"]


def _build_controller(
    method: str,
    artifact: str,
    alpha: float,
    method_kwargs: dict[str, Any],
    concept: str | None = None,
):
    """Construct the right controller from the registry."""
    import inspect

    from src.steering.registry import get_method

    # Trigger registration of every wrapper before lookup. Importing here keeps
    # `--help` light.
    # Single import populates the registry with all 9 methods.
    import src.steering.methods  # noqa: F401

    cls = get_method(method)

    # SAE: features_per_timestep keys come back as strings from JSON; coerce.
    if method == "sae" and "features_per_timestep" in method_kwargs:
        method_kwargs["features_per_timestep"] = {
            int(k): list(v) for k, v in method_kwargs["features_per_timestep"].items()
        }

    # Some controllers take `concept` as a constructor arg; others read it from
    # their artifact. Pass it through only when the target accepts it.
    if concept is not None and "concept" not in method_kwargs:
        accepts_concept = any(
            "concept" in inspect.signature(fn).parameters
            for fn in (cls.__init__, cls.from_pretrained)
        )
        if accepts_concept:
            method_kwargs = {**method_kwargs, "concept": concept}

    return cls.from_pretrained(artifact, alpha=alpha, **method_kwargs)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML experiment config. See ``configs/steering/`` for examples."""
    import yaml

    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _apply_config_defaults(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Fill argparse args from a YAML config when CLI didn't override them.

    CLI flags always win over YAML values, so a config can be tweaked from the
    command line without re-editing the file.
    """
    for key, val in cfg.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            # Unknown YAML key — surface, don't silently drop.
            raise SystemExit(f"YAML config has unknown key {key!r}.")
        current = getattr(args, attr)
        is_default = current is None or (
            isinstance(current, (dict, list)) and not current
        )
        if is_default:
            if attr in ("save_dir", "config") and isinstance(val, str):
                val = Path(val)
            setattr(args, attr, val)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML experiment config. Fields mirror the CLI flags below "
        "(``method``, ``artifact``, ``concept``, ``alphas``, ``method_kwargs``, "
        "etc.). CLI flags override YAML values.",
    )
    parser.add_argument(
        "--method",
        default=None,
        choices=[
            None,
            "caa",
            "sae",
            "austeer",
            "concept_slider",
            "freesliders",
            "textemb",
            "pci",
            "tokemb",
            "audioldm_caa",
        ],
        help="Which registered steering method to run.",
    )
    parser.add_argument("--artifact", default=None, help="Local path or HF repo id.")
    parser.add_argument("--concept", default=None)
    parser.add_argument(
        "--alphas",
        default=None,
        type=_parse_alphas,
        help="Comma-separated alpha sweep, e.g. '--alphas=-20,0,20'. "
        "Overrides --min-range / --max-range when set.",
    )
    parser.add_argument(
        "--min-range",
        type=float,
        default=None,
        help="Most-negative alpha (paired with --max-range). Symmetric sweep is built.",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=None,
        help="Most-positive alpha (paired with --min-range).",
    )
    parser.add_argument(
        "--steps-per-side",
        type=int,
        default=15,
        help="Number of intermediate alphas per side of zero when using --min/--max-range.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help=(
            "Layer config (caa/austeer): 'all' / 'tf6' / 'tf7' / 'tf6tf7' / 'no_tf6tf7'. "
            "Resolved via steering.caa.utils.constants.LAYER_CONFIGS."
        ),
    )
    parser.add_argument(
        "--steer-mode",
        default=None,
        choices=[
            None,
            "cond_only",
            "uncond_only",
            "uncond_for_cond",
            "separate",
            "both_cond",
            "both_uncond",
        ],
        help="CFG steering mode (caa/austeer). Default per-method.",
    )
    parser.add_argument(
        "--prompts", nargs="*", help="Override eval prompts (inline list)."
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Load eval prompts from .txt (one per line) or .csv (named column, see --prompts-column).",
    )
    parser.add_argument(
        "--prompts-column",
        default="prompt",
        help="Column name to use when --prompts-file is a CSV. Default: 'prompt'.",
    )
    parser.add_argument("--lyrics", default="[inst]")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2115)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument(
        "--method-kwargs",
        type=json.loads,
        default={},
        help="JSON dict of method-specific kwargs forwarded to from_pretrained.",
    )
    parser.add_argument("--save-mono", action="store_true", default=True)
    args = parser.parse_args()

    if args.config is not None:
        _apply_config_defaults(args, _load_yaml_config(args.config))

    # After config merge, validate the required-args ourselves (since argparse
    # required=True wouldn't allow them to come from YAML).
    for required in ("method", "artifact", "concept"):
        if getattr(args, required) is None:
            parser.error(
                f"--{required} is required (set it on the CLI or in --config)."
            )

    # Resolve alpha sweep: explicit --alphas > --min/--max-range > default.
    if args.alphas is not None:
        alphas = args.alphas
    elif args.min_range is not None and args.max_range is not None:
        alphas = _alphas_from_range(args.min_range, args.max_range, args.steps_per_side)
    elif args.min_range is not None or args.max_range is not None:
        parser.error("--min-range and --max-range must be supplied together.")
    else:
        alphas = [-40.0, -20.0, 0.0, 20.0, 40.0]
    args.alphas = alphas

    # Fold shell-friendly flags into method_kwargs.
    if args.layers is not None:
        from src.steering.methods.caa.utils.constants import LAYER_CONFIGS

        if args.layers not in LAYER_CONFIGS:
            parser.error(
                f"Unknown --layers {args.layers!r}; choose from {list(LAYER_CONFIGS)}."
            )
        # Every controller now accepts target_layers as either a preset string
        # or an explicit list. Pass the string preset so the controller can do
        # its own resolution and validation.
        args.method_kwargs.setdefault("target_layers", args.layers)
    if args.steer_mode is not None:
        args.method_kwargs.setdefault("steer_mode", args.steer_mode)

    if args.prompts:
        prompts = args.prompts
    elif args.prompts_file is not None:
        prompts = _load_prompts_file(args.prompts_file, column=args.prompts_column)
        print(f"Loaded {len(prompts)} prompt(s) from {args.prompts_file}")
    else:
        prompts = _eval_prompts(args.concept)

    # All methods are expected to condition the model on the same `neutral` text
    # at alpha=0, so their baselines line up and only the steering signal
    # differs. TextEmb / TokEmb / PCI apply the wrap internally (their
    # build_prompt_triple's `neutral` IS the canonical neutral); the other
    # methods leave the prompt untouched, so we pre-wrap here using the same
    # template (CONCEPT_TO_NEUTRAL_ADDON[concept]).
    if args.method not in ("textemb", "tokemb", "pci"):
        from src.steering.methods.caa.utils.constants import CONCEPT_TO_NEUTRAL_ADDON

        addon = CONCEPT_TO_NEUTRAL_ADDON.get(args.concept)
        if addon is not None:
            prompts = [addon.format(p=p) for p in prompts]
            print(
                f"Applied neutral wrap for concept={args.concept!r}: "
                f"template={addon!r}, e.g. prompts[0]={prompts[0]!r}"
            )

    save_dir = args.save_dir or (
        ROOT
        / "steering"
        / "outputs"
        / args.concept
        / args.method
        / f"{args.method}_{torch.utils.data.get_worker_info() is None and 'run' or 'run'}"
    )
    # Simpler: timestamped dir.
    from datetime import datetime

    if args.save_dir is None:
        save_dir = (
            ROOT
            / "steering"
            / "outputs"
            / args.concept
            / args.method
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {save_dir}")

    run_config = {
        "method": args.method,
        "artifact": str(args.artifact),
        "concept": args.concept,
        "alphas": args.alphas,
        "prompts": prompts,
        "lyrics": args.lyrics,
        "duration": args.duration,
        "steps": args.steps,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "method_kwargs": args.method_kwargs,
        "command": " ".join(sys.argv),
    }
    (save_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # AudioLDM2-based methods use a different model wrapper and pipeline call
    # signature. ACE-Step is the default for everything else.
    is_audioldm = args.method.startswith("audioldm")

    if is_audioldm:
        from src.steering import SteerableAudioLDMModel

        print(f"Loading SteerableAudioLDMModel on {args.device}...")
        model = SteerableAudioLDMModel(device=args.device)
    else:
        from src.steering import SteerableACEModel

        print(f"Loading SteerableACEModel on {args.device}...")
        model = SteerableACEModel(device=args.device)
        model.pipeline.load()

    initial_alpha = args.alphas[0]
    print(
        f"Building {args.method} controller from {args.artifact} (initial alpha={initial_alpha})..."
    )
    controller = _build_controller(
        args.method, args.artifact, initial_alpha, args.method_kwargs, args.concept
    )

    # Eval sweeps generate the same prompt-rewrite warnings for each of
    # potentially many alphas; suppress them here. They still fire during
    # interactive `model.generate(...)` calls outside the runner.
    import warnings as _warnings
    from src.steering import PromptRewriteWarning

    _warnings.simplefilter("ignore", PromptRewriteWarning)

    print(f"Sweeping {len(args.alphas)} alpha(s) × {len(prompts)} prompt(s)...")
    with model.steer(controller):
        for alpha in args.alphas:
            controller.set_alpha(alpha)
            controller.reset()
            print(f"  alpha={alpha}")
            if is_audioldm:
                audios = model.generate(
                    prompt=prompts,
                    num_inference_steps=args.steps,
                    audio_length_in_s=args.duration,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                )
            else:
                audios = model.generate(
                    prompt=prompts,
                    lyrics=args.lyrics,
                    audio_duration=args.duration,
                    infer_step=args.steps,
                    manual_seed=args.seed,
                    return_type="audio",
                    guidance_scale=args.guidance_scale,
                    guidance_interval=1.0,
                )
            alpha_dir = save_dir / f"alpha_{alpha}"
            alpha_dir.mkdir(parents=True, exist_ok=True)
            for i, audio in enumerate(audios):
                if is_audioldm:
                    torchaudio.save(
                        str(alpha_dir / f"p{i}.wav"),
                        audio.detach().cpu().unsqueeze(0),  # mono (N,) -> (1, N)
                        model.sample_rate,
                    )
                else:
                    if args.save_mono and audio.dim() == 2:
                        audio = audio.mean(dim=0, keepdim=True)
                    torchaudio.save(
                        str(alpha_dir / f"p{i}.wav"),
                        audio.cpu(),
                        model.sample_rate,
                    )

    print(f"Done. {len(args.alphas)} alpha dirs written under {save_dir}")


if __name__ == "__main__":
    main()
