"""SAE scorers.

* :class:`SAEActivationsScorer` — cache transformer activations for SAE training.
* :class:`SAETrainScorer`        — train the SAE on cached activations.
* :class:`SAEScoresScorer`       — per-concept feature-selection score tables
  (``tf{6,7}_scores.pkl`` with tfidf / diff / mean_pos), consumed at inference
  by :class:`SAESteeringController`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.steering import Scorer
from src.steering.scorer import register_scorer
from src.steering.model import SteerableACEModel

LAYER_TO_HOOK = {
    "tf7": "transformer_blocks.7.cross_attn",
    "tf6": "transformer_blocks.6.cross_attn",
}
LAYER_TO_SAE_REPO = {
    "tf7": "lukasz-staniszewski/ace-step-sae-tf7-cross-attn",
    "tf6": "lukasz-staniszewski/ace-step-sae-tf6-cross-attn",
}
LAYER_TO_SCORES_NAME = {"tf7": "tf7_scores.pkl", "tf6": "tf6_scores.pkl"}
# @TODO: fix to tf7_activations.pkl
LAYER_TO_ACTS_NAME = {"tf7": "activations.pkl", "tf6": "tf6_activations.pkl"}


@register_scorer("sae-activations")
class SAEActivationsScorer(Scorer):
    """Cache ACE-Step activations at one or more hookpoints for SAE training.

    Artifact layout: HuggingFace-Datasets style activations cache (see
    ``CacheActivationsRunner``).
    """

    def compute(
        self,
        model: SteerableACEModel,
        output_dir: str | Path,
        *,
        hook_names: list[str],
        dataset_name: str,
        dataset_type: str = "csv",
        column: str = "caption",
        audio_length_in_s: float = 10.0,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        cache_every_n_timesteps: int = 6,
        batch_size: int = 16,
        max_num_examples: int | None = None,
        **_: Any,
    ) -> Path:
        from src.steering.methods.sae.lib.sae.cache_activations_runner_ace import (
            CacheActivationsRunner,
        )
        from src.steering.methods.sae.lib.sae.config import CacheActivationsRunnerConfig

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        config = CacheActivationsRunnerConfig(
            hook_names=list(hook_names),
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            column=column,
            negative_prompt="",
            model_name="ace-step",
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length_in_s,
            num_waveforms_per_prompt=1,
            guidance_scale=guidance_scale,
            cache_every_n_timesteps=cache_every_n_timesteps,
            batch_size=batch_size,
            max_num_examples=max_num_examples,
            new_cached_activations_path=str(out),
        )
        runner = CacheActivationsRunner(config, pipeline=model.pipeline)
        runner.run()
        return out


@register_scorer("sae-train")
class SAETrainScorer(Scorer):
    """Train an SAE on a cached-activations dataset.

    The existing training script (``steering/sae/lib/scripts/train_ace.py``)
    parses its config via ``simple_parsing``, supports distributed runs, and
    accepts dozens of knobs. This scorer documents the entry point and surfaces
    a clear error if called in-process. The unified runner can still dispatch
    to it via shell-out if needed.
    """

    def compute(
        self,
        model: SteerableACEModel,
        output_dir: str | Path,
        *,
        activations_dir: str | None = None,
        **_: Any,
    ) -> Path:
        raise RuntimeError(
            "SAETrainScorer.compute is intentionally not implemented in-process. "
            "Run the dedicated training entry point instead:\n\n"
            "    python -m steering.sae.lib.scripts.train_ace --help\n\n"
            "It uses simple_parsing/dataclass config and supports DDP. This "
            "scorer exists so the phase is discoverable via list_scorers()."
        )


@register_scorer("sae-scores")
class SAEScoresScorer(Scorer):
    """Compute per-concept SAE feature-selection score tables.

    For each requested layer it collects positive/negative cross-attention
    activations over the concept's prompt sets, runs them through that layer's
    SAE, and writes ``tf{6,7}_scores.pkl`` (dict of tfidf / diff / mean_pos
    tensors of shape ``[num_timesteps, num_features]``) plus the raw
    activations, into ``<output_dir>/``.
    """

    def compute(
        self,
        model: SteerableACEModel,
        output_dir: str | Path,
        *,
        concept: str,
        layers: list[str] = ["tf7", "tf6"],
        sae_paths: dict[str, str] | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        audio_length_in_s: float = 30.0,
        seed: int = 10,
        force: bool = False,
        lyrics: str | None = None,
        reuse_activations: bool = False,
        **_: Any,
    ) -> Path:
        import pickle

        import torch
        import torch.nn.functional as F
        from tqdm import tqdm

        from src.steering.methods.sae.lib.configs.steer_prompts import (
            CONCEPT_TO_PROMPTS,
        )
        from src.steering.methods.sae.lib.hooked_model.hooked_model_acestep import (
            HookedACEStepModel,
        )
        from src.steering.methods.sae.lib.sae.sae import Sae

        if concept not in CONCEPT_TO_PROMPTS:
            raise ValueError(
                f"Unknown concept {concept!r}; choose from {sorted(CONCEPT_TO_PROMPTS)}."
            )
        sae_paths = {**LAYER_TO_SAE_REPO, **(sae_paths or {})}

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # reuse_activations: score only, from activations already staged under
        # output_dir. Model, prompts and latents are not needed in that case.
        pos_prompts = neg_prompts = latents = hooked = None
        if not reuse_activations:
            if model is None:
                model = SteerableACEModel(device="cuda")
            if not getattr(model.pipeline, "loaded", False):
                model.pipeline.load()
            pipeline = model.pipeline
            hooked = HookedACEStepModel(pipeline=pipeline, device="cuda")

            neg_prompts, pos_prompts, concept_lyrics = CONCEPT_TO_PROMPTS[concept]()
            # The concept table's lyrics ("[inst]" for instrumental concepts, "" =
            # free vocals for the two vocal ones) unless overridden. Scoring the
            # vocal concepts on sung lyrics matches how the eval generates them.
            lyrics = concept_lyrics if lyrics is None else lyrics
            print(
                f"[{concept}] neg[0]={neg_prompts[0]!r} pos[0]={pos_prompts[0]!r} "
                f"lyrics={'(concept default) ' if lyrics == concept_lyrics else '(override) '}"
                f"{lyrics[:40]!r}"
            )
            n = max(len(pos_prompts), len(neg_prompts))
            latents = pipeline.prepare_latents(
                batch_size=n, audio_duration=audio_length_in_s, seed=seed
            )

        def _latents(sae, acts):
            with torch.no_grad():
                if acts.dim() == 2:
                    acts = acts.unsqueeze(1)
                # Encode in the SAE's own dtype.
                sae_input, _, _ = sae.preprocess_input(acts.to(sae.W_dec.dtype))
                return F.relu(sae.pre_acts(sae_input))

        for layer in layers:
            if layer not in LAYER_TO_HOOK:
                raise ValueError(
                    f"layer must be one of {list(LAYER_TO_HOOK)}; got {layer!r}"
                )
            hook = LAYER_TO_HOOK[layer]
            acts_path = out / LAYER_TO_ACTS_NAME[layer]
            scores_path = out / LAYER_TO_SCORES_NAME[layer]
            if scores_path.exists() and not force:
                print(
                    f"[{concept}/{layer}] {scores_path.name} exists; pass force=true to recompute."
                )
                continue

            if reuse_activations:
                if not acts_path.exists():
                    raise FileNotFoundError(
                        f"reuse_activations=True but {acts_path} not found — stage the "
                        "precomputed activations under output_dir first."
                    )
                print(f"[{concept}/{layer}] loading existing activations {acts_path.name}")
                with acts_path.open("rb") as f:
                    _d = pickle.load(f)
                pos_acts, neg_acts = _d["positive"], _d["negative"]
            else:
                common = dict(
                    audio_duration=audio_length_in_s,
                    lyrics=lyrics,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    guidance_interval=1.0,
                    guidance_interval_decay=0.0,
                    guidance_scale_text=0.0,
                    guidance_scale_lyric=0.0,
                    manual_seed=seed,
                    return_type="audio",
                    positions_to_cache=[hook],
                )

                print(f"[{concept}/{layer}] collecting activations …")
                out_pos = hooked.run_with_cache(
                    prompt=pos_prompts, latents=latents[: len(pos_prompts)], **common
                )
                pos_acts = out_pos[1]["output"][hook][: len(pos_prompts)].cpu()
                del out_pos
                torch.cuda.empty_cache()
                out_neg = hooked.run_with_cache(
                    prompt=neg_prompts, latents=latents[: len(neg_prompts)], **common
                )
                neg_acts = out_neg[1]["output"][hook][: len(neg_prompts)].cpu()
                del out_neg
                torch.cuda.empty_cache()
                with acts_path.open("wb") as f:
                    pickle.dump(
                        {"positive": pos_acts, "negative": neg_acts, "concept": concept}, f
                    )

            sae_path = sae_paths[layer]
            if (Path(sae_path) / "sae.safetensors").exists():
                sae = Sae.load_from_disk(sae_path, device="cuda").eval()
            else:
                sae = Sae.load_from_hub(sae_path, hookpoint=hook, device="cuda").eval()

            T = pos_acts.shape[1]  # timesteps from the activations themselves (robust in reuse mode)
            tfidf_l, diff_l, mean_pos_l = [], [], []
            for t in tqdm(range(T), desc=f"[{concept}/{layer}] scoring"):
                pos_lat = _latents(sae, pos_acts[:, t].cuda())
                neg_lat = _latents(sae, neg_acts[:, t].cuda())
                mp = pos_lat.mean(dim=0).float().cpu()
                mn = neg_lat.mean(dim=0).float().cpu()
                tfidf_l.append(mp * torch.log(1 + 1 / (mn + 1e-6)))
                diff_l.append(mp - mn)
                mean_pos_l.append(mp)
                del pos_lat, neg_lat

            with scores_path.open("wb") as f:
                pickle.dump(
                    {
                        "tfidf": torch.stack(tfidf_l),
                        "diff": torch.stack(diff_l),
                        "mean_pos": torch.stack(mean_pos_l),
                        "concept": concept,
                    },
                    f,
                )
            # Provenance sidecar: the score pickle records no compute settings, so
            # the artifact cannot otherwise be checked against the config that made it.
            import json as _json
            import subprocess as _sp

            try:
                sha = _sp.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=Path(__file__).resolve().parents[4], text=True
                ).strip()
            except Exception:
                sha = "unknown"
            (out / f"{layer}_scores_config.json").write_text(
                _json.dumps(
                    {
                        "method": "sae-scores",
                        "concept": concept,
                        "layer": layer,
                        "hookpoint": hook,
                        "sae_path": str(sae_path),
                        "num_inference_steps": num_inference_steps,
                        "audio_length_in_s": audio_length_in_s,
                        "guidance_scale": guidance_scale,
                        "seed": seed,
                        "lyrics": lyrics,
                        "reuse_activations": reuse_activations,
                        "num_timesteps": int(T),
                        "num_features": int(tfidf_l[0].shape[0]),
                        "git_sha": sha,
                    },
                    indent=2,
                )
            )
            print(f"[{concept}/{layer}] saved {scores_path.name}")

        return out
