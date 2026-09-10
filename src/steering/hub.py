"""HuggingFace Hub integration for steering artifacts.

Three artifact types:

1. **SAE checkpoints** — already implement ``save_to_disk`` / ``load_from_hub``
   (see ``steering/sae/lib/sae/sae.py``). We add :func:`push_sae_to_hub` which uploads
   one checkpoint to its own repo with a generated model card.

2. **Steering vectors** (CAA, AuSteer, ConceptSlider, ...) — stored as a directory
   of pickles + ``config.json``. :class:`SteeringVectorArtifact` wraps that.

3. **Counterfactual prompt datasets** — uploaded via ``datasets.Dataset.push_to_hub``;
   see ``scripts/hub/push_localization_prompts.py`` (MusicCaps-derived
   ``(clean, corrupted)`` prompts that drive activation patching).

We deliberately recommend **one repo per checkpoint** (per CLAUDE.md / HF guidance)
so download stats and paper-page links work per artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PAPER_URL = "https://huggingface.co/papers/2602.11910"
PAPER_SECTION = (
    "## Paper\n\n"
    "TADA! Tuning Audio Diffusion Models through Activation Steering — "
    f"[{PAPER_URL}]({PAPER_URL})\n\n"
)


def _render_sae_scores_card(
    concept: str,
    repo_id: str,
    tags: "Iterable[str] | None" = None,
) -> str:
    """Model card for per-concept SAE feature-selection score tables.

    Repo contains ``tf6_scores.pkl`` + ``tf7_scores.pkl`` (one per hookpoint).
    """
    base_tags = [
        "audio",
        "music",
        "ace-step",
        "sae",
        "sparse-autoencoder",
        "feature-selection",
        "interpretability",
        "steering",
    ]
    all_tags = sorted(set(base_tags) | set(tags or []))
    cfg_yaml = "\n".join(f"  - {t}" for t in all_tags)
    quickstart = (
        "from src.steering.methods.sae import load_features_from_score_cache\n\n"
        "top20_tf7 = load_features_from_score_cache(\n"
        f'    "{repo_id}", score_filename="tf7_scores.pkl", top_k=20,\n'
        ")\n"
        "top20_tf6 = load_features_from_score_cache(\n"
        f'    "{repo_id}", score_filename="tf6_scores.pkl", top_k=20,\n'
        ")\n"
    )
    return (
        f"---\n"
        f"library_name: audio-interv\n"
        f"tags:\n{cfg_yaml}\n"
        f"---\n\n"
        f"# SAE Feature-Selection Scores — `{concept}` (ACE-Step)\n\n"
        f"Per-concept feature-importance scores for the ACE-Step SAEs at "
        f"`transformer_blocks.6.cross_attn` and `transformer_blocks.7.cross_attn`. "
        f"Consumed at inference time by `SAESteeringController` via "
        f"`load_features_from_score_cache` (top-k features per diffusion step).\n\n"
        f"## Files\n\n"
        f"- `tf7_scores.pkl` — scores for the tf7 SAE.\n"
        f"- `tf6_scores.pkl` — scores for the tf6 SAE.\n\n"
        f"Each pickle is a dict keyed by selection method "
        f"(`tfidf`, `diff`, `mean_pos`, ...); values are tensors of shape "
        f"`(num_timesteps, num_features)`.\n\n"
        f"{PAPER_SECTION}"
        f"## Quickstart\n\n"
        f"```python\n{quickstart}```\n"
    )


# ---------------------------------------------------------------------------
# Steering vectors
# ---------------------------------------------------------------------------


@dataclass
class SteeringVectorArtifact:
    """A directory-backed steering-vector checkpoint.

    Mirrors the on-disk layout produced by ``steering/caa/compute_sv_caa.py``::

        <root>/
          config.json
          sv.pkl
          pos_vectors.pkl    (optional)
          neg_vectors.pkl    (optional)
    """

    root: Path
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, path: str | Path) -> "SteeringVectorArtifact":
        path = Path(path)
        config_path = path / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        return cls(root=path, config=config)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> "SteeringVectorArtifact":
        """Pull a steering-vector repo from the Hub (or use a local path).

        A string with more than one ``/`` (or one that exists on disk) is
        treated as a local path; otherwise it's an HF repo id.
        """
        local = Path(repo_id)
        looks_local = local.exists() or repo_id.count("/") > 1 or repo_id.startswith((".", "/", "~"))
        if looks_local:
            if not local.exists():
                raise FileNotFoundError(
                    f"Local steering-vector artifact not found: {local!s} "
                    f"(cwd={Path.cwd()!s}). Pass an absolute path or an HF repo id."
                )
            return cls.from_dir(local)
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=["*.pkl", "*.json", "*.md"],
            )
        )
        return cls.from_dir(path)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = False,
        commit_message: str = "Upload steering vectors",
        tags: Iterable[str] | None = None,
    ) -> str:
        """Upload the artifact directory to its own Hub repo."""
        from huggingface_hub import HfApi, create_repo

        create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
        # Write/refresh model card before upload.
        card_path = self.root / "README.md"
        card_path.write_text(_render_sv_card(self.config, repo_id, tags=tags))

        api = HfApi()
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(self.root),
            commit_message=commit_message,
            repo_type="model",
        )
        return f"https://huggingface.co/{repo_id}"


def _render_sv_card(
    config: dict[str, Any],
    repo_id: str,
    tags: Iterable[str] | None = None,
) -> str:
    """Minimal model card with HF-discoverable tags.

    Dispatches on ``config["method"]``:
    * ``austeer``: AUSteer header + AUSteerSteeringController quickstart.
    * ``*audioldm*``: AudioLDM header + AudioLDMCAASteeringController quickstart.
    * default: ACE-Step CAA header + CAASteeringController quickstart.
    """
    method = str(config.get("method", "")).lower()
    is_audioldm = "audioldm" in method
    is_austeer = method == "austeer"
    concept = config.get("concept", "<concept>")

    if is_austeer:
        backbone = "ACE-Step"
        method_name = "AUSteer"
        method_tag = "austeer"
        description = (
            f"Per-(step, layer) sparse activation-momentum scores for the "
            f"**{concept}** concept on ACE-Step. At inference, "
            f"`AUSteerSteeringController` adds `alpha` along the top-`k` "
            f"most concept-discriminative bins."
        )
        quickstart = (
            "from src.steering import SteerableACEModel, AUSteerSteeringController\n\n"
            'model = SteerableACEModel(device="cuda")\n'
            "model.pipeline.load()\n"
            f"ctrl = AUSteerSteeringController.from_pretrained(\n"
            f'    "{repo_id}", alpha=15.0, k=256, mode="additive",\n'
            f")\n\n"
            "with model.steer(ctrl):\n"
            "    audio = model.generate(\n"
            '        prompt="instrumental music", lyrics="[inst]",\n'
            "        audio_duration=10.0, infer_step=30, manual_seed=0,\n"
            "    )\n"
        )
    elif is_audioldm:
        backbone = "AudioLDM2"
        method_name = "CAA"
        method_tag = "audioldm2"
        description = (
            f"Steering vectors for the **{concept}** concept on AudioLDM2, "
            f"computed via contrastive activation addition (CAA)."
        )
        quickstart = (
            "from src.steering import SteerableAudioLDMModel, AudioLDMCAASteeringController\n\n"
            'model = SteerableAudioLDMModel(device="cuda")\n'
            f'ctrl = AudioLDMCAASteeringController.from_pretrained("{repo_id}", alpha=1.0)\n\n'
            "with model.steer(ctrl):\n"
            "    out = model.generate(\n"
            '        prompt="instrumental music",\n'
            "        num_inference_steps=30, audio_length_in_s=10.0,\n"
            "        guidance_scale=3.5, seed=0,\n"
            "    )\n"
        )
    else:
        backbone = "ACE-Step"
        method_name = "CAA"
        method_tag = "ace-step"
        description = (
            f"Steering vectors for the **{concept}** concept on ACE-Step, "
            f"computed via contrastive activation addition (CAA)."
        )
        quickstart = (
            "from src.steering import SteerableACEModel, CAASteeringController\n\n"
            'model = SteerableACEModel(device="cuda")\n'
            "model.pipeline.load()\n"
            f'ctrl = CAASteeringController.from_pretrained("{repo_id}", alpha=20.0)\n\n'
            "with model.steer(ctrl):\n"
            "    audio = model.generate(\n"
            '        prompt="instrumental music", lyrics="[inst]",\n'
            "        audio_duration=10.0, infer_step=30, manual_seed=0,\n"
            "    )\n"
        )

    base_tags = [
        "audio",
        "music",
        "diffusion",
        method_tag,
        "interpretability",
        "steering",
        "activation-steering",
    ]
    all_tags = sorted(set(base_tags) | set(tags or []))
    cfg_yaml = "\n".join(f"  - {t}" for t in all_tags)
    return (
        f"---\n"
        f"library_name: audio-interv\n"
        f"tags:\n{cfg_yaml}\n"
        f"---\n\n"
        f"# {method_name} — `{concept}` ({backbone})\n\n"
        f"{description}\n\n"
        f"{PAPER_SECTION}"
        f"## Quickstart\n\n"
        f"```python\n{quickstart}```\n\n"
        f"## Generation config\n\n"
        f"```json\n{json.dumps(config, indent=2)}\n```\n"
    )


def _render_lora_card(
    train_config: dict[str, Any],
    repo_id: str,
    tags: Iterable[str] | None = None,
) -> str:
    """Model card for a Concept-Sliders LoRA artifact."""
    concept = train_config.get("concept", "<concept>")
    rank = train_config.get("lora_config", {}).get("r", "?")
    layers = train_config.get("layers", "?")
    base_tags = [
        "audio",
        "music",
        "diffusion",
        "ace-step",
        "concept-slider",
        "lora",
        "peft",
        "interpretability",
        "steering",
    ]
    all_tags = sorted(set(base_tags) | set(tags or []))
    cfg_yaml = "\n".join(f"  - {t}" for t in all_tags)
    quickstart = (
        "from src.steering import SteerableACEModel, ConceptSlidersSteeringController\n\n"
        'model = SteerableACEModel(device="cuda")\n'
        "model.pipeline.load()\n"
        f'ctrl = ConceptSlidersSteeringController.from_pretrained("{repo_id}", alpha=1.0)\n\n'
        "with model.steer(ctrl):\n"
        "    for alpha in [-2, -1, 0, 1, 2]:\n"
        "        ctrl.set_alpha(alpha)\n"
        "        audio = model.generate(\n"
        '            prompt="instrumental music", lyrics="[inst]",\n'
        "            audio_duration=10.0, infer_step=30, manual_seed=0,\n"
        "        )\n"
    )
    return (
        f"---\n"
        f"library_name: audio-interv\n"
        f"tags:\n{cfg_yaml}\n"
        f"---\n\n"
        f"# Concept Slider LoRA — `{concept}` (r{rank}, layers={layers}) — ACE-Step\n\n"
        f"LoRA adapter trained with the Concept Sliders loss for steering ACE-Step "
        f"audio generation toward the **{concept}** concept. `alpha` modulates the "
        f"active LoRA weight; `set_alpha` swaps weights without reloading.\n\n"
        f"- Rank: r{rank}\n"
        f"- Target layers: {layers}\n\n"
        f"{PAPER_SECTION}"
        f"## Files\n\n"
        f"- `pytorch_lora_weights.safetensors` — the LoRA weights.\n"
        f"- `train_config.json` — training hyperparameters.\n\n"
        f"## Quickstart\n\n"
        f"```python\n{quickstart}```\n\n"
        f"## Training config\n\n"
        f"```json\n{json.dumps(train_config, indent=2)}\n```\n"
    )


def _render_tokemb_card(
    concept: str,
    repo_id: str,
    tags: Iterable[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Model card for a TokEmb direction artifact."""
    base_tags = [
        "audio",
        "music",
        "diffusion",
        "ace-step",
        "tokemb",
        "token-embedding",
        "interpretability",
        "steering",
    ]
    all_tags = sorted(set(base_tags) | set(tags or []))
    cfg_yaml = "\n".join(f"  - {t}" for t in all_tags)
    quickstart = (
        "from src.steering import SteerableACEModel, TokEmbSteeringController\n\n"
        'model = SteerableACEModel(device="cuda")\n'
        "model.pipeline.load()\n"
        f"ctrl = TokEmbSteeringController.from_pretrained(\n"
        f'    "{repo_id}", alpha=1.0, te_split_step=3,\n'
        f")\n\n"
        "with model.steer(ctrl):\n"
        "    audio = model.generate(\n"
        '        prompt="instrumental music", lyrics="[inst]",\n'
        "        audio_duration=10.0, infer_step=30, manual_seed=0,\n"
        "    )\n"
    )
    extra_block = ""
    if extra:
        extra_block = f"\n## Metadata\n\n```json\n{json.dumps(extra, indent=2)}\n```\n"
    return (
        f"---\n"
        f"library_name: audio-interv\n"
        f"tags:\n{cfg_yaml}\n"
        f"---\n\n"
        f"# Token-Embedding (TokEmb) Direction — `{concept}` (ACE-Step)\n\n"
        f"Per-concept T5 hidden-state direction for steering ACE-Step audio "
        f"generation toward the **{concept}** concept. At inference, "
        f"`TokEmbSteeringController` adds `alpha * direction` to the concept "
        f"token's hidden state in the neutral prompt embedding before the "
        f"diffusion process.\n\n"
        f"The `.pt` file is a dict with keys: `direction` (Tensor[hidden_dim]), "
        f"`concept` (str), `hidden_dim` (int).\n\n"
        f"{PAPER_SECTION}"
        f"## Quickstart\n\n"
        f"```python\n{quickstart}```\n"
        f"{extra_block}"
    )


# ---------------------------------------------------------------------------
# SAE checkpoints — wrapper around the existing Sae.save_to_disk / load_from_hub
# ---------------------------------------------------------------------------


def push_sae_to_hub(
    sae: Any,
    repo_id: str,
    *,
    hookpoint: str | None = None,
    private: bool = False,
    tags: Iterable[str] | None = None,
    commit_message: str = "Upload SAE checkpoint",
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Push a trained ``Sae`` to its own Hub repo (one repo per checkpoint).

    Layout matches what ``Sae.load_from_hub`` already understands::

        <repo>/
          [<hookpoint>/]
            cfg.json
            sae.safetensors
          README.md
    """
    import tempfile

    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, private=private, exist_ok=True, repo_type="model")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        target = tmp_root / hookpoint if hookpoint else tmp_root
        sae.save_to_disk(target)
        (tmp_root / "README.md").write_text(_render_sae_card(sae, repo_id, hookpoint, tags=tags, extra=extra_metadata))
        api = HfApi()
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(tmp_root),
            commit_message=commit_message,
            repo_type="model",
        )
    return f"https://huggingface.co/{repo_id}"


def _render_sae_card(
    sae: Any,
    repo_id: str,
    hookpoint: str | None,
    tags: Iterable[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    base_tags = [
        "audio",
        "music",
        "ace-step",
        "interpretability",
        "sparse-autoencoder",
        "sae",
    ]
    all_tags = sorted(set(base_tags) | set(tags or []))
    cfg_yaml = "\n".join(f"  - {t}" for t in all_tags)
    cfg_dict = sae.cfg.to_dict() if hasattr(sae, "cfg") else {}
    meta = json.dumps({**cfg_dict, **(extra or {}), "d_in": getattr(sae, "d_in", None)}, indent=2)
    hp_line = f"\n**Hookpoint:** `{hookpoint}`" if hookpoint else ""
    return (
        f"---\n"
        f"library_name: audio-interv\n"
        f"tags:\n{cfg_yaml}\n"
        f"---\n\n"
        f"# SAE — ACE-Step{hp_line}\n\n"
        f"Sparse autoencoder trained on ACE-Step activations.\n\n"
        f"{PAPER_SECTION}"
        f"## Quickstart\n\n"
        f"```python\n"
        f"from src.steering.methods.sae.lib.sae.sae import Sae\n\n"
        f'sae = Sae.load_from_hub("{repo_id}"'
        f"{f', hookpoint={hookpoint!r}' if hookpoint else ''})\n"
        f"```\n\n"
        f"## Config\n\n```json\n{meta}\n```\n"
    )
