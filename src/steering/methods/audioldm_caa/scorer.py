"""AudioLDM2 CAA scorer — computes steering vectors for one concept.

Wraps the existing ``steering/caa/compute_sv_caa_audioldm.py`` main()-equivalent
into the unified :class:`Scorer` interface.
"""

from __future__ import annotations

import copy
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.steering import Scorer
from src.steering.scorer import register_scorer
from src.steering.model import SteerableACEModel  # only for type hints; we use AudioLDM


@register_scorer("audioldm_caa")
class AudioLDMCaaScorer(Scorer):
    """Compute CAA steering vectors for AudioLDM2 and save them on disk.

    Artifact layout (consumable by ``AudioLDMCAASteeringController.from_pretrained``)::

        <output_dir>/audioldm_<concept>_all<save_all>_norm<norm>_<layers>/
          config.json
          sv.pkl
          pos_vectors.pkl
          neg_vectors.pkl
    """

    def compute(
        self,
        model: Any,
        output_dir: str | Path,
        *,
        concept: str,
        num_inference_steps: int = 30,
        audio_length_in_s: float = 10.0,
        guidance_scale: float = 3.5,
        seed: int = 10,
        save_all_cfg_passes: bool = True,
        layers: str = "all",
        normalize_sv: bool = True,
        device: str = "cuda",
        dtype: str = "float16",
        repo_id: str | None = None,
        **_: Any,
    ) -> Path:
        # Uses SteeredAudioLDMPipeline directly (not SteerableAudioLDMModel):
        # it manages both the diffusers pipe and the controller for the
        # collect-then-snapshot pass.
        from src.models.audioldm.audioldm_steering.controller import (
            resolve_audioldm_layers,
        )
        from src.models.audioldm.steering_audioldm import (
            DEFAULT_REPO_ID,
            SteeredAudioLDMPipeline,
        )
        from src.steering.methods.caa.utils import get_prompts_pairs

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        torch_dtype = dtype_map[dtype]
        layers_to_steer = resolve_audioldm_layers(layers)

        print(f"Loading {repo_id or DEFAULT_REPO_ID} on {device} ({dtype})...")
        pipe = SteeredAudioLDMPipeline(
            repo_id=repo_id or DEFAULT_REPO_ID,
            device=device,
            dtype=torch_dtype,
        )
        pipe.load()

        prompts_pos, prompts_neg, _ = get_prompts_pairs(concept)
        print(f"Collected {len(prompts_pos)} prompt pairs for concept={concept!r}")

        pipe.setup_steering(
            steering_vectors=None,
            layers_to_steer=layers_to_steer,
            steer=False,
            alpha=0.0,
            save_only_cond=not save_all_cfg_passes,
            verbose=True,
        )

        from tqdm import tqdm

        pos_vectors: list[dict] = []
        neg_vectors: list[dict] = []
        for prompt_pos, prompt_neg in tqdm(
            list(zip(prompts_pos, prompts_neg)),
            desc="Collecting activations",
        ):
            for prompt, store in ((prompt_pos, pos_vectors), (prompt_neg, neg_vectors)):
                pipe.generate(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    audio_length_in_s=audio_length_in_s,
                    seed=seed,
                )
                store.append(copy.deepcopy(dict(pipe.controller.vector_store)))

        print("Computing steering vectors...")
        steering_vectors = _compute_steering_vectors(
            pos_vectors, neg_vectors, normalize=normalize_sv
        )

        save_dir = (
            Path(output_dir).resolve()
            / f"audioldm_{concept}_all{save_all_cfg_passes}_norm{normalize_sv}_{layers}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "sv.pkl").open("wb") as f:
            pickle.dump(steering_vectors, f)
        with (save_dir / "pos_vectors.pkl").open("wb") as f:
            pickle.dump(pos_vectors, f)
        with (save_dir / "neg_vectors.pkl").open("wb") as f:
            pickle.dump(neg_vectors, f)
        with (save_dir / "config.json").open("w") as f:
            json.dump(
                {
                    "method": "standard_caa_audioldm",
                    "model": repo_id or DEFAULT_REPO_ID,
                    "concept": concept,
                    "num_inference_steps": num_inference_steps,
                    "audio_length_in_s": audio_length_in_s,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "device": device,
                    "dtype": dtype,
                    "save_all_cfg_passes": save_all_cfg_passes,
                    "layers_preset": layers,
                    "layers_to_steer": layers_to_steer,
                    "normalize_sv": normalize_sv,
                },
                f,
                indent=2,
            )

        print(f"Steering vectors saved to: {save_dir}")
        return save_dir


def _compute_steering_vectors(
    pos_vectors: list[dict],
    neg_vectors: list[dict],
    normalize: bool,
) -> dict:
    """pos-mean − neg-mean per (step, layer); optionally L2-normalize."""
    if not pos_vectors or not neg_vectors:
        raise ValueError("Empty pos or neg activation lists.")

    step_keys = list(pos_vectors[0].keys())
    layer_names = list(pos_vectors[0][step_keys[0]].keys())

    sv: dict = {}
    for step_key in step_keys:
        sv[step_key] = defaultdict(list)
        for layer_name in layer_names:
            pos_stack = np.stack(
                [pv[step_key][layer_name][0] for pv in pos_vectors], axis=0
            )
            neg_stack = np.stack(
                [nv[step_key][layer_name][0] for nv in neg_vectors], axis=0
            )
            # CAA: difference between positive and negative mean activations averaged over all prompts
            v = pos_stack.mean(axis=0) - neg_stack.mean(axis=0)
            if normalize:
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
            sv[step_key][layer_name].append(v)
    return sv
