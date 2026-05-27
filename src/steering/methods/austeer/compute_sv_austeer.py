"""
Compute AUSteer-style activation momentum scores for ACE-Step model.

Instead of computing a mean difference vector (CAA), this computes per-frequency
discriminative scores using activation momentum (AUSteer, ICLR 2026).

Each frequency bin is scored by the *consistency* of its activation difference
across contrastive pairs and timeframes. Scores are stored per layer per step.

The global top-k selection (across all active layers) happens at run time in
AUSteerSteeringController, since the set of active layers may vary between runs.

Usage:
    python steering/caa/compute_sv_austeer.py \
        --concept piano \
        --layers all \
        --num_inference_steps 30 \
        --guidance_scale 5.0 \
        --seed 10
"""

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from tqdm import tqdm

PATH_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PATH_PROJECT)
sys.path.append(os.path.join(PATH_PROJECT, "src", "models", "ace_step", "ACE"))

from src.models.ace_step.ace_steering.controller import compute_num_cfg_passes
from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline
from src.steering.methods.caa.utils import (
    DEFAULT_DEVICE,
    LAYER_CONFIGS,
    collect_raw_activations,
    get_prompts_pairs,
)

DEFAULT_SAVE_DIR = "steering_vectors/austeer"


def compute_austeer_scores(
    pos_activations,
    neg_activations,
    verbose=True,
):
    """
    Compute AUSteer activation momentum scores per frequency bin.

    For each contrastive pair and each timeframe, computes:
        m_i = pos_act_i - neg_act_i  (per-freq momentum)

    Then scores each freq bin by consistency:
        r_pos_i = fraction of samples where m_i > 0
        r_neg_i = fraction of samples where m_i < 0
        beta_i  = sign * max(r_pos, r_neg)

    All timeframes across all contrastive pairs are treated as independent
    samples for the momentum calculation.

    Top-k selection is NOT done here — it happens at run time in
    AUSteerSteeringController, which performs global top-k across all active layers
    (matching the original AUSteer paper).

    Args:
        pos_activations: List of {step: {layer: [tensors]}} for positive prompts
        neg_activations: List of {step: {layer: [tensors]}} for negative prompts
        verbose: Print progress

    Returns:
        austeer_vectors: {step: {layer: {"betas": [N_FREQS], "scores": [N_FREQS]}}}
    """
    austeer_vectors = {}

    steps = sorted(pos_activations[0].keys())
    layers = list(pos_activations[0][steps[0]].keys())

    if verbose:
        print(f"Computing AUSteer scores for {len(steps)} steps, {len(layers)} layers")

    iterator = tqdm(steps, desc="Processing steps") if verbose else steps

    for step in iterator:
        austeer_vectors[step] = {}

        for layer in layers:
            # Collect all (pos, neg) activation pairs across prompts
            # Each tensor is [batch=1, time, freq=2560]
            pos_tensors = []
            neg_tensors = []

            for prompt_idx in range(len(pos_activations)):
                if (
                    step in pos_activations[prompt_idx]
                    and layer in pos_activations[prompt_idx][step]
                ):
                    tensor = pos_activations[prompt_idx][step][layer][0]
                    # tensor shape: [1, time, freq] -> squeeze batch -> [time, freq]
                    arr = tensor.numpy()
                    if arr.ndim == 3:
                        arr = arr[0]
                    pos_tensors.append(arr)

            for prompt_idx in range(len(neg_activations)):
                if (
                    step in neg_activations[prompt_idx]
                    and layer in neg_activations[prompt_idx][step]
                ):
                    tensor = neg_activations[prompt_idx][step][layer][0]
                    arr = tensor.numpy()
                    if arr.ndim == 3:
                        arr = arr[0]
                    neg_tensors.append(arr)

            if not pos_tensors or not neg_tensors:
                continue

            # Each tensor is [time, freq]. Concatenate all timeframes from all pairs
            # into a single sample dimension: [N_samples, freq]
            # where N_samples = num_pairs * time_dim
            assert len(pos_tensors) == len(neg_tensors), (
                f"Mismatched pos/neg counts: {len(pos_tensors)} vs {len(neg_tensors)}"
            )

            # Stack per-pair: compute momentum per timeframe per pair
            all_momentums = []
            for pos_arr, neg_arr in zip(pos_tensors, neg_tensors):
                # pos_arr, neg_arr: [time, freq]
                momentum = pos_arr - neg_arr  # [time, freq]
                all_momentums.append(momentum)

            # [N_samples, freq] where N_samples = sum of time_dims across pairs
            all_momentums = np.concatenate(all_momentums, axis=0)
            n_samples = all_momentums.shape[0]
            n_freqs = all_momentums.shape[1]

            if verbose and step == steps[0]:
                print(
                    f"  Step {step}, Layer {layer}: "
                    f"{n_samples} samples (timeframes×pairs), {n_freqs} freqs"
                )

            # Compute discriminative scores per freq bin
            # r_pos_i = fraction of samples where momentum > 0
            # r_neg_i = fraction of samples where momentum < 0
            r_pos = (all_momentums > 0).sum(axis=0) / n_samples  # [freq]
            r_neg = (all_momentums < 0).sum(axis=0) / n_samples  # [freq]

            # Discriminative score: max(r_pos, r_neg)
            scores = np.maximum(r_pos, r_neg)  # [freq]

            # Signed beta: positive if promotes concept, negative if suppresses
            betas = np.where(r_pos >= r_neg, scores, -scores)  # [freq]

            austeer_vectors[step][layer] = {
                "betas": betas,
                "scores": scores,
            }

    return austeer_vectors


def main(
    concept: str,
    layers: str = "all",
    num_inference_steps: int = 30,
    audio_duration: float = 30.0,
    guidance_scale_text: float = 0.0,
    guidance_scale_lyric: float = 0.0,
    guidance_scale: float = 5.0,
    guidance_interval: float = 1.0,
    guidance_interval_decay: float = 0.0,
    seed: int = 10,
    device: str = DEFAULT_DEVICE,
    save_dir: str = DEFAULT_SAVE_DIR,
):
    """
    Compute AUSteer activation momentum scores for a concept.

    Scores are saved per layer per step. The global top-k selection happens
    at run time in AUSteerSteeringController, since the active
    layer set (and therefore the global ranking) may vary between runs.

    Args:
        concept: Concept name from CONCEPT_TO_PROMPTS
        layers: Which layers to collect from ('all', 'tf6', 'tf7', 'tf6tf7', etc.)
        num_inference_steps: Number of diffusion steps
        audio_duration: Audio duration in seconds
        guidance_scale: Overall guidance scale
        seed: Random seed for generation
        device: Device to run on
        save_dir: Directory to save AUSteer scores
    """
    # Validate layers
    if layers not in LAYER_CONFIGS:
        raise ValueError(
            f"Unknown layers: {layers}. Available: {list(LAYER_CONFIGS.keys())}"
        )
    layers_to_collect = LAYER_CONFIGS[layers]

    # Initialize pipeline
    print("Loading ACE-Step pipeline...")
    pipe = SimpleACEStepPipeline(device=device)
    pipe.load()
    print("Pipeline loaded")

    # Get prompts
    prompts_pos, prompts_neg, lyrics = get_prompts_pairs(concept)

    # Collect raw activations (preserving time dimension)
    print("Collecting raw activations...")
    pos_activations, neg_activations = collect_raw_activations(
        prompts_pos=prompts_pos,
        prompts_neg=prompts_neg,
        pipe=pipe,
        layers_to_steer=layers_to_collect,
        device=device,
        audio_duration=audio_duration,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
        guidance_scale=guidance_scale,
        guidance_interval=guidance_interval,
        guidance_interval_decay=guidance_interval_decay,
    )

    # Compute AUSteer scores
    print("Computing AUSteer activation momentum scores...")
    austeer_vectors = compute_austeer_scores(
        pos_activations=pos_activations,
        neg_activations=neg_activations,
        verbose=True,
    )

    # Save
    save_directory = Path(save_dir).resolve()
    vector_directory = f"ace_{concept}_{layers}"
    save_directory = (save_directory / vector_directory).resolve()
    os.makedirs(save_directory, exist_ok=True)

    with open(save_directory / "austeer.pkl", "wb") as f:
        pickle.dump(austeer_vectors, f)

    with open(save_directory / "config.json", "w") as f:
        json.dump(
            {
                "method": "austeer",
                "concept": concept,
                "lyrics": lyrics,
                "layers": layers,
                "layers_collected": layers_to_collect,
                "num_inference_steps": num_inference_steps,
                "audio_duration": audio_duration,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "guidance_scale_text": guidance_scale_text,
                "guidance_scale_lyric": guidance_scale_lyric,
                "guidance_interval": guidance_interval,
                "guidance_interval_decay": guidance_interval_decay,
            },
            f,
            indent=2,
        )

    # Print summary
    sample_step = list(austeer_vectors.keys())[0]
    n_layers = len(austeer_vectors[sample_step])
    sample_layer = list(austeer_vectors[sample_step].keys())[0]
    sample_data = austeer_vectors[sample_step][sample_layer]
    n_freqs = len(sample_data["betas"])
    max_score = sample_data["scores"].max()
    mean_score = sample_data["scores"].mean()

    print(f"\nAUSteer scores saved to: {save_directory}")
    print(f"Files:")
    print(f"  - austeer.pkl: AUSteer vectors (betas, scores per layer)")
    print(f"  - config.json: Configuration")
    print(f"\nSummary:")
    print(f"  {n_layers} layers, {n_freqs} freqs each")
    print(f"  Total pool: {n_layers * n_freqs} (layer, freq) pairs per step")
    print(f"  Score range (layer {sample_layer}): mean={mean_score:.4f}, max={max_score:.4f}")
    print(f"\nTop-k selection happens at run time via --k in run_eval.py")


if __name__ == "__main__":
    Fire(main)
