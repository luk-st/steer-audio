"""
Utilities for computing steering vectors.

Contains functions for:
- Getting prompt pairs from config
- Collecting activations (standard and raw)
- Computing steering vectors (standard and frequency-based)
"""

import os
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

PATH_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.append(PATH_PROJECT)
sys.path.append(os.path.join(PATH_PROJECT, "src", "models", "ace_step", "ACE"))

from src.steering.methods.sae.lib.configs.steer_prompts import CONCEPT_TO_PROMPTS
from src.models.ace_step.ace_steering.controller import (
    VectorControl,
    VectorStore,
    compute_num_cfg_passes,
    register_vector_control,
)


# =============================================================================
# Prompt Utilities
# =============================================================================


def get_prompts_pairs(concept: str):
    """
    Get prompt pairs from CONCEPT_TO_PROMPTS config.

    Args:
        concept: Concept name (e.g., 'piano', 'mood', 'tempo')

    Returns:
        prompts_pos: List of positive prompts
        prompts_neg: List of negative prompts
        lyrics: Lyrics setting for the concept
    """
    print(f"\nGenerating prompts for concept: {concept}")

    if concept not in CONCEPT_TO_PROMPTS:
        raise ValueError(
            f"Unknown concept: {concept}. Available concepts: {list(CONCEPT_TO_PROMPTS.keys())}"
        )

    prompts_neg, prompts_pos, lyrics = CONCEPT_TO_PROMPTS[concept]()
    print(f"Generated {len(prompts_pos)} prompt pairs")

    return prompts_pos, prompts_neg, lyrics


# =============================================================================
# Standard Activation Collection (averages over time)
# =============================================================================


def generate_vectors_standard(
    prompts_pos,
    prompts_neg,
    pipe,
    device: str,
    save_all_cfg_passes: bool,
    audio_duration: float,
    num_inference_steps: int,
    seed: int,
    guidance_scale_text: float,
    guidance_scale_lyric: float,
    guidance_scale: float,
    guidance_interval: float,
    guidance_interval_decay: float,
):
    """
    Collect activations using VectorStore (averages over time).

    Used for standard CAA steering vectors.

    Returns:
        pos_vectors: List of {step: {layer: [tensor]}} for positive prompts
        neg_vectors: List of {step: {layer: [tensor]}} for negative prompts
    """
    num_cfg_passes = compute_num_cfg_passes(
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
    )

    pos_vectors = []
    neg_vectors = []

    for i, (prompt_pos, prompt_neg) in tqdm(
        enumerate(zip(prompts_pos, prompts_neg)),
        total=len(prompts_pos),
        desc="Collecting activations for pairs",
    ):
        # Positive prompt
        controller = VectorStore(
            device=device,
            save_only_cond=not save_all_cfg_passes,
            num_cfg_passes=num_cfg_passes,
        )
        controller.steer = False  # Just collecting activations, not steering
        register_vector_control(pipe.ace_step_transformer, controller)

        _ = pipe.generate(
            prompt=prompt_pos,
            audio_duration=audio_duration,
            infer_step=num_inference_steps,
            manual_seed=seed,
            return_type="latent",
            use_erg_lyric=False,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
        )

        pos_vectors.append(controller.vector_store)
        controller.reset()

        # Negative prompt
        controller = VectorStore(
            device=device,
            save_only_cond=not save_all_cfg_passes,
            num_cfg_passes=num_cfg_passes,
        )
        controller.steer = False
        register_vector_control(pipe.ace_step_transformer, controller)

        _ = pipe.generate(
            prompt=prompt_neg,
            audio_duration=audio_duration,
            infer_step=num_inference_steps,
            manual_seed=seed,
            return_type="latent",
            use_erg_lyric=False,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
        )
        neg_vectors.append(controller.vector_store)
        controller.reset()

    return pos_vectors, neg_vectors


def compute_standard_steering_vectors(pos_vectors, neg_vectors):
    """
    Compute standard steering vectors (averaged over time).

    Args:
        pos_vectors: List of {step: {layer: [tensor]}} for positive prompts
        neg_vectors: List of {step: {layer: [tensor]}} for negative prompts

    Returns:
        steering_vectors: {step: {layer: [normalized_sv]}}
    """
    print("\nComputing steering vectors...")
    steering_vectors = {}
    all_step_keys = list(pos_vectors[0].keys())
    layer_names = list(pos_vectors[0][all_step_keys[0]].keys())

    for step_key in all_step_keys:
        steering_vectors[step_key] = defaultdict(list)
        for layer_name in layer_names:
            pos_vectors_layer = [
                pos_vectors[i][step_key][layer_name][0] for i in range(len(pos_vectors))
            ]
            pos_vectors_avg = np.mean(pos_vectors_layer, axis=0)

            neg_vectors_layer = [
                neg_vectors[i][step_key][layer_name][0] for i in range(len(neg_vectors))
            ]
            neg_vectors_avg = np.mean(neg_vectors_layer, axis=0)

            steering_vector = pos_vectors_avg - neg_vectors_avg

            norm = np.linalg.norm(steering_vector)
            if norm > 0:
                steering_vector = steering_vector / norm

            steering_vectors[step_key][layer_name].append(steering_vector)

    return steering_vectors


# =============================================================================
# Raw Activation Collector (preserves time dimension for frequency steering)
# =============================================================================


class RawActivationCollector(VectorControl):
    """
    Collects raw activations WITHOUT averaging, preserving freq and time dimensions.

    Used for computing frequency-aware steering vectors that preserve temporal structure.
    """

    def __init__(self, device="cpu", save_only_cond=True, num_cfg_passes=None):
        super().__init__()
        self.device = device
        self.save_only_cond = save_only_cond
        self.num_cfg_passes = num_cfg_passes
        self.cfg_pass_count = 0
        self.actual_denoising_step = 0

        # Store raw activations: {step: {layer: activation_tensor}}
        self.raw_activations = defaultdict(dict)
        self.step_store = defaultdict(list)

    def reset(self):
        super().reset()
        self.raw_activations = defaultdict(dict)
        self.step_store = defaultdict(list)
        self.cfg_pass_count = 0
        self.actual_denoising_step = 0

    def forward(self, vector, place_in_ace: str):
        """Store raw activations without averaging."""
        should_save = (not self.save_only_cond) or (self.cfg_pass_count == 0)

        if should_save:
            # Store raw tensor - shape: [batch, time, freq]
            # Convert to float32 for numpy compatibility
            raw_tensor = vector.detach().cpu().float()
            self.step_store[place_in_ace].append(raw_tensor)

        return vector

    def between_steps(self):
        has_data = self.step_store and any(self.step_store.values())

        if self.save_only_cond:
            if has_data:
                # Store collected activations for this step
                for layer, tensors in self.step_store.items():
                    # tensors is a list with one tensor per layer call
                    self.raw_activations[self.actual_denoising_step][layer] = tensors
                self.actual_denoising_step += 1
                self.cfg_pass_count = 1
            else:
                self.cfg_pass_count += 1
                if (
                    self.num_cfg_passes is not None
                    and self.cfg_pass_count >= self.num_cfg_passes
                ):
                    self.cfg_pass_count = 0

        # Clear step store
        self.step_store = defaultdict(list)


def collect_raw_activations(
    prompts_pos,
    prompts_neg,
    pipe,
    layers_to_steer,
    device: str,
    audio_duration: float,
    num_inference_steps: int,
    seed: int,
    guidance_scale_text: float,
    guidance_scale_lyric: float,
    guidance_scale: float,
    guidance_interval: float,
    guidance_interval_decay: float,
):
    """
    Collect raw activations preserving full [time, freq] shape.

    Used for frequency-based steering vector computation.

    Returns:
        pos_activations: List of {step: {layer: [tensors]}} for positive prompts
        neg_activations: List of {step: {layer: [tensors]}} for negative prompts
    """
    num_cfg_passes = compute_num_cfg_passes(
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
    )

    pos_activations = []
    neg_activations = []

    for i, (prompt_pos, prompt_neg) in tqdm(
        enumerate(zip(prompts_pos, prompts_neg)),
        total=len(prompts_pos),
        desc="Collecting raw activations",
    ):
        # Positive prompt
        collector_pos = RawActivationCollector(
            device=device,
            save_only_cond=True,
            num_cfg_passes=num_cfg_passes,
        )
        register_vector_control(
            pipe.ace_step_transformer,
            collector_pos,
            verbose=False,
            explicit_layers=layers_to_steer,
        )

        _ = pipe.generate(
            prompt=prompt_pos,
            audio_duration=audio_duration,
            infer_step=num_inference_steps,
            manual_seed=seed,
            return_type="latent",
            use_erg_lyric=False,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
        )
        pos_activations.append(collector_pos.raw_activations)

        # Negative prompt
        collector_neg = RawActivationCollector(
            device=device,
            save_only_cond=True,
            num_cfg_passes=num_cfg_passes,
        )
        register_vector_control(
            pipe.ace_step_transformer,
            collector_neg,
            verbose=False,
            explicit_layers=layers_to_steer,
        )

        _ = pipe.generate(
            prompt=prompt_neg,
            audio_duration=audio_duration,
            infer_step=num_inference_steps,
            manual_seed=seed,
            return_type="latent",
            use_erg_lyric=False,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
        )
        neg_activations.append(collector_neg.raw_activations)

    return pos_activations, neg_activations


# =============================================================================
