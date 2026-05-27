"""Steering utilities for CAA (Contrastive Activation Addition).

This package contains shared utilities for computing and evaluating
steering vectors for audio diffusion models.
"""

from src.steering.methods.caa.utils.constants import (
    AUDIO_LENGTH_IN_S,
    DEFAULT_DEVICE,
    GENERATION_SEED,
    GUIDANCE_SCALE,
    LAYER_CONFIGS,
    MULTIPLIERS,
    NUM_INFERENCE_STEPS,
)
from src.steering.methods.caa.utils.eval_utils import (
    eval_audios,
    load_sv_config,
    validate_sv_config,
    warn_config_mismatches,
)
from src.steering.methods.caa.utils.compute_sv_utils import (
    RawActivationCollector,
    collect_raw_activations,
    compute_standard_steering_vectors,
    generate_vectors_standard,
    get_prompts_pairs,
)
from src.steering.methods.caa.utils.controllers import AUSteerSteeringController

__all__ = [
    # Constants
    "AUDIO_LENGTH_IN_S",
    "DEFAULT_DEVICE",
    "GENERATION_SEED",
    "GUIDANCE_SCALE",
    "LAYER_CONFIGS",
    "MULTIPLIERS",
    "NUM_INFERENCE_STEPS",
    # Eval utilities
    "eval_audios",
    "load_sv_config",
    "validate_sv_config",
    "warn_config_mismatches",
    # Compute SV utilities
    "RawActivationCollector",
    "collect_raw_activations",
    "compute_standard_steering_vectors",
    "generate_vectors_standard",
    "get_prompts_pairs",
    # Controllers
    "AUSteerSteeringController",
]
