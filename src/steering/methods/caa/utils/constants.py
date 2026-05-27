"""
Shared constants for CAA steering vector computation and evaluation.
"""

import torch

# =============================================================================
# Device Configuration
# =============================================================================

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Generation Parameters
# =============================================================================

GENERATION_SEED = 2115
GUIDANCE_SCALE = 5.0
AUDIO_LENGTH_IN_S = 30.0
NUM_INFERENCE_STEPS = 30

# =============================================================================
# Alpha Multipliers for Steering
# =============================================================================

MULTIPLIERS = [
    -100.0,
    -90.0,
    -80.0,
    -70.0,
    -60.0,
    -50.0,
    -40.0,
    -30.0,
    -20.0,
    -10.0,
    0.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
]

# =============================================================================
# Layer Configurations
# =============================================================================

LAYER_CONFIGS = {
    "all": [f"tf{i}" for i in range(24)],
    "tf6": ["tf6"],
    "tf7": ["tf7"],
    "tf6tf7": ["tf6", "tf7"],
    "no_tf6tf7": [f"tf{i}" for i in range(24) if i not in [6, 7]],
}

CONCEPT_TO_NEUTRAL_ADDON = {
    "piano": "{p}, with instrument",
    "mood": "a song, {p}",
    "tempo": "a song, {p}",
    "vocal_gender": "{p}, with clean vocal",
    "drums": "{p}, with instrument",
    "vocal_style": "{p}, with clean vocal",
    "guitar_electronic": "{p}, with a guitar",
    "violin": "{p}, with instrument",
    "rock_genre": "a song, {p}",
    "electronic_music": "a song, {p}",
}