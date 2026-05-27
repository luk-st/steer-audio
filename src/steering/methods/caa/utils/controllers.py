"""
Steering controllers for applying steering vectors during generation.
"""

import os
import sys
from collections import defaultdict

import numpy as np
import torch

PATH_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.append(PATH_PROJECT)
sys.path.append(os.path.join(PATH_PROJECT, "src", "models", "ace_step", "ACE"))

from src.models.ace_step.ace_steering.controller import VectorControl


class AUSteerSteeringController(VectorControl):
    """
    Controller that applies AUSteer-style sparse steering using activation
    momentum scoring for dimension selection.

    Uses activation momentum to identify the top-k most discriminative
    freq bins **globally across all active layers** (AUSteer paper).

    The betas vector is used directly as a sparse steering vector:
        sv_austeer = [beta_1, beta_2, ..., 0, 0, ...]  (top-k nonzero)

    Supports two modes:
    - "additive": h += alpha * sv_austeer
        Betas encode both direction (sign) and importance (magnitude).
        Comparable to CAA but sparse and momentum-scored.
    - "multiplicative": Original AUSteer paper formulation.
        h *= (1 + alpha * beta_i)  for selected i

    Data format (from compute_sv_austeer.py):
        austeer_vectors: {step: {layer: {"betas": [N_FREQS], "scores": [N_FREQS]}}}
    """

    def __init__(
        self,
        austeer_vectors,
        k=256,
        alpha=10.0,
        active_layers=None,
        mode="additive",
        device="cuda",
        steer=True,
        steer_mode="cond_only",
        num_cfg_passes=2,
        renorm=False,
        start_step=0,
    ):
        """
        Args:
            austeer_vectors: {step: {layer: {"betas": [N_FREQS], "scores": [N_FREQS]}}}
            k: Global top-k budget across all active layers per step.
            alpha: Global steering strength multiplier.
            active_layers: List of layer names to consider (e.g. ["tf6", "tf7"]).
                If None, uses all layers present in the data.
            mode: "additive" or "multiplicative".
            renorm: If True, renormalize activations after steering (per-timeframe).
        """
        super().__init__()
        self.austeer_vectors = austeer_vectors
        self.k = k
        self.alpha = alpha
        self.mode = mode
        self.device = device
        self.steer = steer
        self.steer_mode = steer_mode
        self.num_cfg_passes = num_cfg_passes
        self.renorm = renorm
        self.start_step = int(start_step)

        self.cfg_pass_count = 0
        self.actual_denoising_step = 0
        self.step_store = defaultdict(list)

        # Precompute sparse steering vectors per step per layer:
        # sv_austeer = betas * mask (top-k nonzero, rest zeroed)
        self._sv, self._masks = self._compute_sparse_vectors(active_layers)

    def _compute_sparse_vectors(self, active_layers):
        """Compute per-step, per-layer sparse steering vectors via global top-k.

        For each step, pool all (layer, freq) |beta| scores across active layers,
        rank them, select top-k, and build masked beta vectors.

        Returns:
            sv:    {step: {layer: np.array([N_FREQS])}}  — betas zeroed outside top-k
            masks: {step: {layer: np.array([N_FREQS])}}  — binary masks
        """
        sv = {}
        masks = {}
        for step, layer_data in self.austeer_vectors.items():
            layers = active_layers if active_layers else list(layer_data.keys())

            # Collect all (|beta|, layer, freq_idx) entries
            entries = []
            for layer in layers:
                if layer not in layer_data:
                    continue
                abs_betas = np.abs(layer_data[layer]["betas"])
                for fi, ab in enumerate(abs_betas):
                    entries.append((ab, layer, fi))

            # Rank and select top-k globally
            entries.sort(key=lambda x: x[0], reverse=True)
            actual_k = min(self.k, len(entries))

            step_masks = {
                layer: np.zeros(len(layer_data[layer]["betas"]))
                for layer in layers
                if layer in layer_data
            }
            for _, layer, fi in entries[:actual_k]:
                step_masks[layer][fi] = 1.0

            # Build sparse SV: betas * mask
            step_sv = {}
            for layer in layers:
                if layer not in layer_data:
                    continue
                step_sv[layer] = layer_data[layer]["betas"] * step_masks[layer]

            sv[step] = step_sv
            masks[step] = step_masks

        return sv, masks

    def reset(self):
        super().reset()
        self.cfg_pass_count = 0
        self.actual_denoising_step = 0
        self.step_store = defaultdict(list)

    def forward(self, vector, place_in_ace: str):  # pyright: ignore
        """Apply AUSteer sparse steering.

        Input vector shape: [batch, time, freq] = [1, 323, 2560]
        """
        should_steer = False
        # Skip the first start_step diffusion steps (no steering applied yet)
        if self.steer and self.actual_denoising_step >= self.start_step:
            if self.steer_mode == "cond_only":
                should_steer = self.cfg_pass_count == 0
            elif self.steer_mode == "both_cond":
                should_steer = True

        if should_steer and self.alpha != 0:
            step_key = self.actual_denoising_step
            if step_key not in self._sv:
                return vector
            if place_in_ace not in self._sv[step_key]:
                return vector

            # Save original norms for renormalization (per timeframe)
            if self.renorm:
                orig_norm = torch.norm(vector, dim=2, keepdim=True)  # [batch, time, 1]

            if self.mode == "additive":
                # h += alpha * sv_austeer  where sv_austeer = betas * mask
                sparse_sv = self._sv[step_key][place_in_ace]  # [N_FREQS]
                sv_tensor = torch.tensor(
                    sparse_sv, dtype=vector.dtype, device=vector.device
                ).view(1, 1, -1)
                vector = vector + self.alpha * sv_tensor.expand(1, vector.size()[1], -1)

            elif self.mode == "multiplicative":
                # Original AUSteer: h *= (1 + alpha * beta_i) for selected i
                sparse_sv = self._sv[step_key][place_in_ace]  # [N_FREQS]
                gamma_tensor = torch.tensor(
                    self.alpha * sparse_sv, dtype=vector.dtype, device=vector.device
                )
                vector = vector * (1.0 + gamma_tensor.unsqueeze(0).unsqueeze(0))

            if self.renorm:
                new_norm = torch.norm(vector, dim=2, keepdim=True)
                vector = vector / (new_norm + 1e-8) * orig_norm

        return vector

    def between_steps(self):
        self.cfg_pass_count += 1
        if self.cfg_pass_count >= self.num_cfg_passes:
            self.cfg_pass_count = 0
            self.actual_denoising_step += 1
        self.step_store = defaultdict(list)
