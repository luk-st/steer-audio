# based on https://github.com/ace-step/ACE-Step/blob/main/acestep/pipeline_ace_step.py

import os

import torch
import torchaudio
from diffusers.utils.torch_utils import randn_tensor
from typing import Literal

from src.models.ace_step.ACE.acestep.cpu_offload import cpu_offload
from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline, SAMPLE_RATE
from src.models.ace_step.ace_steering.controller import (
    VectorStore,
    register_vector_control,
)


class SteeredACEStepPipeline(SimpleACEStepPipeline):
    def __init__(
        self,
        repo_id="",  # mock
        device="cpu",
        dtype="bfloat16",
        persistent_storage_path="res/ace_step",
        torch_compile=False,
        cpu_offload=False,
        quantized=False,
        overlapped_decode=False,
        pad_to_max_len=None,
        steering_vectors=None,
        steer=True,
        alpha=10,
        beta=2,
        steer_back=False,
        save_only_cond=True,
        steer_mode: Literal[
            "cond_only",
            "uncond_only",
            "uncond_for_cond",
            "separate",
            "both_cond",
            "both_uncond",
        ] = "cond_only",
        num_cfg_passes=None,
        explicit_layers=None,
        verbose_register_layers=False,
    ):
        super().__init__(
            dtype=dtype,
            persistent_storage_path=persistent_storage_path,
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            quantized=quantized,
            overlapped_decode=overlapped_decode,
            pad_to_max_len=pad_to_max_len,
            device=device,
        )

        # Initialize steering controller
        self.controller = None
        self.steering_enabled = False
        self._original_forwards = {}  # Store original forward methods for cleanup

        if steering_vectors is not None:
            self.setup_steering(
                steering_vectors=steering_vectors,
                steer=steer,
                alpha=alpha,
                beta=beta,
                steer_back=steer_back,
                save_only_cond=save_only_cond,
                steer_mode=steer_mode,
                num_cfg_passes=num_cfg_passes,
                explicit_layers=explicit_layers,
                verbose_register_layers=verbose_register_layers,
            )

    def setup_steering(
        self,
        steering_vectors,
        steer=True,
        alpha=10,
        beta=2,
        steer_back=False,
        save_only_cond=True,
        steer_mode: Literal[
            "cond_only",
            "uncond_only",
            "uncond_for_cond",
            "separate",
            "both_cond",
            "both_uncond",
        ] = "cond_only",
        num_cfg_passes=None,
        explicit_layers=None,
        verbose_register_layers=False,
        renorm_after_steer=False,
        start_step=0,
    ):
        """
        Setup steering controller and register it with the model.

        Args:
            steering_vectors: Dictionary of steering vectors per layer
            steer: Whether to enable steering
            alpha: Forward steering intensity
            beta: Backward steering intensity (for removal)
            steer_back: Whether to steer backward (remove concept)
            save_only_cond: Whether to only save conditional CFG pass
            steer_mode: How to apply steering vectors
            num_cfg_passes: Number of CFG passes
            renorm_after_steer: If True, renormalize the steered activation to its
                original norm (CAA paper). Default False — discarded after ablation.
        """
        # Clear any existing hooks before registering new ones
        self.clear_steering_hooks()

        # Create controller
        self.controller = VectorStore(
            steering_vectors=steering_vectors,
            steer=steer,
            alpha=alpha,
            beta=beta,
            steer_back=steer_back,
            save_only_cond=save_only_cond,
            device=self.device,
            steer_mode=steer_mode,
            num_cfg_passes=num_cfg_passes,
            renorm_after_steer=renorm_after_steer,
            start_step=start_step,
        )

        # Register controller with the model
        # Note: ace_step_transformer is loaded in load() method
        if hasattr(self, "ace_step_transformer"):
            register_vector_control(
                self.ace_step_transformer,
                self.controller,
                explicit_layers=explicit_layers,
                verbose=verbose_register_layers,
            )
            self.steering_enabled = True
        else:
            raise RuntimeError(
                "Model not loaded. Call load() before setup_steering() or "
                "pass steering_vectors during initialization."
            )

    def load(
        self,
        lora_name_or_path: str = "none",
        lora_weight: float = 1.0,
        explicit_layers=None,
        verbose_register_layers=False,
    ):
        """Override load to register steering after model is loaded."""
        super().load(lora_name_or_path, lora_weight)

        # Store original forward methods for later restoration
        self._store_original_forwards()

        # If controller was created before load, register it now
        if self.controller is not None and not self.steering_enabled:
            register_vector_control(
                self.ace_step_transformer,
                self.controller,
                explicit_layers=explicit_layers,
                verbose=verbose_register_layers,
            )
            self.steering_enabled = True

    def _store_original_forwards(self):
        """Store original forward methods of all LinearTransformerBlocks."""
        if not hasattr(self, "ace_step_transformer"):
            return

        def store_recr(net_, place_in_ace):
            if net_.__class__.__name__ == "LinearTransformerBlock":
                if place_in_ace not in self._original_forwards:
                    # Store the original forward method
                    self._original_forwards[place_in_ace] = net_.forward
            elif hasattr(net_, "children"):
                for net__ in net_.children():
                    store_recr(net__, place_in_ace)

        for net in self.ace_step_transformer.transformer_blocks.named_children():
            name = "tf" + net[0]
            store_recr(net[1], name)

    def clear_steering_hooks(self):
        """Restore original forward methods, removing all steering hooks."""
        if not hasattr(self, "ace_step_transformer"):
            return

        def restore_recr(net_, place_in_ace):
            if net_.__class__.__name__ == "LinearTransformerBlock":
                if place_in_ace in self._original_forwards:
                    net_.forward = self._original_forwards[place_in_ace]
            elif hasattr(net_, "children"):
                for net__ in net_.children():
                    restore_recr(net__, place_in_ace)

        for net in self.ace_step_transformer.transformer_blocks.named_children():
            name = "tf" + net[0]
            restore_recr(net[1], name)

        self.steering_enabled = False

    def reset_controller(self):
        """Reset the controller state between generations."""
        if self.controller is not None:
            self.controller.reset()
