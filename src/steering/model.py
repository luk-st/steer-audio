"""Model wrappers — single user-facing facades for steered generation.

Two wrappers, same surface (``.steer(controller)`` context manager + ``.generate()``):

* :class:`SteerableACEModel`     — ACE-Step (audio-music diffusion transformer)
* :class:`SteerableAudioLDMModel` — AudioLDM2-large (diffusers UNet)

The Controller's lifecycle is managed for you, so generation code is
method-agnostic::

    model = SteerableACEModel(device="cuda")
    controller = CAASteeringController.from_pretrained("user/ace-piano-caa", alpha=20)

    with model.steer(controller):
        audio = model.generate(prompt="instrumental music", seed=0)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .controller import Controller, NullController


class SteerableACEModel:
    """Thin wrapper over ``SimpleACEStepPipeline`` with a Controller slot."""

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "bfloat16",
        persistent_storage_path: str = "res/ace_step",
        **pipeline_kwargs: Any,
    ) -> None:
        # Import locally so the rest of the library doesn't drag the heavy
        # ACE-Step dependency tree on import.
        from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline

        self.pipeline = SimpleACEStepPipeline(
            device=device,
            dtype=dtype,
            persistent_storage_path=persistent_storage_path,
            **pipeline_kwargs,
        )
        self.device = device
        self._active_controller: Controller | None = None

    @property
    def transformer(self):
        """The raw ``ACEStepTransformer2DModel`` (what controllers mount onto)."""
        return self.pipeline.ace_step_transformer

    @property
    def sample_rate(self) -> int:
        return self.pipeline.sample_rate

    # --- core API -----------------------------------------------------------

    @contextmanager
    def steer(self, controller: Controller | None) -> Iterator[Controller]:
        """Mount ``controller`` for the duration of the ``with`` block.

        ``None`` is treated as a no-op (baseline). Nested ``steer`` calls are
        forbidden — caller should ``unmount`` first.
        """
        if self._active_controller is not None:
            raise RuntimeError(
                f"A controller ({type(self._active_controller).__name__}) is "
                "already mounted. Nested steering is not supported."
            )
        ctrl = controller if controller is not None else NullController()
        ctrl.reset()
        ctrl.mount(self.transformer)
        self._active_controller = ctrl
        try:
            yield ctrl
        finally:
            ctrl.unmount()
            self._active_controller = None

    def generate(self, **pipeline_call_kwargs: Any):
        """Delegate to the active controller's :meth:`generate`.

        With a controller mounted, that controller chooses how to drive the
        pipeline (e.g. FreeSliders overrides the diffusion loop entirely;
        activation-hook methods keep the default
        ``self.pipeline.generate(**kwargs)``). Without any controller mounted
        this is a plain ``pipeline.generate`` call.
        """
        if self._active_controller is None:
            return self.pipeline.generate(**pipeline_call_kwargs)
        return self._active_controller.generate(self, **pipeline_call_kwargs)


class SteerableAudioLDMModel:
    """Thin wrapper over diffusers' ``AudioLDM2Pipeline`` with a Controller slot.

    Mirrors :class:`SteerableACEModel`'s surface so the unified runner can drive
    both models through identical CLIs. Controllers can branch on the wrapper
    type via ``isinstance`` if needed.
    """

    DEFAULT_REPO_ID = "cvssp/audioldm2-large"

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float16",
        repo_id: str | None = None,
        disable_progress_bar: bool = True,
    ) -> None:
        import torch
        from diffusers import AudioLDM2Pipeline

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, dtype)
        self.repo_id = repo_id or self.DEFAULT_REPO_ID
        pipe = AudioLDM2Pipeline.from_pretrained(
            self.repo_id, torch_dtype=torch_dtype
        ).to(device)
        if disable_progress_bar:
            pipe.set_progress_bar_config(disable=True)
        self.pipeline = pipe
        self.device = device
        self.dtype = torch_dtype
        self._active_controller: Controller | None = None

    @property
    def transformer(self):
        """The bare UNet (what AudioLDM controllers mount hooks onto)."""
        return self.pipeline.unet

    @property
    def sample_rate(self) -> int:
        return self.pipeline.vocoder.config.sampling_rate

    @contextmanager
    def steer(self, controller: Controller | None) -> Iterator[Controller]:
        if self._active_controller is not None:
            raise RuntimeError(
                f"A controller ({type(self._active_controller).__name__}) is "
                "already mounted. Nested steering is not supported."
            )
        ctrl = controller if controller is not None else NullController()
        ctrl.reset()
        ctrl.mount(self.transformer)
        self._active_controller = ctrl
        try:
            yield ctrl
        finally:
            ctrl.unmount()
            self._active_controller = None

    def generate(self, **pipeline_call_kwargs: Any):
        """If a controller is mounted, dispatch to its :meth:`generate`;
        otherwise call the diffusers pipeline directly. Returns a torch tensor."""
        import torch as _torch

        if self._active_controller is not None:
            return self._active_controller.generate(self, **pipeline_call_kwargs)
        return _torch.as_tensor(self.pipeline(**pipeline_call_kwargs).audios)
