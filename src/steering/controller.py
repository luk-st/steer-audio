"""Universal steering Controller interface.

Every steering method (CAA, SAE, FreeSliders, ConceptSlider, AuSteer, ...) implements
``Controller``. A ``SteerableACEModel`` mounts exactly one controller at a time via
``model.steer(controller)``; the controller decides internally whether to use
``register_forward_hook`` or to patch a block's ``forward``.

Two mount mechanisms are offered as helpers so subclasses don't reinvent the wheel:

* ``_add_forward_hook``       — register/remove standard PyTorch forward hooks.
* ``_patch_method``           — swap an attribute (e.g. a block's ``forward``) and
                                 restore it on unmount. CAA needs this; see
                                 ``register_vector_control`` in
                                 ``src/models/ace_step/ace_steering/controller.py``.

Subclasses only need to override ``mount`` and (optionally) ``reset``.
"""

from __future__ import annotations

import abc
from typing import Any, Callable


class Controller(abc.ABC):
    """Base class for any steering method.

    Lifecycle::

        with model.steer(controller):
            audio = pipeline(prompt)

    is equivalent to::

        controller.mount(model)
        try:
            audio = pipeline(prompt)
        finally:
            controller.unmount()

    Subclasses must implement :meth:`mount`. Everything else has a default.
    """

    # Sentinel: ``attr`` was a class-level fallback before patching, so the
    # correct cleanup is ``delattr(obj, attr)`` rather than restoring an
    # ephemeral bound method.
    _CLASS_FALLBACK = object()

    def __init__(self) -> None:
        self._handles: list[Any] = []
        self._patches: list[tuple[Any, str, Any]] = []  # (obj, attr, original_or_sentinel)
        self._is_mounted = False

    # --- lifecycle ----------------------------------------------------------

    @abc.abstractmethod
    def mount(self, model: Any) -> None:
        """Attach this controller's interventions to ``model``.

        ``model`` is the bare ``ACEStepTransformer2DModel`` (the ``nn.Module``),
        not the pipeline. Implementations should populate ``self._handles`` /
        ``self._patches`` using the helpers below.
        """

    def unmount(self) -> None:
        """Remove all interventions. Idempotent."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for obj, attr, original in reversed(self._patches):
            if original is self._CLASS_FALLBACK:
                # Attribute lived on the class; remove our instance override.
                try:
                    delattr(obj, attr)
                except AttributeError:
                    pass
            else:
                setattr(obj, attr, original)
        self._patches.clear()
        self._is_mounted = False

    def reset(self) -> None:
        """Reset per-generation state (timestep counters, caches, ...).

        Called at the start of every generation. Default is no-op.
        """

    def generate(self, model: Any, **kwargs: Any) -> Any:
        """Run a single generation with this controller mounted.

        Default behaviour: call ``model.pipeline.generate(**kwargs)``. Any
        hooks / patches set up by :meth:`mount` intervene transparently. Methods
        that rewrite the diffusion loop itself (e.g. FreeSliders' 3-pass noise
        blend) override this to take full control of the generation.
        """
        return model.pipeline.generate(**kwargs)

    # --- helpers used by subclasses inside ``mount`` ------------------------

    def _add_forward_hook(self, module: Any, hook: Callable) -> None:
        """Equivalent to ``module.register_forward_hook(hook)``, tracked for cleanup."""
        self._handles.append(module.register_forward_hook(hook))

    def _patch_method(self, obj: Any, attr: str, new_fn: Callable) -> None:
        """Replace ``obj.<attr>`` with ``new_fn``; restored on ``unmount``.

        If ``attr`` was not an instance attribute (i.e. it was provided by the
        class), unmount will ``delattr`` rather than re-assigning a synthesized
        bound method — that way ``obj.attr`` returns the class method again
        instead of a stale instance copy.
        """
        if attr in obj.__dict__:
            self._patches.append((obj, attr, obj.__dict__[attr]))
        else:
            self._patches.append((obj, attr, self._CLASS_FALLBACK))
        setattr(obj, attr, new_fn)

    # --- factory hook -------------------------------------------------------

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs: Any) -> "Controller":
        """Load this controller from a HuggingFace Hub repo (or local path).

        Subclasses override this. The default fails loudly so callers get a
        clear error rather than a silent no-op controller.
        """
        raise TypeError(
            f"{cls.__name__}.from_pretrained is not implemented. "
            "Override it to pull this method's artifacts from the Hub."
        )


class NullController(Controller):
    """No-op controller — useful as the ``alpha=0`` baseline."""

    def mount(self, model: Any) -> None:
        self._is_mounted = True


# ---------------------------------------------------------------------------
# CFG-aware mixin
# ---------------------------------------------------------------------------


class CFGAwareMixin:
    """Counter-based timestep / CFG-pass tracking shared by hook-based methods.

    ACE-Step runs CFG as sequential forward passes per denoising step:
    ``cond_0, [text_only_0], uncond_0, cond_1, ...``. A forward hook sees them
    in this order. This mixin maps the hook-call counter back to
    ``(denoising_step, cfg_pass_index)``.

    Subclasses set ``num_cfg_passes`` (2 or 3, or ``None`` to default to 2).
    """

    def __init__(self, num_cfg_passes: int | None = None, start_step: int = 0) -> None:
        self.num_cfg_passes = num_cfg_passes if num_cfg_passes is not None else 2
        self.start_step = int(start_step)
        self.counter = -1

    def reset_cfg(self) -> None:
        self.counter = -1

    def step_counter(self) -> tuple[int, int]:
        """Advance the counter and return ``(denoising_step, cfg_pass)``."""
        self.counter += 1
        denoising_step, cfg_pass = divmod(self.counter, self.num_cfg_passes)
        return denoising_step, cfg_pass

    def is_cond_pass(self, cfg_pass: int) -> bool:
        return cfg_pass == 0

    def is_uncond_pass(self, cfg_pass: int) -> bool:
        return cfg_pass == self.num_cfg_passes - 1
