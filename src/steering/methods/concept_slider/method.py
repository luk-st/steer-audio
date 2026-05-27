"""Concept Slider (LoRA-based) steering as a unified :class:`Controller`.

A trained Concept Slider is a LoRA adapter; the steering strength is the LoRA
weight. ``mount`` loads the adapter onto the transformer and activates it at
``alpha``; ``unmount`` unloads it cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.steering import Controller
from src.steering.registry import register_method


ADAPTER_NAME = "ace_step_lora"
WEIGHTS_FILENAME = "pytorch_lora_weights.safetensors"


@register_method("concept_slider")
class ConceptSlidersSteeringController(Controller):
    """LoRA-based steering. ``alpha`` becomes the adapter weight."""

    def __init__(
        self,
        lora_path: str | Path,
        alpha: float = 1.0,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.lora_path = str(lora_path)
        self.alpha = alpha
        self.config = config or {}
        self._loaded_on: Any | None = None

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        alpha: float = 1.0,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> "ConceptSlidersSteeringController":
        """Resolve ``repo_id`` to a local directory and construct the controller.

        The pipeline's ``load_lora`` accepts a Hub repo id directly (it calls
        ``snapshot_download`` internally), but we pre-resolve here so the
        controller is constructible without touching the model.
        """
        local = Path(repo_id)
        if local.is_dir():
            root = local
        else:
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    allow_patterns=["*.safetensors", "*.json", "*.md"],
                )
            )
        cfg_path = root / "train_config.json"
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        return cls(lora_path=root, alpha=alpha, config=config, **kwargs)

    def mount(self, model: Any) -> None:
        from diffusers.utils.peft_utils import set_weights_and_activate_adapters

        weights_file = Path(self.lora_path) / WEIGHTS_FILENAME
        if not weights_file.exists():
            raise FileNotFoundError(f"LoRA weights not found at {weights_file}")

        model.load_lora_adapter(
            str(weights_file),
            adapter_name=ADAPTER_NAME,
            with_alpha=True,
            prefix=None,
        )
        set_weights_and_activate_adapters(model, [ADAPTER_NAME], [self.alpha])
        self._loaded_on = model
        self._is_mounted = True

    def unmount(self) -> None:
        if self._loaded_on is not None:
            try:
                self._loaded_on.unload_lora()
            finally:
                self._loaded_on = None
        super().unmount()

    def set_alpha(self, alpha: float) -> None:
        """Change LoRA weight without re-loading the adapter from disk."""
        self.alpha = alpha
        if self._loaded_on is not None:
            from diffusers.utils.peft_utils import set_weights_and_activate_adapters

            set_weights_and_activate_adapters(self._loaded_on, [ADAPTER_NAME], [alpha])
