"""
CAA steering controller for AudioLDM2 via torch forward hooks.

Hooks attach to cross-attention (attn2) outputs inside each BasicTransformerBlock
in the UNet. Each attn2 output has shape [B, seq_len, hidden_dim] where:
  - B = 2*N under classifier-free guidance (first N = uncond, last N = cond)
  - seq_len = H*W of the latent feature map at that UNet stage
  - hidden_dim depends on the stage (256 / 384 / 640)

Saved activations are mean-pooled over batch and seq_len, giving a per-layer
vector of shape [hidden_dim]. Steering vectors (same shape) are added to the
chosen batch rows and, optionally, the rows are renormalized to their original
per-position L2 norm.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from src.models.audioldm.constants import AUDIOLDM2_CROSS_ATTENTION_LAYERS

VALID_STEER_MODES = (
    "cond_only",
    "uncond_only",
    "uncond_for_cond",
    "separate",
    "both_cond",
    "both_uncond",
)


def resolve_audioldm_layers(layers: str) -> list[str]:
    """Resolve a short preset name into a list of fully-qualified attn2 layer names."""
    if layers == "all":
        return list(AUDIOLDM2_CROSS_ATTENTION_LAYERS)
    if layers == "down":
        return [l for l in AUDIOLDM2_CROSS_ATTENTION_LAYERS if ".down_blocks." in l]
    if layers == "mid":
        return [l for l in AUDIOLDM2_CROSS_ATTENTION_LAYERS if ".mid_block." in l]
    if layers == "up":
        return [l for l in AUDIOLDM2_CROSS_ATTENTION_LAYERS if ".up_blocks." in l]
    if layers in ("up0", "up1", "up2"):
        idx = layers[-1]
        return [l for l in AUDIOLDM2_CROSS_ATTENTION_LAYERS if f".up_blocks.{idx}." in l]
    if layers in ("down1", "down2", "down3"):
        idx = layers[-1]
        return [l for l in AUDIOLDM2_CROSS_ATTENTION_LAYERS if f".down_blocks.{idx}." in l]
    # Subsets of up1: 'up1_a5_a10' = up_blocks.1.attentions.{5,10}.transformer_blocks.* = 4 layers
    if layers == "up1_a5_a10":
        return [
            l for l in AUDIOLDM2_CROSS_ATTENTION_LAYERS
            if (".up_blocks.1.attentions.5." in l) or (".up_blocks.1.attentions.10." in l)
        ]
    if layers == "all_minus_up1_a5_a10":
        excluded = set(resolve_audioldm_layers("up1_a5_a10"))
        return [l for l in AUDIOLDM2_CROSS_ATTENTION_LAYERS if l not in excluded]
    if layers == "all_minus_up1":
        excluded = set(resolve_audioldm_layers("up1"))
        return [l for l in AUDIOLDM2_CROSS_ATTENTION_LAYERS if l not in excluded]
    # 7 hand-picked layers in up_blocks.1 (a5/a6/a9 both tfs + a10 tf0 only)
    if layers == "all_minus_up1_late7":
        excluded = set(resolve_audioldm_layers("up1_late7"))
        return [l for l in AUDIOLDM2_CROSS_ATTENTION_LAYERS if l not in excluded]
    if layers == "up1_late7":
        return [
            ".unet.up_blocks.1.attentions.5.transformer_blocks.0.attn2",
            ".unet.up_blocks.1.attentions.5.transformer_blocks.1.attn2",
            ".unet.up_blocks.1.attentions.6.transformer_blocks.0.attn2",
            ".unet.up_blocks.1.attentions.6.transformer_blocks.1.attn2",
            ".unet.up_blocks.1.attentions.9.transformer_blocks.0.attn2",
            ".unet.up_blocks.1.attentions.9.transformer_blocks.1.attn2",
            ".unet.up_blocks.1.attentions.10.transformer_blocks.0.attn2",
        ]
    raise ValueError(
        f"Unknown layers preset: {layers!r}. Options: 'all', 'down', 'mid', 'up', "
        "'up0/1/2', 'down1/2/3', 'up1_a5_a10', 'all_minus_up1_a5_a10', 'all_minus_up1'."
    )


class VectorStoreAudioLDM:
    """
    Collects and/or applies steering vectors at AudioLDM2 cross-attention outputs.

    Storage format of ``vector_store`` after a generate():
      - ``save_only_cond=True``: ``{step_idx: {layer_name: [np.ndarray [hidden_dim]]}}``
      - ``save_only_cond=False``: ``{(step_idx, cfg_pass): {layer_name: [np.ndarray [hidden_dim]]}}``
        where ``cfg_pass == 0`` is cond and ``cfg_pass == 1`` is uncond (matches ACE-Step).

    ``steering_vectors`` passed at init may be in either format; the apply-time
    logic picks the right key based on ``steer_mode``.
    """

    def __init__(
        self,
        steering_vectors: Optional[dict] = None,
        steer: bool = False,
        alpha: float = 0.0,
        beta: float = 2.0,
        steer_back: bool = False,
        steer_mode: str = "cond_only",
        device: str = "cpu",
        save_only_cond: bool = True,
        normalize_sv_at_apply: bool = False,
        renorm_after_steer: bool = False,
        num_layers: int = 0,
    ):
        if steer_mode not in VALID_STEER_MODES:
            raise ValueError(
                f"steer_mode must be one of {VALID_STEER_MODES}, got {steer_mode!r}"
            )

        self.steering_vectors = steering_vectors
        self.steer = steer
        self.alpha = alpha
        self.beta = beta
        self.steer_back = steer_back
        self.steer_mode = steer_mode
        self.device = device
        self.save_only_cond = save_only_cond
        self.normalize_sv_at_apply = normalize_sv_at_apply
        self.renorm_after_steer = renorm_after_steer
        self.num_layers = num_layers

        # True iff the pipeline is doing CFG in this generate() call.
        # Set by SteeredAudioLDMPipeline before each generate.
        self.do_cfg = True

        self.cur_step = 0
        self.cur_layer = 0
        # step_store[pass_idx][layer_name] -> list of np.ndarray [hidden_dim]
        # pass_idx: 0 = cond, 1 = uncond (matches ACE-Step convention)
        self.step_store: dict = defaultdict(lambda: defaultdict(list))
        self.vector_store: dict = defaultdict(dict)

    def reset(self):
        self.cur_step = 0
        self.cur_layer = 0
        self.step_store = defaultdict(lambda: defaultdict(list))
        self.vector_store = defaultdict(dict)

    # ------------------------------------------------------------------ #
    # hook entry point                                                   #
    # ------------------------------------------------------------------ #

    def on_attn2_output(self, layer_name: str, output: torch.Tensor) -> torch.Tensor:
        """
        Called by the forward hook on each attn2 module.
        ``output`` shape: [B, seq_len, hidden_dim].
        Returns a (possibly modified) tensor that diffusers uses downstream.
        """
        if self.steer and self.steering_vectors:
            output = self._apply_steering(layer_name, output)

        self._save_activation(layer_name, output)

        self.cur_layer += 1
        if self.num_layers > 0 and self.cur_layer >= self.num_layers:
            self.cur_layer = 0
            self._between_steps()
            self.cur_step += 1

        return output

    # ------------------------------------------------------------------ #
    # save path                                                          #
    # ------------------------------------------------------------------ #

    def _batch_slices(self, batch_size: int) -> dict:
        """Return the slice objects for cond / uncond halves of the batch."""
        if not self.do_cfg or batch_size < 2 or batch_size % 2 != 0:
            return {"cond": slice(0, batch_size), "uncond": None}
        half = batch_size // 2
        # diffusers CFG batches as [uncond..., cond...]
        return {"cond": slice(half, batch_size), "uncond": slice(0, half)}

    def _save_activation(self, layer_name: str, output: torch.Tensor) -> None:
        slices = self._batch_slices(output.shape[0])

        detached = output.detach().float()

        cond_slice = slices["cond"]
        # average over prompts in batch + patches in activations
        cond_vec = detached[cond_slice].mean(dim=(0, 1)).cpu().numpy()
        self.step_store[0][layer_name].append(cond_vec)

        if not self.save_only_cond and slices["uncond"] is not None:
            uncond_vec = detached[slices["uncond"]].mean(dim=(0, 1)).cpu().numpy()
            self.step_store[1][layer_name].append(uncond_vec)

    def _between_steps(self) -> None:
        if self.save_only_cond:
            # Flatten {0: {layer: [...]}} -> {layer: [...]}
            self.vector_store[self.cur_step] = dict(self.step_store[0])
        else:
            for pass_idx, layers_dict in self.step_store.items():
                if layers_dict:
                    self.vector_store[(self.cur_step, pass_idx)] = dict(layers_dict)
        self.step_store = defaultdict(lambda: defaultdict(list))

    # ------------------------------------------------------------------ #
    # apply path                                                         #
    # ------------------------------------------------------------------ #

    def _lookup_sv(self, layer_name: str, pass_idx: int) -> Optional[np.ndarray]:
        """Fetch the cond (pass_idx=0) or uncond (pass_idx=1) SV for the current step+layer."""
        sv_store = self.steering_vectors
        if not sv_store:
            return None

        first_key = next(iter(sv_store))
        stored_per_pass = isinstance(first_key, tuple)

        if stored_per_pass:
            key = (self.cur_step, pass_idx)
        else:
            if pass_idx != 0:
                raise ValueError(
                    "Steering vectors were saved with save_only_cond=True (cond pass only); "
                    "cannot retrieve uncond-pass vectors. Recompute with save_all_cfg_passes=True."
                )
            key = self.cur_step

        if key not in sv_store or layer_name not in sv_store[key]:
            return None
        entry = sv_store[key][layer_name]
        if isinstance(entry, (list, tuple)):
            return entry[0]
        return entry

    def _apply_steering(self, layer_name: str, output: torch.Tensor) -> torch.Tensor:
        slices = self._batch_slices(output.shape[0])
        cond_slice = slices["cond"]
        uncond_slice = slices["uncond"]

        # Decide: (sv_pass_idx, batch_slice) pairs to apply
        if self.steer_mode == "cond_only":
            pairs = [(0, cond_slice)]
        elif self.steer_mode == "uncond_only":
            if uncond_slice is None:
                return output  # no CFG → nothing to steer on uncond side
            pairs = [(1, uncond_slice)]
        elif self.steer_mode == "uncond_for_cond":
            pairs = [(1, cond_slice)]
        elif self.steer_mode == "separate":
            pairs = [(0, cond_slice)]
            if uncond_slice is not None:
                pairs.append((1, uncond_slice))
        elif self.steer_mode == "both_cond":
            pairs = [(0, cond_slice)]
            if uncond_slice is not None:
                pairs.append((0, uncond_slice))
        elif self.steer_mode == "both_uncond":
            pairs = [(1, cond_slice)]
            if uncond_slice is not None:
                pairs.append((1, uncond_slice))
        else:
            return output

        for sv_pass_idx, target_slice in pairs:
            sv_np = self._lookup_sv(layer_name, sv_pass_idx)
            if sv_np is None:
                continue
            sv = torch.as_tensor(sv_np, dtype=output.dtype, device=output.device).view(1, 1, -1)
            if self.normalize_sv_at_apply:
                n = torch.norm(sv, dim=2, keepdim=True)
                sv = sv / torch.clamp(n, min=1e-8)

            rows = output[target_slice]
            orig_norm = torch.norm(rows, dim=2, keepdim=True)

            if self.steer_back:
                sim = (rows * sv).sum(dim=2, keepdim=True)
                sim = torch.clamp(sim, min=0.0)
                new_rows = rows - self.beta * sim * sv.expand_as(rows)
            else:
                new_rows = rows + self.alpha * sv.expand_as(rows)

            if self.renorm_after_steer:
                new_norm = torch.norm(new_rows, dim=2, keepdim=True)
                new_rows = new_rows / torch.clamp(new_norm, min=1e-8) * orig_norm

            output = output.clone() if not output.is_contiguous() else output
            output[target_slice] = new_rows

        return output


def register_vector_control_audioldm(
    unet: torch.nn.Module,
    controller: VectorStoreAudioLDM,
    layer_names: list[str],
    verbose: bool = False,
) -> list[torch.utils.hooks.RemovableHandle]:
    """
    Register forward hooks on each named cross-attention (attn2) module.

    ``layer_names`` entries should follow the convention in
    ``AUDIOLDM2_CROSS_ATTENTION_LAYERS`` (leading dot, including the ``unet.`` prefix);
    we strip that so names resolve against ``unet.named_modules()``.
    """
    name_to_module = dict(unet.named_modules())
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    registered: list[str] = []

    for full_name in layer_names:
        mod_name = full_name.lstrip(".")
        if mod_name.startswith("unet."):
            mod_name = mod_name[len("unet.") :]
        if mod_name not in name_to_module:
            if verbose:
                print(f"[register] WARNING: {full_name} not found in unet; skipping")
            continue
        mod = name_to_module[mod_name]
        hooks.append(mod.register_forward_hook(_make_hook(controller, full_name)))
        registered.append(full_name)

    controller.num_layers = len(registered)
    if verbose:
        print(f"[register] Hooked {len(registered)}/{len(layer_names)} attn2 modules")
    return hooks


def _make_hook(controller: VectorStoreAudioLDM, layer_name: str):
    def hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return output
        return controller.on_attn2_output(layer_name, output)

    return hook
