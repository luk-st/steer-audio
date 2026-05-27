"""Layer-specific steering hooks for ACE-Step transformer blocks.

Provides context managers that restrict steering to specific transformer
decoder blocks (e.g. tf6, tf7) while keeping other blocks on neutral
conditioning.

Requires PyTorch >= 2.0 (for ``register_forward_pre_hook(with_kwargs=True)``).
"""

import contextlib

import torch

from src.steering.methods.caa.utils.constants import LAYER_CONFIGS

NUM_BLOCKS = 24


# ---------------------------------------------------------------------------
# Layer parsing
# ---------------------------------------------------------------------------


def parse_layers(layers: "str | list[str]") -> list[str]:
    """Resolve a layer specification to a list of block names.

    Accepts either:
    * a preset string in ``LAYER_CONFIGS`` (``"all"``, ``"tf6"``, ``"tf7"``,
      ``"tf6tf7"``, ``"no_tf6tf7"``); or
    * an explicit list / tuple like ``["tf6", "tf7"]``.

    >>> parse_layers("tf6tf7")
    ['tf6', 'tf7']
    >>> parse_layers(["tf3", "tf9"])
    ['tf3', 'tf9']
    """
    if isinstance(layers, (list, tuple)):
        return list(layers)
    if layers in LAYER_CONFIGS:
        return LAYER_CONFIGS[layers]
    raise ValueError(
        f"Unknown layers config: {layers!r}. "
        f"Available presets: {list(LAYER_CONFIGS.keys())}, "
        f"or pass an explicit list like ['tf6', 'tf7']."
    )


def is_all_layers(target_layers: list[str]) -> bool:
    return len(target_layers) == NUM_BLOCKS


def _target_indices(target_layers: list[str]) -> set[int]:
    return {int(name.replace("tf", "")) for name in target_layers}


def _non_target_indices(target_layers: list[str]) -> list[int]:
    return sorted(set(range(NUM_BLOCKS)) - _target_indices(target_layers))


# ---------------------------------------------------------------------------
# Context manager for FreeSliders (custom diffusion loop)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def layer_specific_decode(
    transformer,
    target_layers: list[str],
    replacement_enc_hs: torch.Tensor,
    replacement_enc_mask: torch.Tensor,
):
    """Inject *replacement* conditioning into **target** transformer blocks.

    Non-target blocks see whatever ``encoder_hidden_states`` was passed to
    ``transformer.decode()``.

    Usage (FreeSliders)::

        # All blocks receive neutral_enc_hs as base.
        # Target blocks are overridden with positive_enc_hs.
        with layer_specific_decode(transformer, layers, pos_enc_hs, enc_mask):
            noise_pos = transformer.decode(
                encoder_hidden_states=neutral_enc_hs, ...
            ).sample
    """
    if is_all_layers(target_layers):
        yield
        return

    target_idx = _target_indices(target_layers)

    def _make_hook(repl_hs, repl_mask):
        def hook(_module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs["encoder_hidden_states"] = repl_hs
            kwargs["encoder_attention_mask"] = repl_mask
            return args, kwargs

        return hook

    handles = []
    for idx in target_idx:
        h = transformer.transformer_blocks[idx].register_forward_pre_hook(
            _make_hook(replacement_enc_hs, replacement_enc_mask),
            with_kwargs=True,
        )
        handles.append(h)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Context manager for PI / TE / TokE  (pipeline-based)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def layer_specific_replace_embeds(
    transformer,
    target_layers: list[str],
    neutral_enc_hs: torch.Tensor,
    neutral_enc_mask: torch.Tensor,
    null_enc_hs: torch.Tensor | None,
    null_enc_mask: torch.Tensor | None,
):
    """Restrict ``replace_embeds`` effect to **target** layers only.

    Wrap ``pipe.task_text2music(replace_embeds=True, ...)`` with this
    context manager.  Non-target blocks are restored to neutral
    conditioning (conditional pass) or null conditioning (unconditional
    pass).

    Pass detection uses a lightweight fingerprint comparison: the first
    16 scalar values of the incoming ``encoder_hidden_states`` are
    compared against the pre-computed neutral and null versions to
    determine which one to restore.
    """
    if is_all_layers(target_layers):
        yield
        return

    non_target = _non_target_indices(target_layers)

    # Pre-compute fingerprints for pass-type detection
    _fp_neutral = neutral_enc_hs.reshape(-1)[:16].float()
    _fp_null = (
        null_enc_hs.reshape(-1)[:16].float()
        if null_enc_hs is not None
        else torch.zeros(16, device=neutral_enc_hs.device)
    )

    # Mutable state set by the decode wrapper, read by block hooks
    _restore = {
        "enc_hs": neutral_enc_hs,
        "enc_mask": neutral_enc_mask,
    }

    # -- Patch transformer.decode to detect pass type --
    _orig_decode = transformer.decode

    def _patched_decode(*args, **kwargs):
        enc_hs = kwargs.get("encoder_hidden_states")
        if enc_hs is not None:
            fp = enc_hs.reshape(-1)[:16].float()
            diff_null = (fp - _fp_null).abs().sum().item()
            diff_neutral = (fp - _fp_neutral).abs().sum().item()
            if diff_null < diff_neutral and null_enc_hs is not None:
                _restore["enc_hs"] = null_enc_hs
                _restore["enc_mask"] = null_enc_mask
            else:
                _restore["enc_hs"] = neutral_enc_hs
                _restore["enc_mask"] = neutral_enc_mask
        return _orig_decode(*args, **kwargs)

    transformer.decode = _patched_decode

    # -- Hooks on non-target blocks --
    def _make_hook():
        def hook(_module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs["encoder_hidden_states"] = _restore["enc_hs"]
            kwargs["encoder_attention_mask"] = _restore["enc_mask"]
            return args, kwargs

        return hook

    handles = []
    for idx in non_target:
        h = transformer.transformer_blocks[idx].register_forward_pre_hook(
            _make_hook(), with_kwargs=True
        )
        handles.append(h)

    try:
        yield
    finally:
        transformer.decode = _orig_decode
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Null-conditioning encoder (replicates pipeline logic)
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_null_conditioning(
    pipe,
    neutral_text_mask: torch.Tensor,
    speaker_embds: torch.Tensor,
    lyric_token_ids: torch.Tensor,
    lyric_mask: torch.Tensor,
    null_text_hs: torch.Tensor | None = None,
    use_erg_lyric: bool = True,
    tau: float = 0.01,
    l_min: int = 4,
    l_max: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode null conditioning for the unconditional CFG pass.

    Replicates the pipeline's null-encoding logic so that layer-specific
    hooks can restore non-target blocks to the correct null conditioning.

    Args:
        pipe: The loaded ``SimpleACEStepPipeline``.
        neutral_text_mask: ``[B, seq]`` attention mask from neutral prompts.
        speaker_embds: ``[B, 512]`` speaker embeddings (will be zeroed).
        lyric_token_ids: ``[B, lyric_len]`` lyrics (kept for ERG, zeroed otherwise).
        lyric_mask: ``[B, lyric_len]``.
        null_text_hs: ``[B, seq, 1024]`` ERG T5 output, or *None* for zeros.
        use_erg_lyric: Whether the pipeline uses ERG lyric temperature scaling.

    Returns:
        ``(encoder_hidden_states_null, encoder_hidden_mask)``
    """
    transformer = pipe.ace_step_transformer

    bsz, seq_len = neutral_text_mask.shape
    hidden_dim = 1024

    if null_text_hs is None:
        null_text_hs = torch.zeros(
            bsz, seq_len, hidden_dim, device=pipe.device, dtype=pipe.dtype
        )

    if use_erg_lyric:
        # ERG lyric: temperature-scale lyric encoder Q projections
        handlers = []

        def hook(_module, _input, output):
            output[:] *= tau
            return output

        for i in range(l_min, l_max):
            h = transformer.lyric_encoder.encoders[
                i
            ].self_attn.linear_q.register_forward_hook(hook)
            handlers.append(h)

        enc_hs, enc_mask = transformer.encode(
            null_text_hs,
            neutral_text_mask,
            torch.zeros_like(speaker_embds),
            lyric_token_ids,
            lyric_mask,
        )

        for h in handlers:
            h.remove()
    else:
        enc_hs, enc_mask = transformer.encode(
            torch.zeros(bsz, seq_len, hidden_dim, device=pipe.device, dtype=pipe.dtype),
            neutral_text_mask,
            torch.zeros_like(speaker_embds),
            torch.zeros_like(lyric_token_ids),
            lyric_mask,
        )

    return enc_hs, enc_mask
