"""
ACE-Step SAE Intervention Hooks

This module provides hooks for intervening on ACE-Step model activations using
Sparse Autoencoders (SAEs). These hooks can be registered on model layers to:
- Reconstruct activations through an SAE
- Ablate (zero out) activations
- Apply per-timestep feature interventions during diffusion

All hooks support two SAE modes:
- "sequence": SAE reconstructs along the embedding dimension (default)
- "frequency": SAE reconstructs along the frequency/time dimension (requires transposed training)

Error correction (add_error=True) preserves information the SAE cannot represent
by adding the reconstruction error back:
    output = sae_intervened + (original - sae_clean)
This is equivalent to:
    output = original + (sae_intervened - sae_clean)
Only the SAE-representable component changes; the residual is preserved exactly.

CFG Pass Handling:
ACE-Step calls hooks as: cond_t0, uncond_t0, cond_t1, uncond_t1, ...
By default (uncond_preds=False), hooks only intervene on conditional passes.
Set uncond_preds=True to intervene on both conditional and unconditional passes.
"""

import einops
import torch

# Small epsilon to prevent division by zero in renormalization
RENORM_EPS = 1e-8


def sae_reconstruction(input_to_intervene, sae, sae_mode):
    # input is always in shape (bsz, time, freq)
    batch_size = input_to_intervene.shape[0]
    to_intervene = input_to_intervene
    if sae_mode == "frequency":
        to_intervene = einops.rearrange(input_to_intervene, "b t f -> b f t")

    # reconstruct with sae
    sae_input, _, _ = sae.preprocess_input(to_intervene)
    pre_acts = sae.pre_acts(sae_input)
    top_acts, top_indices = sae.select_topk(pre_acts)
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (sae.W_dec.mT.shape[-1],))
    latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    sae_out = (latents @ sae.W_dec) + sae.b_dec

    # reshape to original shape
    if sae_mode == "frequency":
        sae_out = einops.rearrange(sae_out, "(b f) t -> b t f", b=batch_size)
    else:
        sae_out = einops.rearrange(sae_out, "(b t) f -> b t f", b=batch_size)

    return sae_out


def sae_intervention(
    input_to_intervene,
    sae,
    sae_mode,
    features_to_modify,
    multiplier,
    intervention_mode="pre_topk",
):
    """Encode through SAE, apply intervention, decode.

    Args:
        input_to_intervene: Activation tensor, shape (bsz, time, freq)
        sae: Trained SAE model
        sae_mode: "sequence" or "frequency"
        features_to_modify: List of SAE feature indices to modify
        multiplier: Scale factor (pre_topk/post_topk) or target value (inject)
            For steering_vector mode, this scales the added vector.
        intervention_mode: How to apply the intervention:
            - "pre_topk": Multiply pre-activations before top-k selection.
              Boosted features can enter the active set; suppressed ones drop out.
            - "post_topk": Multiply latents after top-k selection.
              Only affects features already in top-k (0 * mult = 0).
            - "inject": Set latent values directly, regardless of top-k.
              multiplier is the target activation value, not a scale.
            - "steering_vector": Add sum of W_dec directions for selected features.
              No SAE encode/decode. Operates in the SAE input space, respecting
              the frequency/sequence rearrangement.
              output = input + multiplier * sum(W_dec[i] for i in features)
    """
    batch_size = input_to_intervene.shape[0]
    to_intervene = input_to_intervene
    if sae_mode == "frequency":
        to_intervene = einops.rearrange(to_intervene, "b t f -> b f t")

    # steering_vector: add W_dec directions directly in SAE input space
    if intervention_mode == "steering_vector":
        if not features_to_modify or multiplier == 0.0:
            return input_to_intervene
        W_dec = sae.W_dec  # (num_latents, d_in)
        vec = W_dec[features_to_modify].sum(dim=0)  # (d_in,)
        # vec is in the SAE input space (after rearrange for frequency mode)
        # to_intervene is (b, t, f) for sequence or (b, f, t) for frequency
        # broadcast vec over batch and first spatial dim
        vec = vec.unsqueeze(0).unsqueeze(0).to(to_intervene.device, to_intervene.dtype)
        sae_out = to_intervene + multiplier * vec
        # rearrange back to original shape
        if sae_mode == "frequency":
            sae_out = einops.rearrange(sae_out, "b f t -> b t f")
        return sae_out

    # SAE encode
    sae_input, _, _ = sae.preprocess_input(to_intervene)
    pre_acts = sae.pre_acts(sae_input)

    # pre_topk: modify pre-activations before top-k selection
    if intervention_mode == "pre_topk" and features_to_modify:
        for feature_idx in features_to_modify:
            pre_acts[..., feature_idx] *= multiplier

    # top-k selection
    top_acts, top_indices = sae.select_topk(pre_acts)
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (sae.W_dec.mT.shape[-1],))
    latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)

    # post_topk: multiply latents after top-k (only affects features already in top-k)
    if intervention_mode == "post_topk" and features_to_modify:
        for feature_idx in features_to_modify:
            latents[..., feature_idx] *= multiplier

    # inject: set latent values directly (multiplier = target activation value)
    if intervention_mode == "inject" and features_to_modify:
        for feature_idx in features_to_modify:
            latents[..., feature_idx] = multiplier

    sae_out = (latents @ sae.W_dec) + sae.b_dec

    # reshape to original shape
    if sae_mode == "frequency":
        sae_out = einops.rearrange(sae_out, "(b f) t -> b t f", b=batch_size)
    else:
        sae_out = einops.rearrange(sae_out, "(b t) f -> b t f", b=batch_size)
    return sae_out


def build_steering_vectors(sae, scores, top_k, sae_mode="sequence"):
    """Build per-timestep steering vectors from feature scores and SAE decoder weights.

    For each timestep, selects the top-k features by score and constructs:
        v_t = sum_i( score[t, i] * W_dec[i] )  for i in top-k

    The resulting vectors live in the SAE input space and can be added directly
    to model activations without any SAE encode/decode at inference time.

    Args:
        sae: Trained SAE model (only W_dec is used)
        scores: (num_timesteps, num_latents) tensor of per-feature importance scores
        top_k: Number of top features to include per timestep
        sae_mode: "sequence" or "frequency" — determines how the vector is reshaped

    Returns:
        Dict mapping timestep (int) -> steering vector tensor.
        For sequence mode: shape (1, 1, d_in), broadcastable over (batch, time, dim).
        For frequency mode: shape (1, d_in, 1), broadcastable over (batch, dim, freq).
    """
    W_dec = sae.W_dec  # (num_latents, d_in)
    num_timesteps = scores.shape[0]
    steering_vectors = {}

    for t in range(num_timesteps):
        t_scores = scores[t]  # (num_latents,)
        topk_indices = torch.argsort(t_scores, descending=True)[:top_k]
        topk_scores = t_scores[topk_indices].to(W_dec.device, W_dec.dtype)

        # v_t = sum_i( score_i * W_dec[i] )  =>  topk_scores @ W_dec[topk_indices]
        vec = topk_scores @ W_dec[topk_indices]  # (d_in,)

        if sae_mode == "frequency":
            # activations are (batch, dim, freq) after rearrange; vec is in freq space
            steering_vectors[t] = vec.unsqueeze(0).unsqueeze(1)  # (1, 1, d_in)
        else:
            steering_vectors[t] = vec.unsqueeze(0).unsqueeze(0)  # (1, 1, d_in)

    return steering_vectors


class ACEStepSAEReconstructHook:
    """Reconstruct activations through an SAE without modifying latents.

    Useful for measuring SAE reconstruction quality and its effect on generation.

    Note: With add_error=True this becomes a no-op (sae_out + error = original),
    which is useful as a sanity check.

    Args:
        sae: Trained SAE model
        sae_mode: "sequence" (reconstruct embedding dim) or "frequency" (reconstruct time dim)
        uncond_preds: If True, also intervene on unconditional CFG passes
        add_error: If True, add reconstruction error back (output = sae_out + original - sae_clean)
        renorm: If True, preserve original activation magnitude (legacy, prefer add_error)

    Example:
        hook = ACEStepSAEReconstructHook(sae, add_error=True)
        handle = model.layer.register_forward_hook(hook)
        # ... run inference ...
        handle.remove()
    """

    def __init__(
        self,
        sae,
        sae_mode: str = "sequence",
        uncond_preds=False,
        add_error=False,
        renorm=False,
    ):
        assert sae_mode in ["sequence", "frequency"]
        assert not (
            add_error is True and renorm is True
        )  # renorm is not supported with add_error

        self.sae = sae
        self.sae_mode = sae_mode
        self.uncond_preds = uncond_preds
        self.counter = -1
        self.add_error = add_error
        self.renorm = renorm

    @torch.no_grad()
    def __call__(self, module, input, output):
        self.counter += 1
        if self.uncond_preds is False and self.counter % 2 != 0:
            return output
        to_intervene = output
        sae_out = sae_reconstruction(
            input_to_intervene=to_intervene, sae=self.sae, sae_mode=self.sae_mode
        )
        if self.renorm:
            norm_before = torch.norm(to_intervene, dim=2, keepdim=True)
        if self.add_error:
            sae_out = sae_out + (output - sae_out)
        if self.renorm:
            sae_out = sae_out / (torch.norm(sae_out, dim=2, keepdim=True) + RENORM_EPS)
            sae_out = sae_out * norm_before
        return sae_out


class ACEStepAblateHook:
    """Zero out all activations passing through the hooked layer.

    This is a destructive intervention that replaces activations with zeros.
    Useful for measuring the importance of a layer to generation quality.

    Note: renorm is NOT supported for this hook since zeroed activations
    cannot be meaningfully renormalized (would result in NaN from 0/0).

    Args:
        sae: Trained SAE model (unused, kept for API consistency)
        sae_mode: "sequence" or "frequency" (unused, kept for API consistency)
        uncond_preds: If True, also ablate unconditional CFG passes

    Example:
        hook = ACEStepAblateHook(sae)
        handle = model.layer.register_forward_hook(hook)
        # ... run inference to see effect of ablating this layer ...
        handle.remove()
    """

    def __init__(
        self,
        sae=None,
        sae_mode: str = "sequence",
        uncond_preds=False,
    ):
        self.sae = sae  # Kept for API consistency, not used
        self.sae_mode = sae_mode
        self.uncond_preds = uncond_preds
        self.counter = -1

    @torch.no_grad()
    def __call__(self, module, input, output):
        self.counter += 1
        if self.uncond_preds is False and self.counter % 2 != 0:
            return output
        # Simply return zeros with the same shape as output
        return torch.zeros_like(output)


class ACEStepTimestepInterventionHook:
    """Apply per-timestep feature interventions during diffusion.

    Allows different features to be modified at different diffusion timesteps.
    Useful for time-varying steering based on feature importance analysis
    (e.g., from FeatureSelector results).

    Args:
        sae: Trained SAE model (W_dec used for steering_vector mode)
        features_per_timestep: Dict mapping timestep (int) -> list of feature indices
            Example: {0: [1, 5, 10], 5: [2, 8], 10: [3, 7, 9]}
            Timesteps not in dict will have no intervention.
        multiplier: Either:
            - float: Same multiplier for all features at all timesteps
            - dict: Mapping timestep (int) -> multiplier for that timestep
              Example: {0: 2.0, 5: 1.5, 10: 0.5}
        sae_mode: "sequence" or "frequency"
        uncond_preds: If True, also intervene on unconditional CFG passes
        add_error: If True, add reconstruction error back to preserve non-SAE components:
            output = original + (sae_intervened - sae_clean)
            Only the SAE-representable component changes; the residual is preserved.
        renorm: If True, preserve original activation magnitude (legacy, prefer add_error)
        negate_for_uncond: If True (requires uncond_preds=True), negate the multiplier
            for unconditional CFG passes. This amplifies the intervention through CFG:
            cond gets +multiplier, uncond gets -multiplier.
        intervention_mode: How to apply the intervention:
            - "pre_topk": Multiply pre-activations before top-k selection.
              Boosted features can enter the active set; suppressed ones drop out.
            - "post_topk": Multiply latents after top-k selection.
              Only affects features already in top-k (0 * mult = 0).
            - "inject": Set latent values directly, regardless of top-k.
              multiplier is the target activation value, not a scale.
            - "steering_vector": Add sum of W_dec directions for selected features.
              No SAE encode/decode at runtime.
              output = original + multiplier * sum(W_dec[i] for i in features)
    """

    def __init__(
        self,
        sae,
        features_per_timestep: dict[int, list[int]],
        multiplier: float | dict[int, float] = 1.0,
        sae_mode: str = "sequence",
        uncond_preds: bool = False,
        add_error: bool = False,
        renorm: bool = False,
        negate_for_uncond: bool = False,
        intervention_mode: str = "pre_topk",
        start_step: int = 0,
    ):
        assert sae_mode in ["sequence", "frequency"]
        assert intervention_mode in [
            "pre_topk",
            "post_topk",
            "inject",
            "steering_vector",
        ]
        assert not (
            add_error is True and renorm is True
        )  # renorm is not supported with add_error
        assert not (
            negate_for_uncond is True and uncond_preds is False
        )  # negate_for_uncond requires uncond_preds=True

        self.sae = sae
        self.sae_mode = sae_mode
        self.uncond_preds = uncond_preds
        self.counter = -1
        self.add_error = add_error
        self.renorm = renorm
        self.negate_for_uncond = negate_for_uncond
        self.features_per_timestep = features_per_timestep
        self.multiplier = multiplier
        self._is_multiplier_dict = isinstance(multiplier, dict)
        self.intervention_mode = intervention_mode
        self.start_step = int(start_step)

    def _get_current_timestep(self) -> int:
        """Map counter to diffusion timestep."""
        if self.uncond_preds:
            return self.counter // 2
        else:
            return self.counter // 2

    def _get_multiplier_for_timestep(self, timestep: int) -> float:
        """Get multiplier for the given timestep."""
        if self._is_multiplier_dict:
            return self.multiplier.get(timestep, 1.0)
        return self.multiplier

    @torch.no_grad()
    def __call__(self, module, input, output):
        self.counter += 1

        # Skip unconditional passes if uncond_preds=False
        if self.uncond_preds is False and self.counter % 2 != 0:
            return output

        timestep = self._get_current_timestep()

        # Skip the first start_step diffusion steps (no steering applied yet)
        if timestep < self.start_step:
            return output

        multiplier = self._get_multiplier_for_timestep(timestep)

        # Negate multiplier for unconditional passes (odd counter = uncond)
        is_uncond_pass = self.counter % 2 != 0
        if self.negate_for_uncond and is_uncond_pass:
            multiplier = -multiplier

        features_to_modify = self.features_per_timestep.get(timestep, [])

        to_intervene = output
        if self.renorm:
            norm_before = torch.norm(to_intervene, dim=2, keepdim=True)

        if self.add_error:
            sae_reconstruction_out = sae_reconstruction(
                input_to_intervene=to_intervene, sae=self.sae, sae_mode=self.sae_mode
            )
            reconstruction_error = output - sae_reconstruction_out

        sae_out = sae_intervention(
            input_to_intervene=to_intervene,
            sae=self.sae,
            sae_mode=self.sae_mode,
            features_to_modify=features_to_modify,
            multiplier=multiplier,
            intervention_mode=self.intervention_mode,
        )

        if self.add_error:
            sae_out = sae_out + reconstruction_error

        if self.renorm:
            sae_out = sae_out / (torch.norm(sae_out, dim=2, keepdim=True) + RENORM_EPS)
            sae_out = sae_out * norm_before
        return sae_out


class ACEStepPrecomputedSteeringHook:
    """Apply precomputed steering vectors during diffusion.

    Unlike ACEStepTimestepInterventionHook which sums W_dec columns at runtime,
    this hook uses precomputed steering vectors that may include custom weighting.
    No SAE encode/decode is performed - vectors are added directly to activations.

    This is useful when:
    - Steering vectors are computed with weighted feature combinations
    - The same vectors will be reused across multiple generations
    - Feature selection and weighting are separated (e.g., TF-IDF for selection,
      diff scores with softmax weighting for vector construction)

    Args:
        steering_vectors: Dict mapping timestep (int) -> steering vector tensor.
            Vectors should be shape (hidden_dim,) and will be broadcast.
        multiplier: Scale factor for the steering vectors
        sae_mode: "sequence" or "frequency" (determines broadcast shape)
        uncond_preds: If True, also intervene on unconditional CFG passes
        negate_for_uncond: If True (requires uncond_preds=True), negate the multiplier
            for unconditional CFG passes

    Example:
        # Build vectors with custom weighting
        vectors = build_weighted_steering_vectors(scores, W_dec, top_k, weighting="softmax")
        hook = ACEStepPrecomputedSteeringHook(vectors, multiplier=10.0)
        handle = model.layer.register_forward_hook(hook)
    """

    def __init__(
        self,
        steering_vectors: dict[int, torch.Tensor],
        multiplier: float = 1.0,
        sae_mode: str = "sequence",
        uncond_preds: bool = False,
        negate_for_uncond: bool = False,
        start_step: int = 0,
    ):
        assert sae_mode in ["sequence", "frequency"]
        assert not (negate_for_uncond is True and uncond_preds is False)

        self.steering_vectors = steering_vectors
        self.multiplier = multiplier
        self.sae_mode = sae_mode
        self.uncond_preds = uncond_preds
        self.negate_for_uncond = negate_for_uncond
        self.counter = -1
        self.start_step = int(start_step)

    def _get_current_timestep(self) -> int:
        """Map counter to diffusion timestep."""
        return self.counter // 2

    @torch.no_grad()
    def __call__(self, module, input, output):
        self.counter += 1

        # Skip unconditional passes if uncond_preds=False
        if self.uncond_preds is False and self.counter % 2 != 0:
            return output

        timestep = self._get_current_timestep()

        # Skip the first start_step diffusion steps (no steering applied yet)
        if timestep < self.start_step:
            return output

        if timestep not in self.steering_vectors:
            return output

        multiplier = self.multiplier

        # Negate multiplier for unconditional passes
        is_uncond_pass = self.counter % 2 != 0
        if self.negate_for_uncond and is_uncond_pass:
            multiplier = -multiplier

        if multiplier == 0.0:
            return output

        vec = self.steering_vectors[timestep]
        vec = vec.to(output.device, output.dtype)

        # Rearrange output for frequency mode
        if self.sae_mode == "frequency":
            # output: (batch, time, freq) -> (batch, freq, time)
            to_modify = output.permute(0, 2, 1)
        else:
            to_modify = output

        # Add steering vector (broadcast over batch and spatial dims)
        # vec is (hidden_dim,) -> unsqueeze to (1, 1, hidden_dim)
        vec_broadcast = vec.unsqueeze(0).unsqueeze(0)
        result = to_modify + multiplier * vec_broadcast

        # Rearrange back
        if self.sae_mode == "frequency":
            result = result.permute(0, 2, 1)

        return result
