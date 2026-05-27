# ACE-Step SAE Intervention Hooks

This module provides hooks for intervening on ACE-Step model activations using Sparse Autoencoders (SAEs). These hooks can be registered on model layers to reconstruct, ablate, or apply per-timestep feature interventions during audio generation.

## Overview

All hooks follow PyTorch's forward hook API and can be registered using `register_forward_hook()`. They process activations through an SAE and optionally modify specific latent features before decoding back to the original space.

## Key Concepts

### SAE Modes

Two modes are supported for how the SAE processes activations:

- **`"sequence"` (default)**: SAE reconstructs along the embedding dimension
  - Input shape: `(batch, seq_len, dim)` → SAE operates on `dim`
  - Standard mode for most use cases

- **`"frequency"`**: SAE reconstructs along the frequency/time dimension
  - Input shape: `(batch, seq_len, dim)` → transposed to `(batch, dim, seq_len)` → SAE operates on `seq_len`
  - Requires SAE trained on transposed activations

### CFG Pass Handling

ACE-Step uses Classifier-Free Guidance (CFG), which means the model makes multiple forward passes per timestep:

```
cond_t0 → uncond_t0 → cond_t1 → uncond_t1 → ... → cond_tN → uncond_tN
```

- **`uncond_preds=False` (default)**: Only intervene on conditional passes (even counters)
- **`uncond_preds=True`**: Intervene on both conditional and unconditional passes

### Renormalization

Renormalization (`renorm=True`) preserves the original activation magnitude after intervention. This is crucial for preventing loudness changes in generated audio.

**How it works** (following CASteer approach):
1. Save L2 norm of original activations: `norm_before = torch.norm(x, dim=2, keepdim=True)`
2. Process through SAE and apply interventions
3. Normalize output to unit norm and rescale: `x_out = (x_out / ||x_out||) * norm_before`

A small epsilon (`1e-8`) is added to prevent division by zero.

## Available Hooks

### `ACEStepSAEReconstructHook`

Reconstruct activations through an SAE without modifying latents. Useful for measuring SAE reconstruction quality and its effect on generation.

```python
from sae.sae_src.hooked_model.acestep_hooks import ACEStepSAEReconstructHook

hook = ACEStepSAEReconstructHook(
    sae=sae,                    # Trained SAE model
    sae_mode="sequence",        # "sequence" or "frequency"
    uncond_preds=False,         # Only intervene on conditional passes
    renorm=True,                # Preserve activation magnitude
)

handle = model.cross_attn.register_forward_hook(hook)
# ... run inference ...
handle.remove()
```

### `ACEStepTimestepInterventionHook`

Apply per-timestep feature interventions during diffusion. Allows different features to be modified at different timesteps based on feature importance analysis.

```python
from sae.sae_src.hooked_model.acestep_hooks import ACEStepTimestepInterventionHook

# Define features to modify per timestep
features_per_timestep = {
    0: [1, 5, 10],      # Early timesteps: modify features 1, 5, 10
    5: [2, 8],          # Mid timesteps: modify features 2, 8
    10: [3, 7, 9],      # Late timesteps: modify features 3, 7, 9
}

# Same multiplier for all
hook = ACEStepTimestepInterventionHook(
    sae=sae,
    features_per_timestep=features_per_timestep,
    multiplier=2.0,
    renorm=True,
)

# Different multipliers per timestep
hook = ACEStepTimestepInterventionHook(
    sae=sae,
    features_per_timestep=features_per_timestep,
    multiplier={0: 2.0, 5: 1.5, 10: 0.5},  # Decrease effect over time
    renorm=True,
)
```

**Integration with FeatureSelector:**

```python
from sae.sae_src.hooked_model.acestep_hooks import ACEStepTimestepInterventionHook

# Assuming you have FeatureSelector results
features_per_ts = get_top_features_per_timestep(results, 'cohens_d', top_k=10)

hook = ACEStepTimestepInterventionHook(
    sae=sae,
    features_per_timestep=features_per_ts,
    multiplier=2.0,
    renorm=True,
)
```

### `ACEStepAblateHook`

Zero out all activations passing through the hooked layer. This is a destructive intervention for measuring layer importance.

**Note:** Renormalization is NOT supported for this hook since zeroed activations cannot be meaningfully renormalized (would result in NaN from 0/0).

```python
from sae.sae_src.hooked_model.acestep_hooks import ACEStepAblateHook

hook = ACEStepAblateHook(
    sae=None,           # Not used, kept for API consistency
    sae_mode="sequence",
    uncond_preds=False,
)
```

## Usage Pattern

All hooks follow the same usage pattern:

```python
# 1. Create hook
hook = ACEStepSAEReconstructHook(sae, renorm=True)

# 2. Register on model layer
handle = model.transformer_blocks[0].cross_attn.register_forward_hook(hook)

# 3. Run inference
with torch.no_grad():
    output = model.generate(...)

# 4. Remove hook
handle.remove()

# 5. Reset counter for next run (important!)
hook.counter = -1
```

## Important Notes

1. **Counter Reset**: Hooks track CFG passes using an internal counter. Reset it (`hook.counter = -1`) before each new generation.

2. **Memory**: Hooks operate under `@torch.no_grad()` and don't store gradients.

3. **Dtype Handling**: Hooks preserve the input dtype. No explicit dtype conversion is needed.

4. **Multiple Hooks**: You can register multiple hooks on different layers. Each needs its own counter management.

## Comparison with CASteer

These hooks are designed to be compatible with the CASteer (Classifier-free guidance Activation Steering) approach:

| Feature | CASteer | SAE Hooks |
|---------|---------|-----------|
| Steering method | Add steering vector | Modify SAE latents |
| Renormalization | L2 norm preservation | L2 norm preservation (identical) |
| Granularity | Per-layer | Per-layer + per-feature |
| CFG handling | Conditional only (default) | Conditional only (default) |
| Timestep control | Per-step vectors | Per-timestep feature selection |

The renormalization approach is identical to CASteer (lines 239, 263-264 in `controller.py`):
```python
norm = torch.norm(vector, dim=2, keepdim=True)
# ... modify vector ...
vector = vector / torch.norm(vector, dim=2, keepdim=True)
vector = vector * norm
```

## Troubleshooting

### Audio is too loud/quiet
- Enable `renorm=True` to preserve activation magnitudes

### No effect from intervention
- Check that `uncond_preds` matches your CFG setup
- Verify the hook is registered on the correct layer
- Ensure the counter is reset between runs

### NaN in output
- This can happen if renorm is applied to near-zero activations
- The epsilon (`1e-8`) should prevent this, but very small activations may still cause issues
