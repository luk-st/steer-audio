# ACE Steering Modes

This document explains the six steering modes available in the ACE steering implementation.

## Overview

Steering modes control **how** and **which** CFG passes are steered during generation. This affects the interaction between steering strength (α) and CFG scale (w).

## The Six Modes

### 1. `cond_only` (Default)
**Steer only conditional pass using conditional vectors**

```python
controller = VectorStore(
    steering_vectors=vectors,
    steer_mode='cond_only',
    alpha=10.0
)
```

**How it works:**
- Computes: `Δ = E[cond_pos] - E[cond_neg]`
- Applies: `cond' = cond + α·Δ`, `uncond' = uncond` (unchanged)
- CFG: `output = uncond + w·(cond' - uncond)`

**Effect:**
```
output = uncond + w·(cond - uncond) + w·α·Δ
```
- Steering scales with CFG: **effective_steering = w × α**
- Higher CFG → stronger steering
- Need to adjust α when changing CFG scale

**When to use:**
- When you want steering to scale with prompt influence
- When uncond should remain "neutral"
- Default for most use cases

---

### 2. `uncond_only`
**Steer only unconditional pass using unconditional vectors**

```python
controller = VectorStore(
    steering_vectors=vectors,  # Needs all CFG passes saved (unified compute default)
    steer_mode='uncond_only',
    alpha=10.0
)
```

**How it works:**
- Computes: `Δ = E[uncond_pos] - E[uncond_neg]`
- Applies: `cond' = cond` (unchanged), `uncond' = uncond + α·Δ`
- CFG: `output = uncond' + w·(cond - uncond')`

**Effect:**
```
output = (uncond + α·Δ) + w·(cond - uncond - α·Δ)
       = uncond + α·Δ + w·(cond - uncond) - w·α·Δ
       = uncond + w·(cond - uncond) + α·Δ·(1 - w)
```
- Steering has **inverse CFG scaling**: effective_steering = α × (1 - w)
- Higher CFG → weaker steering
- Lower CFG → stronger steering
- Opposite behavior to `cond_only`

**When to use:**
- When you want to modify the "neutral" unconditional baseline
- When you want steering to decrease as CFG increases
- For exploring unconditional space modifications
- Experimental: opposite of `cond_only`

**Requirements:**
- Vectors must include all CFG passes (`save_all_cfg_passes=True`, the unified compute default)
- Need uncond activations from generation

---

### 3. `uncond_for_cond`
**Steer only conditional pass using unconditional vectors**

```python
controller = VectorStore(
    steering_vectors=vectors,  # Needs all CFG passes saved (unified compute default)
    steer_mode='uncond_for_cond',
    alpha=10.0
)
```

**How it works:**
- Computes: `Δ = E[uncond_pos] - E[uncond_neg]`
- Applies: `cond' = cond + α·Δ`, `uncond' = uncond` (unchanged)
- CFG: `output = uncond + w·(cond' - uncond)`

**Effect:**
```
output = uncond + w·(cond + α·Δ - uncond)
       = uncond + w·(cond - uncond) + w·α·Δ
```
- Steering has **CFG scaling**: effective_steering = w × α
- Higher CFG → stronger steering
- Same scaling behavior as `cond_only`
- Uses unconditional concept direction instead of conditional

**When to use:**
- When you want CFG-scaled steering but using unconditional concept space
- Experimental: exploring unconditional concept directions applied to conditional pass
- Comparing unconditional vs conditional concept representations

**Requirements:**
- Vectors must include all CFG passes (`save_all_cfg_passes=True`, the unified compute default)
- Need uncond activations from generation

---

### 4. `separate`
**Steer each CFG pass with its corresponding vectors**

```python
controller = VectorStore(
    steering_vectors=vectors,  # Needs all CFG passes saved (unified compute default)
    steer_mode='separate',
    alpha=10.0
)
```

**How it works:**
- Computes:
  - `Δ_cond = E[cond_pos] - E[cond_neg]`
  - `Δ_uncond = E[uncond_pos] - E[uncond_neg]`
- Applies: `cond' = cond + α·Δ_cond`, `uncond' = uncond + α·Δ_uncond`
- CFG: `output = uncond' + w·(cond' - uncond')`

**Effect:**
```
output = (uncond + α·Δ_uncond) + w·(cond + α·Δ_cond - uncond - α·Δ_uncond)
       = uncond + α·Δ_uncond + w·(cond - uncond) + w·α·(Δ_cond - Δ_uncond)
```
- Most flexible: each pass gets its own concept direction
- Steering depends on difference between cond and uncond vectors
- Requires computing vectors for all CFG passes

**When to use:**
- When you want maximum control
- When uncond and cond have different concept representations
- For research/exploration of CFG pass differences

**Requirements:**
- Vectors must include all CFG passes (`save_all_cfg_passes=True`, the unified compute default)
- More expensive: requires saving activations from all passes

---

### 5. `both_cond`
**Steer all CFG passes using conditional vectors**

```python
controller = VectorStore(
    steering_vectors=vectors,
    steer_mode='both_cond',
    alpha=10.0
)
```

**How it works:**
- Computes: `Δ = E[cond_pos] - E[cond_neg]`
- Applies: `cond' = cond + α·Δ`, `uncond' = uncond + α·Δ` (same vector!)
- CFG: `output = uncond' + w·(cond' - uncond')`

**Effect:**
```
output = (uncond + α·Δ) + w·(cond + α·Δ - uncond - α·Δ)
       = uncond + α·Δ + w·(cond - uncond)
```
- Steering is **independent** of CFG scale
- α controls steering absolutely
- w controls prompt adherence independently

**When to use:**
- When you want steering and CFG to be independent knobs
- When you want consistent steering across different CFG scales
- For global concept shifts (e.g., "make everything more female")
- This is the **CASteer approach**

---

### 6. `both_uncond`
**Steer all CFG passes using unconditional vectors**

```python
controller = VectorStore(
    steering_vectors=vectors,  # Needs all CFG passes saved (unified compute default)
    steer_mode='both_uncond',
    alpha=10.0
)
```

**How it works:**
- Computes: `Δ = E[uncond_pos] - E[uncond_neg]`
- Applies: `cond' = cond + α·Δ`, `uncond' = uncond + α·Δ` (same uncond vector!)
- CFG: `output = uncond' + w·(cond' - uncond')`

**Effect:**
```
output = (uncond + α·Δ) + w·(cond + α·Δ - uncond - α·Δ)
       = uncond + α·Δ + w·(cond - uncond)
```
- Steering is **independent** of CFG scale
- α controls steering absolutely
- w controls prompt adherence independently
- Same behavior as `both_cond` but using unconditional vectors

**When to use:**
- When you want steering and CFG to be independent knobs
- When you want to use unconditional concept space instead of conditional
- For global concept shifts using unconditional representations
- Experimental: comparing conditional vs unconditional concept directions

**Requirements:**
- Vectors must include all CFG passes (`save_all_cfg_passes=True`, the unified compute default)
- Need uncond activations from generation

---

## Comparison Table

| Mode | Steer Cond? | Steer Uncond? | Cond Vector | Uncond Vector | α-w Interaction |
|------|------------|---------------|-------------|---------------|-----------------|
| `cond_only` | ✓ | ✗ | Uses | - | Coupled (w×α) |
| `uncond_only` | ✗ | ✓ | - | Uses | Inverse coupled (α×(1-w)) |
| `uncond_for_cond` | ✓ | ✗ | - | Uses | Coupled (w×α) |
| `separate` | ✓ | ✓ | Uses | Uses | Complex |
| `both_cond` | ✓ | ✓ | Uses | Uses (same as cond) | Independent |
| `both_uncond` | ✓ | ✓ | Uses (same as uncond) | Uses | Independent |

## Mathematical Summary

### Final Output Formula

**cond_only:**
```
output = uncond + w·(cond - uncond) + w·α·Δ_cond
                                      ^^^^^^^^
                                      Scales with CFG
```

**uncond_only:**
```
output = uncond + w·(cond - uncond) + α·Δ_uncond·(1 - w)
                                      ^^^^^^^^^^^^^^^^^^
                                      Inverse scales with CFG
```

**uncond_for_cond:**
```
output = uncond + w·(cond - uncond) + w·α·Δ_uncond
                                      ^^^^^^^^^^^^
                                      Scales with CFG (using uncond vector)
```

**both_cond:**
```
output = uncond + w·(cond - uncond) + α·Δ_cond
                                      ^^^^^^^^
                                      Independent of CFG
```

**both_uncond:**
```
output = uncond + w·(cond - uncond) + α·Δ_uncond
                                      ^^^^^^^^^^^
                                      Independent of CFG (using uncond vector)
```

**separate:**
```
output = uncond + w·(cond - uncond) + α·Δ_uncond + w·α·(Δ_cond - Δ_uncond)
                                      ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
                                      Base shift    CFG-modulated difference
```

## Usage Examples

### Computing Vectors

Vectors are computed by the unified runner (the old ``compute_steering_vectors.py``
CLI was removed):

```bash
python src/steering/run_compute.py --config configs/steering/ace/caa/compute_piano.yaml
```

The CAA scorer saves **all CFG passes by default** (``save_all_cfg_passes=True`` in
``compute_sv_caa``), so the resulting artifact supports every mode below, including
the uncond-based ones.

### Loading and Using

The public API loads an artifact (local dir or HF repo id) and handles
registration for you:

```python
from src.steering import SteerableACEModel, CAASteeringController

model = SteerableACEModel(device="cuda")
model.pipeline.load()
ctrl = CAASteeringController.from_pretrained(
    "lukasz-staniszewski/ace-step-caa-piano", alpha=20, steer_mode="cond_only",
)
with model.steer(ctrl):
    audio = model.generate(prompt="instrumental music", lyrics="[inst]",
                           audio_duration=10.0, infer_step=30, manual_seed=0,
                           return_type="audio")
```

For direct use of this module (what the wrapper does under the hood):

```python
from src.models.ace_step.ace_steering.controller import VectorStore, register_vector_control

# Mode 1: cond_only (default)
controller = VectorStore(
    steering_vectors=vectors,
    steer_mode='cond_only',
    alpha=10.0,
    device='cuda'
)

# Mode 2: uncond_only (needs all CFG passes in the artifact)
controller = VectorStore(
    steering_vectors=vectors,
    steer_mode='uncond_only',
    alpha=10.0,
    device='cuda'
)

# Mode 3: uncond_for_cond (needs all CFG passes in the artifact)
controller = VectorStore(
    steering_vectors=vectors,
    steer_mode='uncond_for_cond',
    alpha=10.0,
    device='cuda'
)

# Mode 4: separate (needs all CFG passes in the artifact)
controller = VectorStore(
    steering_vectors=vectors,
    steer_mode='separate',
    alpha=10.0,
    device='cuda'
)

# Mode 5: both_cond
controller = VectorStore(
    steering_vectors=vectors,
    steer_mode='both_cond',
    alpha=10.0,
    device='cuda'
)

# Mode 6: both_uncond (needs all CFG passes in the artifact)
controller = VectorStore(
    steering_vectors=vectors,
    steer_mode='both_uncond',
    alpha=10.0,
    device='cuda'
)

# Register and generate
register_vector_control(pipe.ace_step_transformer, controller)
audio = pipe.generate(prompt="rock music", ...)
```

## Experimental Recommendations

To determine which mode works best for your use case:

1. **Compute vectors once** with the unified runner (all CFG passes are saved by default)
2. **Test `cond_only` vs `both_cond`** with same α across different CFG scales
3. **Hypothesis:**
   - `cond_only`: Need to adjust α when changing CFG
   - `both_cond`: Same α works across CFG scales
4. **All modes are testable from one artifact**, since the compute saves every CFG pass:
   - Compare conditional vs unconditional concept directions
   - Test CFG scaling behavior differences
   - Explore `uncond_for_cond` vs `cond_only` and `both_uncond` vs `both_cond`

### Test Script Template

```python
# Test CFG-scaled modes (using cond vectors)
modes_cond = ['cond_only', 'both_cond']
# Test CFG-scaled modes (using uncond vectors)
modes_uncond = ['uncond_only', 'uncond_for_cond', 'both_uncond']

cfg_scales = [1.0, 3.0, 5.0, 7.5]
alphas = [5.0, 10.0, 15.0, 20.0]

# Test with conditional vectors
for mode in modes_cond:
    for cfg in cfg_scales:
        for alpha in alphas:
            controller = VectorStore(
                steering_vectors=vectors_cond,
                steer_mode=mode,
                alpha=alpha
            )
            # Generate and evaluate
            # Expected behavior:
            # - cond_only: steering increases with CFG (w×α)
            # - both_cond: steering independent of CFG (α)

# Test with unconditional vectors
for mode in modes_uncond:
    for cfg in cfg_scales:
        for alpha in alphas:
            controller = VectorStore(
                steering_vectors=vectors_uncond,
                steer_mode=mode,
                alpha=alpha
            )
            # Generate and evaluate
            # Expected behavior:
            # - uncond_only: steering decreases with CFG (α×(1-w))
            # - uncond_for_cond: steering increases with CFG (w×α)
            # - both_uncond: steering independent of CFG (α)
```

## Implementation Details

### CFG Pass Tracking

ACE-Step uses **sequential** CFG passes (not batched like UNet):
```
Pass 0: Conditional (with prompt)
Pass 1: Text-only (optional, for double guidance)
Pass 2: Unconditional (empty prompt)
```

The controller tracks `cfg_pass_count` to determine which pass is currently executing.

### Vector Storage Format

**With `save_only_cond=True` (default):**
```python
vectors = {
    step: {
        'tf0': [vector_layer_0, vector_layer_1, ...],
        'tf1': [vector_layer_0, vector_layer_1, ...],
        ...
    }
}
```

**With `save_all_cfg_passes=True`:**
```python
vectors = {
    (step, cfg_pass): {
        'tf0': [vector_layer_0, vector_layer_1, ...],
        'tf1': [vector_layer_0, vector_layer_1, ...],
        ...
    }
}
```

## Questions?

- **Which mode is best?** Depends on your goal. Start with `cond_only`, then try `both_cond`.
- **Do I need all CFG passes in the artifact?** Only for `uncond_only`, `uncond_for_cond`, `both_uncond`, and `separate` modes; the unified compute saves them by default.
- **Can I switch modes with same vectors?**
  - Yes, between `cond_only` and `both_cond` (using cond vectors)
  - Yes, between `uncond_only`, `uncond_for_cond`, and `both_uncond` (using uncond vectors)
  - Need recompute to switch between cond-based and uncond-based modes
- **Which mode does CASteer use?** `both_cond` (but they use batched CFG in UNet, not sequential like ACE)
- **What's the difference between modes using cond vs uncond vectors?**
  - Conditional vectors: concept direction in conditional (prompted) space
  - Unconditional vectors: concept direction in unconditional (unprompted) space
  - These may differ depending on how the model represents concepts with vs without prompts
- **When should I use `uncond_for_cond` vs `cond_only`?**
  - Both have CFG scaling (w×α), but use different concept spaces
  - `uncond_for_cond`: experimental, tests if unconditional concept directions work better
  - `cond_only`: default, uses the more common conditional concept space
