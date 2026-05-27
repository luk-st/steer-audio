# ACE Step Activation Collection

This document explains how to collect activations from the ACE Step model for SAE training.

## Overview

The `collect_activations_acestep.py` script collects intermediate activations from the ACE Step transformer model during audio generation. These activations can be used to train Sparse Autoencoders (SAEs) for interpretability research.

## Architecture

ACE Step uses a transformer architecture with **24 blocks (0-23)**. Each block contains:

- **`attn`**: Self-attention layer - produces attention output before residual connection
- **`cross_attn`**: Cross-attention layer - produces cross-attention output for text/lyric conditioning
- **`ff`**: Feedforward/MLP layer (GLUMBConv) - produces feedforward output before residual

### Forward Pass Flow

Each block processes data in this order:

```python
# 1. Self-attention
attn_output = self.attn(norm_hidden_states)
hidden_states = attn_output + hidden_states  # residual

# 2. Cross-attention (if enabled)
attn_output = self.cross_attn(hidden_states, encoder_hidden_states)
hidden_states = attn_output + hidden_states  # residual

# 3. Feedforward
ff_output = self.ff(norm_hidden_states)
hidden_states = hidden_states + ff_output  # residual
```

**Important**: When you hook `transformer_blocks.X.attn` or `transformer_blocks.X.cross_attn`, you're hooking the **attention output before it's added to the residual stream**. This is the same point where steering vectors are applied in `steering/caa/`.

### CFG Pass Handling

ACE Step runs **separate forward passes** for conditional and unconditional predictions (CFG), not batched together like some other models. This means:

- **2 passes (default)**: conditional, unconditional
- **3 passes (with double guidance)**: conditional, text_only, unconditional

**Important**: This collection script collects activations from **ALL CFG passes** (2-3x more data per prompt). This is because:

1. **Hook-based tracking is unreliable**: Unlike the steering code which replaces forward functions and can track CFG passes precisely, PyTorch hooks can't reliably determine which CFG pass is executing
2. **Simpler and more robust**: Collecting everything avoids fragile tracking logic
3. **Flexible**: You can filter to conditional-only during SAE training if needed

**Data Volume**: Expect 2-3x more activations than conditional-only collection. For a dataset with N prompts and 30 inference steps:
- Conditional only: ~N * 30 activations per layer
- All passes (this script): ~N * 60-90 activations per layer

**Filtering During Training**: If you want only conditional activations for SAE training, you can identify them by pattern (they repeat in sets of 2-3) or add metadata tracking during collection.

## Layer Naming Convention

Layers follow the pattern: `transformer_blocks.{block_num}.{component}`

Examples:
- `transformer_blocks.0.attn` - First block self-attention output
- `transformer_blocks.11.cross_attn` - Middle block cross-attention output
- `transformer_blocks.23.ff` - Last block feedforward output

## Recommended Hookpoints

Based on the steering vector approach from `steering/caa/compute_steering_vectors.py`:

### For Steering-Compatible SAEs:
- **Cross-attention layers** (e.g., `transformer_blocks.{5,11,17}.cross_attn`)
  - These are where text/lyric concepts influence the generation
  - Good for concept-based steering (genre, instrument, mood, etc.)
  - Matches the approach used in ACE steering
  - **Hook the attention output** (before residual addition) to collect/modify representations

## Usage

### Basic Usage

```bash
python sae/src/scripts/collect_activations_acestep.py
```

### Customizing Hookpoints

Edit the `hook_names` list in the script:

```python
hook_names=[
    "transformer_blocks.5.cross_attn",   # Early cross-attention (low-level features)
    "transformer_blocks.11.cross_attn",  # Middle cross-attention (mid-level concepts)
    "transformer_blocks.17.cross_attn",  # Late cross-attention (high-level semantics)
    # Available layers: 0-23
    # Hook outputs are captured BEFORE residual addition
]
```

### Configuration Options

Key parameters in `CacheActivationsRunnerConfig`:

- **`hook_names`**: List of layer names to collect from
- **`dataset_name`**: Path to CSV with prompts (e.g., `"data/musiccaps_public.csv"`)
- **`num_inference_steps`**: Denoising steps (30-60 typical for ACE Step)
- **`audio_length_in_s`**: Audio duration (10s default, up to 240s supported)
- **`guidance_scale`**: CFG scale (3.0 default for ACE Step)
- **`cache_every_n_timesteps`**: How often to cache (10 = every 10 timesteps)
- **`batch_size`**: Samples per batch (adjust for GPU memory)

## Comparison with Steering Vector Collection

The steering vector approach (`steering/caa/compute_steering_vectors.py`) and this SAE collection are related but different:

### Steering Vectors:
- Collects activations from **positive and negative prompt pairs**
- Computes **difference vectors** (pos - neg)
- Focuses on **specific concepts** (genre, mood, instrument)
- Uses `VectorStore` controller to hook during generation
- Saves steering vectors for inference-time control

### SAE Collection:
- Collects activations from **arbitrary prompts**
- Builds a **dataset of activations** for training
- Learns **sparse features** automatically
- Uses `HookedACEStepModel` for caching
- Trains SAEs to decompose activations into interpretable features

### Synergy:
You can use both approaches together:
1. Collect diverse activations with this script
2. Train SAEs to find sparse features
3. Use steering vectors to validate that SAE features capture meaningful concepts

## Output

Activations are saved to:
```
activations/{dataset_name}/{model_name}/
```

Default: `activations/musiccaps_public/ace-step/`

## GPU Memory Requirements

ACE Step is large. Recommended configurations:

- **24GB GPU**: `batch_size=2`, `audio_length_in_s=10.0`
- **40GB GPU**: `batch_size=4`, `audio_length_in_s=10.0`
- **80GB GPU**: `batch_size=8`, `audio_length_in_s=20.0`

Reduce `audio_length_in_s` if you run out of memory.

## Next Steps

After collecting activations:

1. **Train SAEs**: Use `sae/src/scripts/train.py` with collected activations
2. **Analyze Features**: Examine learned SAE features for interpretability
3. **Compare with Steering**: Check if SAE features align with steering vectors
4. **Integrate**: Use SAEs for controllable generation or analysis

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `audio_length_in_s`
- Reduce number of `hook_names`
- Use fewer `num_inference_steps`

### Slow Collection
- Increase `cache_every_n_timesteps` (e.g., 20 instead of 10)
- Reduce `dataset_duplicate_rows`
- Use smaller dataset

### Hook Not Found
- Check layer names with:
  ```python
  for name, module in pipeline.ace_step_transformer.named_modules():
      print(name)
  ```
- Ensure using `transformer_blocks.X.{attn,cross_attn,ff}` format

## References

- ACE Step paper: [Link to paper]
- Steering approach: `steering/caa/compute_steering_vectors.py`
- Base hooked model: `sae/src/hooked_model/hooked_model_acestep.py`
