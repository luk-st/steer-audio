"""
Compute steering vectors for ACE-Step model.

This script generates steering vectors by computing the difference between
activations from positive and negative prompt pairs.

Usage:
    python compute_steering_vectors.py \
        --mode gender \
        --concept_pos "female vocals" \
        --concept_neg "male vocals" \
        --num_prompts 50 \
        --save_dir steering_vectors \
        --num_inference_steps 60
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import torch

from construct_prompts import (
    get_prompts_gender,
    get_prompts_electronic_music,
    get_prompts_instrument,
    get_prompts_mood,
    get_prompts_tempo,
)
from controller import VectorStore, register_vector_control

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_ace import SimpleACEStepPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Compute steering vectors for ACE-Step")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['instrument', 'electronic_music', 'gender', 'tempo', 'mood'],
        default='gender',
        help='Type of steering vectors to compute'
    )
    parser.add_argument(
        '--concept_pos',
        type=str,
        default='female vocals',
        help='Positive concept (target)'
    )
    parser.add_argument(
        '--concept_neg',
        type=str,
        default='male vocals',
        help='Negative concept (source)'
    )
    parser.add_argument(
        '--num_prompts',
        type=int,
        default=50,
        help='Number of prompt pairs to use'
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=60,
        help='Number of denoising steps'
    )
    parser.add_argument(
        '--audio_duration',
        type=float,
        default=10.0,
        help='Duration of generated audio in seconds'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (cuda or cpu)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='steering_vectors',
        help='Directory to save steering vectors'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Path to ACE-Step checkpoint directory'
    )
    parser.add_argument(
        '--save_all_cfg_passes',
        action='store_true',
        help='If set, save activations from all CFG passes (for steer_mode=separate). Default: only save cond.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize pipeline
    print("Loading ACE-Step pipeline...")
    pipe = SimpleACEStepPipeline(
        device=device,
        persistent_storage_path=args.checkpoint_dir,
    )
    pipe.load()

    # Get prompt pairs based on mode
    print(f"\nGenerating prompts for mode: {args.mode}")
    if args.mode == 'instrument':
        prompts_pos, prompts_neg = get_prompts_instrument(
            num=args.num_prompts,
            concept_pos=args.concept_pos,
            concept_neg=args.concept_neg
        )
    elif args.mode == 'electronic_music':
        prompts_pos, prompts_neg = get_prompts_electronic_music(
            num=args.num_prompts,
            concept_pos=args.concept_pos,
            concept_neg=args.concept_neg
        )
    elif args.mode == 'gender':
        prompts_pos, prompts_neg = get_prompts_gender(
            concept_pos=args.concept_pos,
            concept_neg=args.concept_neg
        )
    elif args.mode == 'tempo':
        prompts_pos, prompts_neg = get_prompts_tempo(
            concept_pos=args.concept_pos,
            concept_neg=args.concept_neg
        )
    elif args.mode == 'mood':
        prompts_pos, prompts_neg = get_prompts_mood(
            concept_pos=args.concept_pos,
            concept_neg=args.concept_neg
        )

    print(f"Generated {len(prompts_pos)} prompt pairs")

    # Collect activations for positive and negative prompts
    pos_vectors = []
    neg_vectors = []

    print("\nCollecting activations...")
    for i, (prompt_pos, prompt_neg) in enumerate(zip(prompts_pos, prompts_neg)):
        print(f'\nPrompt pair {i+1}/{len(prompts_pos)}')
        print(f'  Positive: {prompt_pos}')
        print(f'  Negative: {prompt_neg}')

        # Collect positive activations
        controller = VectorStore(
            device=device,
            save_only_cond=not args.save_all_cfg_passes
        )
        controller.steer = False  # Just collecting activations, not steering
        register_vector_control(pipe.ace_step_transformer, controller)

        _ = pipe.generate(
            prompt=prompt_pos,
            audio_duration=args.audio_duration,
            infer_step=args.num_inference_steps,
            manual_seed=args.seed,
            return_type='latent',  # Don't need actual audio
            use_erg_lyric=False
        )

        pos_vectors.append(controller.vector_store)
        controller.reset()

        # Collect negative activations
        controller = VectorStore(
            device=device,
            save_only_cond=not args.save_all_cfg_passes
        )
        controller.steer = False
        register_vector_control(pipe.ace_step_transformer, controller)

        _ = pipe.generate(
            prompt=prompt_neg,
            audio_duration=args.audio_duration,
            infer_step=args.num_inference_steps,
            manual_seed=args.seed,
            return_type='latent',
            use_erg_lyric=False
        )

        neg_vectors.append(controller.vector_store)
        controller.reset()

        # Clean up memory
        if i % 5 == 0:
            pipe.cleanup_memory()

    # Compute steering vectors
    print("\nComputing steering vectors...")
    steering_vectors = {}

    # Get all step keys from first sample
    all_step_keys = list(pos_vectors[0].keys())

    # Filter to ensure all keys have the same type (all tuples or all ints)
    # Detect the expected key type from the first key
    first_key = all_step_keys[0]
    if isinstance(first_key, tuple):
        # Filter to only tuple keys
        all_step_keys = [k for k in all_step_keys if isinstance(k, tuple)]
    else:
        # Filter to only int keys
        all_step_keys = [k for k in all_step_keys if isinstance(k, int)]

    example_step_key = all_step_keys[0]
    layer_names = list(pos_vectors[0][example_step_key].keys())

    # Iterate over actual step keys (could be int or tuple)
    for step_key in all_step_keys:
        steering_vectors[step_key] = defaultdict(list)

        for layer_name in layer_names:
            num_layers = len(pos_vectors[0][step_key][layer_name])

            for layer_idx in range(num_layers):
                # Collect all positive vectors for this layer
                pos_vectors_layer = [
                    pos_vectors[i][step_key][layer_name][layer_idx]
                    for i in range(len(pos_vectors))
                ]
                pos_vectors_avg = np.mean(pos_vectors_layer, axis=0)

                # Collect all negative vectors for this layer
                neg_vectors_layer = [
                    neg_vectors[i][step_key][layer_name][layer_idx]
                    for i in range(len(neg_vectors))
                ]
                neg_vectors_avg = np.mean(neg_vectors_layer, axis=0)

                # Compute steering vector as difference
                steering_vector = pos_vectors_avg - neg_vectors_avg

                # Normalize
                norm = np.linalg.norm(steering_vector)
                if norm > 0:
                    steering_vector = steering_vector / norm

                steering_vectors[step_key][layer_name].append(steering_vector)

    # Save steering vectors
    os.makedirs(args.save_dir, exist_ok=True)
    filename = f"ace_{args.mode}_{args.concept_pos}_{args.concept_neg}.pkl"
    filename = filename.replace(' ', '_').replace('/', '_')
    save_path = os.path.join(args.save_dir, filename)

    with open(save_path, 'wb') as f:
        pickle.dump(steering_vectors, f)

    print(f"\nSteering vectors saved to: {save_path}")
    print(f"  Total steps: {len(steering_vectors)}")
    print(f"  Step keys type: {'tuple (step, cfg_pass)' if isinstance(all_step_keys[0], tuple) else 'int (step only)'}")
    print(f"  Layers per step: {layer_names}")
    print(f"  Vectors per layer: {num_layers}")
    print(f"\nCompatible steering modes:")
    if args.save_all_cfg_passes:
        print(f"  - 'separate': ✓ (cond steered with cond vectors, uncond with uncond vectors)")
        print(f"  - 'cond_only': ✓ (only cond steered, uses cond vectors)")
        print(f"  - 'both_cond': ✓ (both steered with cond vectors)")
    else:
        print(f"  - 'separate': ✗ (need --save_all_cfg_passes)")
        print(f"  - 'cond_only': ✓ (only cond steered, uses cond vectors)")
        print(f"  - 'both_cond': ✓ (both steered with cond vectors)")


if __name__ == '__main__':
    main()
