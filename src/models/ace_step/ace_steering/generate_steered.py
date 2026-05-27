"""
Generate audio with ACE-Step using steering vectors.

Usage:
    python generate_steered.py \
        --prompt "a pop song" \
        --steering_vectors steering_vectors/ace_gender_vocal_gender_male_vocals.pkl \
        --alpha 10 \
        --save_dir outputs/steered
"""

import argparse
import os
import pickle

import torch
import torchaudio

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steering_ace import SteeredACEStepPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio with steering")
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Text prompt for generation'
    )
    parser.add_argument(
        '--steering_vectors',
        type=str,
        required=True,
        help='Path to steering vectors pickle file'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=10.0,
        help='Steering intensity (forward steering)'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=2.0,
        help='Steering intensity (backward steering)'
    )
    parser.add_argument(
        '--steer_back',
        action='store_true',
        help='Use backward steering (remove concept instead of add)'
    )
    parser.add_argument(
        '--steer_all_cfg',
        action='store_true',
        help='Steer all CFG passes (cond, cond_text_only, uncond). Default: only steer conditional pass'
    )
    parser.add_argument(
        '--no_steer',
        action='store_true',
        help='Generate without steering (baseline)'
    )
    parser.add_argument(
        '--audio_duration',
        type=float,
        default=30.0,
        help='Duration of generated audio in seconds'
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=60,
        help='Number of denoising steps'
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=15.0,
        help='Guidance scale for generation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
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
        default='outputs/steered',
        help='Directory to save generated audio'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Path to ACE-Step checkpoint directory'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load steering vectors
    steering_vectors = None
    if not args.no_steer:
        print(f"\nLoading steering vectors from: {args.steering_vectors}")
        with open(args.steering_vectors, 'rb') as f:
            steering_vectors = pickle.load(f)
        print(f"  Loaded {len(steering_vectors)} steps")

    # Initialize pipeline
    print("\nInitializing ACE-Step pipeline...")
    if args.no_steer:
        # Use regular pipeline without steering
        from pipeline_ace import SimpleACEStepPipeline
        pipe = SimpleACEStepPipeline(
            device=device,
            persistent_storage_path=args.checkpoint_dir,
        )
    else:
        # Use steered pipeline
        pipe = SteeredACEStepPipeline(
            device=device,
            persistent_storage_path=args.checkpoint_dir,
            steering_vectors=None,  # Will be set after loading
            steer=True,
            alpha=args.alpha,
            beta=args.beta,
            steer_back=args.steer_back,
            steer_only_cond=not args.steer_all_cfg,
        )

    print("Loading model...")
    pipe.load()

    # Setup steering after model is loaded
    if not args.no_steer:
        print("Setting up steering...")
        pipe.setup_steering(
            steering_vectors=steering_vectors,
            steer=True,
            alpha=args.alpha,
            beta=args.beta,
            steer_back=args.steer_back,
            steer_only_cond=not args.steer_all_cfg,
        )

    # Generate audio
    print(f"\nGenerating audio...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Duration: {args.audio_duration}s")
    print(f"  Steps: {args.num_inference_steps}")
    if not args.no_steer:
        if args.steer_back:
            print(f"  Steering: BACKWARD (against concept), beta={args.beta}")
        else:
            print(f"  Steering: FORWARD (towards concept), alpha={args.alpha}")

    audio = pipe.generate(
        prompt=args.prompt,
        audio_duration=args.audio_duration,
        infer_step=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        manual_seed=args.seed,
        return_type='audio',
        use_erg_lyric=False
    )

    # Save audio
    os.makedirs(args.save_dir, exist_ok=True)

    # Create filename
    steer_suffix = "no_steer" if args.no_steer else (
        f"steer_back_beta{args.beta}" if args.steer_back else f"steer_alpha{args.alpha}"
    )
    filename = f"{args.prompt.replace(' ', '_')[:50]}_seed{args.seed}_{steer_suffix}.wav"
    save_path = os.path.join(args.save_dir, filename)

    # Save audio (assuming audio is tensor with shape [batch, channels, samples])
    if isinstance(audio, torch.Tensor):
        # Take first sample if batch
        if audio.ndim == 3:
            audio = audio[0]
        torchaudio.save(save_path, audio.cpu(), pipe.sample_rate)
    else:
        print(f"Warning: Unexpected audio type: {type(audio)}")
        print("Saving with torch.save instead")
        torch.save(audio, save_path.replace('.wav', '.pt'))

    print(f"\nAudio saved to: {save_path}")

    # Reset controller for next generation
    if not args.no_steer and hasattr(pipe, 'reset_controller'):
        pipe.reset_controller()


if __name__ == '__main__':
    main()
