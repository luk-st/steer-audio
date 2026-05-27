#!/bin/bash
# Run Concept Slider inference with alpha sweep through the unified runner.
#
# Usage:
#   bash steering/cs/scripts/run_eval.sh <lora_path> <concept> [options]
#
# Examples:
#   bash steering/cs/scripts/run_eval.sh steering_vectors/concept_slider/ace_piano_r8_eta7_500_tf6tf7 piano
#   bash steering/cs/scripts/run_eval.sh steering_vectors/concept_slider/ace_piano_r8_eta7_500_tf6tf7 piano --gpu 2 --min -0.5 --max 0.5

set -e

LORA_PATH=$1
CONCEPT=$2
if [ -z "$LORA_PATH" ] || [ -z "$CONCEPT" ]; then
    echo "Usage: $0 <lora_path> <concept> [--gpu N] [--min F] [--max F] [--n_alphas N] [--start_step N] [--save_dir PATH]"
    echo "Concepts: piano, tempo, mood, vocal_gender, drums"
    exit 1
fi
shift 2

GPU=0
MIN_ALPHA=-1.0
MAX_ALPHA=1.0
N_ALPHAS=31
START_STEP=0
SAVE_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU=$2; shift 2;;
        --min) MIN_ALPHA=$2; shift 2;;
        --max) MAX_ALPHA=$2; shift 2;;
        --n_alphas) N_ALPHAS=$2; shift 2;;
        --num_prompts) shift 2;;  # accepted-but-ignored for legacy compat
        --start_step) START_STEP=$2; shift 2;;
        --save_dir) SAVE_DIR=$2; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# Number of intermediate alphas per side of zero — matches the legacy script's
# linspace((N-1)/2 + 1)[:-1] semantics.
N_SIDE=$(( (N_ALPHAS - 1) / 2 ))

if [ -z "$SAVE_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
    SAVE_DIR="steering/outputs/${CONCEPT}/cs/${TIMESTAMP}"
fi

echo "=== Running Concept Slider Inference ==="
echo "  LoRA:       $LORA_PATH"
echo "  Concept:    $CONCEPT"
echo "  GPU:        $GPU"
echo "  Alphas:     ${N_ALPHAS} in [${MIN_ALPHA}, ${MAX_ALPHA}] (${N_SIDE} per side)"
echo "  Start step: $START_STEP"
echo "  Save dir:   $SAVE_DIR"
echo ""

CUDA_VISIBLE_DEVICES=$GPU python src/steering/run_eval.py \
    --method concept_slider \
    --artifact "$LORA_PATH" \
    --concept "$CONCEPT" \
    --min-range "$MIN_ALPHA" \
    --max-range "$MAX_ALPHA" \
    --steps-per-side "$N_SIDE" \
    --save-dir "$SAVE_DIR"

CUDA_VISIBLE_DEVICES=$GPU python src/steering/eval/eval_steering_protocol.py --steering_dir "$SAVE_DIR" --concept "$CONCEPT"
