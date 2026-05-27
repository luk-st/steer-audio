#!/bin/bash
# Train Concept Slider LoRA for ACE-Step through the unified compute runner.
#
# Usage:
#   bash steering/cs/scripts/train.sh <concept> [options]
#
# Examples:
#   bash steering/cs/scripts/train.sh piano
#   bash steering/cs/scripts/train.sh vocal_gender --gpu 3 --eta 7 --iters 500 --rank 8

set -e

CONCEPT=$1
if [ -z "$CONCEPT" ]; then
    echo "Usage: $0 <concept> [--gpu N] [--eta N] [--iters N] [--rank N] [--lr N] [--output_dir PATH]"
    echo "Concepts: piano, tempo, mood, vocal_gender, drums"
    exit 1
fi
shift

GPU=0
ETA=7
ITERS=500
RANK=8
LR=1e-4
LAYERS=all
NO_DENOISING=false
DENOISE_STEPS=50
DENOISE_CFG=5.0
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU=$2; shift 2;;
        --eta) ETA=$2; shift 2;;
        --iters) ITERS=$2; shift 2;;
        --rank) RANK=$2; shift 2;;
        --lr) LR=$2; shift 2;;
        --layers) LAYERS=$2; shift 2;;
        --no_denoising) NO_DENOISING=true; shift;;
        --denoise_steps) DENOISE_STEPS=$2; shift 2;;
        --denoise_cfg) DENOISE_CFG=$2; shift 2;;
        --output_dir) OUTPUT_DIR=$2; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

CS_DIR="steering/cs"
if [ "$RANK" = "8" ]; then
    LORA_CONFIG="${CS_DIR}/lora_config_r8.json"
else
    LORA_CONFIG="${CS_DIR}/lora_config.json"
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="steering_vectors/concept_slider/ace_${CONCEPT}_r${RANK}_eta${ETA}_${ITERS}"
fi

echo "=== Training Concept Slider ==="
echo "  Concept:     $CONCEPT"
echo "  GPU:         $GPU"
echo "  Eta:         $ETA"
echo "  Iterations:  $ITERS"
echo "  Rank:        $RANK"
echo "  LR:          $LR"
echo "  Layers:      $LAYERS"
echo "  LoRA config: $LORA_CONFIG"
echo "  Output:      $OUTPUT_DIR"
echo ""

if [ "$NO_DENOISING" = "true" ]; then
    DENOISING_JSON="\"with_denoising\":false"
else
    DENOISING_JSON="\"with_denoising\":true,\"max_denoising_steps\":${DENOISE_STEPS},\"denoise_cfg_scale\":${DENOISE_CFG}"
fi

CUDA_VISIBLE_DEVICES=$GPU python src/steering/run_compute.py \
    --scorer concept_slider \
    --concept "$CONCEPT" \
    --output "$OUTPUT_DIR" \
    --no-model \
    --scorer-kwargs "{\"iterations\":${ITERS},\"eta\":${ETA},\"lr\":${LR},\"layers\":\"${LAYERS}\",\"lora_config_path\":\"${LORA_CONFIG}\",\"save_every\":500,${DENOISING_JSON}}"
