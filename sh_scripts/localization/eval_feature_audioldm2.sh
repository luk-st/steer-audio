#!/bin/bash
#
# Evaluate patched-vs-clean audios for one feature on AudioLDM 2.
#
# Usage:
#   bash sh_scripts/localization/eval_feature_audioldm2.sh FEATURE FILENAME BLOCK [BLOCK ...]
#
# Args:
#   FEATURE   Concept being patched (e.g. piano, violin, happy, sad).
#             Locates patched audios under
#               outputs/audioldm2/patching/${FEATURE}/${BLOCK}/audios/
#             and the eval prompts (patch_data=musiccaps/${FEATURE}_ace).
#   FILENAME  Output CSV name for the aggregated per-block summary written to
#               outputs/audioldm2/patching/${FEATURE}/${FILENAME}
#             Free-form — include an extension if you want one (e.g. piano_summary.csv).
#   BLOCK...  One or more layer/block presets to evaluate. Each must already
#             have patched + clean audios on disk. Examples: up1tf5, up1tf9attn1, up1tf2tf5tf9tf10.
#
# Example:
#   bash sh_scripts/localization/eval_feature_audioldm2.sh piano piano_summary up1tf5

usage() { sed -n '3,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]]; then usage 0; fi

FEATURE=$1
FILENAME=$2
shift 2

BLOCKS_TO_PATCH=( "$@" )
MODEL_NAME="audioldm2"

for block in "${BLOCKS_TO_PATCH[@]}"; do
    patch_data="musiccaps/${FEATURE}_ace"
    echo "Evaluating ${FEATURE} ${block}"

    python src/eval_audio.py paths.generated_samples="${MODEL_NAME}/patching/${FEATURE}/${block}/audios/patched.npy" paths.reference_samples="${MODEL_NAME}/patching/${FEATURE}/${block}/audios/clean.npy" patch_data="${patch_data}" patch_model="${MODEL_NAME}_patch.yaml" metrics.muqt=true metrics.clap=true metrics.clap_music=true
done

python src/postprocess/collect_feature_metrics.py --feature "${FEATURE}" --blocks "${BLOCKS_TO_PATCH[@]}" --filename "${FILENAME}" --model_name "${MODEL_NAME}" --output_dir "$(pwd)/outputs"