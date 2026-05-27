#!/bin/bash
# Final CS evaluation: r=8, 500it, eta=7, start@0, 31 asymmetric alphas per concept.
# 4 concepts on 4 GPUs in parallel via the unified runner.

set -e
SV_DIR="steering_vectors/cs"
OUT_DIR="steering/outputs"

run() {
    local gpu=$1 concept=$2 alphas=$3
    local lora="${SV_DIR}/ace_${concept}_r8_eta7_5k/checkpoint_500"
    local save="${OUT_DIR}/${concept}/cs/final_r8_eta7_500it"
    echo "GPU${gpu}: ${concept} → ${save}"
    CUDA_VISIBLE_DEVICES=$gpu python src/steering/run_eval.py \
        --method concept_slider \
        --artifact "$lora" \
        --concept "$concept" \
        --alphas="$alphas" \
        --save-dir "$save" 2>&1 | tail -2
}

echo "===== INFERENCE (4 concepts × 31 alphas × 100 prompts) ====="
run 0 piano '-0.32,-0.2987,-0.2773,-0.256,-0.2347,-0.2133,-0.192,-0.1707,-0.1493,-0.128,-0.1067,-0.0853,-0.064,-0.0427,-0.0213,0.0,0.024,0.048,0.072,0.096,0.12,0.144,0.168,0.192,0.216,0.24,0.264,0.288,0.312,0.336,0.36' &
run 1 tempo '-0.92,-0.8587,-0.7973,-0.736,-0.6747,-0.6133,-0.552,-0.4907,-0.4293,-0.368,-0.3067,-0.2453,-0.184,-0.1227,-0.0613,0.0,0.0667,0.1333,0.2,0.2667,0.3333,0.4,0.4667,0.5333,0.6,0.6667,0.7333,0.8,0.8667,0.9333,1.0' &
run 2 mood '-0.54,-0.504,-0.468,-0.432,-0.396,-0.36,-0.324,-0.288,-0.252,-0.216,-0.18,-0.144,-0.108,-0.072,-0.036,0.0,0.026,0.052,0.078,0.104,0.13,0.156,0.182,0.208,0.234,0.26,0.286,0.312,0.338,0.364,0.39' &
run 3 vocal_gender '-0.49,-0.4573,-0.4247,-0.392,-0.3593,-0.3267,-0.294,-0.2613,-0.2287,-0.196,-0.1633,-0.1307,-0.098,-0.0653,-0.0327,0.0,0.0313,0.0627,0.094,0.1253,0.1567,0.188,0.2193,0.2507,0.282,0.3133,0.3447,0.376,0.4073,0.4387,0.47' &
wait
echo "===== INFERENCE DONE ====="

echo "===== EVAL ====="
for i in 0 1 2 3; do
    case $i in
        0) c=piano;;
        1) c=tempo;;
        2) c=mood;;
        3) c=vocal_gender;;
    esac
    CUDA_VISIBLE_DEVICES=$i python src/steering/eval/eval_steering_protocol.py \
        --steering_dir "${OUT_DIR}/${c}/cs/final_r8_eta7_500it" \
        --concept "$c" --skip_aesthetics 2>&1 | tail -1 &
done
wait
echo "===== ALL DONE ====="
