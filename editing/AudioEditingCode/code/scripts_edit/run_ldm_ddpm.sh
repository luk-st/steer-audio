#!/bin/sh

MODE="ddpm"
CUDA_DEVICE_START=4
PARTS=(0 1 2 3)
NUM_DIFFUSION_STEPS=200
TSTART=100
CFG_SRC=3.0
CFG_TAR=12.0
SEED=42
N_PARTS=${#PARTS[@]}

RUN_NAME="audioldm2_${MODE}_ours_cfgs${CFG_SRC}_cfgt${CFG_TAR}_ts${TSTART}_step${NUM_DIFFUSION_STEPS}"

for PART in "${PARTS[@]}"; do
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_START} \
    python edit_audioldm_medleydb.py \
        --mode ${MODE} \
        --part_id ${PART} \
        --n_parts ${N_PARTS} \
        --num_diffusion_steps ${NUM_DIFFUSION_STEPS} \
        --tstart ${TSTART} \
        --cfg_src ${CFG_SRC} \
        --cfg_tar ${CFG_TAR} \
        --seed ${SEED} \
        --run_name ${RUN_NAME} \
        --with_hooks \
        > ../../outputs/medleymd/audioldm2/${RUN_NAME}_part${PART}.log 2>&1 &

    CUDA_DEVICE_START=$((CUDA_DEVICE_START + 1))
done

wait
