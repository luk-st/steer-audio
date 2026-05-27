#!/bin/bash

CUDA_DEVICE=1
DATE="20250820_1"
EDITING_TYPE_ID=3
LAYERS_HOOK_OURS="transformer_blocks.11.attn2"
# LAYERS_HOOK_OURS="transformer_blocks.11.attn2,transformer_blocks.12.attn2"
METHOD="ddpm"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python edit_stableaudio_zome.py --editing_type_id "${EDITING_TYPE_ID}" --mode "${METHOD}" --num_diffusion_steps 100 --cfg_src 1.0 --cfg_tar 3.5 --tstart 50 --seed 42 --target_neg_prompt "" --run_name stableaudio_ours_${METHOD}_${DATE}_cfgsrc1.0_cfgtar3.5_tstart50_steps100__ca11 --layers_to_hook "${LAYERS_HOOK_OURS}"

wait

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python edit_stableaudio_zome.py --editing_type_id "${EDITING_TYPE_ID}" --mode "${METHOD}" --num_diffusion_steps 100 --cfg_src 1.0 --cfg_tar 3.5 --tstart 100 --seed 42 --target_neg_prompt "" --run_name stableaudio_ours_${METHOD}_${DATE}_cfgsrc1.0_cfgtar3.5_tstart100_steps100__ca11 --layers_to_hook "${LAYERS_HOOK_OURS}"

wait


CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python edit_stableaudio_zome.py --editing_type_id "${EDITING_TYPE_ID}" --mode "${METHOD}" --num_diffusion_steps 100 --cfg_src 1.0 --cfg_tar 3.5 --tstart 25 --seed 42 --target_neg_prompt "" --run_name stableaudio_ours_${METHOD}_${DATE}_cfgsrc1.0_cfgtar3.5_tstart25_steps100__ca11 --layers_to_hook "${LAYERS_HOOK_OURS}"

wait

METHOD="ddim"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python edit_stableaudio_zome.py --editing_type_id "${EDITING_TYPE_ID}" --mode "${METHOD}" --num_diffusion_steps 100 --cfg_src 1.0 --cfg_tar 3.5 --tstart 50 --seed 42 --target_neg_prompt "" --run_name stableaudio_ours_${METHOD}_${DATE}_cfgsrc1.0_cfgtar3.5_tstart50_steps100__ca11 --layers_to_hook "${LAYERS_HOOK_OURS}"

wait

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python edit_stableaudio_zome.py --editing_type_id "${EDITING_TYPE_ID}" --mode "${METHOD}" --num_diffusion_steps 100 --cfg_src 1.0 --cfg_tar 3.5 --tstart 100 --seed 42 --target_neg_prompt "" --run_name stableaudio_ours_${METHOD}_${DATE}_cfgsrc1.0_cfgtar3.5_tstart100_steps100__ca11 --layers_to_hook "${LAYERS_HOOK_OURS}"

wait

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python edit_stableaudio_zome.py --editing_type_id "${EDITING_TYPE_ID}" --mode "${METHOD}" --num_diffusion_steps 100 --cfg_src 1.0 --cfg_tar 3.5 --tstart 25 --seed 42 --target_neg_prompt "" --run_name stableaudio_ours_${METHOD}_${DATE}_cfgsrc1.0_cfgtar3.5_tstart25_steps100__ca11 --layers_to_hook "${LAYERS_HOOK_OURS}"

wait

METHOD="sdedit"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python edit_stableaudio_zome.py --editing_type_id "${EDITING_TYPE_ID}" --mode "${METHOD}" --num_diffusion_steps 100 --cfg_src 1.0 --cfg_tar 3.5 --tstart 50 --seed 42 --target_neg_prompt "" --run_name stableaudio_ours_${METHOD}_${DATE}_cfgsrc1.0_cfgtar3.5_tstart50_steps100__ca11 --layers_to_hook "${LAYERS_HOOK_OURS}"

wait

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python edit_stableaudio_zome.py --editing_type_id "${EDITING_TYPE_ID}" --mode "${METHOD}" --num_diffusion_steps 100 --cfg_src 1.0 --cfg_tar 3.5 --tstart 25 --seed 42 --target_neg_prompt "" --run_name stableaudio_ours_${METHOD}_${DATE}_cfgsrc1.0_cfgtar3.5_tstart25_steps100__ca11 --layers_to_hook "${LAYERS_HOOK_OURS}"

wait

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python edit_stableaudio_zome.py --editing_type_id "${EDITING_TYPE_ID}" --mode "${METHOD}" --num_diffusion_steps 100 --cfg_src 1.0 --cfg_tar 3.5 --tstart 10 --seed 42 --target_neg_prompt "" --run_name stableaudio_ours_${METHOD}_${DATE}_cfgsrc1.0_cfgtar3.5_tstart10_steps100__ca11 --layers_to_hook "${LAYERS_HOOK_OURS}"

wait