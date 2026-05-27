#!/bin/bash

ACCOUNT=$GRANT_ACCOUNT
PARTITION=$GRANT_PARTITION
WORKDIR_PATH="/net/tscratch/people/plglukaszst/projects/audio-interv/editing/AudioEditingCode/code"
N_GPUS=1
N_CPUS=16
MEM=$((N_GPUS * 120))

N_PARTS=24
PARTS=(16)
METHOD="sdedit"
NUM_DIFFUSION_STEPS=200
TSTART=100
CFG_SRC=3.0
CFG_TAR=12.0
SEED=42

USE_HOOKS=""
RUN_NAME="ldm_${METHOD}_cfgs${CFG_SRC}_cfgt${CFG_TAR}_ts${TSTART}_step${NUM_DIFFUSION_STEPS}"

# USE_HOOKS="--with_hooks"
# RUN_NAME="ldm_${METHOD}_ours_cfgs${CFG_SRC}_cfgt${CFG_TAR}_ts${TSTART}_step${NUM_DIFFUSION_STEPS}"



main_process_port=12345
for part in "${PARTS[@]}"; do
    main_process_port=$((main_process_port + 1))
    log_file="slurm_out/medleymd/audioldm2/${METHOD}_steps${NUM_DIFFUSION_STEPS}_tstart${TSTART}_cfgt${CFG_TAR}_part${part}_nparts${N_PARTS}${USE_HOOKS}.log"
    job_name="ldm_${METHOD}_part${part}"
    sleep 1
    sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash -l
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --ntasks=${N_GPUS}
#SBATCH --cpus-per-task=${N_CPUS}
#SBATCH --mem=${MEM}G
#SBATCH --time=8:00:00

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate music_edit
export PYTHONPATH=$PWD

cd $WORKDIR_PATH
source .env
export PATH=$SCRATCH/envs/music_edit/bin:$PATH

echo "--------------------------------"
pwd;hostname;date
echo "--------------------------------"
echo "N_GPUS: $N_GPUS"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "--------------------------------"
nvidia-smi
accelerate env
echo "--------------------------------"
echo USER: $USER
which python
echo PYTHONPATH: $PYTHONPATH
echo "--------------------------------"

python edit_audioldm_medleydb.py --mode ${METHOD} --part_id ${part} --n_parts ${N_PARTS} --num_diffusion_steps ${NUM_DIFFUSION_STEPS} --tstart ${TSTART} --cfg_src ${CFG_SRC} --cfg_tar ${CFG_TAR} --seed ${SEED} --run_name ${RUN_NAME} ${USE_HOOKS}
EOT
done