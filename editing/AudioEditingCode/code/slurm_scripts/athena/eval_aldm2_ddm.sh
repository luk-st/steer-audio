#!/bin/bash

ACCOUNT=$GRANT_ACCOUNT
PARTITION=$GRANT_PARTITION
WORKDIR_PATH="/net/tscratch/people/plglukaszst/projects/audio-interv"
N_GPUS=1
N_CPUS=16
MEM=$((N_GPUS * 120))

# METHOD="ddpm" # "ddpm" "ddim"
# NUM_DIFFUSION_STEPS=100
# TSTARTS=(60 70 75 80)
# CFG_SRCS=(1.0 1.5 2.0 2.5)
# CFG_TARS=(4.0 5.0 6.0 7.0)
# HOOKS=("" "--with_hooks")

METHOD="ddim" # "ddpm" "ddim"
NUM_DIFFUSION_STEPS=100
TSTARTS=(40 45 50 55 55 60)
CFG_SRCS=(1.0 1.5 2.0 2.5 3.0)
CFG_TARS=(7.0 8.0 9.0 10.0)
HOOKS=("" "--with_hooks")

for tstart in "${TSTARTS[@]}"; do
    for cfg_src in "${CFG_SRCS[@]}"; do
        for cfg_tar in "${CFG_TARS[@]}"; do
            for hooks in "${HOOKS[@]}"; do
                AUDIO_PATH="/net/tscratch/people/plglukaszst/projects/audio-interv/editing/outputs/medleymd/aldm2_${METHOD}_hparam_search/ldm_${METHOD}${hooks}_cfgs${cfg_src}_cfgt${cfg_tar}_ts${tstart}_step${NUM_DIFFUSION_STEPS}/audios"
                log_file="slurm_out/medleymd/aldm2_${METHOD}_eval.log"
                job_name="eval_medley"
                sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash -l
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --ntasks=${N_GPUS}
#SBATCH --cpus-per-task=${N_CPUS}
#SBATCH --mem=${MEM}G
#SBATCH --time=00:15:00

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate music
export PYTHONPATH=$PWD

cd $WORKDIR_PATH
source .env
export PATH=$SCRATCH/envs/music/bin:$PATH

python editing/legacy/eval_medley.py --path_audio ${AUDIO_PATH}
EOT
            done
        done
    done
done