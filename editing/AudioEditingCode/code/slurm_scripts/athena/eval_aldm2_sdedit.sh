#!/bin/bash

ACCOUNT=$GRANT_ACCOUNT
PARTITION=$GRANT_PARTITION
WORKDIR_PATH="/net/tscratch/people/plglukaszst/projects/audio-interv"
N_GPUS=1
N_CPUS=16
MEM=$((N_GPUS * 120))

METHOD="sdedit"
NUM_DIFFUSION_STEPS=100
TSTARTS=(25 35 40 45)
CFGS=(10.0 11.0 12.0 13.0 14.0 15.0)
HOOKS=("" "--with_hooks")


for tstart in "${TSTARTS[@]}"; do
    for cfg in "${CFGS[@]}"; do
        for hooks in "${HOOKS[@]}"; do
            AUDIO_PATH="/net/tscratch/people/plglukaszst/projects/audio-interv/editing/outputs/medleymd/aldm2_${METHOD}_hparam_search/ldm_${METHOD}${hooks}_cfg${cfg}_ts${tstart}_step${NUM_DIFFUSION_STEPS}/audios"
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