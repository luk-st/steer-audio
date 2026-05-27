#!/bin/bash

ACCOUNT=$GRANT_ACCOUNT
PARTITION=$GRANT_PARTITION
WORKDIR_PATH="/net/tscratch/people/plglukaszst/projects/audio-interv"
N_GPUS=1
N_CPUS=16
MEM=$((N_GPUS * 120))

NUM_DIFFUSION_STEPS=50
TSTARTS=(0.0 0.1 0.2)
TENDS=(0.8 0.9 1.0)
CFGS=(5.0 6.0 7.0 8.0 9.0)
N_AVGS=(3)
CFG_TYPES=("cfg")
HOOKS=("" "--with_hooks")

for tstart in "${TSTARTS[@]}"; do
    for tend in "${TENDS[@]}"; do
        for cfg in "${CFGS[@]}"; do
            for n_avg in "${N_AVGS[@]}"; do
                for cfg_type in "${CFG_TYPES[@]}"; do
                    for hooks in "${HOOKS[@]}"; do
                        AUDIO_PATH="/net/tscratch/people/plglukaszst/projects/audio-interv/editing/outputs/medleymd/ace/ace_flowedit${hooks}_tstart${tstart}_tend${tend}_cfg${cfg}_navg${n_avg}_step${NUM_DIFFUSION_STEPS}_ctype${cfg_type}/audios"
                        log_file="slurm_out/medleymd/eval.log"
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
    done
done