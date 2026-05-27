#!/bin/bash

ACCOUNT=$GRANT_ACCOUNT
PARTITION=$GRANT_PARTITION
WORKDIR_PATH="/net/tscratch/people/plglukaszst/projects/audio-interv"
N_GPUS=1
N_CPUS=16
MEM=$((N_GPUS * 120))

SEED=42
N_PARTS=2
PARTS=(0 1)
METHOD="ode"
NUM_DIFFUSION_STEPS=50
TSTARTS=(20 23 25 28 30)
CFG_SRCS=(1.0 2.0)
CFG_TARS=(1.0 2.0 3.0)
CFG_TYPES=("cfg")
HOOKS=("" "--with_hooks")

for tstart in "${TSTARTS[@]}"; do
    for cfg_src in "${CFG_SRCS[@]}"; do
        for cfg_tar in "${CFG_TARS[@]}"; do
            for cfg_type in "${CFG_TYPES[@]}"; do
                for hooks in "${HOOKS[@]}"; do
                    for part in "${PARTS[@]}"; do
                        RUN_NAME="ace_ode${hooks}_cfgs${cfg_src}_cfgt${cfg_tar}_ts${tstart}_step${NUM_DIFFUSION_STEPS}_ctype${cfg_type}"
                        log_file="slurm_out/medleymd/ace/ode.log"
                        job_name="ace_${METHOD}_part${part}"
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

python editing/edit_audios_ode_medley.py --part_id ${part} --n_parts ${N_PARTS} --tstart ${tstart} --guidance_scale_inversion ${cfg_src} --guidance_scale_editing ${cfg_tar} --cfg_type ${cfg_type} --infer_step ${NUM_DIFFUSION_STEPS} --seed ${SEED} --run_name ${RUN_NAME} ${hooks}
EOT
                    done
                done
            done
        done
    done
done