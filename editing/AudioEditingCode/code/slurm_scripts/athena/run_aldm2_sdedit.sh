#!/bin/bash

ACCOUNT=$GRANT_ACCOUNT
PARTITION=$GRANT_PARTITION
WORKDIR_PATH="/net/tscratch/people/plglukaszst/projects/audio-interv/editing/AudioEditingCode/code"
N_GPUS=1
N_CPUS=16
MEM=$((N_GPUS * 120))

SEED=42
N_PARTS=2
PARTS=(0 1)
METHOD="sdedit"
NUM_DIFFUSION_STEPS=100
TSTARTS=(25 35 40 45)
CFGS=(10.0 11.0 12.0 13.0 14.0 15.0)
HOOKS=("" "--with_hooks")

for tstart in "${TSTARTS[@]}"; do
    for cfg in "${CFGS[@]}"; do
        for hooks in "${HOOKS[@]}"; do
            for part in "${PARTS[@]}"; do
                RUN_NAME="ldm_${METHOD}${hooks}_cfg${cfg}_ts${tstart}_step${NUM_DIFFUSION_STEPS}"
                log_file="slurm_out/medleymd/ldm/${METHOD}_hparam.log"
                job_name="ldm_${METHOD}_part${part}"
                sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash -l
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --ntasks=${N_GPUS}
#SBATCH --cpus-per-task=${N_CPUS}
#SBATCH --mem=${MEM}G
#SBATCH --time=00:45:00

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate music_edit
export PYTHONPATH=$PWD

cd $WORKDIR_PATH
source .env
export PATH=$SCRATCH/envs/music_edit/bin:$PATH

python edit_audioldm_medleydb.py --mode ${METHOD} --part_id ${part} --n_parts ${N_PARTS} --tstart ${tstart} --cfg_tar ${cfg} --num_diffusion_steps ${NUM_DIFFUSION_STEPS} --seed ${SEED} --run_name ${RUN_NAME} ${hooks}
EOT
            done
        done
    done
done