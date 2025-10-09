#!/bin/bash
#SBATCH --job-name=smmil_inst6
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --time=168:00:00
#SBATCH --partition=gpu,hugheslab
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/smmil_inst6_%A_%a.out
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/smmil_inst6_%A_%a.err

# --- env ---
source ~/.bashrc
conda activate jupyter-env
cd /cluster/tufts/hugheslab/dloevl01/pooling

# --- Specific job configurations ---
# Job 0: N=6309, lr=0.01, wd=0.001
# Jobs 1-5: N=10000 with specified LR/WD combinations

IDX=${SLURM_ARRAY_TASK_ID}

if [ ${IDX} -eq 0 ]; then
    NTRAIN=6309
    LR=0.01
    WD=0.001
elif [ ${IDX} -eq 1 ]; then
    NTRAIN=10000
    LR=0.01
    WD=0.1
elif [ ${IDX} -eq 2 ]; then
    NTRAIN=10000
    LR=0.01
    WD=0.01
elif [ ${IDX} -eq 3 ]; then
    NTRAIN=10000
    LR=0.01
    WD=0.001
elif [ ${IDX} -eq 4 ]; then
    NTRAIN=10000
    LR=0.1
    WD=0.01
elif [ ${IDX} -eq 5 ]; then
    NTRAIN=10000
    LR=0.001
    WD=0.01
fi

NVAL=$(( NTRAIN / 4 ))
NTEST=1000

# --- static knobs ---
BATCH=64
CRIT='L1'
DELTA=2
DELTAS=3
EPOCHS=1000
SEED=1001
POOLING='smmil'             # SmMIL pooling
EXP_ROOT="/cluster/tufts/hugheslab/dloevl01/pooling/experiments/smMIL_sweep_EarlySmoothing_instance_6jobs"

EXP_DIR="${EXP_ROOT}/Ntrain_${NTRAIN}"

# model_name contains key hparams for traceability - INSTANCE LEVEL
MODEL_NAME="criterion=${CRIT}_lr=${LR}_pooling=smMILattentionEarly_seed=${SEED}_wd_${WD}_N_${NTRAIN}_instance_level"

# --- create log directory if it doesn't exist ---
LOGDIR="/cluster/tufts/hugheslab/dloevl01/pooling/command_logs_instance_6jobs"
mkdir -p ${LOGDIR}
LOGFILE="${LOGDIR}/smmil_instance_commands.log"

# --- construct and log the command ---
CMD="python src/toy_data.py --batch_size=${BATCH} --criterion=\"${CRIT}\" --delta=${DELTA} --deltaS=${DELTAS} --epochs=${EPOCHS} --experiments_directory=\"${EXP_DIR}\" --lr=${LR} --model_name=\"${MODEL_NAME}\" --N_test=${NTEST} --N_train=${NTRAIN} --N_val=${NVAL} --pooling=\"${POOLING}\" --seed=${SEED} --weight_decay=${WD}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: ${CMD}" >> ${LOGFILE}


# --- run ---
set -x
python src/toy_data.py \
  --batch_size=${BATCH} \
  --criterion="${CRIT}" \
  --delta=${DELTA} \
  --deltaS=${DELTAS} \
  --epochs=${EPOCHS} \
  --experiments_directory="${EXP_DIR}" \
  --lr=${LR} \
  --model_name="${MODEL_NAME}" \
  --N_test=${NTEST} \
  --N_train=${NTRAIN} \
  --N_val=${NVAL} \
  --pooling="${POOLING}" \
  --seed=${SEED} \
  --weight_decay=${WD}
set +x
conda deactivate
