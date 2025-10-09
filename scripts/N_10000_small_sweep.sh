#!/bin/bash
#SBATCH --job-name=smmil_N10k
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --time=168:00:00
#SBATCH --partition=gpu,hugheslab
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/smmil_N10k_%A_%a.out
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/smmil_N10k_%A_%a.err

# --- env ---
source ~/.bashrc
conda activate jupyter-env
cd /cluster/tufts/hugheslab/dloevl01/pooling

# --- grids ---
# 3 LRs × 2 WDs = 6 total combinations
LRS=(0.01 0.001 0.0001)
WDS=(1.0 0.1)
NTRAIN=10000

NUM_LR=${#LRS[@]}          # 3
NUM_WD=${#WDS[@]}          # 2

# --- index mapping (cartesian product) ---
IDX=${SLURM_ARRAY_TASK_ID}
i_lr=$(( IDX % NUM_LR ))
i_wd=$(( IDX / NUM_LR ))

LR=${LRS[$i_lr]}
WD=${WDS[$i_wd]}
NVAL=$(( NTRAIN / 4 ))      # floor: 2500
NTEST=1000

# --- static knobs ---
BATCH=64
CRIT='L1'
DELTA=2
DELTAS=3
EPOCHS=1000
SEED=1001
POOLING='smmil'             # SmMIL pooling
EXP_ROOT="/cluster/tufts/hugheslab/dloevl01/pooling/experiments/smMIL_sweep_EarlySmoothing_N10k_small"

EXP_DIR="${EXP_ROOT}/Ntrain_${NTRAIN}"

# model_name contains key hparams for traceability
MODEL_NAME="criterion=${CRIT}_lr=${LR}_pooling=smMILattentionEarly_seed=${SEED}_wd_${WD}_N_${NTRAIN}_embedding_level"

# --- create log directory if it doesn't exist ---
LOGDIR="/cluster/tufts/hugheslab/dloevl01/pooling/command_logs_N10k_small"
mkdir -p ${LOGDIR}
LOGFILE="${LOGDIR}/smmil_N10k_commands.log"

# --- construct and log the command ---
CMD="python src/toy_data.py --batch_size=${BATCH} --criterion=\"${CRIT}\" --delta=${DELTA} --deltaS=${DELTAS} --epochs=${EPOCHS} --experiments_directory=\"${EXP_DIR}\" --lr=${LR} --model_name=\"${MODEL_NAME}\" --N_test=${NTEST} --N_train=${NTRAIN} --N_val=${NVAL} --pooling=\"${POOLING}\" --seed=${SEED} --weight_decay=${WD} --embedding_level"

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
  --weight_decay=${WD} \
  --embedding_level
set +x
conda deactivate
