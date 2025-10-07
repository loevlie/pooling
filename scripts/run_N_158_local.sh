#!/bin/bash
# Standalone script to run N=10^2.2 (≈158) training locally
# Sweeps over all LR and WD combinations from the original sbatch script

# Disable MPS due to incompatibility with spectral normalization
# The model uses spectral_norm which doesn't work well with MPS backend
export DISABLE_MPS=1

# --- grids ---
LRS=(0.1 0.01 0.001 0.0001)
WDS=(1.0 0.1 0.01 0.001 0.0001 1e-5 1e-6 0.0)
NTRAIN=158  # 10^2.2 ≈ 158.49, rounded to 158

NUM_LR=${#LRS[@]}          # 4
NUM_WD=${#WDS[@]}          # 8
TOTAL_RUNS=$((NUM_LR * NUM_WD))  # 32 runs

NVAL=$(( NTRAIN / 4 ))      # floor division: 39
NTEST=1000

# --- static hyperparameters ---
BATCH=64
CRIT='L1'
DELTA=2
DELTAS=3
EPOCHS=1000
SEED=1001
POOLING='smmil'
EXP_ROOT="./experiments/smMIL_sweep_EarlySmoothing"

EXP_DIR="${EXP_ROOT}/Ntrain_${NTRAIN}"

# --- create log directory ---
LOGDIR="./command_logs"
mkdir -p ${LOGDIR}
LOGFILE="${LOGDIR}/smmil_N158_local.log"

echo "Starting local sweep for N=${NTRAIN}"
echo "Total runs: ${TOTAL_RUNS}"
echo "Progress will be logged to: ${LOGFILE}"
echo ""

# --- run all combinations ---
run_idx=0
for i_lr in $(seq 0 $((NUM_LR - 1))); do
  for i_wd in $(seq 0 $((NUM_WD - 1))); do
    LR=${LRS[$i_lr]}
    WD=${WDS[$i_wd]}

    run_idx=$((run_idx + 1))

    MODEL_NAME="criterion=${CRIT}_lr=${LR}_pooling=smMILattentionEarly_seed=${SEED}_wd_${WD}_N_${NTRAIN}_embedding_level"

    echo "[$run_idx/$TOTAL_RUNS] Running: LR=${LR}, WD=${WD}"

    CMD="python src/toy_data.py --batch_size=${BATCH} --criterion=\"${CRIT}\" --delta=${DELTA} --deltaS=${DELTAS} --epochs=${EPOCHS} --experiments_directory=\"${EXP_DIR}\" --lr=${LR} --model_name=\"${MODEL_NAME}\" --N_test=${NTEST} --N_train=${NTRAIN} --N_val=${NVAL} --pooling=\"${POOLING}\" --seed=${SEED} --weight_decay=${WD} --embedding_level"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run ${run_idx}/${TOTAL_RUNS}: ${CMD}" >> ${LOGFILE}

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

    echo ""
  done
done

echo "All runs complete!"
echo "Results saved to: ${EXP_DIR}"
