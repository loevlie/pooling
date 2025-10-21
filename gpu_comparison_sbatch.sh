#!/bin/bash
#SBATCH --job-name=gpu_compare
#SBATCH --array=0-14%15
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --time=4:00:00
#SBATCH --partition=gpu,hugheslab
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/gpu_compare_%A_%a.out
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/gpu_compare_%A_%a.err

# GPU Comparison Test
# Runs the SAME configuration 15 times to test determinism across different GPUs
# The scheduler will assign different GPUs randomly, allowing us to compare results

# --- env ---
source ~/.bashrc
conda activate jupyter-env
cd /cluster/tufts/hugheslab/dloevl01/pooling

# --- FIXED configuration (same for all runs) ---
LR=0.01
WD=0.001  # This will be used as alpha for L1 loss
NTRAIN=158
NVAL=$(( NTRAIN / 4 ))  # 39
NTEST=1000

# Static parameters
BATCH=64
CRIT='L1'
DELTA=2
DELTAS=3
EPOCHS=200  # Using 200 for faster testing (change to 1000 if needed)
SEED=1001
POOLING='smmil'

# Array index for run ID
RUN_ID=$((SLURM_ARRAY_TASK_ID + 1))

# --- Get GPU info ---
GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1)
GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | sed 's/ /_/g')
GPU_CAP=$(echo $GPU_INFO | cut -d',' -f2 | sed 's/ //g')

echo "==============================================================================="
echo "GPU COMPARISON RUN ${RUN_ID}/15"
echo "==============================================================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "GPU: ${GPU_NAME}"
echo "Compute Capability: ${GPU_CAP}"
echo "Configuration: LR=${LR}, WD=${WD}, N_train=${NTRAIN}, epochs=${EPOCHS}, seed=${SEED}"
echo "==============================================================================="

# --- Python script to capture detailed GPU info ---
python -c "
import torch
import json
import os

info = {
    'run_id': ${RUN_ID},
    'slurm_job_id': '${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}',
    'hostname': '$(hostname)',
    'torch_version': torch.__version__,
    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
}

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        info[f'gpu_{i}_name'] = torch.cuda.get_device_name(i)
        info[f'gpu_{i}_capability'] = '.'.join(map(str, torch.cuda.get_device_capability(i)))

# Save to file
output_dir = '/cluster/tufts/hugheslab/dloevl01/pooling/experiments/gpu_comparison'
os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/gpu_info_run_{${RUN_ID}}.json', 'w') as f:
    json.dump(info, f, indent=2)

print('GPU Info saved')
print(json.dumps(info, indent=2))
"

# --- Set output directory ---
EXP_ROOT="/cluster/tufts/hugheslab/dloevl01/pooling/experiments/gpu_comparison"
EXP_DIR="${EXP_ROOT}/results"

# Model name includes GPU info for traceability
MODEL_NAME="gpu_compare_run${RUN_ID}_${GPU_NAME}_lr=${LR}_wd=${WD}_seed=${SEED}_N=${NTRAIN}"

# --- Create log directory ---
LOGDIR="/cluster/tufts/hugheslab/dloevl01/pooling/command_logs_gpu_compare"
mkdir -p ${LOGDIR}
LOGFILE="${LOGDIR}/gpu_comparison_commands.log"

# --- Log the command ---
CMD="python src/toy_data.py --batch_size=${BATCH} --criterion=\"${CRIT}\" --delta=${DELTA} --deltaS=${DELTAS} --epochs=${EPOCHS} --experiments_directory=\"${EXP_DIR}\" --lr=${LR} --model_name=\"${MODEL_NAME}\" --N_test=${NTEST} --N_train=${NTRAIN} --N_val=${NVAL} --pooling=\"${POOLING}\" --seed=${SEED} --alpha=${WD} --embedding_level"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run ${RUN_ID} on ${GPU_NAME}: ${CMD}" >> ${LOGFILE}

# --- Run the experiment ---
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
  --alpha=${WD} \
  --embedding_level
set +x

echo "==============================================================================="
echo "Run ${RUN_ID} completed on ${GPU_NAME}"
echo "==============================================================================="

conda deactivate