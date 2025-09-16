#!/bin/bash
#SBATCH --array=0-14%8
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/delta_exp_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --ntasks=2
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/delta_exp_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=6:00:00
#SBATCH --job-name=delta_experiment

source ~/.bashrc
conda activate jupyter-env

# Create experiment directory
EXPERIMENT_BASE="/cluster/tufts/hugheslab/dloevl01/pooling/experiments/delta_experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_BASE"

# Define experiment configurations
# Based on the quick hyperparameter search, using these best configs:
# - Attention: lr=0.001, batch_size=32
# - MT ERM: lr=0.001, batch_size=32, local_window=5, num_layers=2, num_heads=4
# - MT Entropy: lr=0.005, batch_size=32, local_window=5, num_layers=2, num_heads=4, alpha=0.1

# Array of experiments: "method delta pooling criterion lr batch_size local_window num_layers num_heads alpha"
experiments=(
    # Delta = 1
    "attention 1 attention ERM 0.001 32 3 2 4 0.0"
    "multilayer_erm 1 multilayer_transformer ERM 0.001 32 5 2 4 0.0"
    "multilayer_entropy 1 multilayer_transformer EntropyRegularization 0.005 32 5 2 4 0.1"

    # Delta = 2
    "attention 2 attention ERM 0.001 32 3 2 4 0.0"
    "multilayer_erm 2 multilayer_transformer ERM 0.001 32 5 2 4 0.0"
    "multilayer_entropy 2 multilayer_transformer EntropyRegularization 0.005 32 5 2 4 0.1"

    # Delta = 3
    "attention 3 attention ERM 0.001 32 3 2 4 0.0"
    "multilayer_erm 3 multilayer_transformer ERM 0.001 32 5 2 4 0.0"
    "multilayer_entropy 3 multilayer_transformer EntropyRegularization 0.005 32 5 2 4 0.1"

    # Delta = 4
    "attention 4 attention ERM 0.001 32 3 2 4 0.0"
    "multilayer_erm 4 multilayer_transformer ERM 0.001 32 5 2 4 0.0"
    "multilayer_entropy 4 multilayer_transformer EntropyRegularization 0.005 32 5 2 4 0.1"

    # Delta = 5
    "attention 5 attention ERM 0.001 32 3 2 4 0.0"
    "multilayer_erm 5 multilayer_transformer ERM 0.001 32 5 2 4 0.0"
    "multilayer_entropy 5 multilayer_transformer EntropyRegularization 0.005 32 5 2 4 0.1"
)

# Parse experiment parameters
IFS=' ' read -r method delta pooling criterion lr batch_size local_window num_layers num_heads alpha <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

model_name="${method}_delta${delta}_best"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running delta experiment: $model_name"
echo "Method: $method, Delta: $delta, DeltaS: 3"
echo "Config: pooling=$pooling, criterion=$criterion, lr=$lr"
echo "=========================================="

# Run the delta experiment
python ../src/toy_data.py \
    --pooling="$pooling" \
    --criterion="$criterion" \
    --lr="$lr" \
    --batch_size="$batch_size" \
    --alpha="$alpha" \
    --delta="$delta" \
    --deltaS=3 \
    --epochs=150 \
    --model_name="$model_name" \
    --experiments_directory="$EXPERIMENT_BASE" \
    --save

echo "Delta experiment completed for: $model_name (delta=$delta)"

# Create/append to summary file
SUMMARY_FILE="$EXPERIMENT_BASE/delta_results.csv"

# Create header if file doesn't exist
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "method,config_name,delta,deltaS,best_val_auroc,best_test_auroc,best_test_acc,epochs_run" > "$SUMMARY_FILE"
fi

# Parse results and append to summary
if [ -f "$EXPERIMENT_BASE/${model_name}.csv" ]; then
    python3 << EOF
import pandas as pd
import numpy as np

try:
    df = pd.read_csv("$EXPERIMENT_BASE/${model_name}.csv")

    # Find best validation performance
    best_val_idx = df['val_auroc'].idxmax()
    best_row = df.iloc[best_val_idx]

    best_val_auroc = best_row['val_auroc']
    best_test_auroc = best_row['test_auroc']
    best_test_acc = best_row['test_acc']
    epochs_run = len(df)

    print(f"Results: Delta {delta}, Val AUROC={best_val_auroc:.4f}, Test AUROC={best_test_auroc:.4f}")

    # Append to summary
    summary_line = f"$method,best,$delta,3,{best_val_auroc:.4f},{best_test_auroc:.4f},{best_test_acc:.4f},{epochs_run}"

    with open("$SUMMARY_FILE", "a") as f:
        f.write(summary_line + "\n")

except Exception as e:
    print(f"Error processing results: {e}")
    with open("$SUMMARY_FILE", "a") as f:
        f.write(f"$method,best,$delta,3,0,0,0,0\n")
EOF
fi

conda deactivate