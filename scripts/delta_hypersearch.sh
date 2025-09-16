#!/bin/bash
#SBATCH --array=0-13%6
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/delta_hypersearch_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --ntasks=2
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/delta_hypersearch_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=4:00:00
#SBATCH --job-name=delta_hypersearch

source ~/.bashrc
conda activate jupyter-env

# Create base experiment directory
EXPERIMENT_BASE="/cluster/tufts/hugheslab/dloevl01/pooling/experiments/delta_experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_BASE/hypersearch"

# Define hyperparameter search experiments
# Format: "pooling criterion lr batch_size local_window num_layers num_heads alpha model_name"
experiments=(
    # Attention baselines
    "attention ERM 0.001 32 3 2 4 0.0 search_attention_lr001"
    "attention ERM 0.005 32 3 2 4 0.0 search_attention_lr005"
    "attention ERM 0.01 32 3 2 4 0.0 search_attention_lr01"

    # MultiLayer Transformer ERM configs
    "multilayer_transformer ERM 0.001 32 3 2 4 0.0 search_mt_erm_lr001_lw3"
    "multilayer_transformer ERM 0.001 32 5 2 4 0.0 search_mt_erm_lr001_lw5"
    "multilayer_transformer ERM 0.005 32 3 3 4 0.0 search_mt_erm_lr005_lw3_nl3"
    "multilayer_transformer ERM 0.005 32 5 2 1 0.0 search_mt_erm_lr005_lw5_nh1"

    # MultiLayer Transformer Entropy configs
    "multilayer_transformer EntropyRegularization 0.001 32 3 2 4 0.05 search_mt_entropy_lr001_lw3_a005"
    "multilayer_transformer EntropyRegularization 0.001 32 5 2 4 0.1 search_mt_entropy_lr001_lw5_a01"
    "multilayer_transformer EntropyRegularization 0.005 32 3 2 4 0.1 search_mt_entropy_lr005_lw3_a01"
    "multilayer_transformer EntropyRegularization 0.005 32 5 3 1 0.2 search_mt_entropy_lr005_lw5_nl3_nh1_a02"
    "multilayer_transformer EntropyRegularization 0.01 32 5 2 4 0.1 search_mt_entropy_lr01_lw5_a01"
)

# Parse experiment parameters
IFS=' ' read -r pooling criterion lr batch_size local_window num_layers num_heads alpha model_name <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running hyperparameter search: $model_name"
echo "Config: pooling=$pooling, criterion=$criterion, lr=$lr"
echo "=========================================="

# Run the hyperparameter search experiment
python ../src/toy_data.py \
    --pooling="$pooling" \
    --criterion="$criterion" \
    --lr="$lr" \
    --batch_size="$batch_size" \
    --alpha="$alpha" \
    --delta=3.0 \
    --deltaS=3 \
    --epochs=100 \
    --model_name="$model_name" \
    --experiments_directory="$EXPERIMENT_BASE/hypersearch" \
    --save

echo "Hyperparameter search completed for: $model_name"

conda deactivate