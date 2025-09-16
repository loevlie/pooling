#!/bin/bash

# Hyperparameter search script for MultiLayerTransformer
# This script runs multiple experiments with different hyperparameters

# Create results directory
RESULTS_DIR="./experiments/hypersearch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file for the search
LOG_FILE="$RESULTS_DIR/hypersearch.log"
SUMMARY_FILE="$RESULTS_DIR/summary.csv"

echo "Starting hyperparameter search at $(date)" | tee "$LOG_FILE"
echo "Results will be saved to: $RESULTS_DIR" | tee -a "$LOG_FILE"

# Create summary CSV header
echo "model_name,pooling,criterion,lr,batch_size,local_window,num_layers,num_heads,alpha,epochs_run,best_val_auroc,best_test_auroc,best_test_acc,final_test_auroc,final_test_acc,converged" > "$SUMMARY_FILE"

# Function to run a single experiment with early stopping
run_experiment() {
    local model_name="$1"
    local pooling="$2"
    local criterion="$3"
    local lr="$4"
    local batch_size="$5"
    local local_window="$6"
    local num_layers="$7"
    local num_heads="$8"
    local alpha="$9"
    local max_epochs="${10:-200}"

    echo "Running experiment: $model_name" | tee -a "$LOG_FILE"
    echo "  pooling=$pooling, criterion=$criterion, lr=$lr, batch_size=$batch_size" | tee -a "$LOG_FILE"
    echo "  local_window=$local_window, num_layers=$num_layers, num_heads=$num_heads, alpha=$alpha" | tee -a "$LOG_FILE"

    # Run the experiment
    python src/toy_data.py \
        --pooling="$pooling" \
        --criterion="$criterion" \
        --lr="$lr" \
        --batch_size="$batch_size" \
        --alpha="$alpha" \
        --epochs="$max_epochs" \
        --model_name="$model_name" \
        --experiments_directory="$RESULTS_DIR" \
        --save \
        2>&1 | tee -a "$LOG_FILE"

    # Check if the experiment completed successfully
    if [ -f "$RESULTS_DIR/${model_name}.csv" ]; then
        # Parse results using Python
        python3 << EOF | tee -a "$LOG_FILE"
import pandas as pd
import numpy as np

try:
    df = pd.read_csv("$RESULTS_DIR/${model_name}.csv")

    # Early stopping logic: stop if no improvement for 20 epochs
    best_val_auroc = 0
    no_improvement_count = 0
    best_epoch = 0
    converged = False

    for idx, row in df.iterrows():
        if row['val_auroc'] > best_val_auroc:
            best_val_auroc = row['val_auroc']
            no_improvement_count = 0
            best_epoch = idx
        else:
            no_improvement_count += 1

        # Early stopping after 20 epochs without improvement
        if no_improvement_count >= 20 and idx >= 30:  # At least 30 epochs
            converged = True
            break

    # Get metrics at best validation AUROC epoch
    best_row = df.iloc[best_epoch]
    final_row = df.iloc[-1]

    best_val_auroc = best_row['val_auroc']
    best_test_auroc = best_row['test_auroc']
    best_test_acc = best_row['test_acc']
    final_test_auroc = final_row['test_auroc']
    final_test_acc = final_row['test_acc']
    epochs_run = len(df)

    print(f"Experiment completed: {epochs_run} epochs")
    print(f"Best val AUROC: {best_val_auroc:.4f} at epoch {best_epoch}")
    print(f"Best test AUROC: {best_test_auroc:.4f}")
    print(f"Final test AUROC: {final_test_auroc:.4f}")
    print(f"Converged: {converged}")

    # Append to summary
    summary_line = f"$model_name,$pooling,$criterion,$lr,$batch_size,$local_window,$num_layers,$num_heads,$alpha,{epochs_run},{best_val_auroc:.4f},{best_test_auroc:.4f},{best_test_acc:.4f},{final_test_auroc:.4f},{final_test_acc:.4f},{converged}"

    with open("$SUMMARY_FILE", "a") as f:
        f.write(summary_line + "\n")

except Exception as e:
    print(f"Error processing results: {e}")
    summary_line = f"$model_name,$pooling,$criterion,$lr,$batch_size,$local_window,$num_layers,$num_heads,$alpha,0,0,0,0,0,0,False"
    with open("$SUMMARY_FILE", "a") as f:
        f.write(summary_line + "\n")
EOF
    else
        echo "Experiment failed - no results file found" | tee -a "$LOG_FILE"
        echo "$model_name,$pooling,$criterion,$lr,$batch_size,$local_window,$num_layers,$num_heads,$alpha,0,0,0,0,0,0,False" >> "$SUMMARY_FILE"
    fi

    echo "----------------------------------------" | tee -a "$LOG_FILE"
}

# Hyperparameter grid
pooling_methods=("multilayer_transformer" "attention" "transformer")
criteria=("ERM" "EntropyRegularization")
learning_rates=(0.001 0.01 0.005)
batch_sizes=(16 32 64)
local_windows=(3 5 7)
num_layers_list=(2 3 4)
num_heads_list=(1 4 8)
alphas=(0.01 0.1 0.5)  # For entropy regularization

# Quick baseline runs first
echo "=== Running Baseline Comparisons ===" | tee -a "$LOG_FILE"

# Standard attention baseline
run_experiment "baseline_attention" "attention" "ERM" 0.001 32 3 2 4 0.0 100

# Standard transformer baseline
run_experiment "baseline_transformer" "transformer" "ERM" 0.001 32 3 2 8 0.0 100

# Quick MultiLayerTransformer baseline
run_experiment "baseline_multilayer" "multilayer_transformer" "ERM" 0.001 32 5 2 4 0.0 100

echo "=== Starting Hyperparameter Search ===" | tee -a "$LOG_FILE"

# Counter for experiments
exp_count=0

# MultiLayerTransformer hyperparameter search
for lr in "${learning_rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for local_window in "${local_windows[@]}"; do
            for num_layers in "${num_layers_list[@]}"; do
                for num_heads in "${num_heads_list[@]}"; do

                    # ERM experiments
                    exp_count=$((exp_count + 1))
                    model_name="exp_${exp_count}_multilayer_erm_lr${lr}_bs${batch_size}_lw${local_window}_nl${num_layers}_nh${num_heads}"
                    run_experiment "$model_name" "multilayer_transformer" "ERM" "$lr" "$batch_size" "$local_window" "$num_layers" "$num_heads" 0.0 150

                    # Entropy regularization experiments (subset)
                    if [ $((exp_count % 2)) -eq 0 ]; then  # Every other experiment
                        for alpha in "${alphas[@]}"; do
                            exp_count=$((exp_count + 1))
                            model_name="exp_${exp_count}_multilayer_entropy_lr${lr}_bs${batch_size}_lw${local_window}_nl${num_layers}_nh${num_heads}_a${alpha}"
                            run_experiment "$model_name" "multilayer_transformer" "EntropyRegularization" "$lr" "$batch_size" "$local_window" "$num_layers" "$num_heads" "$alpha" 150
                        done
                    fi
                done
            done
        done
    done
done

# Additional comparison experiments with best hyperparameters found so far
echo "=== Running Additional Comparison Experiments ===" | tee -a "$LOG_FILE"

# Best learning rates for attention pooling
for lr in 0.001 0.005; do
    for batch_size in 32 64; do
        exp_count=$((exp_count + 1))
        model_name="comp_${exp_count}_attention_lr${lr}_bs${batch_size}"
        run_experiment "$model_name" "attention" "ERM" "$lr" "$batch_size" 3 2 4 0.0 150

        exp_count=$((exp_count + 1))
        model_name="comp_${exp_count}_attention_entropy_lr${lr}_bs${batch_size}"
        run_experiment "$model_name" "attention" "EntropyRegularization" "$lr" "$batch_size" 3 2 4 0.1 150
    done
done

echo "=== Hyperparameter Search Complete ===" | tee -a "$LOG_FILE"
echo "Completed at $(date)" | tee -a "$LOG_FILE"
echo "Total experiments run: $exp_count" | tee -a "$LOG_FILE"

# Generate final analysis
python3 << EOF | tee -a "$LOG_FILE"
import pandas as pd
import numpy as np

print("\n=== FINAL ANALYSIS ===")

try:
    df = pd.read_csv("$SUMMARY_FILE")
    df = df[df['epochs_run'] > 0]  # Filter out failed experiments

    if len(df) == 0:
        print("No successful experiments found!")
        exit()

    print(f"\nSuccessful experiments: {len(df)}")
    print(f"Failed experiments: {df['epochs_run'].eq(0).sum()}")

    # Best performing models
    print("\n=== TOP 10 MODELS BY BEST VAL AUROC ===")
    top_models = df.nlargest(10, 'best_val_auroc')[['model_name', 'pooling', 'criterion', 'lr', 'batch_size', 'local_window', 'num_layers', 'num_heads', 'alpha', 'best_val_auroc', 'best_test_auroc']]
    print(top_models.to_string(index=False))

    print("\n=== TOP 10 MODELS BY BEST TEST AUROC ===")
    top_test_models = df.nlargest(10, 'best_test_auroc')[['model_name', 'pooling', 'criterion', 'lr', 'batch_size', 'local_window', 'num_layers', 'num_heads', 'alpha', 'best_val_auroc', 'best_test_auroc']]
    print(top_test_models.to_string(index=False))

    # Best by pooling method
    print("\n=== BEST MODEL PER POOLING METHOD ===")
    for pooling in df['pooling'].unique():
        subset = df[df['pooling'] == pooling]
        best = subset.loc[subset['best_val_auroc'].idxmax()]
        print(f"{pooling}: {best['model_name']} - Val AUROC: {best['best_val_auroc']:.4f}, Test AUROC: {best['best_test_auroc']:.4f}")

    # Hyperparameter analysis for MultiLayerTransformer
    mt_df = df[df['pooling'] == 'multilayer_transformer']
    if len(mt_df) > 0:
        print("\n=== MULTILAYER TRANSFORMER HYPERPARAMETER ANALYSIS ===")

        # Best hyperparameters
        best_mt = mt_df.loc[mt_df['best_val_auroc'].idxmax()]
        print(f"Best MultiLayerTransformer config:")
        print(f"  lr={best_mt['lr']}, batch_size={best_mt['batch_size']}, local_window={best_mt['local_window']}")
        print(f"  num_layers={best_mt['num_layers']}, num_heads={best_mt['num_heads']}, alpha={best_mt['alpha']}")
        print(f"  Val AUROC: {best_mt['best_val_auroc']:.4f}, Test AUROC: {best_mt['best_test_auroc']:.4f}")

        # Average performance by hyperparameters
        print("\nAverage performance by learning rate:")
        lr_perf = mt_df.groupby('lr')['best_val_auroc'].agg(['mean', 'std', 'count'])
        print(lr_perf)

        print("\nAverage performance by local window:")
        lw_perf = mt_df.groupby('local_window')['best_val_auroc'].agg(['mean', 'std', 'count'])
        print(lw_perf)

        print("\nAverage performance by num_layers:")
        nl_perf = mt_df.groupby('num_layers')['best_val_auroc'].agg(['mean', 'std', 'count'])
        print(nl_perf)

    # Save detailed analysis
    with open("$RESULTS_DIR/analysis.txt", "w") as f:
        f.write("=== HYPERPARAMETER SEARCH ANALYSIS ===\n\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Best overall Val AUROC: {df['best_val_auroc'].max():.4f}\n")
        f.write(f"Best overall Test AUROC: {df['best_test_auroc'].max():.4f}\n\n")

        f.write("Top 5 models:\n")
        f.write(df.nlargest(5, 'best_val_auroc')[['model_name', 'pooling', 'best_val_auroc', 'best_test_auroc']].to_string(index=False))

except Exception as e:
    print(f"Error in final analysis: {e}")
EOF

echo "Analysis complete. Check $RESULTS_DIR for detailed results." | tee -a "$LOG_FILE"