#!/bin/bash

# Quick hyperparameter search script for testing
# Smaller grid with fewer epochs for faster iteration

# Create results directory
RESULTS_DIR="./experiments/quick_search_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file for the search
LOG_FILE="$RESULTS_DIR/hypersearch.log"
SUMMARY_FILE="$RESULTS_DIR/summary.csv"

echo "Starting quick hyperparameter search at $(date)" | tee "$LOG_FILE"
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
    local max_epochs="${10:-50}"

    echo "Running experiment: $model_name" | tee -a "$LOG_FILE"
    echo "  pooling=$pooling, criterion=$criterion, lr=$lr, batch_size=$batch_size" | tee -a "$LOG_FILE"

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
        2>&1 | tee -a "$LOG_FILE"

    # Parse results
    if [ -f "$RESULTS_DIR/${model_name}.csv" ]; then
        python3 << EOF
import pandas as pd
import numpy as np

try:
    df = pd.read_csv("$RESULTS_DIR/${model_name}.csv")

    # Find best validation performance
    best_val_idx = df['val_auroc'].idxmax()
    best_row = df.iloc[best_val_idx]
    final_row = df.iloc[-1]

    best_val_auroc = best_row['val_auroc']
    best_test_auroc = best_row['test_auroc']
    best_test_acc = best_row['test_acc']
    final_test_auroc = final_row['test_auroc']
    final_test_acc = final_row['test_acc']
    epochs_run = len(df)

    print(f"Completed: {epochs_run} epochs, Best val AUROC: {best_val_auroc:.4f}")

    # Append to summary
    summary_line = f"$model_name,$pooling,$criterion,$lr,$batch_size,$local_window,$num_layers,$num_heads,$alpha,{epochs_run},{best_val_auroc:.4f},{best_test_auroc:.4f},{best_test_acc:.4f},{final_test_auroc:.4f},{final_test_acc:.4f},True"

    with open("$SUMMARY_FILE", "a") as f:
        f.write(summary_line + "\n")

except Exception as e:
    print(f"Error: {e}")
    with open("$SUMMARY_FILE", "a") as f:
        f.write(f"$model_name,$pooling,$criterion,$lr,$batch_size,$local_window,$num_layers,$num_heads,$alpha,0,0,0,0,0,0,False\n")
EOF
    fi
    echo "----------------------------------------" | tee -a "$LOG_FILE"
}

# Reduced hyperparameter grid for quick testing
learning_rates=(0.001 0.005)
batch_sizes=(32)
local_windows=(3 5)
num_layers_list=(2 3)
num_heads_list=(1 4)
alphas=(0.1)

echo "=== Quick Baseline Comparisons ===" | tee -a "$LOG_FILE"

# Baselines
run_experiment "quick_attention" "attention" "ERM" 0.001 32 3 2 4 0.0 50
run_experiment "quick_multilayer" "multilayer_transformer" "ERM" 0.001 32 5 2 4 0.0 50

echo "=== Quick MultiLayerTransformer Search ===" | tee -a "$LOG_FILE"

exp_count=0
for lr in "${learning_rates[@]}"; do
    for local_window in "${local_windows[@]}"; do
        for num_layers in "${num_layers_list[@]}"; do
            for num_heads in "${num_heads_list[@]}"; do
                # ERM experiment
                exp_count=$((exp_count + 1))
                model_name="quick_${exp_count}_erm_lr${lr}_lw${local_window}_nl${num_layers}_nh${num_heads}"
                run_experiment "$model_name" "multilayer_transformer" "ERM" "$lr" 32 "$local_window" "$num_layers" "$num_heads" 0.0 50

                # Entropy experiment
                exp_count=$((exp_count + 1))
                model_name="quick_${exp_count}_entropy_lr${lr}_lw${local_window}_nl${num_layers}_nh${num_heads}"
                run_experiment "$model_name" "multilayer_transformer" "EntropyRegularization" "$lr" 32 "$local_window" "$num_layers" "$num_heads" 0.1 50
            done
        done
    done
done

echo "=== Quick Search Complete ===" | tee -a "$LOG_FILE"

# Generate analysis
python3 << EOF | tee -a "$LOG_FILE"
import pandas as pd

print("\n=== QUICK SEARCH RESULTS ===")

try:
    df = pd.read_csv("$SUMMARY_FILE")
    df = df[df['epochs_run'] > 0]

    print(f"Experiments completed: {len(df)}")

    if len(df) > 0:
        print("\nTop 5 models by validation AUROC:")
        top_models = df.nlargest(5, 'best_val_auroc')[['model_name', 'pooling', 'criterion', 'lr', 'local_window', 'num_layers', 'num_heads', 'alpha', 'best_val_auroc', 'best_test_auroc']]
        print(top_models.to_string(index=False))

        print(f"\nBest validation AUROC: {df['best_val_auroc'].max():.4f}")
        print(f"Best test AUROC: {df['best_test_auroc'].max():.4f}")

        # Best MultiLayerTransformer config
        mt_df = df[df['pooling'] == 'multilayer_transformer']
        if len(mt_df) > 0:
            best_mt = mt_df.loc[mt_df['best_val_auroc'].idxmax()]
            print(f"\nBest MultiLayerTransformer:")
            print(f"  Config: lr={best_mt['lr']}, lw={best_mt['local_window']}, nl={best_mt['num_layers']}, nh={best_mt['num_heads']}")
            print(f"  Val AUROC: {best_mt['best_val_auroc']:.4f}, Test AUROC: {best_mt['best_test_auroc']:.4f}")

except Exception as e:
    print(f"Error in analysis: {e}")
EOF

echo "Quick search complete! Check $RESULTS_DIR for results."