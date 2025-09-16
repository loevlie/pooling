#!/bin/bash

# Two-stage experiment:
# 1. Find best hyperparameters for each method
# 2. Test these configs across different delta values (1-5) with deltaS=3

RESULTS_DIR="./experiments/delta_experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

LOG_FILE="$RESULTS_DIR/delta_experiment.log"
SUMMARY_FILE="$RESULTS_DIR/delta_results.csv"

echo "Starting delta experiment at $(date)" | tee "$LOG_FILE"
echo "Results will be saved to: $RESULTS_DIR" | tee -a "$LOG_FILE"

# Create summary CSV header
echo "method,config_name,delta,deltaS,best_val_auroc,best_test_auroc,best_test_acc,epochs_run" > "$SUMMARY_FILE"

# Function to run a single delta experiment
run_delta_experiment() {
    local method="$1"
    local config_name="$2"
    local pooling="$3"
    local criterion="$4"
    local lr="$5"
    local batch_size="$6"
    local local_window="$7"
    local num_layers="$8"
    local num_heads="$9"
    local alpha="${10}"
    local delta="${11}"
    local deltaS="${12}"

    local model_name="${method}_delta${delta}_${config_name}"

    echo "Running delta experiment: $model_name" | tee -a "$LOG_FILE"
    echo "  delta=$delta, deltaS=$deltaS" | tee -a "$LOG_FILE"

    # Run the experiment
    python src/toy_data.py \
        --pooling="$pooling" \
        --criterion="$criterion" \
        --lr="$lr" \
        --batch_size="$batch_size" \
        --alpha="$alpha" \
        --delta="$delta" \
        --deltaS="$deltaS" \
        --epochs=100 \
        --model_name="$model_name" \
        --experiments_directory="$RESULTS_DIR" \
        --save \
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

    best_val_auroc = best_row['val_auroc']
    best_test_auroc = best_row['test_auroc']
    best_test_acc = best_row['test_acc']
    epochs_run = len(df)

    print(f"Delta {delta}: Val AUROC={best_val_auroc:.4f}, Test AUROC={best_test_auroc:.4f}")

    # Append to summary
    summary_line = f"$method,$config_name,$delta,$deltaS,{best_val_auroc:.4f},{best_test_auroc:.4f},{best_test_acc:.4f},{epochs_run}"

    with open("$SUMMARY_FILE", "a") as f:
        f.write(summary_line + "\n")

except Exception as e:
    print(f"Error: {e}")
    with open("$SUMMARY_FILE", "a") as f:
        f.write(f"$method,$config_name,$delta,$deltaS,0,0,0,0\n")
EOF
    fi

    echo "----------------------------------------" | tee -a "$LOG_FILE"
}

# ===== STAGE 1: QUICK HYPERPARAMETER SEARCH =====
echo "=== STAGE 1: Finding Best Hyperparameters ===" | tee -a "$LOG_FILE"

# Quick search for best configs (reduced from your hypersearch_quick.sh)
SEARCH_DIR="$RESULTS_DIR/hypersearch"
mkdir -p "$SEARCH_DIR"
SEARCH_SUMMARY="$SEARCH_DIR/summary.csv"

echo "model_name,pooling,criterion,lr,batch_size,local_window,num_layers,num_heads,alpha,epochs_run,best_val_auroc,best_test_auroc,best_test_acc" > "$SEARCH_SUMMARY"

# Function for hyperparameter search
run_hypersearch() {
    local model_name="$1"
    local pooling="$2"
    local criterion="$3"
    local lr="$4"
    local batch_size="$5"
    local local_window="$6"
    local num_layers="$7"
    local num_heads="$8"
    local alpha="$9"

    echo "Hypersearch: $model_name" | tee -a "$LOG_FILE"

    python src/toy_data.py \
        --pooling="$pooling" \
        --criterion="$criterion" \
        --lr="$lr" \
        --batch_size="$batch_size" \
        --alpha="$alpha" \
        --delta=3.0 \
        --deltaS=3 \
        --epochs=50 \
        --model_name="$model_name" \
        --experiments_directory="$SEARCH_DIR" \
        2>&1 | tee -a "$LOG_FILE"

    if [ -f "$SEARCH_DIR/${model_name}.csv" ]; then
        python3 << EOF
import pandas as pd

try:
    df = pd.read_csv("$SEARCH_DIR/${model_name}.csv")
    best_val_idx = df['val_auroc'].idxmax()
    best_row = df.iloc[best_val_idx]

    best_val_auroc = best_row['val_auroc']
    best_test_auroc = best_row['test_auroc']
    best_test_acc = best_row['test_acc']
    epochs_run = len(df)

    print(f"Hypersearch result: {best_val_auroc:.4f} val AUROC")

    summary_line = f"$model_name,$pooling,$criterion,$lr,$batch_size,$local_window,$num_layers,$num_heads,$alpha,{epochs_run},{best_val_auroc:.4f},{best_test_auroc:.4f},{best_test_acc:.4f}"

    with open("$SEARCH_SUMMARY", "a") as f:
        f.write(summary_line + "\n")

except Exception as e:
    print(f"Error: {e}")
    with open("$SEARCH_SUMMARY", "a") as f:
        f.write(f"$model_name,$pooling,$criterion,$lr,$batch_size,$local_window,$num_layers,$num_heads,$alpha,0,0,0,0\n")
EOF
    fi
}

# Quick hyperparameter search for key methods
echo "Searching for best attention config..." | tee -a "$LOG_FILE"
run_hypersearch "search_attention_001" "attention" "ERM" 0.001 32 3 2 4 0.0
run_hypersearch "search_attention_005" "attention" "ERM" 0.005 32 3 2 4 0.0

echo "Searching for best multilayer transformer configs..." | tee -a "$LOG_FILE"
run_hypersearch "search_mt_erm_001_lw3" "multilayer_transformer" "ERM" 0.001 32 3 2 4 0.0
run_hypersearch "search_mt_erm_005_lw5" "multilayer_transformer" "ERM" 0.005 32 5 2 4 0.0
run_hypersearch "search_mt_entropy_001_lw3" "multilayer_transformer" "EntropyRegularization" 0.001 32 3 2 4 0.1
run_hypersearch "search_mt_entropy_005_lw5" "multilayer_transformer" "EntropyRegularization" 0.005 32 5 2 4 0.1

# Find best configurations
echo "=== Finding Best Configurations ===" | tee -a "$LOG_FILE"

python3 << EOF | tee -a "$LOG_FILE"
import pandas as pd

try:
    df = pd.read_csv("$SEARCH_SUMMARY")
    df = df[df['epochs_run'] > 0]  # Filter successful experiments

    if len(df) == 0:
        print("No successful hyperparameter search results!")
        exit(1)

    print(f"Successful hyperparameter experiments: {len(df)}")

    # Find best attention config
    attention_df = df[df['pooling'] == 'attention']
    if len(attention_df) > 0:
        best_attention = attention_df.loc[attention_df['best_val_auroc'].idxmax()]
        print(f"\nBest Attention config:")
        print(f"  Model: {best_attention['model_name']}")
        print(f"  LR: {best_attention['lr']}, Batch: {best_attention['batch_size']}")
        print(f"  Val AUROC: {best_attention['best_val_auroc']:.4f}")

        # Save best attention config
        with open("$RESULTS_DIR/best_attention_config.txt", "w") as f:
            f.write(f"{best_attention['lr']},{best_attention['batch_size']},{best_attention['criterion']},{best_attention['alpha']}")

    # Find best multilayer transformer configs
    mt_df = df[df['pooling'] == 'multilayer_transformer']
    if len(mt_df) > 0:
        # Best ERM config
        mt_erm = mt_df[mt_df['criterion'] == 'ERM']
        if len(mt_erm) > 0:
            best_mt_erm = mt_erm.loc[mt_erm['best_val_auroc'].idxmax()]
            print(f"\nBest MultiLayer ERM config:")
            print(f"  Model: {best_mt_erm['model_name']}")
            print(f"  LR: {best_mt_erm['lr']}, Local Window: {best_mt_erm['local_window']}")
            print(f"  Val AUROC: {best_mt_erm['best_val_auroc']:.4f}")

            with open("$RESULTS_DIR/best_mt_erm_config.txt", "w") as f:
                f.write(f"{best_mt_erm['lr']},{best_mt_erm['batch_size']},{best_mt_erm['local_window']},{best_mt_erm['num_layers']},{best_mt_erm['num_heads']},{best_mt_erm['alpha']}")

        # Best Entropy config
        mt_entropy = mt_df[mt_df['criterion'] == 'EntropyRegularization']
        if len(mt_entropy) > 0:
            best_mt_entropy = mt_entropy.loc[mt_entropy['best_val_auroc'].idxmax()]
            print(f"\nBest MultiLayer Entropy config:")
            print(f"  Model: {best_mt_entropy['model_name']}")
            print(f"  LR: {best_mt_entropy['lr']}, Local Window: {best_mt_entropy['local_window']}, Alpha: {best_mt_entropy['alpha']}")
            print(f"  Val AUROC: {best_mt_entropy['best_val_auroc']:.4f}")

            with open("$RESULTS_DIR/best_mt_entropy_config.txt", "w") as f:
                f.write(f"{best_mt_entropy['lr']},{best_mt_entropy['batch_size']},{best_mt_entropy['local_window']},{best_mt_entropy['num_layers']},{best_mt_entropy['num_heads']},{best_mt_entropy['alpha']}")

except Exception as e:
    print(f"Error analyzing hyperparameter results: {e}")
    # Fallback to default configs
    with open("$RESULTS_DIR/best_attention_config.txt", "w") as f:
        f.write("0.001,32,ERM,0.0")
    with open("$RESULTS_DIR/best_mt_erm_config.txt", "w") as f:
        f.write("0.001,32,5,2,4,0.0")
    with open("$RESULTS_DIR/best_mt_entropy_config.txt", "w") as f:
        f.write("0.001,32,5,2,4,0.1")
    print("Using fallback default configurations")
EOF

# ===== STAGE 2: DELTA EXPERIMENTS =====
echo "=== STAGE 2: Delta Experiments ===" | tee -a "$LOG_FILE"

# Read best configurations
if [ -f "$RESULTS_DIR/best_attention_config.txt" ]; then
    IFS=',' read -r att_lr att_batch att_criterion att_alpha < "$RESULTS_DIR/best_attention_config.txt"
    echo "Using attention config: lr=$att_lr, batch=$att_batch" | tee -a "$LOG_FILE"
else
    att_lr=0.001; att_batch=32; att_criterion="ERM"; att_alpha=0.0
    echo "Using default attention config" | tee -a "$LOG_FILE"
fi

if [ -f "$RESULTS_DIR/best_mt_erm_config.txt" ]; then
    IFS=',' read -r mt_erm_lr mt_erm_batch mt_erm_lw mt_erm_nl mt_erm_nh mt_erm_alpha < "$RESULTS_DIR/best_mt_erm_config.txt"
    echo "Using MT ERM config: lr=$mt_erm_lr, lw=$mt_erm_lw" | tee -a "$LOG_FILE"
else
    mt_erm_lr=0.001; mt_erm_batch=32; mt_erm_lw=5; mt_erm_nl=2; mt_erm_nh=4; mt_erm_alpha=0.0
    echo "Using default MT ERM config" | tee -a "$LOG_FILE"
fi

if [ -f "$RESULTS_DIR/best_mt_entropy_config.txt" ]; then
    IFS=',' read -r mt_ent_lr mt_ent_batch mt_ent_lw mt_ent_nl mt_ent_nh mt_ent_alpha < "$RESULTS_DIR/best_mt_entropy_config.txt"
    echo "Using MT Entropy config: lr=$mt_ent_lr, lw=$mt_ent_lw, alpha=$mt_ent_alpha" | tee -a "$LOG_FILE"
else
    mt_ent_lr=0.001; mt_ent_batch=32; mt_ent_lw=5; mt_ent_nl=2; mt_ent_nh=4; mt_ent_alpha=0.1
    echo "Using default MT Entropy config" | tee -a "$LOG_FILE"
fi

# Run delta experiments for each method
deltas=(1 2 3 4 5)
deltaS=3

for delta in "${deltas[@]}"; do
    echo "=== Testing delta = $delta ===" | tee -a "$LOG_FILE"

    # Attention baseline
    run_delta_experiment "attention" "best" "attention" "$att_criterion" "$att_lr" "$att_batch" 3 2 4 "$att_alpha" "$delta" "$deltaS"

    # MultiLayer Transformer ERM
    run_delta_experiment "multilayer_erm" "best" "multilayer_transformer" "ERM" "$mt_erm_lr" "$mt_erm_batch" "$mt_erm_lw" "$mt_erm_nl" "$mt_erm_nh" "$mt_erm_alpha" "$delta" "$deltaS"

    # MultiLayer Transformer with Entropy Regularization
    run_delta_experiment "multilayer_entropy" "best" "multilayer_transformer" "EntropyRegularization" "$mt_ent_lr" "$mt_ent_batch" "$mt_ent_lw" "$mt_ent_nl" "$mt_ent_nh" "$mt_ent_alpha" "$delta" "$deltaS"
done

echo "=== Delta Experiment Complete ===" | tee -a "$LOG_FILE"
echo "Results saved to: $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "Run: python3 plot_delta_results.py $RESULTS_DIR" | tee -a "$LOG_FILE"