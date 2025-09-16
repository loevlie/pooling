# Hyperparameter Search Scripts

This directory contains scripts for comprehensive hyperparameter search of the MultiLayerTransformer implementation.

## Scripts Available

### 1. `hypersearch_quick.sh` (Recommended for testing)
- **Runtime**: ~30-60 minutes
- **Experiments**: ~18 experiments with 50 epochs each
- **Grid**: Reduced hyperparameter space for quick validation
- **Use case**: Initial testing and validation of hyperparameters

### 2. `hypersearch.sh` (Full search)
- **Runtime**: Several hours to overnight
- **Experiments**: 100+ experiments with up to 200 epochs each
- **Grid**: Comprehensive hyperparameter space
- **Use case**: Production hyperparameter optimization

## Usage

```bash
# Quick search (recommended first)
./hypersearch_quick.sh

# Full search (run after quick search validates setup)
./hypersearch.sh
```

## Hyperparameters Tested

### MultiLayerTransformer
- **Learning rates**: 0.001, 0.005, 0.01
- **Batch sizes**: 16, 32, 64
- **Local windows**: 3, 5, 7
- **Number of layers**: 2, 3, 4
- **Number of heads**: 1, 4, 8
- **Criteria**: ERM, EntropyRegularization
- **Entropy weights**: 0.01, 0.1, 0.5

### Baselines
- Standard attention pooling
- Transformer-based pooling
- Various learning rates and batch sizes

## Early Stopping

The scripts implement early stopping logic:
- Stops if no validation AUROC improvement for 20 epochs
- Minimum 30 epochs before early stopping
- Saves best model based on validation performance

## Output Structure

```
experiments/
├── quick_search_YYYYMMDD_HHMMSS/
│   ├── hypersearch.log              # Detailed execution log
│   ├── summary.csv                  # Summary of all experiments
│   ├── analysis.txt                 # Automated analysis
│   ├── exp_1_*.csv                  # Individual experiment results
│   ├── exp_1_*.pt                   # Saved model weights (best val)
│   └── ...
```

## Key Output Files

### `summary.csv`
Contains one row per experiment with:
- Model configuration (lr, batch_size, etc.)
- Best validation AUROC and corresponding test metrics
- Final epoch metrics
- Convergence status

### `hypersearch.log`
Detailed execution log with:
- Experiment configurations
- Training progress
- Error messages
- Final analysis

## Analysis Features

The scripts automatically generate:

1. **Top performing models** by validation and test AUROC
2. **Best model per pooling method**
3. **Hyperparameter analysis** for MultiLayerTransformer:
   - Performance by learning rate
   - Performance by local window size
   - Performance by architecture (layers/heads)
4. **Statistical summaries** (mean, std, count)

## Example Results Format

```
=== TOP 10 MODELS BY BEST VAL AUROC ===
model_name                              pooling              criterion  lr    batch_size  best_val_auroc  best_test_auroc
quick_5_erm_lr0.005_lw5_nl3_nh4        multilayer_transformer  ERM      0.005    32          0.6234         0.5987
baseline_attention                      attention              ERM      0.001    32          0.6123         0.5876
...
```

## Monitoring Progress

To monitor a running search:
```bash
# Check current progress
tail -f experiments/quick_search_*/hypersearch.log

# Check summary so far
cat experiments/quick_search_*/summary.csv
```

## Customization

You can modify the hyperparameter grids by editing the arrays in the scripts:

```bash
# In hypersearch_quick.sh or hypersearch.sh
learning_rates=(0.001 0.005 0.01)
batch_sizes=(16 32 64)
local_windows=(3 5 7)
# ... etc
```

## Hardware Requirements

- **Quick search**: Can run on CPU, ~2-4GB RAM
- **Full search**: GPU recommended, ~4-8GB RAM
- **Storage**: ~100MB for quick search, ~1GB for full search

## Troubleshooting

### Common Issues:
1. **Permission denied**: Run `chmod +x hypersearch*.sh`
2. **Module not found**: Ensure all dependencies are installed
3. **Out of memory**: Reduce batch_size in the grid
4. **Slow performance**: Use GPU if available

### Error Recovery:
The scripts are designed to continue even if individual experiments fail. Failed experiments are marked in the summary with `epochs_run=0`.

## Integration with toy_data.py

The scripts use the existing `toy_data.py` script with different parameters:
- No modifications needed to existing code
- All hyperparameters passed as command line arguments
- Results saved in standard CSV format
- Models saved with `--save` flag when validation performance peaks

## Next Steps After Search

1. **Analyze results** in `summary.csv` and `analysis.txt`
2. **Select best configurations** based on validation performance
3. **Run longer training** with best configs for final evaluation
4. **Compare against baselines** using the automated analysis
5. **Visualize attention patterns** using saved model weights

The scripts provide a comprehensive framework for systematically evaluating the MultiLayerTransformer implementation and comparing it against baseline methods.