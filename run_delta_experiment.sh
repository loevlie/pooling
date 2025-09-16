#!/bin/bash

# Simple wrapper to run the complete delta experiment

echo "=========================================="
echo "Delta Experiment: Signal Strength Analysis"
echo "=========================================="
echo ""
echo "This experiment will:"
echo "1. Find best hyperparameters for each method"
echo "2. Test performance across delta values 1-5"
echo "3. Generate plots and analysis"
echo ""
echo "Expected runtime: ~2-3 hours"
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Experiment cancelled."
    exit 1
fi

echo "Starting delta experiment..."

# Run the experiment
./delta_experiment.sh

# Check if it completed successfully
LATEST_DIR=$(ls -td experiments/delta_experiment_* 2>/dev/null | head -1)

if [ -n "$LATEST_DIR" ] && [ -f "$LATEST_DIR/delta_results.csv" ]; then
    echo ""
    echo "Experiment completed successfully!"
    echo "Generating plots and analysis..."

    # Install matplotlib if needed
    python3 -c "import matplotlib" 2>/dev/null || {
        echo "Installing matplotlib..."
        pip install matplotlib --user
    }

    # Generate plots and analysis
    python3 plot_delta_results.py "$LATEST_DIR" --save-plot

    echo ""
    echo "==========================================="
    echo "EXPERIMENT COMPLETE!"
    echo "==========================================="
    echo "Results directory: $LATEST_DIR"
    echo "Plot: $LATEST_DIR/delta_performance_plot.png"
    echo "Table: $LATEST_DIR/delta_results_table.txt"
    echo ""
    echo "To view the plot:"
    echo "  python3 plot_delta_results.py $LATEST_DIR --show-plot"
    echo ""

else
    echo ""
    echo "Experiment may have failed or is still running."
    echo "Check the log files in experiments/delta_experiment_*/"
fi