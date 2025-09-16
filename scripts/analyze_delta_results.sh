#!/bin/bash
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/delta_analysis_%j.err
#SBATCH --mem=4g
#SBATCH --ntasks=1
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/delta_analysis_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=1:00:00
#SBATCH --job-name=delta_analysis

source ~/.bashrc
conda activate jupyter-env

# Find the most recent delta experiment directory
LATEST_DIR=$(ls -td /cluster/tufts/hugheslab/dloevl01/pooling/experiments/delta_experiment_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No delta experiment results found!"
    exit 1
fi

echo "Analyzing results from: $LATEST_DIR"

# Install matplotlib if needed (should already be in jupyter-env)
python3 -c "import matplotlib" 2>/dev/null || {
    echo "Installing matplotlib..."
    pip install matplotlib --user
}

# Run the analysis
python3 ../plot_delta_results.py "$LATEST_DIR" --save-plot

echo "Analysis complete!"
echo "Results saved to: $LATEST_DIR"
echo "Plot: $LATEST_DIR/delta_performance_plot.png"
echo "Table: $LATEST_DIR/delta_results_table.txt"

conda deactivate