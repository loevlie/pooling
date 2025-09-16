#!/bin/bash

# Script to analyze the latest hyperparameter search results

# Find the most recent results directory
LATEST_DIR=$(ls -td experiments/quick_search_* experiments/hypersearch_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No hyperparameter search results found!"
    echo "Make sure you've run ./hypersearch_quick.sh or ./hypersearch.sh first."
    exit 1
fi

echo "Analyzing results from: $LATEST_DIR"
echo "========================================"

# Run the analysis
python3 analyze_results.py "$LATEST_DIR" --save --top-n 5

echo ""
echo "Analysis complete!"
echo "Detailed results saved to: $LATEST_DIR/detailed_analysis.txt"