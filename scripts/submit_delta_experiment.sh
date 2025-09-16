#!/bin/bash

# Submit delta experiment SLURM jobs

echo "=========================================="
echo "Delta Experiment SLURM Submission"
echo "=========================================="
echo ""
echo "This will submit:"
echo "  - Hyperparameter search (12 jobs, ~4 hours each)"
echo "  - Delta experiments (15 jobs, ~6 hours each)"
echo "  - Analysis job (1 job, ~1 hour)"
echo ""
echo "Total expected runtime: ~6-8 hours"
echo ""

read -p "Submit jobs? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 1
fi

# Submit hyperparameter search first
echo "Submitting hyperparameter search jobs..."
HYPERSEARCH_JOB=$(sbatch delta_hypersearch.sh | awk '{print $4}')
echo "Hyperparameter search job ID: $HYPERSEARCH_JOB"

# Submit main delta experiments (depends on hypersearch completion)
echo "Submitting delta experiment jobs..."
DELTA_JOB=$(sbatch --dependency=afterok:$HYPERSEARCH_JOB delta_experiment.sh | awk '{print $4}')
echo "Delta experiment job ID: $DELTA_JOB"

# Submit analysis job (depends on delta experiments completion)
echo "Submitting analysis job..."
ANALYSIS_JOB=$(sbatch --dependency=afterok:$DELTA_JOB analyze_delta_results.sh | awk '{print $4}')
echo "Analysis job ID: $ANALYSIS_JOB"

echo ""
echo "=========================================="
echo "Jobs submitted successfully!"
echo "=========================================="
echo "Hyperparameter search: $HYPERSEARCH_JOB"
echo "Delta experiments: $DELTA_JOB"
echo "Analysis: $ANALYSIS_JOB"
echo ""
echo "Monitor with:"
echo "  squeue -u dloevl01"
echo ""
echo "Check results in:"
echo "  /cluster/tufts/hugheslab/dloevl01/pooling/experiments/delta_experiment_*/"
echo ""
echo "View logs in:"
echo "  /cluster/tufts/hugheslab/dloevl01/slurmlog/out/"
echo "  /cluster/tufts/hugheslab/dloevl01/slurmlog/err/"