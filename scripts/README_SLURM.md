# SLURM Delta Experiment

SLURM array jobs for comprehensive delta experiment comparing attention vs masked attention with regularization.

## Quick Start

```bash
cd scripts/
./submit_delta_experiment.sh
```

## What This Does

**Two-stage experiment:**

1. **Stage 1**: Hyperparameter search (12 jobs)
   - Find best configs for attention, multilayer transformer (ERM), and multilayer transformer (entropy)
   - Each job tests different lr/window/layer configurations
   - ~4 hours per job, runs in parallel

2. **Stage 2**: Delta experiments (15 jobs)
   - Test best configs from Stage 1 across delta values 1-5
   - Fixed deltaS=3 for all experiments
   - ~6 hours per job, runs in parallel

3. **Stage 3**: Analysis (1 job)
   - Generate plots and statistical analysis
   - Create performance tables
   - ~1 hour

## Files

### SLURM Scripts
- `delta_hypersearch.sh` - Hyperparameter search array job
- `delta_experiment.sh` - Main delta experiments array job
- `analyze_delta_results.sh` - Analysis job
- `submit_delta_experiment.sh` - Submit all jobs with dependencies

### Output
Results saved to: `/cluster/tufts/hugheslab/dloevl01/pooling/experiments/delta_experiment_*/`

- `delta_results.csv` - Summary of all results
- `delta_performance_plot.png` - Performance vs delta plot
- `delta_results_table.txt` - Detailed analysis
- Individual `.csv` files for each experiment

### Logs
- Output: `/cluster/tufts/hugheslab/dloevl01/slurmlog/out/`
- Errors: `/cluster/tufts/hugheslab/dloevl01/slurmlog/err/`

## Monitoring

```bash
# Check job status
squeue -u dloevl01

# Check specific job
scontrol show job <job_id>

# Cancel jobs if needed
scancel <job_id>
```

## Expected Results

The experiment will generate:

1. **Performance plot**: Test AUROC vs delta (1-5) for each method
2. **Statistical analysis**: Best configs, trends, comparisons
3. **Data table**: Complete results in CSV format

**Key comparisons:**
- Attention (baseline) vs MultiLayer Transformer
- ERM vs Entropy Regularization
- Performance across different signal strengths (delta)

## Resource Requirements

- **GPU**: 1 per job
- **Memory**: 8GB per job
- **Time**: 4-6 hours per job
- **Total jobs**: 28 (12 + 15 + 1)

## Manual Submission

If you prefer to submit jobs individually:

```bash
# Stage 1: Hyperparameter search
sbatch delta_hypersearch.sh

# Stage 2: Wait for Stage 1 to complete, then:
sbatch delta_experiment.sh

# Stage 3: Wait for Stage 2 to complete, then:
sbatch analyze_delta_results.sh
```

## Configuration

The scripts use optimized configurations based on initial testing:

- **Attention**: lr=0.001, batch_size=32
- **MT ERM**: lr=0.001, local_window=5, num_layers=2, num_heads=4
- **MT Entropy**: lr=0.005, local_window=5, alpha=0.1

These can be modified in the `experiments` arrays within each script.

## Troubleshooting

### Common Issues:
1. **Permission denied**: Run `chmod +x scripts/*.sh`
2. **Module not found**: Check conda environment name
3. **Path not found**: Verify directory paths match your setup
4. **Job fails**: Check error logs in `/cluster/tufts/hugheslab/dloevl01/slurmlog/err/`

### Error Recovery:
If jobs fail, you can resubmit individual array indices:
```bash
sbatch --array=5 delta_experiment.sh  # Run only job 5
```