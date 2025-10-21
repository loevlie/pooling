#!/usr/bin/env python
"""
Analyze results from GPU comparison SBATCH array job.
This script reads all results from different GPU runs and compares them.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from datetime import datetime

def load_all_results(base_dir='/cluster/tufts/hugheslab/dloevl01/pooling/experiments/gpu_comparison'):
    """Load all GPU comparison results and metadata."""

    results = {}

    # Load GPU info files
    gpu_info_files = glob.glob(f'{base_dir}/gpu_info_run_*.json')
    print(f"Found {len(gpu_info_files)} GPU info files")

    for info_file in sorted(gpu_info_files):
        with open(info_file) as f:
            info = json.load(f)
            run_id = info['run_id']
            results[f'run_{run_id}'] = {'info': info}

    # Load result CSVs
    result_dir = f'{base_dir}/results'
    csv_files = glob.glob(f'{result_dir}/gpu_compare_run*.csv')
    print(f"Found {len(csv_files)} result CSV files")

    for csv_file in sorted(csv_files):
        # Extract run ID from filename
        filename = Path(csv_file).name
        # Parse run ID from filename like "gpu_compare_run1_..."
        run_id = filename.split('_')[2].replace('run', '')

        df = pd.read_csv(csv_file)
        key = f'run_{run_id}'

        if key in results:
            results[key]['data'] = df
            results[key]['csv_file'] = filename

            # Extract GPU name from filename
            parts = filename.split('_')
            gpu_idx = parts.index('compare') + 2  # Skip "gpu_compare_runX_"
            gpu_name_end = parts.index('lr')
            gpu_name = '_'.join(parts[gpu_idx:gpu_name_end])
            results[key]['gpu_name'] = gpu_name
        else:
            results[key] = {'data': df, 'csv_file': filename}

    return results

def summarize_gpus(results):
    """Summarize which GPUs were used."""

    print("\n" + "="*80)
    print("GPU SUMMARY")
    print("="*80)

    gpu_types = {}

    for run_key, run_data in sorted(results.items()):
        if 'info' in run_data:
            info = run_data['info']
            gpu_name = info.get('gpu_0_name', 'Unknown')
            gpu_cap = info.get('gpu_0_capability', 'Unknown')

            if gpu_name not in gpu_types:
                gpu_types[gpu_name] = {
                    'capability': gpu_cap,
                    'runs': []
                }
            gpu_types[gpu_name]['runs'].append(run_key)

            print(f"{run_key}: {gpu_name} (Compute {gpu_cap})")

    print("\n" + "-"*80)
    print("GPU Distribution:")
    for gpu_name, gpu_info in gpu_types.items():
        print(f"  {gpu_name} (Compute {gpu_info['capability']}): {len(gpu_info['runs'])} runs")
        print(f"    Runs: {', '.join(gpu_info['runs'])}")

    return gpu_types

def check_determinism_per_gpu(results, gpu_types):
    """Check if runs on the same GPU type are deterministic."""

    print("\n" + "="*80)
    print("DETERMINISM CHECK (Same GPU Type)")
    print("="*80)

    for gpu_name, gpu_info in gpu_types.items():
        runs = gpu_info['runs']
        if len(runs) < 2:
            print(f"\n{gpu_name}: Only 1 run, cannot check determinism")
            continue

        print(f"\n{gpu_name}: Comparing {len(runs)} runs")

        # Get data for all runs on this GPU
        dfs = []
        for run_key in runs:
            if run_key in results and 'data' in results[run_key]:
                dfs.append(results[run_key]['data'])

        if len(dfs) < 2:
            print("  Insufficient data for comparison")
            continue

        # Compare first 100 epochs
        deterministic = True
        for epoch in range(min(100, len(dfs[0]))):
            values = [df.iloc[epoch]['test_auroc'] for df in dfs]
            if max(values) - min(values) > 1e-8:
                print(f"  ❌ Divergence at epoch {epoch}: range = {max(values) - min(values):.2e}")
                deterministic = False
                break

        if deterministic:
            print(f"  ✓ Deterministic for first 100 epochs")

        # Final epoch comparison
        final_epoch = min(len(df) for df in dfs) - 1
        final_values = [df.iloc[final_epoch]['test_auroc'] for df in dfs]
        print(f"  Final test AUROC: mean={np.mean(final_values):.6f}, std={np.std(final_values):.2e}")

def compare_across_gpus(results, gpu_types):
    """Compare results across different GPU types."""

    print("\n" + "="*80)
    print("CROSS-GPU COMPARISON")
    print("="*80)

    # Get one representative run from each GPU type
    representatives = {}
    for gpu_name, gpu_info in gpu_types.items():
        for run_key in gpu_info['runs']:
            if run_key in results and 'data' in results[run_key]:
                representatives[gpu_name] = results[run_key]['data']
                break

    if len(representatives) < 2:
        print("Insufficient GPU types for comparison")
        return

    gpu_names = list(representatives.keys())
    print(f"\nComparing: {' vs '.join(gpu_names)}")

    # Find first divergence
    df1 = representatives[gpu_names[0]]
    df2 = representatives[gpu_names[1]]

    for epoch in range(min(len(df1), len(df2), 100)):
        diff = abs(df1.iloc[epoch]['test_auroc'] - df2.iloc[epoch]['test_auroc'])
        if diff > 1e-8:
            print(f"\nFirst divergence at epoch {epoch}:")
            print(f"  {gpu_names[0]}: {df1.iloc[epoch]['test_auroc']:.10f}")
            print(f"  {gpu_names[1]}: {df2.iloc[epoch]['test_auroc']:.10f}")
            print(f"  Difference: {diff:.2e}")
            break
    else:
        print("\nNo divergence in first 100 epochs!")

    # Final epoch comparison
    final_epoch = min(len(df1), len(df2)) - 1
    print(f"\nFinal epoch ({final_epoch}) comparison:")

    metrics = ['train_loss', 'val_loss', 'test_loss', 'test_auroc', 'val_auroc']
    for metric in metrics:
        values = {gpu: df.iloc[final_epoch][metric] for gpu, df in representatives.items()}
        value_list = list(values.values())

        print(f"\n{metric}:")
        for gpu, val in values.items():
            print(f"  {gpu}: {val:.8f}")
        print(f"  Range: {max(value_list) - min(value_list):.8f}")
        print(f"  Relative diff: {(max(value_list) - min(value_list))/np.mean(value_list)*100:.4f}%")

def plot_comparison(results, gpu_types, output_file='gpu_comparison_results.png'):
    """Create comparison plots."""

    # Get representative runs
    representatives = {}
    for gpu_name, gpu_info in gpu_types.items():
        for run_key in gpu_info['runs']:
            if run_key in results and 'data' in results[run_key]:
                representatives[gpu_name] = results[run_key]['data']
                break

    if len(representatives) < 2:
        print("Insufficient data for plotting")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(representatives)))

    # Plot metrics
    metrics = [
        ('train_loss', 'Training Loss', axes[0, 0]),
        ('val_loss', 'Validation Loss', axes[0, 1]),
        ('test_auroc', 'Test AUROC', axes[0, 2]),
    ]

    for metric, title, ax in metrics:
        for (gpu_name, df), color in zip(representatives.items(), colors):
            ax.plot(df['epoch'], df[metric], label=gpu_name,
                   linewidth=2, alpha=0.8, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Plot differences from first GPU
    base_gpu = list(representatives.keys())[0]
    base_df = representatives[base_gpu]

    diff_metrics = [
        ('train_loss', f'Train Loss Diff from {base_gpu}', axes[1, 0]),
        ('val_loss', f'Val Loss Diff from {base_gpu}', axes[1, 1]),
        ('test_auroc', f'Test AUROC Diff from {base_gpu}', axes[1, 2]),
    ]

    for metric, title, ax in diff_metrics:
        for (gpu_name, df), color in zip(representatives.items(), colors):
            if gpu_name != base_gpu:
                min_len = min(len(df), len(base_df))
                diff = df[metric].values[:min_len] - base_df[metric].values[:min_len]
                ax.plot(range(min_len), diff, label=f'{gpu_name} - {base_gpu}',
                       linewidth=1.5, alpha=0.8, color=color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Difference')
        ax.set_title(title)
        if ax.get_lines():  # Only add legend if there are lines
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    plt.suptitle('GPU Comparison Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")

def save_summary_report(results, gpu_types, output_file='gpu_comparison_report.txt'):
    """Save a summary report."""

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GPU COMPARISON EXPERIMENT REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")

        # Configuration
        f.write("CONFIGURATION:\n")
        f.write("  Learning Rate: 0.01\n")
        f.write("  Weight Decay (alpha): 0.001\n")
        f.write("  N_train: 158\n")
        f.write("  Epochs: 200\n")
        f.write("  Seed: 1001\n")
        f.write("  Pooling: smmil\n\n")

        # GPU Summary
        f.write("GPU TYPES TESTED:\n")
        for gpu_name, gpu_info in gpu_types.items():
            f.write(f"  {gpu_name} (Compute {gpu_info['capability']}): {len(gpu_info['runs'])} runs\n")

        # Results summary
        f.write("\nFINAL RESULTS SUMMARY:\n")
        all_final_aurocs = []
        for run_key, run_data in sorted(results.items()):
            if 'data' in run_data:
                final_auroc = run_data['data'].iloc[-1]['test_auroc']
                gpu_name = run_data.get('gpu_name', 'Unknown')
                f.write(f"  {run_key} ({gpu_name}): {final_auroc:.6f}\n")
                all_final_aurocs.append(final_auroc)

        if all_final_aurocs:
            f.write(f"\nOverall Statistics:\n")
            f.write(f"  Mean: {np.mean(all_final_aurocs):.6f}\n")
            f.write(f"  Std:  {np.std(all_final_aurocs):.6f}\n")
            f.write(f"  Range: {max(all_final_aurocs) - min(all_final_aurocs):.6f}\n")

    print(f"\nReport saved to: {output_file}")

def main():
    """Main analysis function."""

    print("="*80)
    print("GPU COMPARISON ANALYSIS")
    print("="*80)

    # Load results
    results = load_all_results()

    if not results:
        print("No results found!")
        return

    # Analyze
    gpu_types = summarize_gpus(results)
    check_determinism_per_gpu(results, gpu_types)
    compare_across_gpus(results, gpu_types)

    # Generate outputs
    plot_comparison(results, gpu_types)
    save_summary_report(results, gpu_types)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()