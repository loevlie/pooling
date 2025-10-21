#!/usr/bin/env python
"""
Compare results from different GPUs to identify source of differences.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_results(local_dir='determinism_quick_test', hpc_dir='hpc_determinism_results'):
    """Load results from both local and HPC runs."""

    results = {'local': {}, 'hpc': {}}

    # Load local results
    local_files = list(Path(local_dir).glob('*.csv'))
    if local_files:
        # Take first file as representative since all 3 are identical
        results['local']['data'] = pd.read_csv(local_files[0])
        results['local']['file'] = local_files[0].name
        print(f"Loaded local results: {local_files[0].name}")

    # Load HPC results
    hpc_files = list(Path(hpc_dir).glob('*.csv'))
    for f in hpc_files:
        df = pd.read_csv(f)
        results['hpc'][f.name] = df
        print(f"Loaded HPC results: {f.name}")

    # Load system info if available
    local_gpu_info = get_local_gpu_info()
    results['local']['system'] = local_gpu_info

    hpc_info_file = Path(hpc_dir) / 'system_info.json'
    if hpc_info_file.exists():
        with open(hpc_info_file) as f:
            results['hpc']['system'] = json.load(f)

    return results

def get_local_gpu_info():
    """Get local GPU information."""
    import torch
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_capability'] = '.'.join(map(str, torch.cuda.get_device_capability(0)))

    return info

def compare_systems(results):
    """Compare system configurations."""
    print("\n" + "="*70)
    print("SYSTEM COMPARISON")
    print("="*70)

    local_sys = results['local'].get('system', {})
    hpc_sys = results['hpc'].get('system', {})

    print(f"{'Property':<25} {'Local':<25} {'HPC':<25}")
    print("-"*75)

    # Compare key properties
    props = [
        ('PyTorch Version', 'torch_version'),
        ('CUDA Version', 'cuda_version'),
        ('cuDNN Version', 'cudnn_version'),
        ('GPU Name', 'gpu_name', lambda x: x.get('gpus', [{}])[0].get('name', 'N/A') if 'gpus' in x else x.get('gpu_name', 'N/A')),
        ('Compute Capability', 'gpu_capability', lambda x: x.get('gpus', [{}])[0].get('capability', 'N/A') if 'gpus' in x else x.get('gpu_capability', 'N/A')),
        ('cuDNN Deterministic', 'cudnn_deterministic'),
    ]

    for display_name, key, *custom_getter in props:
        if custom_getter:
            getter = custom_getter[0]
            local_val = getter(local_sys)
            hpc_val = getter(hpc_sys)
        else:
            local_val = local_sys.get(key, 'N/A')
            hpc_val = hpc_sys.get(key, 'N/A')

        # Highlight differences
        if str(local_val) != str(hpc_val):
            print(f"{display_name:<25} {str(local_val):<25} {str(hpc_val):<25} ⚠️")
        else:
            print(f"{display_name:<25} {str(local_val):<25} {str(hpc_val):<25}")

def find_divergence_point(df1, df2, metric='test_auroc', threshold=1e-8):
    """Find where two runs diverge."""
    for epoch in range(min(len(df1), len(df2))):
        diff = abs(df1.iloc[epoch][metric] - df2.iloc[epoch][metric])
        if diff > threshold:
            return epoch, diff
    return None, 0

def plot_comparison(results):
    """Create comparison plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Get data
    local_df = results['local'].get('data')
    hpc_dfs = [df for name, df in results['hpc'].items() if name != 'system']

    if local_df is None or len(hpc_dfs) == 0:
        print("Insufficient data for plotting")
        return

    # Take first HPC run for comparison
    hpc_df = list(results['hpc'].values())[0] if isinstance(list(results['hpc'].values())[0], pd.DataFrame) else None

    if hpc_df is None:
        print("No HPC data to plot")
        return

    # Plot metrics
    metrics = [
        ('train_loss', 'Training Loss', axes[0, 0]),
        ('val_loss', 'Validation Loss', axes[0, 1]),
        ('test_auroc', 'Test AUROC', axes[0, 2]),
    ]

    for metric, title, ax in metrics:
        ax.plot(local_df['epoch'], local_df[metric], label='Local (RTX 4090)', linewidth=2, alpha=0.8)
        ax.plot(hpc_df['epoch'], hpc_df[metric], label='HPC GPU', linewidth=2, alpha=0.8, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot differences
    diff_metrics = [
        ('train_loss', 'Train Loss Difference', axes[1, 0]),
        ('val_loss', 'Val Loss Difference', axes[1, 1]),
        ('test_auroc', 'Test AUROC Difference', axes[1, 2]),
    ]

    for metric, title, ax in diff_metrics:
        diff = hpc_df[metric].values[:len(local_df)] - local_df[metric].values[:len(hpc_df)]
        ax.plot(range(len(diff)), diff, color='red', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('HPC - Local')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        # Add statistics
        max_diff = np.max(np.abs(diff))
        ax.text(0.02, 0.98, f'Max |diff|: {max_diff:.2e}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('GPU Comparison: Local vs HPC', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gpu_comparison.png', dpi=150)
    plt.show()

def analyze_differences(results):
    """Detailed analysis of differences."""

    print("\n" + "="*70)
    print("DIVERGENCE ANALYSIS")
    print("="*70)

    local_df = results['local'].get('data')
    if local_df is None:
        print("No local data available")
        return

    # Compare with each HPC run
    for name, hpc_df in results['hpc'].items():
        if name == 'system' or not isinstance(hpc_df, pd.DataFrame):
            continue

        print(f"\nComparing local vs {name}:")

        # Find divergence points for different metrics
        metrics = ['train_loss', 'val_loss', 'test_loss', 'test_auroc']

        for metric in metrics:
            epoch, diff = find_divergence_point(local_df, hpc_df, metric)
            if epoch is not None:
                print(f"  {metric:12s}: First divergence at epoch {epoch:3d} (diff: {diff:.2e})")
                print(f"    Local: {local_df.iloc[epoch][metric]:.10f}")
                print(f"    HPC:   {hpc_df.iloc[epoch][metric]:.10f}")
            else:
                print(f"  {metric:12s}: No divergence detected")

        # Final epoch comparison
        final_epoch = min(len(local_df), len(hpc_df)) - 1
        print(f"\n  Final epoch ({final_epoch}) differences:")

        for metric in metrics:
            local_val = local_df.iloc[final_epoch][metric]
            hpc_val = hpc_df.iloc[final_epoch][metric]
            diff = hpc_val - local_val
            rel_diff = (diff / local_val * 100) if local_val != 0 else 0

            print(f"    {metric:12s}: {diff:+.8f} ({rel_diff:+.4f}%)")

def main():
    """Main analysis function."""

    print("="*70)
    print("GPU RESULTS COMPARISON")
    print("="*70)

    # Check if directories exist
    if not Path('determinism_quick_test').exists():
        print("ERROR: Local results not found. Run test_determinism_3runs_quick.py first.")
        return

    if not Path('hpc_determinism_results').exists():
        print("\nHPC results not found!")
        print("Please:")
        print("1. Copy hpc_determinism_test.py to your HPC")
        print("2. Run it: python hpc_determinism_test.py")
        print("3. Download the hpc_determinism_results/ folder")
        print("4. Place it in this directory")
        print("5. Run this script again")
        return

    # Load and analyze
    results = load_results()
    compare_systems(results)
    analyze_differences(results)
    plot_comparison(results)

    print("\n" + "="*70)
    print("Analysis complete. Check gpu_comparison.png for visual comparison.")
    print("="*70)

if __name__ == "__main__":
    main()