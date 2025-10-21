#!/usr/bin/env python
"""
HPC Determinism Test Script
Run this on HPC to compare with local results.
Collects system info and runs identical configuration.
"""

import torch
import numpy as np
import pandas as pd
import subprocess
import platform
import os
import sys
import json
from datetime import datetime

def collect_system_info():
    """Collect detailed system and GPU information."""
    info = {
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['cudnn_enabled'] = torch.backends.cudnn.enabled
        info['cudnn_benchmark'] = torch.backends.cudnn.benchmark
        info['cudnn_deterministic'] = torch.backends.cudnn.deterministic
        info['gpu_count'] = torch.cuda.device_count()

        # Get info for each GPU
        info['gpus'] = []
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'capability': '.'.join(map(str, torch.cuda.get_device_capability(i))),
                'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 1e9,
            }
            info['gpus'].append(gpu_info)

    return info

def run_determinism_test(output_dir='hpc_determinism_results'):
    """Run the same test configuration as local."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configuration - MUST match local test exactly
    CONFIG = {
        'N_train': 158,
        'N_val': 39,
        'N_test': 1000,
        'lr': 0.01,
        'alpha': 0.001,  # Used for L1 loss regularization
        'batch_size': 64,
        'epochs': 200,
        'seed': 1001,
        'pooling': 'smmil',
        'criterion': 'L1',
        'delta': 2,
        'deltaS': 3
    }

    # Save configuration
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)

    # Collect system info
    print("Collecting system information...")
    sys_info = collect_system_info()

    # Pretty print GPU info
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Hostname: {sys_info['hostname']}")
    print(f"PyTorch: {sys_info['torch_version']}")
    print(f"CUDA: {sys_info.get('cuda_version', 'N/A')}")
    print(f"cuDNN: {sys_info.get('cudnn_version', 'N/A')}")
    print(f"cuDNN deterministic: {sys_info.get('cudnn_deterministic', 'N/A')}")

    if sys_info.get('gpus'):
        print(f"\nGPUs detected: {len(sys_info['gpus'])}")
        for gpu in sys_info['gpus']:
            print(f"  GPU {gpu['index']}: {gpu['name']}")
            print(f"    Compute capability: {gpu['capability']}")
            print(f"    Memory: {gpu['total_memory_gb']:.2f} GB")
    print("="*60)

    # Save system info
    with open(f'{output_dir}/system_info.json', 'w') as f:
        json.dump(sys_info, f, indent=2)

    # Run 3 identical experiments
    print(f"\nRunning 3 identical experiments with seed {CONFIG['seed']}...")
    print(f"Configuration: N_train={CONFIG['N_train']}, epochs={CONFIG['epochs']}, lr={CONFIG['lr']}")

    for run_id in range(1, 4):
        print(f"\n--- Run {run_id}/3 ---")

        model_name = f"hpc_run_{run_id}_gpu_{sys_info.get('gpus', [{}])[0].get('name', 'unknown').replace(' ', '_')}"

        # Build command
        cmd = [
            'python', 'src/toy_data.py',
            f'--batch_size={CONFIG["batch_size"]}',
            f'--criterion={CONFIG["criterion"]}',
            f'--delta={CONFIG["delta"]}',
            f'--deltaS={CONFIG["deltaS"]}',
            f'--epochs={CONFIG["epochs"]}',
            f'--experiments_directory={output_dir}',
            f'--lr={CONFIG["lr"]}',
            f'--model_name={model_name}',
            f'--N_test={CONFIG["N_test"]}',
            f'--N_train={CONFIG["N_train"]}',
            f'--N_val={CONFIG["N_val"]}',
            f'--pooling={CONFIG["pooling"]}',
            f'--seed={CONFIG["seed"]}',
            f'--alpha={CONFIG["alpha"]}',
            '--embedding_level'
        ]

        print(f"Command: {' '.join(cmd)}")

        # Run experiment
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error in Run {run_id}:")
            print(result.stderr)
        else:
            print(f"Run {run_id} completed successfully")

    print("\n" + "="*60)
    print("ANALYZING RESULTS")
    print("="*60)

    # Analyze results
    csv_files = [f'{output_dir}/{f}' for f in os.listdir(output_dir) if f.endswith('.csv')]

    if len(csv_files) >= 2:
        dfs = [pd.read_csv(f) for f in sorted(csv_files)]

        # Compare first 2 runs
        print("\nComparing Run 1 vs Run 2:")

        # Find first divergence
        for epoch in range(min(len(dfs[0]), 100)):
            test_auroc_diff = abs(dfs[0].iloc[epoch]['test_auroc'] - dfs[1].iloc[epoch]['test_auroc'])
            if test_auroc_diff > 1e-8:
                print(f"  First divergence at epoch {epoch}")
                print(f"    Run 1 test_auroc: {dfs[0].iloc[epoch]['test_auroc']:.10f}")
                print(f"    Run 2 test_auroc: {dfs[1].iloc[epoch]['test_auroc']:.10f}")
                print(f"    Difference: {test_auroc_diff:.2e}")
                break
        else:
            print("  No divergence detected in first 100 epochs!")

        # Final epoch comparison
        final_epoch = len(dfs[0]) - 1
        print(f"\nFinal epoch ({final_epoch}) comparison:")

        metrics = ['train_loss', 'val_loss', 'test_loss', 'test_auroc']
        for metric in metrics:
            values = [df.iloc[final_epoch][metric] for df in dfs[:3] if final_epoch < len(df)]
            if len(values) >= 2:
                diff = max(values) - min(values)
                print(f"  {metric:12s}: Range={diff:.8f}, Values={values}")

    print("\n" + "="*60)
    print(f"Results saved to: {output_dir}/")
    print("Download the entire folder for comparison with local results")
    print("="*60)

if __name__ == "__main__":
    # Optional: Set deterministic mode (uncomment to test with determinism enabled)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    run_determinism_test()