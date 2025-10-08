#!/usr/bin/env python3
"""Find the best model (highest val AUROC) in an experiment directory."""
import argparse
import os
import pandas as pd
from pathlib import Path


def parse_model_name(filename):
    """Extract hyperparameters from model filename."""
    # Remove .csv extension
    name = filename.replace('.csv', '')

    # Parse key=value pairs
    parts = name.split('_')
    params = {}

    i = 0
    while i < len(parts):
        if '=' in parts[i]:
            key, value = parts[i].split('=', 1)
            # Handle multi-word values (e.g., pooling=smMILattentionEarly)
            # Check if next part doesn't contain '='
            full_value = value
            while i + 1 < len(parts) and '=' not in parts[i + 1]:
                i += 1
                full_value += '_' + parts[i]
            params[key] = full_value
        i += 1

    return params


def find_best_model(exp_dir):
    """Find the model with the best val AUROC."""
    exp_path = Path(exp_dir)

    if not exp_path.exists():
        print(f"Error: Directory {exp_dir} does not exist")
        return None

    csv_files = list(exp_path.glob("*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in {exp_dir}")
        return None

    print(f"Found {len(csv_files)} CSV files in {exp_dir}")
    print()

    best_val_auroc = -1
    best_model_info = None

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if 'val_auroc' not in df.columns:
                print(f"Warning: {csv_file.name} missing 'val_auroc' column, skipping")
                continue

            # Find best val_auroc in this file
            max_val_auroc_idx = df['val_auroc'].idxmax()
            max_val_auroc = df.loc[max_val_auroc_idx, 'val_auroc']

            if max_val_auroc > best_val_auroc:
                best_val_auroc = max_val_auroc
                best_epoch = df.loc[max_val_auroc_idx, 'epoch']
                test_auroc = df.loc[max_val_auroc_idx, 'test_auroc']

                # Parse hyperparameters from filename
                params = parse_model_name(csv_file.name)

                best_model_info = {
                    'filename': csv_file.name,
                    'epoch': int(best_epoch),
                    'val_auroc': max_val_auroc,
                    'test_auroc': test_auroc,
                    'params': params,
                    'csv_path': str(csv_file)
                }

        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue

    return best_model_info


def main():
    parser = argparse.ArgumentParser(description="Find best model in experiment directory")
    parser.add_argument("exp_dir", type=str, help="Experiment directory containing CSV files")
    parser.add_argument("--verbose", action="store_true", help="Show additional details")
    args = parser.parse_args()

    best = find_best_model(args.exp_dir)

    if best is None:
        print("No valid models found")
        return

    print("="*70)
    print("BEST MODEL")
    print("="*70)
    print(f"File: {best['filename']}")
    print(f"Epoch: {best['epoch']}")
    print(f"Val AUROC: {best['val_auroc']:.6f}")
    print(f"Test AUROC: {best['test_auroc']:.6f}")
    print()
    print("Hyperparameters:")
    for key, value in sorted(best['params'].items()):
        print(f"  {key:20s}: {value}")

    if args.verbose:
        print()
        print(f"Full path: {best['csv_path']}")


if __name__ == "__main__":
    main()
