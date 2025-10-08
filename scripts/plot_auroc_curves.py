#!/usr/bin/env python3
"""Plot training and validation AUROC curves from experiment CSVs."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plot_best_model_curves(exp_dir, output_file=None):
    """Plot train/val AUROC curves for the best model (highest val AUROC)."""
    exp_path = Path(exp_dir)
    csv_files = list(exp_path.glob("*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in {exp_dir}")
        return

    # Find best model
    best_val_auroc = -1
    best_csv = None
    best_params = None

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'val_auroc' not in df.columns:
                continue

            max_val_auroc = df['val_auroc'].max()
            if max_val_auroc > best_val_auroc:
                best_val_auroc = max_val_auroc
                best_csv = csv_file
                # Extract lr and wd from filename
                name = csv_file.stem
                parts = name.split('_')
                params = {}
                for part in parts:
                    if '=' in part:
                        key, val = part.split('=', 1)
                        params[key] = val
                best_params = params

        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue

    if best_csv is None:
        print("No valid models found")
        return

    # Read best model
    df = pd.read_csv(best_csv)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['epoch'], df['train_auroc'], label='Train AUROC', linewidth=2)
    ax.plot(df['epoch'], df['val_auroc'], label='Val AUROC', linewidth=2)

    # Mark best val epoch
    best_epoch_idx = df['val_auroc'].idxmax()
    best_epoch = df.loc[best_epoch_idx, 'epoch']
    best_val = df.loc[best_epoch_idx, 'val_auroc']
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Val (epoch {int(best_epoch)})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)

    title = f"Best Model: LR={best_params.get('lr', '?')}, WD={best_params.get('wd', '?')}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def plot_all_models_curves(exp_dir, output_file=None):
    """Plot train/val AUROC curves for all models."""
    exp_path = Path(exp_dir)
    csv_files = list(exp_path.glob("*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in {exp_dir}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    models_plotted = 0
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'val_auroc' not in df.columns or 'train_auroc' not in df.columns:
                continue

            # Extract lr and wd from filename
            name = csv_file.stem
            parts = name.split('_')
            params = {}
            for part in parts:
                if '=' in part:
                    key, val = part.split('=', 1)
                    params[key] = val

            label = f"LR={params.get('lr', '?')}, WD={params.get('wd', '?')}"

            ax1.plot(df['epoch'], df['train_auroc'], label=label, alpha=0.6, linewidth=1)
            ax2.plot(df['epoch'], df['val_auroc'], label=label, alpha=0.6, linewidth=1)
            models_plotted += 1

        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train AUROC', fontsize=12)
    ax1.set_title('Training AUROC - All Models', fontsize=14)
    ax1.grid(True, alpha=0.3)
    if models_plotted <= 10:
        ax1.legend(fontsize=8)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Val AUROC', fontsize=12)
    ax2.set_title('Validation AUROC - All Models', fontsize=14)
    ax2.grid(True, alpha=0.3)
    if models_plotted <= 10:
        ax2.legend(fontsize=8)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

    print(f"Plotted {models_plotted} models")


def main():
    parser = argparse.ArgumentParser(description="Plot AUROC curves from experiment CSVs")
    parser.add_argument("exp_dir", type=str, help="Experiment directory containing CSV files")
    parser.add_argument("--all", action="store_true", help="Plot all models instead of just best")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: show plot)")
    args = parser.parse_args()

    if args.all:
        plot_all_models_curves(args.exp_dir, args.output)
    else:
        plot_best_model_curves(args.exp_dir, args.output)


if __name__ == "__main__":
    main()
