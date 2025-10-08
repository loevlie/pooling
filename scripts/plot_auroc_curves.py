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

                i = 0
                while i < len(parts):
                    part = parts[i]
                    if '=' in part:
                        key, val = part.split('=', 1)
                        params[key] = val
                    elif part == 'wd' and i + 1 < len(parts):
                        params['wd'] = parts[i + 1]
                        i += 1
                    i += 1

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


def plot_all_models_curves(exp_dir, output_dir=None):
    """Plot train/val AUROC curves for all models individually."""
    exp_path = Path(exp_dir)
    csv_files = list(exp_path.glob("*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in {exp_dir}")
        return

    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving individual plots to: {output_path}")
    else:
        print("No output directory specified, will show plots interactively")
        return

    models_plotted = 0
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'val_auroc' not in df.columns or 'train_auroc' not in df.columns:
                continue

            # Extract lr and wd from filename
            # Format: criterion=L1_lr=0.01_pooling=smMILattentionEarly_seed=1001_wd_0.001_N_158_embedding_level
            name = csv_file.stem
            parts = name.split('_')
            params = {}

            i = 0
            while i < len(parts):
                part = parts[i]
                if '=' in part:
                    key, val = part.split('=', 1)
                    params[key] = val
                elif part == 'wd' and i + 1 < len(parts):
                    # wd is followed by the value
                    params['wd'] = parts[i + 1]
                    i += 1  # Skip next part since we consumed it
                i += 1

            lr = params.get('lr', 'unknown')
            wd = params.get('wd', 'unknown')

            # Create individual plot
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(df['epoch'], df['train_auroc'], label='Train AUROC', linewidth=2, color='#1f77b4')
            ax.plot(df['epoch'], df['val_auroc'], label='Val AUROC', linewidth=2, color='#ff7f0e')

            # Mark best val epoch
            best_epoch_idx = df['val_auroc'].idxmax()
            best_epoch = df.loc[best_epoch_idx, 'epoch']
            best_val = df.loc[best_epoch_idx, 'val_auroc']
            ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5,
                      label=f'Best Val (epoch {int(best_epoch)})')
            ax.scatter([best_epoch], [best_val], color='red', s=100, zorder=5)

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('AUROC', fontsize=12)
            ax.set_title(f'LR={lr}, WD={wd}', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save with sanitized filename
            safe_lr = str(lr).replace('.', 'p').replace('-', 'n')
            safe_wd = str(wd).replace('.', 'p').replace('-', 'n').replace('+', 'p')
            output_file = output_path / f"auroc_lr_{safe_lr}_wd_{safe_wd}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            models_plotted += 1

        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue

    print(f"Successfully plotted {models_plotted} models")


def main():
    parser = argparse.ArgumentParser(description="Plot AUROC curves from experiment CSVs")
    parser.add_argument("exp_dir", type=str, help="Experiment directory containing CSV files")
    parser.add_argument("--all", action="store_true", help="Plot all models individually (requires --output as directory)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path: directory for --all mode, file for single plot mode")
    args = parser.parse_args()

    if args.all:
        if not args.output:
            print("Error: --all mode requires --output directory to save individual plots")
            print("Usage: python plot_auroc_curves.py <exp_dir> --all --output <output_dir>")
            return
        plot_all_models_curves(args.exp_dir, args.output)
    else:
        plot_best_model_curves(args.exp_dir, args.output)


if __name__ == "__main__":
    main()
