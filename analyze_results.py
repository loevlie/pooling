#!/usr/bin/env python3
"""
Comprehensive analysis script for hyperparameter search results.
Analyzes summary.csv files and generates detailed performance reports.
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path

def load_results(results_dir):
    """Load results from summary.csv file."""
    summary_file = os.path.join(results_dir, "summary.csv")
    if not os.path.exists(summary_file):
        print(f"Error: {summary_file} not found!")
        return None

    df = pd.read_csv(summary_file)
    # Filter out failed experiments
    df = df[df['epochs_run'] > 0]

    if len(df) == 0:
        print("No successful experiments found!")
        return None

    return df

def print_section(title, char="=", width=80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_subsection(title, char="-", width=60):
    """Print a formatted subsection header."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")

def analyze_top_models(df, n=10):
    """Analyze top performing models."""
    print_section("TOP PERFORMING MODELS")

    # Top by validation AUROC
    print_subsection("Top Models by Validation AUROC")
    top_val = df.nlargest(n, 'best_val_auroc')
    display_cols = ['model_name', 'pooling', 'criterion', 'lr', 'batch_size',
                   'local_window', 'num_layers', 'num_heads', 'alpha',
                   'best_val_auroc', 'best_test_auroc', 'best_test_acc']
    print(top_val[display_cols].to_string(index=False, float_format='%.4f'))

    # Top by test AUROC
    print_subsection("Top Models by Test AUROC")
    top_test = df.nlargest(n, 'best_test_auroc')
    print(top_test[display_cols].to_string(index=False, float_format='%.4f'))

    return top_val.iloc[0], top_test.iloc[0]

def analyze_by_pooling_method(df):
    """Analyze best models by pooling method."""
    print_section("BEST MODEL PER POOLING METHOD")

    results = {}
    for pooling in sorted(df['pooling'].unique()):
        subset = df[df['pooling'] == pooling]
        best = subset.loc[subset['best_val_auroc'].idxmax()]
        results[pooling] = best

        print(f"\n{pooling.upper()}:")
        print(f"  Model: {best['model_name']}")
        print(f"  Criterion: {best['criterion']}")
        print(f"  Config: lr={best['lr']}, batch_size={best['batch_size']}")
        if pooling == 'multilayer_transformer':
            print(f"  Architecture: local_window={best['local_window']}, "
                  f"num_layers={best['num_layers']}, num_heads={best['num_heads']}")
            if best['criterion'] == 'EntropyRegularization':
                print(f"  Entropy weight: {best['alpha']}")
        print(f"  Performance: Val AUROC={best['best_val_auroc']:.4f}, "
              f"Test AUROC={best['best_test_auroc']:.4f}, "
              f"Test Acc={best['best_test_acc']:.4f}")

    return results

def analyze_hyperparameters(df, pooling_method='multilayer_transformer'):
    """Analyze hyperparameter effects for a specific pooling method."""
    subset = df[df['pooling'] == pooling_method]

    if len(subset) == 0:
        print(f"No experiments found for {pooling_method}")
        return

    print_section(f"HYPERPARAMETER ANALYSIS: {pooling_method.upper()}")

    # Best configuration
    best = subset.loc[subset['best_val_auroc'].idxmax()]
    print_subsection("Best Configuration")
    print(f"Model: {best['model_name']}")
    print(f"Learning rate: {best['lr']}")
    print(f"Batch size: {best['batch_size']}")
    print(f"Local window: {best['local_window']}")
    print(f"Number of layers: {best['num_layers']}")
    print(f"Number of heads: {best['num_heads']}")
    print(f"Criterion: {best['criterion']}")
    print(f"Alpha: {best['alpha']}")
    print(f"Performance: Val AUROC={best['best_val_auroc']:.4f}, "
          f"Test AUROC={best['best_test_auroc']:.4f}")

    # Hyperparameter effects
    hyperparams = ['lr', 'batch_size', 'local_window', 'num_layers', 'num_heads', 'criterion']

    for param in hyperparams:
        if param in subset.columns and subset[param].nunique() > 1:
            print_subsection(f"Effect of {param}")
            grouped = subset.groupby(param)['best_val_auroc'].agg(['mean', 'std', 'count', 'max'])
            grouped = grouped.sort_values('mean', ascending=False)
            print(grouped.to_string(float_format='%.4f'))

    # Entropy regularization analysis
    entropy_subset = subset[subset['criterion'] == 'EntropyRegularization']
    if len(entropy_subset) > 0:
        print_subsection("Entropy Regularization Effect")
        grouped = entropy_subset.groupby('alpha')['best_val_auroc'].agg(['mean', 'std', 'count', 'max'])
        grouped = grouped.sort_values('mean', ascending=False)
        print(grouped.to_string(float_format='%.4f'))

def compare_methods(df):
    """Compare different pooling methods statistically."""
    print_section("STATISTICAL COMPARISON")

    print_subsection("Performance by Pooling Method")
    pooling_stats = df.groupby('pooling')['best_val_auroc'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    pooling_stats = pooling_stats.sort_values('mean', ascending=False)
    print(pooling_stats.to_string())

    print_subsection("Performance by Criterion")
    criterion_stats = df.groupby('criterion')['best_val_auroc'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    criterion_stats = criterion_stats.sort_values('mean', ascending=False)
    print(criterion_stats.to_string())

    # MultiLayerTransformer vs baselines
    mt_df = df[df['pooling'] == 'multilayer_transformer']
    baseline_df = df[df['pooling'].isin(['attention', 'transformer'])]

    if len(mt_df) > 0 and len(baseline_df) > 0:
        print_subsection("MultiLayerTransformer vs Baselines")
        print(f"MultiLayerTransformer:")
        print(f"  Count: {len(mt_df)}")
        print(f"  Mean Val AUROC: {mt_df['best_val_auroc'].mean():.4f} ± {mt_df['best_val_auroc'].std():.4f}")
        print(f"  Best Val AUROC: {mt_df['best_val_auroc'].max():.4f}")
        print(f"  Mean Test AUROC: {mt_df['best_test_auroc'].mean():.4f} ± {mt_df['best_test_auroc'].std():.4f}")

        print(f"\nBaselines (attention + transformer):")
        print(f"  Count: {len(baseline_df)}")
        print(f"  Mean Val AUROC: {baseline_df['best_val_auroc'].mean():.4f} ± {baseline_df['best_val_auroc'].std():.4f}")
        print(f"  Best Val AUROC: {baseline_df['best_val_auroc'].max():.4f}")
        print(f"  Mean Test AUROC: {baseline_df['best_test_auroc'].mean():.4f} ± {baseline_df['best_test_auroc'].std():.4f}")

def generate_recommendations(df, pooling_results):
    """Generate recommendations based on results."""
    print_section("RECOMMENDATIONS")

    # Overall best model
    best_model = df.loc[df['best_val_auroc'].idxmax()]
    print_subsection("Best Overall Configuration")
    print(f"Use {best_model['pooling']} with {best_model['criterion']}")
    print(f"Hyperparameters:")
    print(f"  - Learning rate: {best_model['lr']}")
    print(f"  - Batch size: {best_model['batch_size']}")
    if best_model['pooling'] == 'multilayer_transformer':
        print(f"  - Local window: {best_model['local_window']}")
        print(f"  - Number of layers: {best_model['num_layers']}")
        print(f"  - Number of heads: {best_model['num_heads']}")
        if best_model['criterion'] == 'EntropyRegularization':
            print(f"  - Entropy weight: {best_model['alpha']}")
    print(f"Expected performance: {best_model['best_val_auroc']:.4f} Val AUROC")

    # Method-specific recommendations
    print_subsection("Method-Specific Recommendations")

    for pooling, result in pooling_results.items():
        print(f"\nFor {pooling}:")
        print(f"  Best config achieves {result['best_val_auroc']:.4f} Val AUROC")
        if pooling == 'multilayer_transformer':
            # Analyze what works best for MultiLayerTransformer
            mt_df = df[df['pooling'] == pooling]

            # Best hyperparameters
            best_lr = mt_df.groupby('lr')['best_val_auroc'].mean().idxmax()
            best_lw = mt_df.groupby('local_window')['best_val_auroc'].mean().idxmax()
            best_nl = mt_df.groupby('num_layers')['best_val_auroc'].mean().idxmax()
            best_nh = mt_df.groupby('num_heads')['best_val_auroc'].mean().idxmax()

            print(f"  Recommended: lr={best_lr}, local_window={best_lw}, "
                  f"num_layers={best_nl}, num_heads={best_nh}")

            # ERM vs Entropy
            erm_performance = mt_df[mt_df['criterion'] == 'ERM']['best_val_auroc'].mean()
            entropy_performance = mt_df[mt_df['criterion'] == 'EntropyRegularization']['best_val_auroc'].mean()

            if not pd.isna(erm_performance) and not pd.isna(entropy_performance):
                if entropy_performance > erm_performance:
                    print(f"  Entropy regularization improves performance by {entropy_performance - erm_performance:.4f}")
                else:
                    print(f"  ERM performs better than entropy regularization")

def save_analysis(df, results_dir, pooling_results, best_val, best_test):
    """Save detailed analysis to file."""
    analysis_file = os.path.join(results_dir, "detailed_analysis.txt")

    with open(analysis_file, 'w') as f:
        f.write("HYPERPARAMETER SEARCH DETAILED ANALYSIS\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Best validation AUROC: {df['best_val_auroc'].max():.4f}\n")
        f.write(f"Best test AUROC: {df['best_test_auroc'].max():.4f}\n")
        f.write(f"Mean validation AUROC: {df['best_val_auroc'].mean():.4f} ± {df['best_val_auroc'].std():.4f}\n\n")

        f.write("BEST MODELS BY POOLING METHOD:\n")
        f.write("-" * 40 + "\n")
        for pooling, result in pooling_results.items():
            f.write(f"{pooling}: {result['model_name']}\n")
            f.write(f"  Val AUROC: {result['best_val_auroc']:.4f}\n")
            f.write(f"  Test AUROC: {result['best_test_auroc']:.4f}\n\n")

        f.write("TOP 5 MODELS OVERALL:\n")
        f.write("-" * 30 + "\n")
        top5 = df.nlargest(5, 'best_val_auroc')
        for idx, row in top5.iterrows():
            f.write(f"{row['model_name']}: {row['best_val_auroc']:.4f} val, {row['best_test_auroc']:.4f} test\n")

    print(f"\nDetailed analysis saved to: {analysis_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter search results")
    parser.add_argument("results_dir", help="Directory containing summary.csv")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top models to show")
    parser.add_argument("--save", action="store_true", help="Save detailed analysis to file")
    parser.add_argument("--method", default="multilayer_transformer",
                       help="Pooling method to analyze in detail")

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Directory {args.results_dir} does not exist!")
        sys.exit(1)

    # Load results
    df = load_results(args.results_dir)
    if df is None:
        sys.exit(1)

    print(f"Loaded {len(df)} successful experiments from {args.results_dir}")
    print(f"Experiment date: {os.path.basename(args.results_dir)}")

    # Run analysis
    best_val, best_test = analyze_top_models(df, args.top_n)
    pooling_results = analyze_by_pooling_method(df)
    analyze_hyperparameters(df, args.method)
    compare_methods(df)
    generate_recommendations(df, pooling_results)

    # Save if requested
    if args.save:
        save_analysis(df, args.results_dir, pooling_results, best_val, best_test)

    print_section("ANALYSIS COMPLETE")
    print(f"Best overall model: {best_val['model_name']}")
    print(f"Best validation AUROC: {best_val['best_val_auroc']:.4f}")
    print(f"Corresponding test AUROC: {best_val['best_test_auroc']:.4f}")

if __name__ == "__main__":
    main()