#!/usr/bin/env python3
"""
Plot delta experiment results comparing different methods across signal strengths.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

def load_and_clean_data(results_dir):
    """Load delta experiment results."""
    summary_file = os.path.join(results_dir, "delta_results.csv")
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

def create_delta_plot(df, results_dir):
    """Create the main delta vs performance plot."""
    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Define colors and markers for each method
    method_styles = {
        'attention': {'color': '#1f77b4', 'marker': 'o', 'label': 'Attention (Baseline)'},
        'multilayer_erm': {'color': '#ff7f0e', 'marker': 's', 'label': 'MultiLayer Transformer (ERM)'},
        'multilayer_entropy': {'color': '#2ca02c', 'marker': '^', 'label': 'MultiLayer Transformer (Entropy Reg.)'}
    }

    # Plot each method
    for method in method_styles.keys():
        method_data = df[df['method'] == method].sort_values('delta')

        if len(method_data) > 0:
            deltas = method_data['delta'].values
            test_aurocs = method_data['best_test_auroc'].values
            val_aurocs = method_data['best_val_auroc'].values

            style = method_styles[method]

            # Plot test AUROC (main line)
            plt.plot(deltas, test_aurocs,
                    color=style['color'], marker=style['marker'],
                    linewidth=2, markersize=8,
                    label=style['label'])

            # Add validation AUROC as smaller markers (for reference)
            plt.plot(deltas, val_aurocs,
                    color=style['color'], marker=style['marker'],
                    linewidth=1, markersize=4, alpha=0.6, linestyle='--')

    # Customize the plot
    plt.xlabel('Signal Strength (δ)', fontsize=14)
    plt.ylabel('Test AUROC', fontsize=14)
    plt.title('Performance vs Signal Strength (δS=3)\nSolid lines: Test AUROC, Dashed lines: Validation AUROC', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Set reasonable axis limits
    plt.xlim(0.5, 5.5)
    if len(df) > 0:
        min_auroc = df['best_test_auroc'].min() - 0.05
        max_auroc = df['best_test_auroc'].max() + 0.05
        plt.ylim(max(0.4, min_auroc), min(1.0, max_auroc))

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(results_dir, "delta_performance_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    return plot_path

def create_performance_table(df, results_dir):
    """Create a detailed performance table."""
    print("\n" + "="*80)
    print("DELTA EXPERIMENT RESULTS")
    print("="*80)

    # Pivot table for easy viewing
    pivot_test = df.pivot(index='delta', columns='method', values='best_test_auroc')
    pivot_val = df.pivot(index='delta', columns='method', values='best_val_auroc')

    print("\nTest AUROC by Method and Delta:")
    print("-" * 50)
    print(pivot_test.to_string(float_format='%.4f'))

    print("\nValidation AUROC by Method and Delta:")
    print("-" * 50)
    print(pivot_val.to_string(float_format='%.4f'))

    # Performance analysis
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS")
    print("="*50)

    for method in df['method'].unique():
        method_data = df[df['method'] == method].sort_values('delta')
        if len(method_data) > 0:
            print(f"\n{method.upper()}:")
            print(f"  Best Test AUROC: {method_data['best_test_auroc'].max():.4f} at δ={method_data.loc[method_data['best_test_auroc'].idxmax(), 'delta']}")
            print(f"  Worst Test AUROC: {method_data['best_test_auroc'].min():.4f} at δ={method_data.loc[method_data['best_test_auroc'].idxmin(), 'delta']}")
            print(f"  Range: {method_data['best_test_auroc'].max() - method_data['best_test_auroc'].min():.4f}")

            # Trend analysis
            deltas = method_data['delta'].values
            aurocs = method_data['best_test_auroc'].values
            if len(deltas) > 2:
                slope = np.polyfit(deltas, aurocs, 1)[0]
                if slope > 0.01:
                    trend = "Improving with higher δ"
                elif slope < -0.01:
                    trend = "Declining with higher δ"
                else:
                    trend = "Stable across δ"
                print(f"  Trend: {trend} (slope: {slope:.4f})")

    # Save detailed results
    table_path = os.path.join(results_dir, "delta_results_table.txt")
    with open(table_path, 'w') as f:
        f.write("DELTA EXPERIMENT DETAILED RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write("Test AUROC by Method and Delta:\n")
        f.write(pivot_test.to_string(float_format='%.4f') + "\n\n")
        f.write("Validation AUROC by Method and Delta:\n")
        f.write(pivot_val.to_string(float_format='%.4f') + "\n\n")

        # Full data
        f.write("Complete Results:\n")
        f.write(df.to_string(index=False, float_format='%.4f'))

    print(f"\nDetailed results saved to: {table_path}")

    return table_path

def generate_summary_report(df, results_dir):
    """Generate a summary report with key findings."""
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)

    # Find overall best performing method
    best_overall = df.loc[df['best_test_auroc'].idxmax()]
    print(f"\nBest Overall Performance:")
    print(f"  Method: {best_overall['method']}")
    print(f"  Delta: {best_overall['delta']}")
    print(f"  Test AUROC: {best_overall['best_test_auroc']:.4f}")
    print(f"  Val AUROC: {best_overall['best_val_auroc']:.4f}")

    # Compare methods at each delta
    print(f"\nMethod Ranking by Delta:")
    for delta in sorted(df['delta'].unique()):
        delta_data = df[df['delta'] == delta].sort_values('best_test_auroc', ascending=False)
        print(f"  δ={delta}: ", end="")
        rankings = []
        for idx, row in delta_data.iterrows():
            rankings.append(f"{row['method']} ({row['best_test_auroc']:.4f})")
        print(" > ".join(rankings))

    # Key insights
    print(f"\nKey Insights:")

    # Check if MultiLayer methods outperform attention
    attention_mean = df[df['method'] == 'attention']['best_test_auroc'].mean()
    mt_erm_mean = df[df['method'] == 'multilayer_erm']['best_test_auroc'].mean()
    mt_entropy_mean = df[df['method'] == 'multilayer_entropy']['best_test_auroc'].mean()

    print(f"  • Average Test AUROC - Attention: {attention_mean:.4f}")
    print(f"  • Average Test AUROC - MT ERM: {mt_erm_mean:.4f}")
    print(f"  • Average Test AUROC - MT Entropy: {mt_entropy_mean:.4f}")

    if mt_erm_mean > attention_mean:
        print(f"  • MultiLayer Transformer (ERM) outperforms attention by {mt_erm_mean - attention_mean:.4f}")
    if mt_entropy_mean > attention_mean:
        print(f"  • MultiLayer Transformer (Entropy) outperforms attention by {mt_entropy_mean - attention_mean:.4f}")
    if mt_entropy_mean > mt_erm_mean:
        print(f"  • Entropy regularization improves MT performance by {mt_entropy_mean - mt_erm_mean:.4f}")

    # Signal strength effects
    print(f"  • Signal strength (δ) effects:")
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        if len(method_data) >= 3:
            best_delta = method_data.loc[method_data['best_test_auroc'].idxmax(), 'delta']
            worst_delta = method_data.loc[method_data['best_test_auroc'].idxmin(), 'delta']
            print(f"    - {method}: Best at δ={best_delta}, Worst at δ={worst_delta}")

def main():
    parser = argparse.ArgumentParser(description="Plot delta experiment results")
    parser.add_argument("results_dir", help="Directory containing delta_results.csv")
    parser.add_argument("--save-plot", action="store_true", help="Save plot to file")
    parser.add_argument("--show-plot", action="store_true", help="Display plot interactively")

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Directory {args.results_dir} does not exist!")
        sys.exit(1)

    # Load data
    df = load_and_clean_data(args.results_dir)
    if df is None:
        sys.exit(1)

    print(f"Loaded {len(df)} successful experiments from {args.results_dir}")

    # Create visualizations and analysis
    plot_path = create_delta_plot(df, args.results_dir)
    table_path = create_performance_table(df, args.results_dir)
    generate_summary_report(df, args.results_dir)

    # Show plot if requested
    if args.show_plot:
        plt.show()

    print(f"\nAnalysis complete!")
    print(f"Plot: {plot_path}")
    print(f"Table: {table_path}")

if __name__ == "__main__":
    main()