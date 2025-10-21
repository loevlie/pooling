#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For headless environments


def find_metric_column(columns, split="val"):
    """
    Try to find a column for AUROC/AUC for a given split ('val' or 'test').
    Returns the column name or None.
    """
    cand = None
    # Normalize columns once for matching
    cols_norm = {c: c.lower() for c in columns}
    # Preferred order of patterns
    patterns = [
        rf"^{split}[_\- ]*auroc$",
        rf"^{split}[_\- ]*auc$",
        rf"auroc[_\- ]*{split}$",
        rf"auc[_\- ]*{split}$",
        rf"{split}.*(auroc|auc)",
        rf"(auroc|auc).*{split}",
    ]
    for p in patterns:
        regex = re.compile(p)
        for orig, low in cols_norm.items():
            if regex.search(low):
                cand = orig
                return cand
    return None


def extract_ntrain_from_source(folder_name: str, csv_filename: str = None):
    """
    Tries to extract N_train from either folder names like 'Ntrain_6309' 
    or CSV filenames like 'something_N_100_something.csv'.
    Returns (int_ntrain, source_label).
    """
    # First try folder name patterns
    patterns = [
        r"[Nn]train[_\-]?(\d+)",  # Ntrain_6309 or Ntrain6309
        r"[Nn][_\-]?(\d+)",        # N_100 or N100
    ]
    
    for pattern in patterns:
        m = re.search(pattern, folder_name)
        if m:
            return int(m.group(1)), folder_name
    
    # If folder doesn't have it and csv_filename is provided, try the filename
    if csv_filename:
        for pattern in patterns:
            m = re.search(pattern, csv_filename)
            if m:
                return int(m.group(1)), f"{folder_name} (N from file)"
    
    # Last resort: just extract any number from folder
    m = re.search(r"(\d+)", folder_name)
    if m:
        return int(m.group(1)), folder_name
        
    return None, folder_name


def summarize_csv(csv_path: Path):
    """
    Load a CSV, find the row with max val AUROC, and return a dict summary.
    If metrics are missing/invalid, returns None.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {
            "ok": False,
            "reason": f"read_error: {e}",
            "csv_path": str(csv_path),
        }

    if df.empty:
        return {"ok": False, "reason": "empty_csv", "csv_path": str(csv_path)}

    val_col = find_metric_column(df.columns, split="val")
    if val_col is None:
        return {
            "ok": False,
            "reason": "no_val_auroc_column",
            "csv_path": str(csv_path),
            "columns": ",".join(df.columns.astype(str)),
        }

    # coerce to numeric
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    if df[val_col].notna().sum() == 0:
        return {
            "ok": False,
            "reason": "val_auroc_all_nan",
            "csv_path": str(csv_path),
        }

    # Find the row with maximum validation AUROC
    best_idx = df[val_col].astype(float).idxmax()
    best_row = df.loc[best_idx]
    
    print(f"  Best val AUROC: {float(best_row[val_col]):.4f} at index {best_idx}")

    test_col = find_metric_column(df.columns, split="test")
    test_auroc = None
    if test_col is not None:
        # coerce and pull from the SAME row
        df[test_col] = pd.to_numeric(df[test_col], errors="coerce")
        val = best_row.get(test_col, None)
        if pd.notna(val):
            test_auroc = float(val)
            print(f"  Corresponding test AUROC: {test_auroc:.4f}")

    epoch = None
    for ep_col in ["epoch", "Epoch", "step", "Step"]:
        if ep_col in df.columns:
            try:
                epoch = int(best_row[ep_col])
                break
            except Exception:
                pass

    return {
        "ok": True,
        "csv_path": str(csv_path),
        "epoch_at_best_val": epoch,
        "val_col": val_col,
        "test_col": test_col,
        "val_auroc_best": float(best_row[val_col]),
        "test_auroc_at_best_val": (float(test_auroc) if test_auroc is not None else None),
    }


def create_ntrain_vs_test_plot(df, save_path="ntrain_vs_test_auroc.png"):
    """
    Create a plot of N_Train vs Test AUROC for the best models.
    """
    # Filter for successful runs with test AUROC
    plot_df = df[df["ok"] & df["test_auroc_at_best_val"].notna()].copy()
    
    if plot_df.empty:
        print("No valid data for plotting (no successful runs with test AUROC)")
        return
    
    # Get best result per N_train
    best_per_n = (
        plot_df.sort_values(by=["N_train", "val_auroc_best"], ascending=[True, False])
        .groupby("N_train", as_index=False)
        .first()
    )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main plot
    ax.plot(best_per_n["N_train"], best_per_n["test_auroc_at_best_val"], 
            'o-', linewidth=2, markersize=8, label='Test AUROC')
    
    # Add validation AUROC for comparison
    ax.plot(best_per_n["N_train"], best_per_n["val_auroc_best"], 
            's--', linewidth=1.5, markersize=6, alpha=0.7, 
            color='orange', label='Validation AUROC')
    
    # Formatting
    ax.set_xlabel('N_Train (Number of Training Samples)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Test AUROC vs Training Set Size', 
                 fontsize=14, fontweight='bold')
    
    # Set x-axis to log scale if we have a wide range
    if best_per_n["N_train"].max() / best_per_n["N_train"].min() > 10:
        ax.set_xscale('log')
        ax.set_xlabel('N_Train', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Add value annotations
    for _, row in best_per_n.iterrows():
        ax.annotate(f'{row["test_auroc_at_best_val"]:.3f}', 
                   (row["N_train"], row["test_auroc_at_best_val"]),
                   textcoords="offset points", xytext=(0,5), 
                   ha='center', fontsize=8)
    
    # Set y-axis limits for better visibility
    y_min = min(best_per_n["test_auroc_at_best_val"].min(), 
                best_per_n["val_auroc_best"].min()) - 0.02
    y_max = max(best_per_n["test_auroc_at_best_val"].max(), 
                best_per_n["val_auroc_best"].max()) + 0.02
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    # Print summary statistics
    print("\n=== Plot Summary ===")
    print(f"N_train values: {sorted(best_per_n['N_train'].unique())}")
    print(f"Test AUROC range: {best_per_n['test_auroc_at_best_val'].min():.4f} - "
          f"{best_per_n['test_auroc_at_best_val'].max():.4f}")
    print(f"Best test AUROC: {best_per_n['test_auroc_at_best_val'].max():.4f} "
          f"at N_train={best_per_n.loc[best_per_n['test_auroc_at_best_val'].idxmax(), 'N_train']}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize best val AUROC (and corresponding test AUROC) from CSV logs grouped by N_train folders."
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root folder containing N_train subfolders (e.g., Ntrain_100/, Ntrain_6309/, ...) or a single folder with CSVs.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="summary_best_val_auroc.csv",
        help="Path to save the combined summary CSV.",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="ntrain_vs_test_auroc.png",
        help="Path to save the N_train vs Test AUROC plot.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="**/*.csv",
        help="Glob pattern (relative to each N_train folder) to find CSV log files (default: '**/*.csv').",
    )
    parser.add_argument(
        "--single_folder",
        action="store_true",
        help="Treat root as a single folder with CSVs (extract N_train from filenames).",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root folder does not exist: {root}")

    rows = []
    
    if args.single_folder:
        # Process single folder, extract N_train from CSV filenames
        print(f"Processing single folder: {root}")
        csv_files = sorted(root.glob(args.glob))
        
        for csv_path in csv_files:
            print(f"\nProcessing: {csv_path.name}")
            int_ntrain, ntrain_label = extract_ntrain_from_source(
                root.name, csv_path.name
            )
            
            info = summarize_csv(csv_path)
            rows.append(
                {
                    "N_train": int_ntrain,
                    "N_train_folder": ntrain_label,
                    "csv_path": str(csv_path),
                    "ok": info.get("ok", False),
                    "reason": info.get("reason"),
                    "val_auroc_best": info.get("val_auroc_best"),
                    "test_auroc_at_best_val": info.get("test_auroc_at_best_val"),
                    "epoch_at_best_val": info.get("epoch_at_best_val"),
                    "val_metric_col": info.get("val_col"),
                    "test_metric_col": info.get("test_col"),
                }
            )
    else:
        # Original behavior: process N_train subfolders
        for ntrain_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            print(f"\nProcessing folder: {ntrain_dir.name}")
            int_ntrain, ntrain_label = extract_ntrain_from_source(ntrain_dir.name)
            csv_files = sorted(ntrain_dir.glob(args.glob))

            if not csv_files:
                rows.append(
                    {
                        "N_train": int_ntrain,
                        "N_train_folder": ntrain_label,
                        "csv_path": None,
                        "ok": False,
                        "reason": "no_csv_found",
                        "val_auroc_best": None,
                        "test_auroc_at_best_val": None,
                        "epoch_at_best_val": None,
                        "val_metric_col": None,
                        "test_metric_col": None,
                    }
                )
                continue

            for csv_path in csv_files:
                print(f"  Processing: {csv_path.name}")
                info = summarize_csv(csv_path)
                rows.append(
                    {
                        "N_train": int_ntrain,
                        "N_train_folder": ntrain_label,
                        "csv_path": str(csv_path),
                        "ok": info.get("ok", False),
                        "reason": info.get("reason"),
                        "val_auroc_best": info.get("val_auroc_best"),
                        "test_auroc_at_best_val": info.get("test_auroc_at_best_val"),
                        "epoch_at_best_val": info.get("epoch_at_best_val"),
                        "val_metric_col": info.get("val_col"),
                        "test_metric_col": info.get("test_col"),
                    }
                )

    df = pd.DataFrame(rows)

    # Helpful sort: by N_train then by best val desc (placing failures last)
    df["__val_sort"] = df["val_auroc_best"].fillna(-1)
    df.sort_values(by=["N_train", "__val_sort"], ascending=[True, False], inplace=True)
    df.drop(columns="__val_sort", inplace=True)

    # Save + print a concise view
    df.to_csv(args.save_csv, index=False)

    # Also print a minimal table to stdout
    display_cols = [
        "N_train",
        "N_train_folder",
        "val_auroc_best",
        "test_auroc_at_best_val",
        "epoch_at_best_val",
        "csv_path",
        "ok",
        "reason",
    ]
    print("\n" + "="*80)
    print("SUMMARY OF ALL CSV FILES")
    print("="*80)
    print(df[display_cols].to_string(index=False))

    # If you also want one top-per-N_train folder, show a quick pivot:
    top_per_n = (
        df[df["ok"]]
        .sort_values(by=["N_train", "val_auroc_best"], ascending=[True, False])
        .groupby("N_train", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    if not top_per_n.empty:
        print("\n" + "="*80)
        print("BEST CSV PER N_TRAIN (by validation AUROC)")
        print("="*80)
        print(
            top_per_n[
                ["N_train", "val_auroc_best", "test_auroc_at_best_val", "epoch_at_best_val", "csv_path"]
            ].to_string(index=False)
        )
        # save alongside main summary for convenience
        top_per_n.to_csv(Path(args.save_csv).with_name("summary_best_per_Ntrain.csv"), index=False)
        
        # Create the plot
        create_ntrain_vs_test_plot(df, save_path=args.save_plot)


if __name__ == "__main__":
    main()