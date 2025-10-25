import argparse
import os
import pandas as pd
import glob
import torch
import torch.nn.functional as F
# Importing our custom module(s)
import losses
import models
import utils

def find_best_model(experiments_directory):
    """
    Find the CSV file with the highest overall val_auroc value.

    Args:
        experiments_directory: Path to the directory containing CSV files

    Returns:
        tuple: (best_csv_path, best_model_name, best_val_auroc)
    """
    csv_files = glob.glob(os.path.join(experiments_directory, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {experiments_directory}")

    best_val_auroc = -1
    best_csv_path = None
    best_model_name = None

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if 'val_auroc' in df.columns:
                max_val_auroc = df['val_auroc'].max()
                if max_val_auroc > best_val_auroc:
                    best_val_auroc = max_val_auroc
                    best_csv_path = csv_path
                    # Extract model name from filename (without .csv extension)
                    best_model_name = os.path.splitext(os.path.basename(csv_path))[0]
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

    if best_csv_path is None:
        raise ValueError("No valid CSV files with val_auroc found")

    print(f"Best model: {best_model_name}")
    print(f"Best val_auroc: {best_val_auroc}")
    print(f"CSV path: {best_csv_path}")

    return best_csv_path, best_model_name, best_val_auroc


def parse_hyperparameters_from_filename(model_name):
    """
    Parse hyperparameters from the model filename.
    Expected format: contains parameters separated by underscores or hyphens

    Args:
        model_name: The model filename (without extension)

    Returns:
        dict: Dictionary of hyperparameters
    """
    # Parse hyperparameters from filename
    # This assumes a naming convention - adjust based on actual naming
    params = {}

    parts = model_name.split('_')
    for part in parts:
        if '=' in part:
            key, value = part.split('=')
            # Try to convert to appropriate type
            try:
                # Try int first
                params[key] = int(value)
            except ValueError:
                try:
                    # Try float
                    params[key] = float(value)
                except ValueError:
                    # Keep as string
                    params[key] = value

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate best model on test data")
    parser.add_argument("--experiments_directory", required=True, help="Directory containing saved CSV files and model weights", type=str)
    parser.add_argument("--output_file", default="test_predictions.csv", help="Output CSV file for predictions (default: test_predictions.csv)", type=str)
    parser.add_argument("--data_seed_test", default=2, help="Random seed for test data generation (default: 2)", type=int)
    parser.add_argument("--N_test", default=1000, help="Number of test samples (default: 1000)", type=int)
    parser.add_argument("--batch_size", default=64, help="Batch size for evaluation (default: 64)", type=int)

    # Model architecture parameters (with defaults matching toy_data.py)
    parser.add_argument("--alpha", default=0.0, help="Alpha parameter for loss (default: 0.0)", type=float)
    parser.add_argument("--beta", default=0.0, help="Beta parameter for loss (default: 0.0)", type=float)
    parser.add_argument("--criterion", default="ERM", help="Loss criterion (default: \"ERM\")", type=str)
    parser.add_argument("--delta", default=1.0, help="Delta parameter for toy data (default: 1.0)", type=float)
    parser.add_argument("--deltaS", default=3, help="DeltaS parameter for toy data (default: 3)", type=int)
    parser.add_argument("--embedding_level", action='store_true', default=False, help='Whether to use embedding-level approach (default: False)')
    parser.add_argument("--instance_conv", action='store_true', default=False, help='Whether to use instance convolutions (default: False)')
    parser.add_argument("--kernel_size", default=3, help="Kernel size for instance convolutions (default: 3)", type=int)
    parser.add_argument("--pooling", default="max", help="Pooling operation (default: \"max\")", type=str)
    parser.add_argument("--use_pos_embedding", action='store_true', default=False, help='Whether to use positional embedding (default: False)')

    args = parser.parse_args()

    # Find the best model
    best_csv_path, best_model_name, best_val_auroc = find_best_model(args.experiments_directory)

    # Path to the corresponding .pt weights file
    weights_path = os.path.join(args.experiments_directory, f"{best_model_name}.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"Loading weights from: {weights_path}")

    # Generate test data (using seed=2 as per toy_data.py default)
    print(f"Generating test data with seed={args.data_seed_test}, N={args.N_test}")
    X_test, lengths_test, u_test, y_test = utils.generate_toy_data(
        args.N_test,
        delta=args.delta,
        deltaS=args.deltaS,
        seed=args.data_seed_test
    )

    test_dataset = utils.ToyDataset(X_test, lengths_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=utils.collate_fn,
        shuffle=False  # Important: don't shuffle for consistent ordering
    )

    # Initialize model with same architecture
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.embedding_level:
        model = models.PoolClf(
            in_features=768,
            out_features=1,
            pooling=args.pooling,
            use_pos_embedding=args.use_pos_embedding
        )
    else:
        model = models.ClfPool(
            in_features=768,
            out_features=1,
            instance_conv=args.instance_conv,
            kernel_size=args.kernel_size,
            pooling=args.pooling,
            use_pos_embedding=args.use_pos_embedding
        )

    # Load the trained weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    print("Evaluating on test data...")

    # Collect predictions
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, lengths, labels in test_loader:
            if device.type == "cuda":
                images = images.to(device)

            logits, _ = model(images, lengths)

            # Convert logits to probabilities using sigmoid
            probabilities = torch.sigmoid(logits)

            if device.type == "cuda":
                probabilities = probabilities.cpu()
                labels = labels.cpu()

            all_predictions.extend(probabilities.squeeze().numpy().tolist())
            all_labels.extend(labels.squeeze().numpy().tolist())

    # Create output dataframe
    results_df = pd.DataFrame({
        'predicted_probability': all_predictions,
        'ground_truth': all_labels
    })

    # Save to CSV
    results_df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to: {args.output_file}")
    print(f"Number of predictions: {len(results_df)}")
    print(f"\nSample of results:")
    print(results_df.head(10))

    # Print some statistics
    print(f"\nStatistics:")
    print(f"Mean predicted probability: {results_df['predicted_probability'].mean():.4f}")
    print(f"Ground truth distribution: {results_df['ground_truth'].value_counts().to_dict()}")
