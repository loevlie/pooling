"""Benchmark epoch time for different N_train values.

This script times training epochs without saving results or modifying code permanently.
"""
import argparse
import os
import pandas as pd
import tempfile
import time
import torch
import losses
import models
import utils


def benchmark(N_train, seed, num_warmup_epochs=3, num_timed_epochs=10,
              batch_size=64, pooling='smmil', embedding_level=True, include_overhead=False):
    """Run benchmark and return average time per epoch.

    Args:
        include_overhead: If True, includes val/test evaluation and CSV writing overhead
    """

    torch.manual_seed(seed)

    # Generate data (using same parameters as run_N_158_local.sh)
    delta = 2
    deltaS = 3
    N_val = N_train // 4
    N_test = 1000

    X_train, lengths_train, u_train, y_train = utils.generate_toy_data(
        N_train, delta=delta, deltaS=deltaS, seed=0
    )

    if include_overhead:
        X_val, lengths_val, u_val, y_val = utils.generate_toy_data(
            N_val, delta=delta, deltaS=deltaS, seed=1
        )
        X_test, lengths_test, u_test, y_test = utils.generate_toy_data(
            N_test, delta=delta, deltaS=deltaS, seed=2
        )

    # Setup device and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = utils.ToyDataset(X_train, lengths_train, y_train)
    shuffled_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=utils.collate_fn, drop_last=True
    )

    if include_overhead:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=utils.collate_fn
        )
        val_dataset = utils.ToyDataset(X_val, lengths_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=utils.collate_fn
        )
        test_dataset = utils.ToyDataset(X_test, lengths_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, collate_fn=utils.collate_fn
        )

    if embedding_level:
        model = models.PoolClf(in_features=768, out_features=1, pooling=pooling)
    else:
        model = models.ClfPool(in_features=768, out_features=1, pooling=pooling)
    model.to(device)

    criterion = losses.L1Loss(alpha=0.0, criterion=torch.nn.BCEWithLogitsLoss())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0, momentum=0.9)

    # Create temporary CSV file if including overhead
    if include_overhead:
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "benchmark.csv")
        columns = ["epoch", "test_acc", "test_auroc", "test_auprc", "test_loss", "test_nll",
                   "train_acc", "train_auroc", "train_auprc", "train_loss", "train_nll",
                   "val_acc", "val_auroc", "val_auprc", "val_loss", "val_nll"]
        model_history_df = pd.DataFrame(columns=columns)

    # Warmup epochs (not timed)
    print(f"Running {num_warmup_epochs} warmup epochs...")
    for epoch in range(num_warmup_epochs):
        if include_overhead:
            shuffled_train_metrics = utils.train_one_epoch(model, criterion, optimizer, shuffled_train_loader)
            train_metrics = shuffled_train_metrics
            val_metrics = utils.evaluate(model, criterion, val_loader)
            test_metrics = utils.evaluate(model, criterion, test_loader)
            row = [epoch, test_metrics["acc"], test_metrics["auroc"], test_metrics["auprc"],
                   test_metrics["loss"], test_metrics["nll"], train_metrics["acc"],
                   train_metrics["auroc"], train_metrics["auprc"], train_metrics["loss"],
                   train_metrics["nll"], val_metrics["acc"], val_metrics["auroc"],
                   val_metrics["auprc"], val_metrics["loss"], val_metrics["nll"]]
            model_history_df.loc[epoch] = row
            model_history_df.to_csv(csv_path)
        else:
            utils.train_one_epoch(model, criterion, optimizer, shuffled_train_loader)

    # Synchronize GPU before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed epochs
    print(f"Timing {num_timed_epochs} epochs...")
    epoch_times = []
    for epoch in range(num_timed_epochs):
        start_time = time.time()

        if include_overhead:
            shuffled_train_metrics = utils.train_one_epoch(model, criterion, optimizer, shuffled_train_loader)
            train_metrics = shuffled_train_metrics
            val_metrics = utils.evaluate(model, criterion, val_loader)
            test_metrics = utils.evaluate(model, criterion, test_loader)

            epoch_idx = num_warmup_epochs + epoch
            row = [epoch_idx, test_metrics["acc"], test_metrics["auroc"], test_metrics["auprc"],
                   test_metrics["loss"], test_metrics["nll"], train_metrics["acc"],
                   train_metrics["auroc"], train_metrics["auprc"], train_metrics["loss"],
                   train_metrics["nll"], val_metrics["acc"], val_metrics["auroc"],
                   val_metrics["auprc"], val_metrics["loss"], val_metrics["nll"]]
            model_history_df.loc[epoch_idx] = row
            model_history_df.to_csv(csv_path)

            # Match toy_data.py line 96: print the row
            print(model_history_df.iloc[epoch_idx])

            # Match toy_data.py line 100: check for best model (idxmax call)
            best_epoch = model_history_df.val_auroc.idxmax()
        else:
            utils.train_one_epoch(model, criterion, optimizer, shuffled_train_loader)

        # Synchronize GPU to get accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

    # Cleanup temp directory
    if include_overhead:
        import shutil
        shutil.rmtree(temp_dir)

    avg_time = sum(epoch_times) / len(epoch_times)
    return avg_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark epoch time")
    parser.add_argument("--N_train", type=int, required=True, help="Number of training samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_warmup_epochs", type=int, default=3, help="Number of warmup epochs")
    parser.add_argument("--num_timed_epochs", type=int, default=10, help="Number of epochs to time")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--pooling", type=str, default="smmil", help="Pooling method")
    parser.add_argument("--embedding_level", action='store_true', default=False,
                        help="Use embedding-level model")
    parser.add_argument("--include_overhead", action='store_true', default=False,
                        help="Include val/test evaluation and CSV writing overhead")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"N_train: {args.N_train}, Seed: {args.seed}")
    if args.include_overhead:
        print("Including overhead: val/test evaluation + CSV writing")

    avg_time = benchmark(
        N_train=args.N_train,
        seed=args.seed,
        num_warmup_epochs=args.num_warmup_epochs,
        num_timed_epochs=args.num_timed_epochs,
        batch_size=args.batch_size,
        pooling=args.pooling,
        embedding_level=args.embedding_level,
        include_overhead=args.include_overhead
    )

    print(f"Average time per epoch: {avg_time:.4f} seconds")
    print(f"RESULT: {args.N_train},{args.seed},{avg_time:.6f}")
