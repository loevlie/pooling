#!/usr/bin/env python3
"""Parse command logs and extract job information by N_train value."""
import argparse
import re
from pathlib import Path
from collections import defaultdict


def parse_log_line(line):
    """Parse a single log line and extract relevant information."""
    # Pattern: [timestamp] Job jobid: python ... --lr=X ... --N_train=Y ... --weight_decay=Z
    match = re.match(r'\[([\d\-: ]+)\]\s+Job\s+([\w_]+):\s+(.+)', line)

    if not match:
        return None

    timestamp = match.group(1)
    job_id = match.group(2)
    command = match.group(3)

    # Extract parameters from command
    info = {
        'timestamp': timestamp,
        'job_id': job_id,
        'command': command
    }

    # Extract N_train
    n_train_match = re.search(r'--N_train[= ](\d+)', command)
    if n_train_match:
        info['N_train'] = int(n_train_match.group(1))

    # Extract lr
    lr_match = re.search(r'--lr[= ]([\d.e\-]+)', command)
    if lr_match:
        info['lr'] = float(lr_match.group(1))

    # Extract weight_decay
    wd_match = re.search(r'--weight_decay[= ]([\d.e\-]+)', command)
    if wd_match:
        info['weight_decay'] = float(wd_match.group(1))

    return info


def parse_log_file(log_file, n_train_filter=None):
    """Parse entire log file and return job information."""
    jobs = []

    with open(log_file, 'r') as f:
        for line in f:
            info = parse_log_line(line.strip())
            if info is None:
                continue

            # Filter by N_train if specified
            if n_train_filter is not None:
                if 'N_train' not in info or info['N_train'] != n_train_filter:
                    continue

            jobs.append(info)

    return jobs


def summarize_jobs(jobs):
    """Summarize jobs by LR and WD combinations."""
    # Group by (lr, wd)
    combinations = defaultdict(list)

    for job in jobs:
        if 'lr' in job and 'weight_decay' in job:
            key = (job['lr'], job['weight_decay'])
            combinations[key].append(job['job_id'])

    return combinations


def main():
    parser = argparse.ArgumentParser(description="Parse command logs and extract job info")
    parser.add_argument("log_file", type=str, help="Path to command log file")
    parser.add_argument("--N_train", type=int, required=True, help="N_train value to filter by")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file {args.log_file} does not exist")
        return

    print(f"Parsing log file: {args.log_file}")
    print(f"Filtering by N_train = {args.N_train}")
    print()

    jobs = parse_log_file(log_path, args.N_train)

    if not jobs:
        print(f"No jobs found for N_train = {args.N_train}")
        return

    print(f"Found {len(jobs)} jobs for N_train = {args.N_train}")
    print()

    # Summarize by combinations
    combinations = summarize_jobs(jobs)

    print("="*70)
    print(f"LR/WD COMBINATIONS (N_train = {args.N_train})")
    print("="*70)
    print(f"{'LR':<15} {'WD':<15} {'# Jobs':<10} {'Job IDs'}")
    print("-"*70)

    for (lr, wd), job_ids in sorted(combinations.items()):
        job_ids_str = ', '.join(job_ids)
        print(f"{lr:<15.6f} {wd:<15.6e} {len(job_ids):<10} {job_ids_str}")

    print()
    print(f"Total combinations: {len(combinations)}")
    print(f"Total jobs: {len(jobs)}")

    if args.verbose:
        print()
        print("="*70)
        print("DETAILED JOB INFORMATION")
        print("="*70)
        for job in jobs:
            print(f"Job ID: {job['job_id']}")
            print(f"  Timestamp: {job['timestamp']}")
            print(f"  N_train: {job.get('N_train', 'N/A')}")
            print(f"  LR: {job.get('lr', 'N/A')}")
            print(f"  WD: {job.get('weight_decay', 'N/A')}")
            if args.verbose:
                print(f"  Command: {job['command'][:100]}...")
            print()


if __name__ == "__main__":
    main()
