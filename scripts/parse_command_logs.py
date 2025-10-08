#!/usr/bin/env python3
"""Parse command logs and extract job information by N_train value."""
import argparse
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd


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


def get_active_jobs():
    """Query squeue to get currently running/pending jobs."""
    try:
        import os
        username = os.environ.get('USER', os.environ.get('USERNAME', ''))

        if not username:
            print("Warning: Could not determine username")
            return {}

        # Run squeue with format to get job ID, state, and time
        result = subprocess.run(
            ['squeue', '-u', username, '-o', '%.18i %.2t %.10M'],
            capture_output=True,
            text=True,
            shell=False
        )

        # If squeue not available, return empty dict
        if result.returncode != 0:
            return {}

        active_jobs = {}
        lines = result.stdout.strip().split('\n')

        # Skip header
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                job_id = parts[0]
                state = parts[1]  # R=running, PD=pending, etc
                time_str = parts[2]  # Format: days-HH:MM:SS or HH:MM:SS or MM:SS

                active_jobs[job_id] = {
                    'state': state,
                    'time': time_str
                }

        return active_jobs

    except Exception as e:
        print(f"Warning: Could not query squeue: {e}")
        return {}


def parse_slurm_time(time_str):
    """Parse SLURM time format to timedelta."""
    # Formats: "days-HH:MM:SS", "HH:MM:SS", "MM:SS"
    try:
        if '-' in time_str:
            # days-HH:MM:SS
            days_part, time_part = time_str.split('-')
            days = int(days_part)
            time_parts = time_part.split(':')
        else:
            days = 0
            time_parts = time_str.split(':')

        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
        elif len(time_parts) == 2:
            hours = 0
            minutes, seconds = map(int, time_parts)
        else:
            return None

        return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    except:
        return None


def format_timedelta(td):
    """Format timedelta as human-readable string."""
    if td is None:
        return "N/A"

    total_seconds = int(td.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def get_current_epoch(exp_dir, lr, wd, n_train, seed=1001, criterion='L1', pooling='smMILattentionEarly', embedding_level=True):
    """Get current epoch from CSV file if it exists."""
    try:
        # Construct expected filename
        level_suffix = "embedding_level" if embedding_level else "instance_level"
        model_name = f"criterion={criterion}_lr={lr}_pooling={pooling}_seed={seed}_wd_{wd}_N_{n_train}_{level_suffix}.csv"

        csv_path = Path(exp_dir) / model_name

        if not csv_path.exists():
            return None

        # Read CSV and get last epoch
        df = pd.read_csv(csv_path)
        if 'epoch' not in df.columns or len(df) == 0:
            return None

        return int(df['epoch'].max())

    except Exception as e:
        return None


def summarize_jobs(jobs, check_slurm=False, exp_dir=None, n_train=None):
    """Summarize jobs by LR and WD combinations."""
    # Group by (lr, wd)
    combinations = defaultdict(list)

    # Get active jobs if requested
    active_jobs = {}
    if check_slurm:
        active_jobs = get_active_jobs()

    for job in jobs:
        if 'lr' in job and 'weight_decay' in job:
            key = (job['lr'], job['weight_decay'])

            # Check if job is active
            job_id = job['job_id']
            if job_id in active_jobs:
                job['status'] = active_jobs[job_id]['state']
                job['runtime'] = active_jobs[job_id]['time']
            else:
                job['status'] = 'FINISHED' if check_slurm else 'UNKNOWN'
                job['runtime'] = None

            # Get current epoch if exp_dir provided
            if exp_dir and n_train:
                current_epoch = get_current_epoch(exp_dir, job['lr'], job['weight_decay'], n_train)
                job['current_epoch'] = current_epoch

            combinations[key].append(job)

    return combinations


def main():
    parser = argparse.ArgumentParser(description="Parse command logs and extract job info")
    parser.add_argument("log_file", type=str, help="Path to command log file")
    parser.add_argument("--N_train", type=int, required=True, help="N_train value to filter by")
    parser.add_argument("--exp-dir", type=str, default=None, help="Experiments directory to check current epoch")
    parser.add_argument("--check-slurm", action="store_true", help="Check job status via squeue")
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
    combinations = summarize_jobs(jobs, check_slurm=args.check_slurm, exp_dir=args.exp_dir, n_train=args.N_train)

    # Count status
    status_counts = defaultdict(int)
    for job_list in combinations.values():
        for job in job_list:
            if 'status' in job:
                status_counts[job['status']] += 1

    if args.check_slurm:
        print("="*70)
        print("JOB STATUS SUMMARY")
        print("="*70)
        for status, count in sorted(status_counts.items()):
            status_label = {
                'R': 'Running',
                'PD': 'Pending',
                'FINISHED': 'Finished'
            }.get(status, status)
            print(f"  {status_label}: {count}")
        print()

    print("="*110)
    print(f"LR/WD COMBINATIONS (N_train = {args.N_train})")
    print("="*110)

    if args.check_slurm or args.exp_dir:
        header = f"{'LR':<12} {'WD':<12}"
        if args.check_slurm:
            header += f" {'Status':<12} {'Runtime':<12}"
        if args.exp_dir:
            header += f" {'Epoch':<12}"
        header += f" {'Job ID'}"
        print(header)
        print("-"*110)
    else:
        print(f"{'LR':<15} {'WD':<15} {'# Jobs':<10} {'Job IDs'}")
        print("-"*110)

    for (lr, wd), job_list in sorted(combinations.items()):
        if args.check_slurm or args.exp_dir:
            for job in job_list:
                line = f"{lr:<12.6f} {wd:<12.6e}"

                if args.check_slurm:
                    status = job.get('status', 'UNKNOWN')
                    status_label = {
                        'R': 'RUNNING',
                        'PD': 'PENDING',
                        'FINISHED': 'FINISHED'
                    }.get(status, status)
                    runtime_str = job.get('runtime', 'N/A') if job.get('runtime') else 'N/A'
                    line += f" {status_label:<12} {runtime_str:<12}"

                if args.exp_dir:
                    current_epoch = job.get('current_epoch')
                    epoch_str = f"{current_epoch}/1000" if current_epoch is not None else "N/A"
                    line += f" {epoch_str:<12}"

                job_id = job['job_id']
                line += f" {job_id}"
                print(line)
        else:
            job_ids_str = ', '.join([j['job_id'] for j in job_list])
            print(f"{lr:<15.6f} {wd:<15.6e} {len(job_list):<10} {job_ids_str}")

    print()
    print(f"Total combinations: {len(combinations)}")
    print(f"Total jobs: {len(jobs)}")

    if args.verbose:
        print()
        print("="*70)
        print("DETAILED JOB INFORMATION")
        print("="*70)
        for (lr, wd), job_list in sorted(combinations.items()):
            for job in job_list:
                print(f"Job ID: {job['job_id']}")
                print(f"  Timestamp: {job['timestamp']}")
                print(f"  N_train: {job.get('N_train', 'N/A')}")
                print(f"  LR: {job.get('lr', 'N/A')}")
                print(f"  WD: {job.get('weight_decay', 'N/A')}")
                if args.check_slurm and 'status' in job:
                    status_label = {
                        'R': 'RUNNING',
                        'PD': 'PENDING',
                        'FINISHED': 'FINISHED'
                    }.get(job['status'], job['status'])
                    print(f"  Status: {status_label}")
                    if job.get('runtime'):
                        print(f"  Runtime: {job['runtime']}")
                if args.exp_dir and 'current_epoch' in job:
                    current_epoch = job.get('current_epoch')
                    if current_epoch is not None:
                        print(f"  Current Epoch: {current_epoch}/1000")
                print(f"  Command: {job['command'][:100]}...")
                print()


if __name__ == "__main__":
    main()
