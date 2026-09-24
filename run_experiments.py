"""
Run the full experiment matrix from IMPLEMENTATION_PLAN.md (Step 4) so you don't
have to invoke train.py by hand for each one.

  (a) Baseline    - 4 networks x 3 datasets, lr=0.01, batch_size=4, epochs=20
  (b) Learning rate - three-layer MLP w/ residual, full dataset, lr in
                      {0.001, 0.01, 1.0, 10, 100}
  (c) Batch size    - same network/dataset, lr=0.01, batch_size in {1, 4, 8, 16}

Configs shared between groups (e.g. the lr=0.01/batch_size=4 baseline run for the
residual three-layer MLP) are only run once. Every run also gets skipped if a
matching config is already present in results/experiment_log.csv, so re-running
this script after an interruption resumes instead of redoing finished work.

Each run's console output (including numpy RuntimeWarnings, which are expected
evidence for the lr=10/lr=100 instability question) is streamed live and saved
to results/logs/.

Usage:
    python run_experiments.py                # run everything not already done
    python run_experiments.py --dry-run       # list the run matrix, run nothing
    python run_experiments.py --only a,c      # restrict to experiment group(s)
    python run_experiments.py --limit 2       # cap number of runs (smoke test)
    python run_experiments.py --rerun         # ignore experiment_log.csv, redo all
"""

import argparse
import csv
import os
import subprocess
import sys
import time

from experiment_utils import LOG_PATH, dataset_tag

DATA_DIR = "./data"
TEST_FILE = "fashion-mnist_test.csv.gz"
FULL = "fashion-mnist_train.csv.gz"
PRUNED_50 = "fashion-mnist_train_pruned_50percent.csv.gz"
PRUNED_10 = "fashion-mnist_train_pruned_10percent.csv.gz"

LOG_DIR = os.path.join("results", "logs")

MODEL_NAME = {"linear": "Linear Classifier", "mlp": "Two-Layer MLP", "mlp3": "Three-Layer MLP"}

NETWORKS = [
    dict(model="linear"),
    dict(model="mlp", nhidden=30),
    dict(model="mlp3", nhidden=30, nhidden2=30),
    dict(model="mlp3", nhidden=30, nhidden2=30, residual=True),
]

RESIDUAL_MLP3 = dict(model="mlp3", nhidden=30, nhidden2=30, residual=True)


def _run(train_file, lr=0.01, batch_size=4, epochs=20, **network):
    run = dict(train_file=train_file, lr=float(lr), batch_size=int(batch_size), epochs=int(epochs))
    run.update(network)
    return run


def build_matrix():
    """Returns an ordered list of {"groups": set[str], "run": dict}, deduplicated."""
    runs = {}

    def add(group, run):
        sig = signature(run)
        if sig in runs:
            runs[sig]["groups"].add(group)
        else:
            runs[sig] = {"groups": {group}, "run": run}

    # (a) baseline: 4 networks x 3 datasets
    for network in NETWORKS:
        for train_file in (FULL, PRUNED_50, PRUNED_10):
            add("a", _run(train_file, **network))

    # (b) learning rate sweep, residual three-layer MLP, full dataset
    for lr in (0.001, 0.01, 1.0, 10, 100):
        add("b", _run(FULL, lr=lr, **RESIDUAL_MLP3))

    # (c) batch size sweep, same network/dataset
    for batch_size in (1, 4, 8, 16):
        add("c", _run(FULL, batch_size=batch_size, **RESIDUAL_MLP3))

    return list(runs.values())


def signature(run):
    """Normalized tuple identifying a config, matching how train.py logs it."""
    model = run["model"]
    is_mlp = model in ("mlp", "mlp3")
    is_mlp3 = model == "mlp3"
    return (
        MODEL_NAME[model],
        dataset_tag(run["train_file"]),
        str(run.get("nhidden", "")) if is_mlp else "",
        str(run.get("nhidden2", "")) if is_mlp3 else "",
        str(run.get("residual", False)) if is_mlp3 else "",
        str(float(run["lr"])),
        str(int(run["batch_size"])),
        str(int(run["epochs"])),
    )


def already_done():
    """Signatures of runs already present in results/experiment_log.csv."""
    if not os.path.exists(LOG_PATH):
        return set()
    with open(LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    done = set()
    for row in rows:
        try:
            done.add((
                row["model"], row["dataset"], row["nhidden"], row["nhidden2"],
                row["residual"], str(float(row["lr"])), str(int(row["batch_size"])),
                str(int(row["epochs"])),
            ))
        except (KeyError, ValueError):
            continue  # tolerate rows from an older log schema
    return done


def build_command(run):
    cmd = [
        sys.executable, "train.py",
        "--model", run["model"],
        "--ninput", "784", "--noutput", "10",
        "--epochs", str(run["epochs"]),
        "--lr", str(run["lr"]),
        "--batch_size", str(run["batch_size"]),
        "--train_file", run["train_file"],
        "--test_file", TEST_FILE,
        "--data_dir", DATA_DIR,
        "--plot",
    ]
    if run.get("nhidden") is not None:
        cmd += ["--nhidden", str(run["nhidden"])]
    if run.get("nhidden2") is not None:
        cmd += ["--nhidden2", str(run["nhidden2"])]
    if run.get("residual"):
        cmd.append("--residual")
    return cmd


def slug(index, groups, run):
    ds = dataset_tag(run["train_file"])
    bits = [f"{index:02d}", run["model"]]
    if run.get("nhidden2") is not None:
        bits.append(f"h{run.get('nhidden', '')}-h2{run['nhidden2']}")
    elif run.get("nhidden") is not None:
        bits.append(f"h{run['nhidden']}")
    if run.get("residual"):
        bits.append("res")
    bits += [ds, f"lr{run['lr']}", f"bs{run['batch_size']}", f"[{''.join(sorted(groups))}]"]
    return "_".join(str(b) for b in bits)


def run_one(cmd, log_path):
    """Stream subprocess output live to the console and to log_path. Returns exit code."""
    env = dict(os.environ, MPLBACKEND="Agg")
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Run the Step 4 experiment matrix")
    parser.add_argument("--dry-run", action="store_true", help="List planned runs without training")
    parser.add_argument("--only", type=str, default="a,b,c", help="Comma-separated experiment groups to include (a,b,c)")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of runs (for a smoke test)")
    parser.add_argument("--rerun", action="store_true", help="Ignore experiment_log.csv and redo every run")
    args = parser.parse_args()

    wanted_groups = set(args.only.split(","))
    matrix = [entry for entry in build_matrix() if entry["groups"] & wanted_groups]
    done = set() if args.rerun else already_done()

    os.makedirs(LOG_DIR, exist_ok=True)

    planned, skipped = [], []
    for entry in matrix:
        if signature(entry["run"]) in done:
            skipped.append(entry)
        else:
            planned.append(entry)
    if args.limit is not None:
        planned = planned[:args.limit]

    print(f"Matrix: {len(matrix)} unique configs across groups {sorted(wanted_groups)} "
          f"({len(skipped)} already logged, {len(planned)} to run)")
    for i, entry in enumerate(planned, 1):
        print(f"  [{i}/{len(planned)}] groups={sorted(entry['groups'])} {build_command(entry['run'])}")

    if args.dry_run:
        return

    failures = []
    sweep_start = time.time()
    for i, entry in enumerate(planned, 1):
        run, groups = entry["run"], entry["groups"]
        cmd = build_command(run)
        log_path = os.path.join(LOG_DIR, f"{slug(i, groups, run)}.log")

        print(f"\n{'='*60}\n[{i}/{len(planned)}] groups={sorted(groups)} -> {log_path}\n{' '.join(cmd)}\n{'='*60}")
        start = time.time()
        returncode = run_one(cmd, log_path)
        elapsed = time.time() - start

        if returncode != 0:
            print(f"*** Run {i} FAILED (exit {returncode}) after {elapsed:.1f}s - see {log_path} ***")
            failures.append((i, cmd, log_path))
        else:
            print(f"Run {i} finished in {elapsed:.1f}s")

    total_elapsed = time.time() - sweep_start
    print(f"\n{'='*60}\nSweep complete: {len(planned)} run, {len(skipped)} skipped (already logged), "
          f"{len(failures)} failed, {total_elapsed/60:.1f} min total")
    if failures:
        print("Failed runs:")
        for i, cmd, log_path in failures:
            print(f"  [{i}] {' '.join(cmd)}  (see {log_path})")
    print(f"Results: {LOG_PATH}")


if __name__ == "__main__":
    main()
