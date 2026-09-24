"""
Helper functions for tracking Fashion-MNIST experiment results and avoiding
plot filename collisions.

Every run gets a unique run_id (hyperparameters + timestamp) so that repeated
experiments with the same model type never overwrite each other's plots or
history. Results are written under results/:
  results/plots/<run_id>.png     - loss/accuracy curves (path chosen by caller)
  results/history/<run_id>.json  - full per-epoch training history
  results/experiment_log.csv     - one row per run, for comparing experiments
"""

import os
import csv
import json
from datetime import datetime

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
HISTORY_DIR = os.path.join(RESULTS_DIR, "history")
LOG_PATH = os.path.join(RESULTS_DIR, "experiment_log.csv")

LOG_FIELDS = [
    "run_id", "timestamp", "model", "activation", "nhidden", "nhidden2",
    "residual", "lr", "batch_size", "epochs", "val_ratio", "seed",
    "best_epoch", "best_val_accuracy", "best_test_accuracy",
    "final_train_loss", "final_val_loss", "final_test_loss",
    "training_time_sec", "plot_path",
]


def ensure_dirs():
    """Create the results/plots/history directories if they don't exist yet."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)


def get_unique_filepath(filepath):
    """
    Return a filepath that doesn't collide with an existing file.
    If `filepath` already exists, append _1, _2, ... before the extension
    until a free name is found.
    """
    if not os.path.exists(filepath):
        return filepath

    base, ext = os.path.splitext(filepath)
    counter = 1
    while True:
        candidate = f"{base}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def build_run_id(model_name, args):
    """
    Build a short, descriptive, (near-certainly) unique identifier for a run,
    e.g. 'two-layer_mlp_relu_h30_lr0.01_bs4_ep20_seed42_20260924-153000'.
    Uses getattr with defaults so it works whether or not `args` has
    MLP-specific fields (activation, nhidden, residual, ...).
    """
    tag = model_name.lower().replace(" ", "_")
    parts = [tag]
    is_mlp = "mlp" in tag

    activation = getattr(args, "activation", None)
    if is_mlp and activation:
        parts.append(activation)

    nhidden = getattr(args, "nhidden", None)
    if is_mlp and nhidden:
        parts.append(f"h{nhidden}")

    nhidden2 = getattr(args, "nhidden2", None)
    if nhidden2 and tag.startswith("three"):
        parts.append(f"h2-{nhidden2}")

    if getattr(args, "residual", False):
        parts.append("residual")

    parts.append(f"lr{args.lr}")
    parts.append(f"bs{args.batch_size}")
    parts.append(f"ep{args.epochs}")
    parts.append(f"seed{args.seed}")
    parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))

    return "_".join(parts)


def save_history_json(training_history, run_id):
    """Dump the full per-epoch training history to results/history/<run_id>.json."""
    ensure_dirs()

    def to_plain(value):
        if isinstance(value, list):
            return [float(v) for v in value]
        if hasattr(value, "item"):
            return value.item()
        return value

    serializable = {k: to_plain(v) for k, v in training_history.items()}

    history_path = get_unique_filepath(os.path.join(HISTORY_DIR, f"{run_id}.json"))
    with open(history_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return history_path


def log_experiment(run_id, model_name, args, training_history, best_test_accuracy,
                    training_time, plot_path=""):
    """
    Append one row summarizing this run to results/experiment_log.csv,
    creating the file with a header on first use.
    """
    ensure_dirs()
    is_mlp = "mlp" in model_name.lower()
    is_mlp3 = model_name.lower().startswith("three")

    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "activation": getattr(args, "activation", "") if is_mlp else "",
        "nhidden": getattr(args, "nhidden", "") if is_mlp else "",
        "nhidden2": getattr(args, "nhidden2", "") if is_mlp3 else "",
        "residual": getattr(args, "residual", False) if is_mlp3 else "",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "best_epoch": training_history["best_epoch"] + 1,
        "best_val_accuracy": round(float(training_history["best_val_accuracy"]), 4),
        "best_test_accuracy": round(float(best_test_accuracy), 4),
        "final_train_loss": round(float(training_history["train_losses"][-1]), 4),
        "final_val_loss": round(float(training_history["val_losses"][-1]), 4),
        "final_test_loss": round(float(training_history["test_losses"][-1]), 4),
        "training_time_sec": round(training_time, 2),
        "plot_path": plot_path,
    }

    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, restval="")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return LOG_PATH
