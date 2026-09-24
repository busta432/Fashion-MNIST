# Fashion-MNIST

Exploration of the Fashion-MNIST dataset with four architectures built from scratch with NumPy/autograd: Softmax Regression, a Two-Layer MLP, and a Three-Layer MLP (with and without residual connections).

## Files

- `nn.py` — model definitions (`LinearClassifier`, `TwoLayerMLP`, `ThreeLayerMLP`)
- `train.py` — data loading, training loop, evaluation, and plotting
- `experiment_utils.py` — experiment tracking helpers (unique run IDs, collision-safe filenames, CSV/JSON logging)
- `run_experiments.py` — runs the full assignment experiment matrix unattended (see below)

## Usage

```
python train.py --model linear --epochs 20 --lr 0.01 --batch_size 4 --plot
python train.py --model mlp --nhidden 30 --activation relu --epochs 20 --plot
python train.py --model mlp3 --nhidden 30 --nhidden2 30 --residual --activation relu --plot
```

Data is expected under `./data` as `fashion-mnist_train.csv.gz` / `fashion-mnist_test.csv.gz` (override with `--data_dir`, `--train_file`, `--test_file`).

## Experiment tracking

Every run is assigned a unique `run_id` from its model type and hyperparameters (e.g. `two-layer_mlp_relu_h30_lr0.01_bs4_ep20_seed42_20260924-153000`), so repeated experiments never overwrite each other's outputs. Results are written to `results/` (gitignored):

- `results/plots/<run_id>_training_history.png` — loss/accuracy curves, only saved with `--plot`
- `results/history/<run_id>.json` — full per-epoch training history
- `results/experiment_log.csv` — one row per run (hyperparameters + final metrics), appended automatically so runs can be compared across a whole sweep

## Running the full experiment sweep

`run_experiments.py` runs every config required by the assignment (12 baseline runs across 4 networks x 3 datasets, a 5-point learning-rate sweep, and a 4-point batch-size sweep on the residual three-layer MLP) — 19 unique runs after deduplicating configs shared across those groups. Each run's console output (epoch progress, timing, any numpy RuntimeWarnings) is streamed live and also saved under `results/logs/`.

```
python run_experiments.py --dry-run     # preview the 19 runs without training
python run_experiments.py               # run everything not already in experiment_log.csv
python run_experiments.py --only b      # just the learning-rate sweep (a, b, or c)
python run_experiments.py --limit 1     # smoke-test on a single run first
```

Re-running the script skips any config already logged in `results/experiment_log.csv`, so it's safe to stop and resume (use `--rerun` to force redoing everything). Budget several hours for the full sweep — the residual three-layer MLP on the full dataset takes ~10-15 min per run, and `--batch_size 1` runs are the slowest (~40+ min); start it and let it run in the background rather than waiting on it.
