# Fashion-MNIST

Exploration of the Fashion-MNIST dataset with four architectures built from scratch with NumPy/autograd: Softmax Regression, a Two-Layer MLP, and a Three-Layer MLP (with and without residual connections).

## Files

- `nn.py` — model definitions (`LinearClassifier`, `TwoLayerMLP`, `ThreeLayerMLP`)
- `train.py` — data loading, training loop, evaluation, and plotting
- `experiment_utils.py` — experiment tracking helpers (unique run IDs, collision-safe filenames, CSV/JSON logging)

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
