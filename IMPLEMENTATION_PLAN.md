# 2802ICT Assignment 2 — Fill in the NotImplementedError placeholders

## Context

`nn.py` and `train.py` are a supplied framework with 7 `raise NotImplementedError`
placeholders. The assignment (Task 1, 30 marks) requires three verified architectures —
Softmax regression 784-10, two-layer MLP 784-30-10 with sigmoid, and a three-layer MLP
784-30-30-10 with a residual connection from hidden layer 1 to hidden layer 2 — trained by
mini-batch gradient descent using `autograd` for automatic differentiation. The spec says
to insert code **only** at the placeholders and to vary behaviour through command-line
arguments rather than rewriting the framework.

You want the code snippets to type in yourself, so everything below is copy-paste ready
with the surrounding context needed to locate each spot.

**Two decisions you made:**
- Both sigmoid and ReLU available, selected by a new `--activation` CLI flag (default
  `sigmoid`).
- Experiments (b) and (c) run on the three-layer MLP **with** residual connections.

**One caveat on the `--activation` flag.** It is *additive* — it does not replace or alter
any of the arguments the spec lists, and with the default `sigmoid` every run behaves
exactly as the template intends, so the Appendix 3 output is unchanged. It is the only
edit below that goes beyond a `NotImplementedError` marker. Keep the **two-layer MLP
reported with sigmoid**, since requirement (2) names sigmoid explicitly; ReLU is the free
choice for the three-layer net under "Choose your activation functions", and is a good
source of an extra comparison row in the report.

---

## Step 0 — Environment setup

`numpy 2.4.4` and `matplotlib 3.11.1` are present on Python 3.14.3. **`autograd` is not
installed.**

```powershell
python -m pip install autograd
```

Extract the data (the zip already contains a top-level `data/` folder, so unzip into the
assignment directory and `--data_dir ./data` resolves correctly):

```powershell
Expand-Archive -Path ".\fashion-mnist_data.zip" -DestinationPath "." -Force
Get-ChildItem .\data\*.gz
```

Expected: `fashion-mnist_train.csv.gz`, `fashion-mnist_test.csv.gz`,
`fashion-mnist_train_pruned_50percent.csv.gz`, `fashion-mnist_train_pruned_10percent.csv.gz`.

---

## Step 1 — `nn.py`

### 1a. `LinearClassifier.forward` — replace the `raise NotImplementedError` at line 52

```python
        # Linear transformation: (batch, n_input) @ (n_input, n_output) -> (batch, n_output).
        # anp.dot (not np.dot) so autograd records this op on the computational graph and
        # can backpropagate through it. b has shape (n_output,) and NumPy broadcasting
        # adds it to every row of the batch.
        return anp.dot(X, self.W) + self.b
```

No softmax here — `forward` returns raw logits; `predict` and `cross_entropy_loss` apply
softmax themselves. That is what makes this network a *softmax regression*.

---

### 1b. `TwoLayerMLP.__init__` — add the activation option

Change the signature (line 91) and its docstring, then store the choice:

```python
    def __init__(self, n_input, n_hidden, n_output, activation='sigmoid'):
        """
        Initialize MLP parameters
        Args:
            n_input: Number of input features (784)
            n_hidden: Number of hidden units (30)
            n_output: Number of output classes (10)
            activation: Hidden-layer activation, 'sigmoid' or 'relu'
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.activation = activation          # <-- add this line
```

Leave the weight/bias initialisation below it untouched.

### 1c. `TwoLayerMLP` — add two helper methods

Insert directly **after** the existing `sigmoid` method (after line 114):

```python
    def relu(self, z):
        """ReLU activation function"""
        return anp.maximum(0, z)

    def activate(self, z):
        """Apply the hidden-layer activation selected by --activation"""
        if self.activation == 'relu':
            return self.relu(z)
        return self.sigmoid(z)
```

### 1d. `TwoLayerMLP.forward` — replace **both** `raise NotImplementedError` (lines 150, 167)

First one:

```python
        # Layer 1: affine map then non-linearity.
        # X (batch, 784) @ W1 (784, 30) -> z1 (batch, 30); b1 broadcasts over the batch.
        z1 = anp.dot(X, self.W1) + self.b1
        h1 = self.activate(z1)   # hidden representation, shape (batch, 30)
```

Second one:

```python
        # Layer 2: affine map only. No activation here — these are raw logits, and
        # softmax is applied inside cross_entropy_loss / predict. Applying softmax twice
        # would flatten the gradients.
        z2 = anp.dot(h1, self.W2) + self.b2   # (batch, 10)
```

Keep the existing `return z2` at the end. Both `raise` lines must be **deleted**, not just
commented out around — a `raise` aborts before anything below it runs.

### 1e. `TwoLayerMLP.get_params` — replace the `raise NotImplementedError` at line 196

```python
        # autograd's grad() differentiates w.r.t. a single array argument, so every weight
        # and bias is flattened into one 1-D vector. The order here MUST match the
        # unpacking order in set_params below: W1, b1, W2, b2.
        return anp.concatenate([self.W1.flatten(), self.b1.flatten(),
                                self.W2.flatten(), self.b2.flatten()])
```

`set_params` already slices in exactly that order — do not reorder it.

---

### 1f. `ThreeLayerMLP.__init__` — add the activation option

Signature (line 223) and docstring:

```python
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, use_residual=False,
                 activation='sigmoid'):
```

Add to the docstring Args list:

```
            activation: Hidden-layer activation, 'sigmoid' or 'relu'
```

and store it alongside the other attributes (after `self.use_residual = use_residual`):

```python
        self.activation = activation
```

`activation` goes **last** so the existing positional call
`ThreeLayerMLP(ninput, nhidden, nhidden2, noutput, residual)` keeps working.

### 1g. `ThreeLayerMLP` — add the same two helpers

Insert after its `sigmoid` method (after line 258) — identical to 1c:

```python
    def relu(self, z):
        """ReLU activation function"""
        return anp.maximum(0, z)

    def activate(self, z):
        """Apply the hidden-layer activation selected by --activation"""
        if self.activation == 'relu':
            return self.relu(z)
        return self.sigmoid(z)
```

### 1h. `ThreeLayerMLP.forward` — replace **all four** `raise NotImplementedError`

Lines 290, 303, the `if/else` pair at 317-320, and line 325. Final body:

```python
        # Hidden layer 1: X (batch, 784) @ W1 (784, 30) -> (batch, 30)
        z1 = anp.dot(X, self.W1) + self.b1
        h1 = self.activate(z1)

        # Hidden layer 2: h1 (batch, 30) @ W2 (30, 30) -> (batch, 30)
        z2 = anp.dot(h1, self.W2) + self.b2
        h2_raw = self.activate(z2)

        # Residual (skip) connection from hidden layer 1 to hidden layer 2.
        # h2 = F(h1) + h1 — the identity shortcut gives the gradient a path that bypasses
        # the second layer's weights, so it does not get attenuated by that layer's
        # activation derivative. This is what stops the vanishing-gradient problem that
        # stacked sigmoids cause (sigmoid' <= 0.25 everywhere, so gradients shrink by at
        # least 4x per layer without the shortcut).
        # __init__ already guarantees n_hidden1 == n_hidden2, so the shapes line up and no
        # projection matrix is needed.
        if self.use_residual:
            h2 = h2_raw + h1   # residual connection
        else:
            h2 = h2_raw        # plain feed-forward

        # Output layer: h2 (batch, 30) @ W3 (30, 10) -> (batch, 10) logits
        z3 = anp.dot(h2, self.W3) + self.b3
        return z3
```

The trailing `return z3` already in the template covers the last line — keep only one.

### 1i. `ThreeLayerMLP.get_params` — replace the `raise NotImplementedError` at line 348

```python
        # Flatten all six parameter tensors into one vector for autograd.
        # Order MUST match set_params: W1, b1, W2, b2, W3, b3.
        return anp.concatenate([self.W1.flatten(), self.b1.flatten(),
                                self.W2.flatten(), self.b2.flatten(),
                                self.W3.flatten(), self.b3.flatten()])
```

---

## Step 2 — `train.py`

### 2a. `cross_entropy_loss` — replace the `raise NotImplementedError` at line 160

`probs = softmax(y_pred)` is already computed above it. Add:

```python
    # Cross-entropy:  L = -(1/N) * sum_i sum_c  y_true[i,c] * log(probs[i,c])
    # Because y_true is one-hot, the inner sum picks out the log-probability of the single
    # correct class for each sample; the mean averages over the mini-batch.
    # The 1e-12 floor keeps log() finite if a probability underflows to exactly 0, which
    # happens once logits get large (e.g. the lr=10 / lr=100 runs in experiment (b)).
    # It is far below float precision on a healthy run, so it does not change the reported
    # loss, it just stops -inf from poisoning the gradient before divergence is visible.
    log_probs = anp.log(probs + 1e-12)
    loss = -anp.mean(anp.sum(y_true * log_probs, axis=1))
```

Keep the existing `return loss` below it.

### 2b. Parameter update — replace the `raise NotImplementedError` at line 274

```python
            # Gradient descent step: move each parameter a small distance DOWN the
            # gradient, i.e. opposite to the direction of steepest increase of the loss.
            # `gradients` comes straight from autograd's reverse-mode pass over the whole
            # network, so this single vectorised line updates every layer's weights and
            # biases at once.
            new_params = params - learning_rate * gradients
```

Keep the existing `model.set_params(new_params)` on the next line.

### 2c. Add the `--activation` argument

Insert after the `--residual` argument (after line 415):

```python
    parser.add_argument('--activation', type=str, default='sigmoid',
                       choices=['sigmoid', 'relu'],
                       help='Hidden-layer activation function for mlp/mlp3 '
                            '(default: sigmoid)')
```

Add one line to the config banner, **inside** the existing `if args.model in ['mlp', 'mlp3']:`
block at line 443 so that the `--model linear` output still matches Appendix 3 exactly:

```python
    if args.model in ['mlp', 'mlp3']:
        print(f"Hidden size 1: {args.nhidden}")
        print(f"Activation: {args.activation}")          # <-- add this line
```

### 2d. Pass the activation through to the constructors (lines 471-476)

```python
    elif args.model == 'mlp':
        model = TwoLayerMLP(args.ninput, args.nhidden, args.noutput,
                            activation=args.activation)
        model_name = "Two-Layer MLP"
    elif args.model == 'mlp3':
        model = ThreeLayerMLP(args.ninput, args.nhidden, args.nhidden2, args.noutput,
                              args.residual, activation=args.activation)
        model_name = "Three-Layer MLP"
```

`LinearClassifier` is unchanged — it has no hidden layer to activate.

---

## Step 3 — Verification

**Fast smoke test first** (2 epochs on the 10% set, ~1 min) before committing to any long
run. Confirms loss decreases, accuracy climbs above 10%, and no shape errors:

```powershell
python train.py --model linear --ninput 784 --noutput 10 --epochs 2 --lr 0.01 --batch_size 64 --train_file fashion-mnist_train_pruned_10percent.csv.gz --test_file fashion-mnist_test.csv.gz --data_dir ./data
python train.py --model mlp --ninput 784 --nhidden 30 --noutput 10 --epochs 2 --lr 0.01 --batch_size 64 --train_file fashion-mnist_train_pruned_10percent.csv.gz --test_file fashion-mnist_test.csv.gz --data_dir ./data
python train.py --model mlp3 --ninput 784 --nhidden 30 --nhidden2 30 --noutput 10 --residual --epochs 2 --lr 0.01 --batch_size 64 --train_file fashion-mnist_train_pruned_10percent.csv.gz --test_file fashion-mnist_test.csv.gz --data_dir ./data
python train.py --model mlp3 --ninput 784 --nhidden 30 --nhidden2 30 --noutput 10 --residual --activation relu --epochs 2 --lr 0.01 --batch_size 64 --train_file fashion-mnist_train_pruned_10percent.csv.gz --test_file fashion-mnist_test.csv.gz --data_dir ./data
```

**Two independent correctness checks** worth running once each:

1. *Residual is actually wired in.* Run `mlp3` twice with the same `--seed`, once with
   `--residual` and once without. The printed losses must differ from epoch 1. Identical
   numbers mean the `if self.use_residual:` branch is not taking effect.
2. *Parameter packing round-trips.* In a Python shell:
   ```python
   from nn import ThreeLayerMLP
   import numpy as np
   m = ThreeLayerMLP(784, 30, 30, 10, True)
   p = m.get_params(); m.set_params(p)
   assert np.allclose(p, m.get_params())   # get/set must be exact inverses
   assert p.size == 784*30 + 30 + 30*30 + 30 + 30*10 + 10   # == 24700
   ```
   A mismatched order in `get_params` still *runs* but silently scrambles the weights every
   optimisation step — this assert is the only thing that catches it.

**Baseline reproduction.** `--model linear` with all defaults should land near the
Appendix 3 figures (~85% test accuracy, best val ~84.9%). Exact digits will differ —
`np.random.seed` is only set inside `create_validation_split`, so the weight
initialisation is not seeded — but being within a percent or so confirms correctness.

---

## Step 4 — Experiments

Set the matplotlib backend once per shell so `plt.show()` does not block on a GUI window
(the PNG is still written by `savefig` before the `show` call):

```powershell
$env:MPLBACKEND = "Agg"
```

**Filename collision warning:** `plot_training_history` names the PNG from `model_name`
only, so every `mlp3` run overwrites `three-layer_mlp_training_history.png`. Rename the
file after each run before starting the next, e.g.
`Rename-Item .\three-layer_mlp_training_history.png "mlp3_res_full_lr0.01_bs4.png"`.

### (a) Baseline — 4 networks x 3 datasets = 12 runs
Defaults throughout: `--epochs 20 --lr 0.01 --batch_size 4`.

| Network | `--model` flags |
|---|---|
| Softmax regression | `--model linear --ninput 784 --noutput 10` |
| Two-layer MLP | `--model mlp --ninput 784 --nhidden 30 --noutput 10` |
| Three-layer, no residual | `--model mlp3 --ninput 784 --nhidden 30 --nhidden2 30 --noutput 10` |
| Three-layer, residual | `--model mlp3 --ninput 784 --nhidden 30 --nhidden2 30 --noutput 10 --residual` |

Datasets via `--train_file`: `fashion-mnist_train.csv.gz`,
`fashion-mnist_train_pruned_50percent.csv.gz`, `fashion-mnist_train_pruned_10percent.csv.gz`.
`--test_file fashion-mnist_test.csv.gz --data_dir ./data` for all.

### (b) Learning rate — 5 runs
Three-layer MLP **with** `--residual`, full dataset, `--batch_size 4 --epochs 20`, sweeping
`--lr 0.001 0.01 1.0 10 100`. Expect lr=100 (and likely lr=10) to overflow inside
`softmax`'s `anp.exp` and produce `nan` losses with ~10% accuracy — that *is* the
"totally unstable" answer the question asks for. Record the RuntimeWarnings as evidence
rather than suppressing them.

### (c) Mini-batch size — 4 runs
Same network and dataset, `--lr 0.01 --epochs 20`, sweeping `--batch_size 1 4 8 16`.
`--batch_size 4` is shared with (a) and need not be rerun. Batch size 1 will be by far the
slowest (4x the gradient calls of batch 4) — that is the expected answer to "which is
slowest to train", and the per-epoch `Time:` field in the output is your evidence.

### Runtime expectations
`train.py` calls `grad()` once per mini-batch. On the full set at batch size 4 that is
12,000 autograd passes per epoch, 240,000 over 20 epochs. The PDF's own timing is ~86 s
total for `linear`; `mlp3` is several times heavier, so budget **~10-15 min per full-dataset
mlp3 run** and **40+ min for the `--batch_size 1` run**. Add ~1-2 min per run just for
`np.loadtxt` to parse the gzipped CSV. Total across all ~20 runs is a few hours — start
the long ones in the background and capture stdout with `| Tee-Object -FilePath log.txt`
so you have the numbers and screenshots for the report without rerunning anything.

### Optional: `run_experiments.ps1`
A loop that runs the matrix, renames each PNG, and tees each log to a uniquely named file
would save a lot of manual bookkeeping. It is a new file, not a modification to `nn.py` or
`train.py`, so it does not conflict with the "do not overwrite the framework" instruction.
Say the word and I will write it.

---

## Files touched
- [nn.py](nn.py) — 5 placeholders + `activation` plumbing on both MLP classes
- [train.py](train.py) — 2 placeholders + `--activation` argument and its 2 call sites
