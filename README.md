# Clinical Risk Assessment System

Infection risk prediction for cirrhotic patients based on the **NAM-TabR**
hybrid neural architecture.

This repository contains the core code only: the model framework, the
training scripts, and the web interfaces. 

## Model Architecture

NAM-TabR combines gated residual terms on a linear foundation:

- **Linear baseline** — a global linear layer over all features plus a
  dedicated linear pathway for the prespecified core features.
- **NAM** — one small MLP per feature; zero-meaned per-feature outputs are
  summed into additive, interpretable contributions.
- **Core-feature interaction network** — a compact MLP over the core features.
- **TabR (Siamese retrieval)** — dynamic pools of positive AND negative
  training anchors; each sample is scored by core-weighted cosine similarity
  to its nearest neighbours in each pool, and the positive-versus-negative
  similarity difference becomes a bonus logit.
- Each nonlinear component has its own learnable gate (α / β / γ), initialised
  near zero so the model defaults to the interpretable linear prediction and
  only activates nonlinear terms when they reduce the loss. During an initial
  linear-warmup phase the nonlinear modules are frozen.
- Training uses focal loss (label smoothing) + class-balanced sampling.

`src/models/nam_tabr.py` implements the framework (`LRTabREnhanced`,
`LRTabRModel`, `NAMTabRClassifier`).

## Project Structure

```
├── app_gradio_sbp.py      # SBP risk-assessment web interface (port 7860)
├── app_gradio_sbe.py      # SBE risk-assessment web interface (port 7861)
├── train_sbp_model.py     # SBP training script
├── train_sbe_model.py     # SBE training script
├── src/
│   ├── data/              # data loading
│   └── models/            # NAM-TabR framework
```

## Requirements

Python 3.9+, then:

## Usage

### 1. Prepare your data

Place three CSV files per fluid in `data/sbp/` and `data/sbe/`:

```
data/<fluid>/train.csv
data/<fluid>/internal_test.csv
data/<fluid>/external_test.csv
```

To adapt the code to your own feature set, edit `FEATURES` / `CORE_FEATURES` / `LABEL_COL` at the
top of the training scripts — nothing else is column-dependent.

### 2. Train

```bash
python train_sbp_model.py   # -> checkpoints/sbp/model.pt + outputs/sbp/
python train_sbe_model.py   # -> checkpoints/sbe/model.pt + outputs/sbe/
```

### 3. Launch the web interfaces

```bash
python app_gradio_sbp.py    # http://localhost:7860
python app_gradio_sbe.py    # http://localhost:7861
```

## Output Description

- Risk level: HIGH RISK / LOW RISK (Youden-J decision threshold from training)
- Feature contribution chart: per-feature gated NAM contributions to the
  log-odds

## Notes

- For research purposes only; not a medical device.
- The web interfaces include a few illustrative example inputs for the
  "Load a Case" button — replace them with examples from your own cohort.
