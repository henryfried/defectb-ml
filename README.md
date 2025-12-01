# Defectb_ai


This repository contains a scaled dot-product self-attention based neural network for training and prediction of
tight-binding parameters based on the projected density of states. The main components of the project include
a training pipeline, a data loading module, a neural network, and prediction handling.

## Project Structure
Datahandling is done in the data_loader directory.
Neural network architectures are stored in the models directory.

Dependencies are declared in `pyproject.toml` and will be installed automatically (torch, pytorch-lightning, numpy, pandas).

## Installation

Use a Python 3.9+ environment (virtualenv/conda recommended).

Standard install from the repo root:

```bash
python3 -m pip install .
```

For editable development with tests (includes pytest):

```bash
python3 -m pip install -e .[dev]
```

For TensorBoard logging support (used by the training scripts):

```bash
python3 -m pip install tensorboard
```

## Tests

```bash
python3 -m pytest
```

## Examples

- Train: adjust hyperparameters in `defectb_ai/examples/conv_attention/config_transf.py`, then run the training script (e.g., `python defectb_ai/examples/conv_attention/train_Conv1DSelfAtten.py`).
- Predict: load a checkpoint with the matching config and run the prediction script (e.g., `python defectb_ai/examples/conv_attention/pred_Conv1DSelfAtten.py`).
