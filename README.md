# DefecTB-ML

This repository contains a scaled dot-product self-attention based neural network for training and prediction of
tight-binding parameters based on the projected density of states. The main components of the project include
a training pipeline, a data loading module, a neural network, and prediction handling.

## Project Structure
Datahandling is done in the data_loader directory.
Neural network architectures are stored in the models directory.

Dependencies are declared in `pyproject.toml` and will be installed automatically (torch, pytorch-lightning, numpy, pandas).

Package import name: `defectb_ml`.

## Installation

Requirements
- Python 3.8+
- NumPy
- Pandas
- PyTorch
- PyTorch Lightning

Install from source
```bash
python -m pip install -e .
```

For TensorBoard logging support (used by the training scripts):

```bash
python3 -m pip install tensorboard
```

## Quickstart

Train with the example config
```bash
python examples/conv_attention/train_Conv1DSelfAtten.py
```

Run prediction
```bash
python examples/conv_attention/pred_Conv1DSelfAtten.py
```

Configuration
- Edit `examples/conv_attention/config_transf.py` to set paths, hyperparameters, and model options.
- Example datasets live in `examples/data_sets`.

## Project Structure
- `defectb_ml/data_loader`: dataset and datamodule logic.
- `defectb_ml/models`: network and callback implementations.
- `examples`: training/prediction scripts and example datasets.

## License
See `LICENCE`.
