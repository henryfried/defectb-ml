# defectb-ml

defectb-ml provides a scaled dot-product self-attention network for predicting tight-binding parameters from projected density of states (PDOS). The repo includes data loaders, model definitions, and runnable training/prediction scripts.

Package import name: `defectb_ml`.

## Highlights
- Self-attention architecture for tight-binding parameter regression.
- Modular data loading in `defectb_ml/data_loader`.
- Model definitions in `defectb_ml/models`.
- Training and prediction scripts in `examples/conv_attention`.

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
