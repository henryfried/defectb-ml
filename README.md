# Defectb_ai


This repository contains a scaled dot-product self-attention based neural network for training and prediction of
tight-binding parameters based on the projected density of states. The main components of the project include
a training pipeline, a data loading module, a neural network, and prediction handling.

## Project Structure
Datahandling is done in the data_loader directory.
Neural network architectures are stored in the models directory.

Requirements

    NumPy: pip install numpy
    Pandas: pip install pandas
    PyTorch: pip install torch
    PyTorch-Lightning: pip install pytorch-lightning
    and dependencies.
    Other Utilities: Custom utilities such as nearest-neighbor calculations and stencil writing (defined in util).

## Installation

    pip install .

## Tests

    - run python train with desired parameters in the config_transf.py
    - run predict to obtain parametes




