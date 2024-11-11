import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from defectb_ai.models.callbacks import LearningRatePrinter, MyPrintingCallback
from defectb_ai.models.network import Conv1DSelfAtten
from defectb_ai.data_loader.dataloader import DataModule
import config_transf as config
from pytorch_lightning.callbacks import LearningRateMonitor  # , ModelSummary
'''
Main training file. Hyperparameters are defined in a separate configuration file (config_transf.py).
To make modifications to hyperparameters, update them directly in the config file.

- Logger: Manages logging of training runs and saves logs to the directory `tb_logs`.

- Profiler: Monitors and identifies performance bottlenecks (e.g., memory usage, data loading time) across different 
  stages such as training and validation.

- Model: Loads the neural network model with architecture and hyperparameters defined in the models directory. Any 
  architectural changes should be made there.

- dm (DataModule): Loads and prepares the dataset for training or prediction. 
  For training, validation, and testing, no prediction dataset is needed (dm.pred_ds can be ignored). 
  For predictions, both training and prediction datasets are required for rescaling purposes.

- model_trained: Loads a pre-trained model from a checkpoint for prediction tasks.

- trainer: Initializes the PyTorch Lightning Trainer to coordinate model training or predictions.
  - Training: Only requires loading the primary dataset and neural network (NN).
  - Prediction: Requires loading both the dataset and prediction data, with model_trained as the NN.
  - Profiler: Can be set to default values (e.g., 'simple', 'advanced') or customized in the profiler.
  - Logger: Currently set to use TensorBoard for tracking metrics.
  - Hyperparameters: Set in the config file.

'''

# Configure Learning Rate monitoring
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Main training file, with hyperparameters set in config file

torch.set_float32_matmul_precision("medium")  # Set precision level to avoid PyTorch Lightning warnings

if __name__ == "__main__":

    # Initialize TensorBoard logger for logging training progress
    logger = TensorBoardLogger("tb_logs", name=f"{config.TRAIN_SET_NAME}")

    # Set up profiling to track performance bottlenecks and memory usage
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"tb_logs/profiler/lin/{config.TRAIN_SET_NAME}"),
        schedule=torch.profiler.schedule(skip_first=0, wait=1, warmup=2, active=3),
    )

    # Define Transformer model architecture with config parameters
    model = Conv1DSelfAtten(
        input_size=config.INPUT_SIZE,
        num_heads=config.NUM_HEADS,
        head_dim=config.HEAD_DIM,
        conv_layer=config.CONV_LAYER,
        output_dims=config.OUTPUT_DIMS,
        target_size=config.TARGET_SIZE,
        learning_rate=config.LEARNING_RATE,
        dropout=config.DROPOUT,
        alpha=config.ALPHA,
        dr=config.DECAY_RATE,
    )

    # Load data module for training; 'train_data' required, 'pred_data' optional for predictions
    dm = DataModule(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_data_path=config.TRAIN_SET,
        pred_data_path=None,  # Optional for prediction data loading
        num_pdos_features=config.LDOS_NUM,
        normalize_per_site=config.SEP_DOS_SCALE,
        include_pdos_sum=config.INCLUDE_SUM,
    )

    # Initialize PyTorch Lightning Trainer with core settings
    trainer = pl.Trainer(
        profiler=profiler,
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
    )

    # Start training
    trainer.fit(model, dm)

    # Display data min-max values after training
    print(dm.dataset.data_mins_maxs)

    # Run validation and testing
    trainer.validate(model, dm)
    trainer.test(model, dm)
