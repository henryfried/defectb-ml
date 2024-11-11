import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from defectb_ai.models.callbacks import LearningRatePrinter, MyPrintingCallback
from defectb_ai.models.network import Conv1DSelfAtten

from defectb_ai.data_loader.dataloader import DataModule
import config_transf as config

torch.set_float32_matmul_precision("medium")  # Adjust matrix multiplication precision to avoid precision warnings

'''
Main prediction file. Hyperparameters are defined in a separate configuration file (config_transf.py).
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
  - Prediction: Requires loading both the dataset and prediction data, with model_trained as the NN.
  - Profiler: Can be set to default values (e.g., 'simple', 'advanced') or customized in the profiler.
  - Logger: Currently set to use TensorBoard for tracking metrics.
  - Hyperparameters: Set in the config file.
'''

if __name__ == "__main__":
    #     logger = Tenmy_nn_configrdLogger("tb_logs", name=f"{my_nn_config.TRAIN_SET}")
    #     profiler = PyTorchProfiler(
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(f"tb_logs/profiler/{my_nn_config.TRAIN_SET}"),
    #         schedule=torch.profiler.schedule(skip_first=0, wait=2, warmup=2, active=2),
    #     )
    # Initialize DataModule for training and prediction data handling.
    dm = DataModule(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_data_path=config.TRAIN_SET,
        pred_data_path=config.PRED_DATA_DIR,
        num_pdos_features=config.LDOS_NUM,
        normalize_per_site=config.SEP_DOS_SCALE,
        include_pdos_sum=config.INCLUDE_SUM,
        train=False,  # Set to True for training mode, False for predictions
    )

    # Load the pre-trained Transformer model with parameters from the config file.
    model = Conv1DSelfAtten.load_from_checkpoint(
        checkpoint_path=config.TRAINED_MODEL_DIR,
        num_heads=config.NUM_HEADS,
        head_dim=config.HEAD_DIM,
        conv_layer=config.CONV_LAYER,
        input_size=config.INPUT_SIZE,
        output_dims=config.OUTPUT_DIMS,
        target_size=config.TARGET_SIZE,
        learning_rate=config.LEARNING_RATE,
        dropout=config.DROPOUT,
        alpha=config.ALPHA,
        dr=config.DECAY_RATE,
        train=False,  # Set to False for prediction mode
    )

    # Initialize the PyTorch Lightning Trainer with configuration settings
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
    )


    def predict():
        """
        Run the model prediction on the dataset, rescale the outputs, and save the results.
        """
        # Run predictions and retrieve output
        results = np.array(trainer.predict(model, dm)[0]).squeeze()
        print(results)

        # Extract min and max values for rescaling output to original value range
        min_max_out = dm.pred_ds.data_trained_min_max_values[-config.TARGET_SIZE:]

        # Rescale results to original scale
        results_rescaled = [
            result * (min_max_out[i][1] - min_max_out[i][0]) + min_max_out[i][0]
            for i, result in enumerate(results)
        ]

        # Save rescaled results to a file
        print(results_rescaled)
        np.savetxt(f'results_c_n_{config.VERSION_NUM}.txt', results_rescaled)


    # Run the prediction function
    predict()
