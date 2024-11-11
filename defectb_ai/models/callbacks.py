from pytorch_lightning.callbacks import EarlyStopping, Callback

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

class LearningRatePrinter(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, trainer, pl_module):
        # Access the learning rate from the optimizer
        lr = trainer.optimizers[0].param_groups[0]['lr']
        print(f'Learning Rate: {lr}')