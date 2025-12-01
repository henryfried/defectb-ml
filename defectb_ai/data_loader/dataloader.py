from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from defectb_ai.data_loader.mydataset import MyTBDataSet, MyDosDFTDataSet

class DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule to handle loading, splitting, and preparing datasets for training,
    validation, testing, and prediction.
    """

    def __init__(self, batch_size, num_workers, train_data_path, pred_data_path, num_pdos_features, normalize_per_site, include_pdos_sum, train=True):
        """
        Initializes the DataModule with key parameters.

        Args:
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of workers for data loading.
            train (bool): If True, sets up data for training; if False, for predictions.
            train_data_path (str): Directory path for training data.
            pred_data_path (str): Directory path for prediction data.
            num_pdos_features (int): Number of LDOS channels.
            normalize_per_site (bool): Whether to scale PDOS channels separately.
            include_pdos_sum (bool): Whether to include the sum of all PDOS at each site.
            num_folds (int): Number of folds for cross-validation (if used).
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.train_data = train_data_path
        self.pred_data = pred_data_path
        self.num_pdos_features = num_pdos_features
        self.normalize_per_site = normalize_per_site
        self.include_pdos_sum = include_pdos_sum

        # Initialize the dataset for training or validation
        self.dataset = MyTBDataSet(f'{self.train_data}.npy', num_pdos_features, normalize_per_site, include_pdos_sum)

    def setup(self, stage=None):
        """
        Splits data into training, validation, and test sets if training.
        Otherwise, loads prediction dataset.
        """
        if self.train:
            # Define split sizes and randomly split the dataset
            train_size = int(len(self.dataset) * 0.8)
            val_size = (len(self.dataset) - train_size) // 2
            test_size = len(self.dataset) - train_size - val_size

            self.train_ds, self.val_ds, self.test_ds = random_split(
                self.dataset, [train_size, val_size, test_size]
            )
        else:
            # Load prediction dataset with preprocessing adjustments if not training
            self.pred_ds = MyDosDFTDataSet(
                f'{self.pred_data}',
                self.dataset.data_mins_maxs,
                self.num_pdos_features,
                self.normalize_per_site,
                self.include_pdos_sum,
            )

    def train_dataloader(self):
        # DataLoader for training data
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        # DataLoader for validation data
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        # DataLoader for test data
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        # DataLoader for prediction data
        return DataLoader(
            self.pred_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
