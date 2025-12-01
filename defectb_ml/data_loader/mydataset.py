import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MyTBDataSet(Dataset):
    """
    Custom PyTorch Dataset for loading and processing projected density of states (PDOS)
    and parameters (labels) for a tight-binding dataset.
    """

    def __init__(self, file_name: str, num_pdos_features: int, normalize_per_site: bool, include_pdos_sum: bool):
        """
        Initialize the dataset with the data file and preprocessing parameters.

        Parameters:
            file_name (str): Path to the .npy file containing the dataset.
            num_pdos_features (int): Number of local density of states (LDOS) features per sample.
            normalize_per_site (bool): If True, scales PDOS for each site separately.
            include_pdos_sum (bool): If True, includes the sum of all PDOS at each site as an additional feature.
        """
        super().__init__()

        # Load the dataset and transpose it to create a DataFrame
        self.data_df = pd.DataFrame(np.load(file_name, allow_pickle=True).item()).T
        self.data_mins_maxs = []  # Store min and max values for scaling
        self.num_pdos_features = num_pdos_features
        self.normalize_per_site = normalize_per_site
        self.include_pdos_sum = include_pdos_sum
        self.features = None  # Will hold input features
        self.target_labels = None  # Will hold labels
        # Parse and normalize data
        self.read_parse_data()

    def read_parse_data(self):
        """
        Parse and preprocess PDOS and parameter data. Normalizes the features and labels.
        """
        # Extract and stack PDOS (local_dos) for all samples
        pdos_features = np.vstack(self.data_df['local_dos'], dtype=np.float32)

        # Extract labels and separate each parameter into individual arrays
        parameters_label = np.array(self.data_df['label'].values.tolist())
        label_list = [parameters_label[:, i] for i in range(parameters_label.shape[1])]

        # Process PDOS data according to scaling options
        feature_list = self.dos_extraction(pdos_features)

        # Scale PDOS and labels
        features_torch_scaled = self.scale(feature_list)
        label_torch_scaled = self.scale(label_list)

        # Concatenate scaled PDOS data along the feature dimension
        self.features = torch.cat(features_torch_scaled, dim=1)

        # Stack scaled labels into columns
        self.target_labels = torch.column_stack(label_torch_scaled)

    def dos_extraction(self, pdos_features):
        """
        Extract and organize PDOS data with options for separate scaling and sum inclusion.

        Parameters:
            pdos_features (np.ndarray): Array of PDOS data for each sample.

        Returns:
            List[np.ndarray]: List of processed PDOS arrays.
        """
        feature_list = []
        # Option 1: Separate scaling with sum inclusion
        if self.normalize_per_site and self.include_pdos_sum:
            print(
                '           -------------------------------------------\n'
                '           PDOS normalized separately, sum included\n'
                f'                    num PDOS {self.num_pdos_features}         \n'
                '           -------------------------------------------\n')
            pdos_split = np.hsplit(pdos_features, self.num_pdos_features)
            pdos_sum = np.sum(pdos_split[:self.num_pdos_features], axis=0)
            feature_list.extend(pdos_split)
            feature_list.append(pdos_sum)

        # Option 2: Separate scaling without sum inclusion
        elif self.normalize_per_site and not self.include_pdos_sum:
            print(
                '           -------------------------------------------\n'
                '                   PDOS normalized separately\n'
                '           -------------------------------------------\n')
            feature_list = np.hsplit(pdos_features, self.num_pdos_features)

        # Option 3: Unified scaling with sum inclusion
        elif not self.normalize_per_site and self.include_pdos_sum:
            raise NotImplementedError('The option to  normalize PDOS together and simultaneously  include the sum is '
                                      'not implemented yet!')

        # Option 4: Unified scaling without sum inclusion
        else:
            print(
                '           -------------------------------------------\n'
                '                   PDOS normalized together\n'
                '           -------------------------------------------\n')
            feature_list = [pdos_features]
        return feature_list

    def scale(self, list_arr):
        """
        Scale each array in the list to the range [0, 1].

        Parameters:
            list_arr (List[np.ndarray]): List of arrays to scale.

        Returns:
            List[torch.FloatTensor]: List of scaled tensors.
        """
        torch_list = []
        for array in list_arr:
            array_min = np.min(array)
            array_max = np.max(array)
            denom = array_max - array_min
            if denom == 0:
                # Avoid divide-by-zero when the feature has no variation
                scaled_array = np.zeros_like(array, dtype=np.float32)
            else:
                scaled_array = (array - array_min) / denom
            torch_list.append(torch.as_tensor(scaled_array, dtype=torch.float32))
            # Store min and max for each array for possible later use
            self.data_mins_maxs.append([array_min, array_max])
        return torch_list

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.target_labels)

    def __getitem__(self, idx):
        # Retrieve a single sample with its features (x_data) and labels (y_data)
        return self.features[idx], self.target_labels[idx]


def min_max_scaler(arr, min_max):
    """
    Scale an array to the range [0, 1] using predefined min and max values.

    Parameters:
        arr (np.ndarray): Array to scale.
        min_max (Tuple[float, float]): Min and max values for scaling.

    Returns:
        np.ndarray: Scaled array.
    """
    denom = min_max[1] - min_max[0]
    if denom == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_max[0]) / denom


class MyDosDFTDataSet(Dataset):
    """
    Custom PyTorch Dataset for loading and scaling DFT-based PDOS data
    according to precomputed min and max values from training data.
    """

    def __init__(self, file_name, min_max_values, num_pdos_features, normalize_per_site, include_pdos_sum):
        """
        Initialize the DFT dataset with file and scaling options.

        Parameters:
            file_name (str): Path to the file containing DFT PDOS data.
            min_max_values (List[Tuple[float, float]]): Min and max values from training set.
            num_pdos_features (int): Number of local/projected density of states (PDOS) features per sample
            normalize_per_site (bool): If True, scales PDOS for each site separately.
            include_pdos_sum (bool): If True, includes the sum of all PDOS at each site.
        """
        print(file_name)
        # Load only the necessary columns, from column 1 up to `num_pdos_features + 1`
        self.dft_feature = np.loadtxt(file_name, usecols=range(1, num_pdos_features + 1)).T
        print(self.dft_feature.shape)
        self.data_trained_min_max_values = min_max_values
        self.num_pdos_features = num_pdos_features
        self.normalize_per_site = normalize_per_site
        self.include_pdos_sum = include_pdos_sum
        self.dft_pdos = None
        # Scale data according to predefined min and max values
        self.scale_data()

    def scale_data(self):
        """
        Scale DFT PDOS data with precomputed min and max values based on training data.
        """
        # Option 1: Separate scaling with sum inclusion
        if self.normalize_per_site and self.include_pdos_sum:
            print(
                '           -------------------------------------------\n'
                '           PDOS normalized separately, sum included\n'
                f'           num PDOS {self.num_pdos_features}                              \n'
                '           -------------------------------------------\n')
            pdos_sum = np.sum(self.dft_feature, axis=0)
            dft_dos = np.concatenate((self.dft_feature, pdos_sum[np.newaxis, ...]), axis=0)

        # Option 2: Separate scaling without sum inclusion
        elif self.normalize_per_site and not self.include_pdos_sum:
            print(
                '           -------------------------------------------\n'
                '                   PDOS normalized separately\n'
                '           -------------------------------------------\n')
            dft_dos = np.split(self.dft_feature, self.num_pdos_features)

        # Option 3: Unified scaling with sum inclusion
        elif not self.normalize_per_site and self.include_pdos_sum:
            raise NotImplementedError('The option to  normalize PDOS together and simultaneously  include the sum is '
                                      'not implemented yet!')

        # Option 4: Unified scaling without sum inclusion
        else:
            print(
                '           -------------------------------------------\n'
                '                   PDOS normalized together\n'
                '           -------------------------------------------\n')
            dft_dos = self.dft_feature

        print(np.asarray(dft_dos).shape)

        # Scale each dos array using predefined min and max values from training data
        dos_scaled = []
        for ind, dos in enumerate(dft_dos):
            ds_min, ds_max = self.data_trained_min_max_values[ind]
            dos_scaled.append(min_max_scaler(dos, (ds_min, ds_max)))

        # Convert to Torch tensor
        self.dft_pdos = torch.FloatTensor(np.array(dos_scaled)).unsqueeze(0)

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.dft_pdos)

    def __getitem__(self, idx):
        # Retrieve a single sample from the dataset
        return self.dft_pdos
