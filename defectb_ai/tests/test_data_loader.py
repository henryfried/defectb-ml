import numpy as np
import torch

from defectb_ai.data_loader.mydataset import MyTBDataSet, min_max_scaler


def _build_dummy_dataset(tmp_path):
    local_dos_len = 6
    sample = {
        "sample_0": {
            "local_dos": np.ones(local_dos_len, dtype=np.float32),
            "label": np.array([0.5, 0.5], dtype=np.float32),
        },
        "sample_1": {
            "local_dos": np.full(local_dos_len, 2.0, dtype=np.float32),
            "label": np.array([1.0, 1.0], dtype=np.float32),
        },
    }
    file_path = tmp_path / "dummy.npy"
    np.save(file_path, sample)
    return file_path


def test_dataset_scaling_handles_constant_features(tmp_path):
    file_path = _build_dummy_dataset(tmp_path)

    dataset = MyTBDataSet(
        file_name=str(file_path),
        num_pdos_features=2,
        normalize_per_site=True,
        include_pdos_sum=True,
    )

    features, targets = dataset[0]

    assert torch.isfinite(dataset.features).all()
    assert torch.isfinite(dataset.target_labels).all()
    assert torch.isfinite(features).all()
    assert torch.isfinite(targets).all()
    assert features.shape[0] == dataset.features.shape[1]
    assert targets.shape[0] == dataset.target_labels.shape[1]


def test_min_max_scaler_returns_zeros_for_constant_array():
    arr = np.array([5.0, 5.0, 5.0], dtype=np.float32)

    scaled = min_max_scaler(arr, (5.0, 5.0))

    assert np.all(scaled == 0.0)
