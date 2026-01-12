import h5py
import numpy as np

from dataset import H5PairedDataset


def _write_sample_h5(path) -> None:
    data = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    with h5py.File(path, "w") as handle:
        lr_group = handle.create_group("TFM")
        hr_group = handle.create_group("hr")
        lr_sample = lr_group.create_group("000001")
        lr_sample.create_dataset("I", data=data)
        lr_sample.create_dataset("X", data=data)
        lr_sample.create_dataset("Y", data=data)
        hr_sample = hr_group.create_group("000001")
        hr_sample.create_dataset("data", data=data)


def test_h5_dataset_tfm_channels(tmp_path) -> None:
    h5_path = tmp_path / "sample.h5"
    _write_sample_h5(h5_path)

    dataset = H5PairedDataset(
        h5_path=str(h5_path),
        lr_key="TFM",
        hr_key="hr",
        lr_dataset_name="I",
        hr_dataset_name="data",
        transpose_lr=False,
        transpose_hr=False,
        use_tfm_channels=True,
        coord_range=((0.0, 1.0), (0.0, 1.0)),
        augment=False,
    )
    lr_tensor, hr_tensor, name = dataset[0]
    dataset.close()

    assert name == "000001"
    assert lr_tensor.shape == (3, 4, 4)
    assert hr_tensor.shape == (1, 4, 4)
    assert lr_tensor.min().item() >= -1.01
    assert lr_tensor.max().item() <= 1.01


def test_h5_dataset_single_channel(tmp_path) -> None:
    h5_path = tmp_path / "sample.h5"
    _write_sample_h5(h5_path)

    dataset = H5PairedDataset(
        h5_path=str(h5_path),
        lr_key="TFM",
        hr_key="hr",
        lr_dataset_name="I",
        hr_dataset_name="data",
        transpose_lr=False,
        transpose_hr=False,
        use_tfm_channels=False,
        augment=False,
    )
    lr_tensor, hr_tensor, _ = dataset[0]
    dataset.close()

    assert lr_tensor.shape == (1, 4, 4)
    assert hr_tensor.shape == (1, 4, 4)
