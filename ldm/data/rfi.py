import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision.transforms as tf
import torchvision.transforms.functional as F

class BaselineDataset(Dataset):
    def __init__(self, dataset: str, polarization=0, offset=0, count=-1):
        print("Dataset instantiated")
        self.file = h5py.File(dataset, "r")
        self.baselines = list(self.file.keys())
        self.polarization = polarization

        self.count = len(self.baselines) if count == -1 else count
        self.offset = offset
        self.transforms = tf.Compose(
            [
                tf.ToTensor(),
                RandomCrop(128)
            ]
        )

    def __len__(self):
        return self.count

    def __del__(self):
        self.file.close()

    def __getitem__(self, idx: int) -> np.ndarray:
        idx += self.offset
        data = self.file[f"{self.baselines[idx]}/CORRUPTED_DATA"][:, :, self.polarization]
        clean_data = self.file[f"{self.baselines[idx]}/CORRECTED_DATA"][:, :, self.polarization]
        assert data.shape == clean_data.shape, f"Data shape is different: {data.shape} != {clean_data.shape}"

        data, clean_data = np.abs(data), np.abs(clean_data)
        data, clean_data = self.transforms(np.dstack((data, clean_data)))

        return {
            "x": data.unsqueeze(0),
            "t": clean_data.unsqueeze(0)
        }

class RandomCrop:
    """Transform (img, target) by randomly cropping. Compatable with both tensor and PIL image"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, target = sample

        if torch.is_tensor(image):
            h, w = image.shape[:2]
        else:
            w, h = image.size
        new_h, new_w = self.output_size

        if h == new_h:
            top=0
        else:
            top = torch.randint(0, h - new_h, (1,)).item()

        if w == new_w:
            left = 0
        else:
            left = torch.randint(0, w - new_w, (1,)).item()

        return F.crop(image, top, left, new_h, new_w), F.crop(
            target, top, left, new_h, new_w
        )
