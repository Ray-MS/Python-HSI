from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

from . import transforms as T


class HsiDataset(data.Dataset):
    def __init__(
            self,
            root: str,

            window_size: int = 15,

            train: bool = False,
            train_rate: float = 1.,
            valid: bool = False,

            augmentation: bool = False,
            crop: float = 0.5,
            flip: float = 0.5,
            noise: float = 0.5,
    ):
        self.root = Path(root, self.__class__.__name__)
        self.window_size = window_size

        self.data, self.target = self._load_data()
        self.xyl = self._load_xyl(train, train_rate, valid)
        self.transform = self._load_transform(augmentation, crop, flip, noise)

    @property
    def data_file(self):
        return Path(self.root, "data.pt")

    @property
    def target_file(self):
        return Path(self.root, "target.pt")

    def _load_data(self):
        if self.data_file.exists():
            data = torch.load(self.data_file)
        else:
            data_file = Path(self.root, "data.npy")
            if data_file.exists():
                data = np.load(data_file)
                data = TF.to_tensor(data)
                torch.save(data, self.data_file)
            else:
                raise FileNotFoundError("No DATA File.")

        if self.target_file.exists():
            target = torch.load(self.target_file)
        else:
            target_file = Path(self.root, "target.npy")
            if target_file.exists():
                target = np.load(target_file)
                target = torch.LongTensor(target)
                torch.save(target, self.target_file)
            else:
                raise FileNotFoundError("No TARGET File.")

        if data.shape[-2:] != target.shape:
            raise ValueError("DATA and Target do not match.")

        mean = torch.mean(data, dim=(1, 2), keepdim=True)
        std = torch.std(data, dim=(1, 2), keepdim=True)
        data = (data - mean) / std
        data = TF.pad(data, self.window_size // 2)
        return data, target

    def _load_xyl(self, train, train_rate, valid):
        xyl_folder = Path(self.root, "xyl")
        xyl_folder.mkdir(exist_ok=True)
        if train and train_rate > 0:
            file = Path(xyl_folder, f"train_{train_rate}.pt")
            if file.exists():
                return torch.load(file)
            else:
                xyl_file = Path(self.root, f"{train_rate}", "train_coordinates.npy")
                xyl = torch.LongTensor(np.load(xyl_file))
                torch.save(xyl, file)
                return xyl
        elif valid:
            file = Path(xyl_folder, "valid.pt")
            if file.exists():
                return torch.load(file)
            else:
                xyl_file = Path(self.root, "test_coordinates.npy")
                xyl = torch.LongTensor(np.load(xyl_file))
                torch.save(xyl, Path(xyl_folder, "valid.pt"))
                return xyl
        else:
            file = Path(xyl_folder, "test.pt")
            if file.exists():
                return torch.load(file)
            else:
                xyl = []
                h, w = self.target.shape
                for x in range(h):
                    for y in range(w):
                        xyl.append((x, y, self.target[x, y]))
                xyl = torch.LongTensor(xyl)
                torch.save(xyl, Path(xyl_folder, "test.pt"))
            return xyl

    def _load_transform(self, augmentation, crop, flip, noise):
        transforms = []
        if augmentation:
            transforms.append(T.RandomCrop(self.window_size, self.window_size // 4, p=crop))
            transforms.append(T.RandomHorizontalFlip(flip))
            transforms.append(T.RandomVerticalFlip(flip))
            transforms.append(T.RandomNoise(noise))

        return T.Compose(transforms)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.xyl)

    def __repr__(self) -> str:
        head = "Dataset: " + self.__class__.__name__
        body = [f"Number of datapoints: {len(self)}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        return "\n".join((head, *body))
