import torch
from torch import Tensor
from torchvision.transforms import *


class RandomNoise(torch.nn.Module):
    def __init__(self, mean=0, std=0.01, p=0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, tensor: Tensor) -> Tensor:
        if torch.rand(1) < self.p:
            tensor += torch.randn(tensor.shape) * self.std + self.mean
        return tensor

class RandomCrop(RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", p=0.5):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.p = p

    def forward(self, tensor: Tensor) -> Tensor:
        if torch.randn(1) < self.p:
            return super().forward(tensor)
        return tensor
