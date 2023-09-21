import torch
from torch import Tensor
from torch.nn import Module


def diff_map_to_anomaly_map(diff_map: Tensor, threshold: float, transform: Module = None) -> Tensor:
    if transform is not None:
        diff_map = transform(diff_map)
    return torch.where(diff_map >= threshold, 1, 0)
