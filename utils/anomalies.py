import torch
from torch import Tensor


def diff_map_to_anomaly_map(diff_map: Tensor, threshold: float) -> Tensor:
    return torch.where(diff_map >= threshold, diff_map, 0)
