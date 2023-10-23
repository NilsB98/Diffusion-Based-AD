from collections import Counter
from typing import List, Dict

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


def diff_maps_to_anomaly_map(diff_maps: List[Tensor], thresholds: List[float], transform: Module = None) -> Tensor:

    # apply transform
    if transform is not None:
        for i in range(len(diff_maps)):
            diff_maps[i] = transform(diff_maps[i])

    # init aggregation diffmap
    diff_map = torch.zeros_like(diff_maps[0])

    # aggregate diffmaps and normalize each by threshold, s.t. the threshold for each one is at 1 and summation makes
    # sense
    for i in range(len(diff_maps)):
        diff_map += diff_maps[i] / thresholds[i] * .7

    return torch.where(diff_map >= 1, 1, 0)


def count_values(tensor: Tensor, factor=1, counter: Counter=None) -> Dict[int, int]:
    tensor *= factor
    counts = torch.bincount(tensor.to(torch.int).reshape(-1))

    counts_dict = {key: counts[key].item() for key in range(len(counts))}

    if counter is not None:
        counter.update(counts_dict)

    return counts_dict


def calc_threshold(bin_dict: Dict[int, int], quantile: float, factor=1) -> float:
    """
    Given a counter of occurrences per value (e.g. number pf pixels per pixel value), calculate a threshold
    at which the quantile is reached.
    Since binning doesn't make sense with float values this method is best used by first scaling potential float values
    up and then converting them to integers to create the bin_dict.
    The keys of the bin_dict are interpreted as the bins, the values as the occurrences.

    e.g. calculate the threshold at which 99.9% of values would be included, assuming ascending order of keys within the
    dict.

    :param bin_dict: a dict with bins of values, i.e. number of occurrences per key.
    :param quantile: percentage as float indicating the percentage of how much data should be smaller than the threshold
    :param factor: factor with which the returned value/index will be scaled.
    :return: threshold to fulfill the quantile condition.
    """

    for key in range(max(bin_dict.keys())):
        if bin_dict.get(key) is None:
            bin_dict[key] = 0

    bin_dict = dict(sorted(bin_dict.items()))
    cumsum = np.cumsum(list(bin_dict.values()))

    full_sum = cumsum[-1]
    for idx in range(len(cumsum)):
        if cumsum[idx] / full_sum >= quantile:
            return idx / factor


if __name__ == '__main__':
    count_values(torch.rand((8, 3, 256, 256)), factor=1000, counter=Counter())
