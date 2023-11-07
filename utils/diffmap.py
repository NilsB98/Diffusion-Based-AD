from collections import Counter
from typing import List, Dict, TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

import feature_extraction
from utils.visualize import split_batch_into_patch, stitch_batch_patches


class DiffMaps(TypedDict):
    diffmap_pl: torch.Tensor
    diffmap_fl: torch.Tensor


def diff_maps_to_anomaly_map(diff_maps: DiffMaps,
                             diff_map_contrib: TypedDict('Influence', {'diffmap_pl': float, 'diffmap_fl': float}),
                             transform: Module = None) -> Tensor:
    # apply transform
    if transform is not None:
        for name, d_map in diff_maps.items():
            diff_maps[name] = transform(d_map)

    # init aggregation diffmap
    diff_map = torch.zeros_like(diff_maps['diffmap_pl'])

    # aggregate diffmaps and normalize each by threshold, s.t. the threshold for each one is at 1 and summation makes
    # sense
    for key in diff_maps.keys():
        diff_map += diff_maps[key] * diff_map_contrib[key]

    return torch.where(diff_map >= 1, 1, 0)


def normalize_diffmaps(diffmaps: DiffMaps, normalization: TypedDict('DiffMapNormalization', {'threshold_pl': float,
                                                                                             'threshold_fl': float})) -> DiffMaps:
    """
    Normalize the given diffmaps by some factors.

    :param diffmaps: Diffmaps to normalize
    :param normalization: normalization factor
    :return: normalized diffmaps
    """

    if "diffmap_fl" in diffmaps and "threshold_fl" in normalization:
        diffmaps["diffmap_fl"] /= normalization["threshold_fl"]
    if "diffmap_pl" in diffmaps and "threshold_pl" in normalization:
        diffmaps["diffmap_pl"] /= normalization["threshold_pl"]

    return diffmaps


def count_values(tensor: Tensor, factor=1, counter: Counter = None) -> Dict[int, int]:
    tensor *= factor
    counts = torch.bincount(tensor.to(torch.int).reshape(-1))

    counts_dict = {key: counts[key].item() for key in range(len(counts))}

    if counter is not None:
        counter.update(counts_dict)

    return counts_dict


def calc_threshold(bin_dict: Dict[int, int], quantile: float, factor=1) -> float:
    """
    Calculate the threshold for a diffMap at which a pixel is anomalous.

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


def create_diffmaps(original, reconstruction, extractor, extractor_resolution: int, fl_smoothing_size=3) -> DiffMaps:
    """
    Create the diffMaps, showing the difference between the original and reconstructed image.
    I.e. the pixel-level diffmap and the feature-level diffMap based on a feature extractor.

    :param original: The original samples
    :param reconstruction: The reconstructede samples
    :param extractor: The extractor model, to extract features from the samples.
    :param extractor_resolution: The resolution/input size on which the extractor was trained on. (Should match the input size of the diffusion-model)
    :param fl_smoothing_size: Kernel used to smoothen the features found by the extractor. No smoothing if set to 1.
    :return: Dict with the diffMaps.
    """

    with torch.no_grad():
        diff_maps: DiffMaps = {}

        # pixel-level
        diff_map = (original - reconstruction) ** 2
        pl_diff_map = torch.amax(diff_map, (1))[:, None, :, :]
        diff_maps['diffmap_pl'] = pl_diff_map

        # feature-level
        num_imgs = len(original)
        original = split_batch_into_patch(original, extractor_resolution)
        reconstruction = split_batch_into_patch(reconstruction, extractor_resolution)

        if extractor is not None:
            resnet_diffmap = feature_extraction.utils.create_fl_diffmap(extractor, original, reconstruction,
                                                                        fl_smoothing_size)
            resnet_diffmap = stitch_batch_patches(resnet_diffmap, num_imgs)
            diff_maps['diffmap_fl'] = resnet_diffmap

        return diff_maps
