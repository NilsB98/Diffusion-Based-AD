import torch
import torch.nn.functional as F
from torchvision import transforms


def create_fl_diffmap(extractor: torch.nn.Module, batch1: torch.Tensor, batch2: torch.Tensor):
    """
    Extract and compare features to a single feature-level difference map, based on the given extractor.

    :param extractor: Model to extract the features from different layers. Expects the model to return a dict of
    layer name to layer result
    :param batch1: compare this batch of images against batch2
    :param batch2: compare this batch of images against batch1
    :return Feature-Level Difference Map
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    batch1 = normalize(batch1)
    batch2 = normalize(batch2)

    features1 = extractor(batch1)
    features2 = extractor(batch2)

    features1 = list(features1.values())
    features2 = list(features2.values())

    diff_map = torch.zeros((batch1.shape[0], 1, *batch1.shape[2:])).to(batch1.device)

    for f1, f2 in zip(features1, features2):
        d_map = 1 - F.cosine_similarity(_mean_conv(f1), _mean_conv(f2))
        d_map = torch.unsqueeze(d_map, dim=1)
        d_map = F.interpolate(d_map, size=batch1.shape[-1], mode='bilinear', align_corners=True)
        diff_map += d_map

    return diff_map


def _mean_conv(x: torch.Tensor, kernel_size=3):
    """
    Smoothing operation with a mean kernel.

    :param x: Tensor to be smoothened
    :param kernel_size: kernel size of the operation
    :return: x after the operation
    """

    in_shape = x.shape
    weights = torch.ones((kernel_size, kernel_size), device=x.device) / (kernel_size ** 2)
    weights = weights.view(1, 1, kernel_size, kernel_size)
    x = x.view(-1, 1, x.shape[2], x.shape[3])
    out = F.conv2d(x, weights, stride=1, padding=kernel_size // 2)

    return out.view(in_shape)
