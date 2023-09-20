from torch import Tensor
from typing import Dict
import torch

def pro(gt: Tensor, prediction: Tensor) -> float:
    """
    calculate the per-region overlap score between two gray scale images.
    'the percentage of correctly predicted pixels is computed for each annotated defect region in the ground-truth.
    The average over all defects yields the final PRO value'
    --Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization

    :param gt: Ground truth image
    :param prediction: Predicted anomaly regions
    :return: PRO-Score
    """

    return 0


def scores(gt: Tensor, prediction: Tensor) -> Dict[str, float]:
    """
    Calculate multiple metrics at once for a given GT segmentation and its prediction (on a pixel level).

    :param gt: Ground Truth segmentation image
    :param prediction: Predicted anomaly regions image
    :return: dict with metrics
    """

    num_pixels = gt.shape[-1] ** 2
    _true = gt.squeeze().int() == prediction.squeeze().int()
    tp = len(torch.argwhere(torch.logical_and(_true, gt.squeeze().int() == 1)))
    tn = len(torch.argwhere(torch.logical_and(_true, gt.squeeze().int() == 0)))
    fp = len(torch.argwhere(gt.squeeze().int() < prediction.squeeze().int()))
    fn = len(torch.argwhere(gt.squeeze().int() > prediction.squeeze().int()))

    return {
        # 'tp': tp,
        # 'tn': tn,
        # 'fp': fp,
        # 'fn': fn,
        'tpr': tp/(tp + fn) if tp+fn > 0 else 1,
        'fnr': fn/(fn + tp) if tp+fn > 0 else 0,
        'fpr': fp/(tn + fp),
        'tnr': tn/(tn + fp),
        'acc': (tp + tn) / num_pixels,
        'precision': tp / (tp + fp) if tp+fp > 0 else 1,
        'f1': 2*tp / (2*tp + fp + fn) if (tp + fp + fn) > 0 else 1,
        'img_acc': 1 if gt.max() == 0 and prediction.max() == 0 or gt.max() == 1 and prediction.max() == 1 else 0
    }
