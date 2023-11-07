from collections import Counter

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


def scores_batch(gts: Tensor, predictions: Tensor) -> Dict[str, float]:
    batch_score = Counter()

    for gt, pred in zip(gts, predictions):
        batch_score.update(scores(gt, pred))

    for key in batch_score:
        if not key.startswith('img'):
            batch_score[key] /= len(gts)
    return dict(batch_score)


def scores(gt: Tensor, prediction: Tensor) -> Dict[str, float]:
    """
    Calculate multiple metrics at once for a given GT segmentation and its prediction (on a pixel level).

    :param gt: Ground Truth segmentation image
    :param prediction: Predicted anomaly regions image
    :return: dict with metrics
    """

    gt = gt.cpu()
    prediction = prediction.cpu()

    num_pixels = gt.shape[-1] ** 2
    _true = gt.squeeze().int() == prediction.squeeze().int()
    pxl_tp = len(torch.argwhere(torch.logical_and(_true, gt.squeeze().int() == 1)))
    pxl_tn = len(torch.argwhere(torch.logical_and(_true, gt.squeeze().int() == 0)))
    pxl_fp = len(torch.argwhere(gt.squeeze().int() < prediction.squeeze().int()))
    pxl_fn = len(torch.argwhere(gt.squeeze().int() > prediction.squeeze().int()))

    return {
        'pxl_tpr': _calc_tpr(pxl_tp, pxl_fn),
        'pxl_fnr': _calc_fnr(pxl_tp, pxl_fn),
        'pxl_fpr': _calc_fpr(pxl_fp, pxl_tn),
        'pxl_tnr': _calc_tnr(pxl_fp, pxl_tn),
        'pxl_acc': _calc_acc(num_pixels, pxl_tp, pxl_tn),
        'pxl_precision': _calc_precision(pxl_tp, pxl_fp),
        'pxl_f1': _calc_f1(pxl_tp, pxl_fp, pxl_fn),
        'img_tp': int(gt.max() == 1 and prediction.max() == 1),
        'img_fp': int(gt.max() == 0 and prediction.max() == 1),
        'img_tn': int(gt.max() == 0 and prediction.max() == 0),
        'img_fn': int(gt.max() == 1 and prediction.max() == 0),
    }


def normalize_pxl_scores(factor: int, counter: Counter) -> Counter:
    """
    normalize all entries in the counter which start with 'pxl' by the given factor (probably the length of the loader).

    :param factor: normalization factor
    :param counter: Counter to normalize
    :return: counter
    """

    for key in counter:
        if key.startswith('pxl'):
            counter[key] /= factor

    return counter

def aggregate_img_scores(counter: Counter) -> Counter:
    """
    Turn img_tp, img_fp, img_tn, img_fn into tpr, fnr, fpr, tnr, accuracy, precision and f1-score.
    Remove img_xx values in the process.

    :param counter: Counter containing img_tp, img_fp, img_tn, img_fn
    :return: Counter with new scores
    """

    tp = counter['img_tp']
    tn = counter['img_tn']
    fp = counter['img_fp']
    fn = counter['img_fn']

    counter['img_tpr'] = _calc_tpr(tp, fn)
    counter['img_fnr'] = _calc_fnr(tp, fn)
    counter['img_fpr'] = _calc_fpr(fp, tn)
    counter['img_tnr'] = _calc_tnr(fp, tn)
    counter['img_acc'] = _calc_acc(tp + tn + fp + fn, tp, tn)
    counter['img_precision'] = _calc_precision(tp, fp)
    counter['img_f1'] = _calc_f1(tp, fp, fn)

    del counter['img_tp']
    del counter['img_fp']
    del counter['img_tn']
    del counter['img_fn']

    return counter


def _calc_tpr(num_tps, num_fns):
    """

    :param num_tps: Number of true positives
    :param num_fns: Number of false negatives
    :return: TPR
    """

    return num_tps / (num_tps + num_fns) if num_tps + num_fns > 0 else 1


def _calc_fnr(num_tps, num_fns):
    """

    :param num_tps: Number of true positives
    :param num_fns: Number of false negatives
    :return: FNR
    """

    return num_fns / (num_fns + num_tps) if num_tps + num_fns > 0 else 0


def _calc_fpr(num_fps, num_tns):
    """

    :param num_fps: Number of false positives
    :param num_tns: Number of true negatives
    :return: FPR
    """

    return num_fps / (num_fps + num_tns) if num_fps + num_tns > 0 else 0


def _calc_tnr(num_fps, num_tns):
    """

    :param num_fps: Number of false positives
    :param num_tns: Number of true negatives
    :return: TNR
    """

    return num_tns / (num_tns + num_fps) if num_fps + num_tns > 0 else 1


def _calc_acc(num_samples, num_tps, num_tns):
    """

    :param num_samples: Number of all samples
    :param num_tps: number of true positives
    :param num_tns: number of false positives
    :return: ACC
    """

    return (num_tps + num_tns) / num_samples


def _calc_precision(num_tps, num_fps):
    """

    :param num_tps: number of true positives
    :param num_fps:
    :return:  PRECISION
    """

    return num_tps / (num_tps + num_fps) if num_tps + num_fps > 0 else 1


def _calc_f1(num_tps, num_fps, num_fns):
    """

    :param num_tps: number of true positives
    :param num_fps: number of false positives
    :param num_fns: number of false negatives
    :return: F1-Score
    """

    return 2 * num_tps / (2 * num_tps + num_fps + num_fns) if (num_tps + num_fps + num_fns) > 0 else 1
