# -*- coding: utf-8 -*-
# 🐐-🐐-🐐-🐐-🐐-🐐-🐐-🐐-🐐

from typing import *

import numpy as np

np.warnings.filterwarnings('ignore')

Seq = Sequence[Union[int, float]]


def dist(p1, p2, th):
    # type: (Seq, Seq, float) -> float
    """
    3D Point Distance
    :param p1: predicted point
    :param p2: GT point
    :param th: max acceptable distance
    :return: euclidean distance between the positions of the two joints
    """
    if p1[0] != p2[0]:
        return np.nan
    d = np.linalg.norm(np.array(p1) - np.array(p2))
    return d if d <= th else np.nan


def non_minima_suppression(x: np.ndarray) -> np.ndarray:
    """
    :return: non-minima suppressed version of the input array;
    supressed values become np.nan
    """
    min = np.nanmin(x)
    x[x != min] = np.nan
    if len(x[x == min]) > 1:
        ok = True
        for i in range(len(x)):
            if x[i] == min and ok:
                ok = False
            else:
                x[i] = np.nan
    return x


def not_nan_count(x: np.ndarray) -> int:
    """
    :return: number of not np.nan elements of the array
    """
    return len(x[~np.isnan(x)])


def joint_det_metrics(points_pred, points_true, th=7.0):
    # type: (List[Seq], List[Seq], float) -> Dict[str, Union[int, float]]
    """
    Joint Detection Metrics
    :param points_pred: list of predicted points
    :param points_true: list of GT points
    :param th: distance threshold; all distances > th will be considered 'np.nan'.
    :return: a dictionary of metrics, 'met', related to joint detection;
             the the available metrics are:
             (1) met['tp'] = number of True Positives
             (2) met['fn'] = number of False Negatives
             (3) met['fp'] = number of False Positives
             (4) met['pr'] = PRecision
             (5) met['re'] = REcall
             (6) met['f1'] = F1-score
    """
    # create distance matrix
    # the number of rows of the matrix corresponds to the number of GT joints
    # the number of columns of the matrix corresponds to the number of predicted joints
    # mat[i,j] contains the njd-distance between joints_true[i] and joints_pred[j]

    if len(points_pred) > 0 and len(points_true) > 0:
        mat = []
        for p_true in points_true:
            row = np.array([dist(p_pred, p_true, th=th) for p_pred in points_pred])
            mat.append(row)
        mat = np.array(mat)
        mat = np.apply_along_axis(non_minima_suppression, 1, mat)
        mat = np.apply_along_axis(non_minima_suppression, 0, mat)

        # calculate joint detection metrics
        nr = np.apply_along_axis(not_nan_count, 1, mat)
        tp = len(nr[nr != 0])  # number of True Positives
        fn = len(nr[nr == 0])  # number of False Negatives
        fp = len(points_pred) - tp  # number of False Positives
        pr = tp / (tp + fp)  # PRecision
        re = tp / (tp + fn)  # REcall
        f1 = 2 * tp / (2 * tp + fn + fp)  # F1-score

    elif len(points_pred) == 0 and len(points_true) == 0:
        tp = 0  # number of True Positives
        fn = 0  # number of False Negatives
        fp = 0
        pr = 1.0
        re = 1.0
        f1 = 1.0
    elif len(points_pred) == 0:
        tp = 0  # number of True Positives
        fn = len(points_true)  # number of False Negatives
        fp = 0
        pr = 0.0  # PRecision
        re = 0.0  # REcall
        f1 = 0.0  # F1-score
    else:
        tp = 0
        fn = 0
        fp = len(points_pred)
        pr = 0.0  # PRecision
        re = 0.0  # REcall
        f1 = 0.0  # F1-score

    # build the metrics dictionary
    metrics = {
        'tp': tp, 'fn': fn, 'fp': fp,
        'pr': pr, 're': re, 'f1': f1,
    }

    return metrics
