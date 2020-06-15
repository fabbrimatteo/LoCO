# -*- coding: utf-8 -*-
# ---------------------

from typing import *

import PIL
import numpy as np
import torch
from PIL.Image import Image
from path import Path


MAX_LS = np.array([0.27, 0.41, 0.67, 0.93, 0.41, 0.67, 0.92, 0.88, 1.28, 1.69, 0.88, 1.29, 1.70])


def imread(path):
    # type: (Union[Path, str]) -> Image
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


def torch_to_list(y):
    # type: (torch.Tensor) -> List[List[int]]
    if len(y.shape) == 3:
        y = y[0]
    return [list(point3d.cpu().numpy()) for point3d in y]


def local_maxima_3d(hmaps3d, threshold, device='cuda', ret_confs=False):
    # type: (torch.Tensor, float, str, bool) -> Union[List[List[int]], Tuple[List[List[int]], List[float]]]
    """
    :param hmaps3d: 3D heatmaps with shape (N_joints, D, H, W)
    :param threshold: peaks with values < of that threshold will not be returned
    :param device: device where you want to run the operation
    :param ret_confs: do you want to return confidence values (one for each peak)?
    :return: list of detected peak(s) and (optional) confidence values
    """
    d = torch.device(device)

    peaks = []
    confidences = []
    for jtype, hmap3d in enumerate(hmaps3d):
        m_f = torch.zeros(hmap3d.shape).to(d)
        m_f[1:, :, :] = hmap3d[:-1, :, :]
        m_b = torch.zeros(hmap3d.shape).to(d)
        m_b[:-1, :, :] = hmap3d[1:, :, :]

        m_u = torch.zeros(hmap3d.shape).to(d)
        m_u[:, 1:, :] = hmap3d[:, :-1, :]
        m_d = torch.zeros(hmap3d.shape).to(d)
        m_d[:, :-1, :] = hmap3d[:, 1:, :]

        m_r = torch.zeros(hmap3d.shape).to(d)
        m_r[:, :, 1:] = hmap3d[:, :, :-1]
        m_l = torch.zeros(hmap3d.shape).to(d)
        m_l[:, :, :-1] = hmap3d[:, :, 1:]

        p = torch.zeros(hmap3d.shape).to(d)
        p[hmap3d >= m_f] = 1
        p[hmap3d >= m_b] += 1
        p[hmap3d >= m_u] += 1
        p[hmap3d >= m_d] += 1
        p[hmap3d >= m_r] += 1
        p[hmap3d >= m_l] += 1

        p[hmap3d >= threshold] += 1
        p[p != 7] = 0

        tmp_j = torch.nonzero(p).cpu()
        tmp_j = [[jtype, z, y, x] for z, y, x in torch_to_list(tmp_j)]
        peaks += tmp_j

        if ret_confs:
            tmp_c = torch.nonzero(p).cpu()
            tmp_c = [hmap3d[z, y, x].item() for z, y, x in torch_to_list(tmp_c)]
            confidences += tmp_c

    if ret_confs:
        return peaks, confidences
    else:
        return peaks


def rescale_to_real(x2d, y2d, cam_dist, q):
    # type: (int, int, float, float) -> Tuple[int, int, float]
    """
    :param x2d: predicted `x2d` (real_x2d // 8)
    :param y2d: predicted `y2d` (real_y2d // 8)
    :param cam_dist: predicted `cam_distance` (real_cam_distance // 0.317)
    :param q: quantization coefficient
    :return: (real_x2d, real_y2d, real_cam_distance)
    """
    return x2d * 8, y2d * 8, (cam_dist * q)


def to3d(x2d, y2d, cam_dist, fx, fy, cx, cy):
    # type: (int, int, float, float, float, float, float) -> Tuple[float, float, float]
    """
    Converts a 2D point on the image plane into a 3D point in  in the standard
    coordinate system using the intrinsic camera parameters.

    :param x2d: 2D x coordinate [px]
    :param y2d: 2D y coordinate [px]
    :param cam_dist: distance from camera [m]
    :param fx: x component of the focal len
    :param fy: y component of the focal len
    :param cx: x component of the central point
    :param cy: y component of the central point
    :return: 3D coordinates
    """

    k = (-1) * np.sqrt(
        fx ** 2 * fy ** 2 + fx ** 2 * cy ** 2 - 2 * fx ** 2 * cy * y2d + fx ** 2 * y2d ** 2 +
        fy ** 2 * cx ** 2 - 2 * fy ** 2 * cx * x2d + fy ** 2 * x2d ** 2
    )

    x3d = ((fy * cam_dist * cx) - (fy * cam_dist * x2d)) / k
    y3d = ((fx * cy * cam_dist) - (fx * cam_dist * y2d)) / k
    z3d = -(fx * fy * cam_dist) / k

    return x3d, y3d, z3d


def normalize_rr_pose(rr_pose, max_ls=MAX_LS):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    :param rr_pose: root relative pose
    :param max_ls: normalization array
    :return: normalized version of the input `rr_pose`
    """
    for i in range(3):
        rr_pose[:, i] = 0.5 + rr_pose[:, i] / (2 * max_ls)
    return rr_pose


def denormalize_rr_pose(rr_pose, max_ls=MAX_LS):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    :param rr_pose: normalized root relative pose
    :param max_ls: normalization array
    :return: denormalized version of the input `rr_pose`
    """
    for i in range(3):
        rr_pose[:, i] = (rr_pose[:, i] - 0.5) * 2 * max_ls
    return rr_pose
