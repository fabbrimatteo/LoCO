# -*- coding: utf-8 -*-
# ---------------------

from typing import Tuple

import click
import cv2
import numpy as np
from mayavi import mlab

from pose import Pose
import torch


# useful colors
LIMB_COLORS = [(231 / 255, 76 / 255, 60 / 255), (60 / 255, 222 / 255, 157 / 255)]
BLUE = (55 / 255, 135 / 255, 195 / 255)
ALMOST_BLACK = (0.104, 0.146, 0.189)

# left/right limbs
LIMBS_LR = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]


def draw_sphere(point3d, color):
    # type: (Tuple[float, float, float], Tuple[float, float, float]) -> None
    """
    Draws a sphere of color `color` centered in `point3d`
    """
    mlab.points3d(
        np.array(point3d[0]), np.array(point3d[1]), np.array(point3d[2]),
        scale_factor=0.15, color=color,
    )


def draw_tube(p1, p2, color):
    # type: (Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]) -> None
    """
    Draws a tube of color `color` linking point `p1` with point `p2`
    """
    xs = np.array([p1[0], p2[0]])
    ys = np.array([p1[1], p2[1]])
    zs = np.array([p1[2], p2[2]])
    mlab.plot3d(xs, ys, zs, [1, 2], tube_radius=0.04, color=color)


def dist(p1, p2):
    # type: (Tuple[int, float, float, float], Tuple[int, float, float, float]) -> float
    """
    Returns the Euclidean distance between points `p1` and `p2`
    """
    return np.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2 + (p1[3] - p2[3]) ** 2)


def show_poses(poses):
    """
    Shows a visual representation of the HPE results.
    :param poses: list of poses to show
    """

    # open figure
    mlab.figure(1, bgcolor=ALMOST_BLACK, size=(960, 540))
    mlab.view(azimuth=180, elevation=0)

    # draw each pose
    for coords in poses:

        # draw links
        for c, limb in enumerate(Pose.LIMBS_14):
            type_a, type_b = limb
            jas = [j for j in coords if j[0] == type_a]  # all joints of type 'type_a'
            jbs = [j for j in coords if j[0] == type_b]  # all joints of type 'type_b'
            a = jas[0] if len(jas) == 1 else None
            b = jbs[0] if len(jbs) == 1 else None
            if a is not None and b is not None and dist(a, b) < 1:
                draw_tube(p1=(a[1], a[2], a[3]), p2=(b[1], b[2], b[3]), color=LIMB_COLORS[LIMBS_LR[c]])

        # draw a sphere for each 3D point
        for c in coords:
            jtype, x3d, y3d, z3d = c
            point3d = (x3d, y3d, z3d)
            draw_sphere(point3d, color=BLUE)

    mlab.show()


@click.command()
@click.option('--ex', type=str, default='1')
def main(ex):
    # type: (str) -> None

    print(f'\n▶ Demo \'{ex}\'')

    rgb = cv2.imread(f'demo/{ex}_rgb.jpg')
    cv2.imshow('rgb', rgb)

    rgb = cv2.imread(f'demo/{ex}_res.jpg')
    cv2.imshow('res', rgb)

    data = torch.load(f'demo/{ex}_res.data')
    show_poses(data[1])


if __name__ == '__main__':
    main()
