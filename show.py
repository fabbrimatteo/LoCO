# -*- coding: utf-8 -*-
# ---------------------

from typing import Tuple

import click
import cv2
import numpy as np
from mayavi import mlab
from torch.utils.data import DataLoader
import nms3d_cuda
import torch

import utils
from association import coords_to_poses
from conf import Conf
from dataset.test_set import JTATestSet
from models import Autoencoder
from models import CodePredictor
from models import Refiner
from pose import Pose

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
    return np.sqrt((p1[1] - p2[1])**2 + (p1[2] - p2[2])**2 + (p1[3] - p2[3])**2)


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


def results(cnf):
    # type: (Conf) -> None
    """
    Shows a visual representation of the obtained results
    using the test set images as input
    """

    # init Code Predictor
    code_predictor = CodePredictor()
    code_predictor.to(cnf.device)
    code_predictor.eval()
    code_predictor.requires_grad(False)
    code_predictor.load_w(f'log/{cnf.exp_name}/best.pth')

    # init Decoder
    autoencoder = Autoencoder(pretrained=True)
    autoencoder.to(cnf.device)
    autoencoder.eval()
    autoencoder.requires_grad(False)

    # init Hole Filler
    refiner = Refiner(pretrained=True)
    refiner.to(cnf.device)
    refiner.eval()
    refiner.requires_grad(False)

    # init data loader
    ts = JTATestSet(cnf=cnf)
    loader = DataLoader(dataset=ts, batch_size=1, shuffle=False, num_workers=0)

    for step, sample in enumerate(loader):

        x, _, fx, fy, cx, cy, frame_path = sample

        x = x.to(cnf.device)
        fx, fy, cx, cy = fx.item(), fy.item(), cx.item(), cy.item()

        # image --> [code_predictor] --> code
        code_pred = code_predictor.forward(x).unsqueeze(0)

        # code --> [decode] --> hmap
        hmap_pred = autoencoder.decode(code_pred).squeeze()

        # hmap --> [local maxima search] --> pseudo-3D coordinates
        # coords2d_pred, confs = utils.local_maxima_3d(hmap_pred, threshold=0.2, device=cnf.device, ret_confs=True)

        # hmap --> [local maxima search with cuda kernel] --> pseudo-3D coordinates
        coords2d_pred = []
        confs = []
        for jtype, hmp in enumerate(hmap_pred):
            res = nms3d_cuda.NMSFilter3d(torch.nn.ConstantPad3d(1, 0)(hmp), 3, 1)
            nz = torch.nonzero(res).cpu()
            for el in nz:
                confid = res[tuple(el)]
                if confid > 0.1:
                    coords2d_pred.append((jtype, el[0].item(), el[1].item(), el[2].item()))
                    confs.append(confid.cpu())

        # pseudo-3D coordinates --> [to_3d] --> real 3D coordinates
        coords3d_pred = []
        for i in range(len(coords2d_pred)):
            joint_type, cam_dist, y2d, x2d = coords2d_pred[i]
            x2d, y2d, cam_dist = utils.rescale_to_real(x2d, y2d, cam_dist)
            x3d, y3d, z3d = utils.to3d(x2d, y2d, cam_dist, fx=fx, fy=fy, cx=cx, cy=cy)
            coords3d_pred.append((joint_type, x3d, y3d, z3d))

        # real 3D coordinates --> [association] --> list of poses
        poses = coords_to_poses(coords3d_pred, confs)

        # list of poses ---> [pose refiner] ---> refined list of poses
        refined_poses = []
        for person_id, pose in enumerate(poses):
            confidences = [j.confidence for j in pose]
            pose = [(joint.type, joint.x3d, joint.y3d, joint.z3d) for joint in pose]
            refined_pose = refiner.refine(pose=pose, hole_th=0.2, confidences=confidences, replace_th=1)
            refined_poses.append(refined_pose)

        # show input
        img = cv2.imread(frame_path[0])
        cv2.imshow('input image', img)

        # show output
        show_poses(refined_poses)


@click.command()
@click.option('--exp_name', type=str, default='default')
@click.option('--conf_file_path', type=str, default=None)
@click.option('--seed', type=int, default=None)
def main(exp_name, conf_file_path, seed):
    # type: (str, str, int) -> None

    # if `exp_name` contains a '@' character,
    # the number following '@' is considered as
    # the desired random seed for the experiment
    split = exp_name.split('@')
    if len(split) == 2:
        seed = int(split[1])
        exp_name = split[0]

    cnf = Conf(conf_file_path=conf_file_path, seed=seed, exp_name=exp_name)

    print(f'\n▶ Showing visual results of experiment \'{exp_name}\'')
    results(cnf=cnf)


if __name__ == '__main__':
    main()
