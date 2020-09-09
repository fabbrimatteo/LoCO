# -*- coding: utf-8 -*-
# ---------------------

from typing import Tuple

import click
import numpy as np
import torch
import nms3d_cuda
from mayavi import mlab
from torch.utils.data import DataLoader

import utils
from conf import Conf
from dataset.validation_set import JTAValidationSet
from models import Autoencoder
from models import CodePredictor
from models import Refiner
from pose import Pose
from post_processing import joint_association, filter_joints, refine_pose


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
            joint_a = coords[limb[0]]
            joint_b = coords[limb[1]]
            draw_tube(p1=joint_a, p2=joint_b, color=LIMB_COLORS[LIMBS_LR[c]])

        # draw a sphere for each 3D point
        for c in coords:
            draw_sphere(c, color=BLUE)

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
    # refiner.to(cnf.device)
    refiner.eval()
    refiner.requires_grad(False)

    # init data loader
    ts = JTAValidationSet(cnf=cnf)
    loader = DataLoader(dataset=ts, batch_size=1, shuffle=False, num_workers=0)

    for step, sample in enumerate(loader):

        x, _, fx, fy, cx, cy, frame_path = sample

        x = x.to(cnf.device)
        fx, fy, cx, cy = fx.item(), fy.item(), cx.item(), cy.item()

        # image --> [code_predictor] --> code
        code_pred = code_predictor.forward(x).unsqueeze(0)

        # code --> [decode] --> hmap
        hmap_pred = autoencoder.decode(code_pred).squeeze()

        # hmap --> [local maxima search with cuda kernel] --> pseudo-3D coordinates
        pseudo3d_coords_pred = []
        confidences = []
        for jtype, hmp in enumerate(hmap_pred):
            suppressed_hmap = nms3d_cuda.NMSFilter3d(torch.nn.ConstantPad3d(1, 0)(hmp), 3, 1)
            nonzero_coords = torch.nonzero(suppressed_hmap).cpu()
            for coord in nonzero_coords:
                confidence = suppressed_hmap[tuple(coord)]
                if confidence > cnf.nms_th:
                    pseudo3d_coords_pred.append((jtype, coord[0].item(), coord[1].item(), coord[2].item()))
                    confidences.append(confidence.cpu())

        # pseudo-3D coordinates --> [reverse projection] --> real 3D coordinates
        coords3d_pred = []
        for i in range(len(pseudo3d_coords_pred)):
            joint_type, cam_dist, y2d, x2d = pseudo3d_coords_pred[i]
            x2d, y2d, cam_dist = utils.rescale_to_real(x2d, y2d, cam_dist, q=cnf.q)
            x3d, y3d, z3d = utils.to3d(x2d, y2d, cam_dist, fx=fx, fy=fy, cx=cx, cy=cy)
            coords3d_pred.append((joint_type, x3d, y3d, z3d))
        filter_joints(coords3d_pred, duplicate_th=0.05)

        # real 3D coordinates --> [association] --> list of poses
        poses = joint_association(coords3d_pred)

        # 3D poses -> [refiner] -> refined 3D poses
        refined_poses = []
        for _pose in poses:
            refined_pose = refine_pose(pose=_pose, refiner=refiner)
            if refined_pose is not None:
                refined_poses.append(refined_pose)

        # show output
        print(f'\n\t▶▶ Showing results of \'{frame_path[0]}\'')
        print(f'\t▶▶ It may take some time: please wait')
        print(f'\t▶▶ Close mayavi window to continue')
        show_poses(refined_poses)


@click.command()
@click.argument('exp_name', type=str, default='default')
def main(exp_name):
    # type: (str) -> None

    cnf = Conf(exp_name=exp_name)

    print(f'▶ Results of experiment \'{exp_name}\'')
    results(cnf=cnf)


if __name__ == '__main__':
    main()
