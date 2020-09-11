# -*- coding: utf-8 -*-
# ---------------------

import json

import click
import numpy as np
import torch
import nms3d_cuda
from path import Path
from torch import nn
from torch.utils.data import DataLoader

import utils
from association import coords_to_poses
from conf import Conf
from dataset.testing_set import JTATestingSet
from models import Autoencoder
from models import BaseModel
from models import CodePredictor
from models import Refiner
from test_metrics import joint_det_metrics


MAX_LS = np.array([0.27, 0.41, 0.67, 0.93, 0.41, 0.67, 0.92, 0.88, 1.28, 1.69, 0.88, 1.29, 1.70])

# metrics thresholds
THS = [0.4, 0.8, 1.2]


@click.command()
@click.argument('exp_name', type=str, default='pretrained')
def compute(exp_name):
    # type: (str) -> None

    cnf = Conf(exp_name=exp_name)

    # init Code Predictor
    predictor = CodePredictor()  # type: BaseModel
    predictor.to(cnf.device)
    predictor.eval()
    predictor.requires_grad(False)
    predictor.load_w(cnf.exp_log_path / 'best.pth')

    # init Decoder
    autoencoder = Autoencoder()  # type: BaseModel
    autoencoder.to(cnf.device)
    autoencoder.eval()
    autoencoder.requires_grad(False)
    autoencoder.load_w(Path(__file__).parent / 'models/weights/vha.pth')

    # init Hole Filler
    hole_filler = Refiner(pretrained=True)
    hole_filler.to(cnf.device)
    hole_filler.eval()
    hole_filler.requires_grad(False)
    hole_filler.load_w(Path(__file__).parent / 'models/weights/pose_refiner.pth')

    # init data loader
    ts = JTATestingSet(cnf=cnf)
    loader = DataLoader(dataset=ts, batch_size=1, shuffle=False, num_workers=0)

    metrics_dict = {}
    for th in THS:
        for key in ['pr', 're', 'f1']:
            metrics_dict[f'{key}@{th}'] = []  # without refinement
            metrics_dict[f'{key}@{th}+'] = []  # with refinement

    for step, sample in enumerate(loader):

        x, coords3d_true, fx, fy, cx, cy, frame_path = sample
        x = x.to(cnf.device)
        coords3d_true = json.loads(coords3d_true[0])
        fx, fy, cx, cy = fx.item(), fy.item(), cx.item(), cy.item()

        # image --> [code_predictor] --> code
        code_pred = predictor.forward(x).unsqueeze(0)

        # code --> [decoder] --> hmap
        hmap_pred = autoencoder.decode(code_pred).squeeze()

        # hmap --> [local maxima search] --> pseudo-3D coordinates
        coords2d_pred = []
        confs = []
        for jtype, hmp in enumerate(hmap_pred.squeeze()):
            res = nms3d_cuda.NMSFilter3d(nn.ConstantPad3d(1, 0)(hmp), 3, 1)
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
            x2d, y2d, cam_dist = utils.rescale_to_real(x2d, y2d, cam_dist, q=cnf.q)
            x3d, y3d, z3d = utils.to3d(x2d, y2d, cam_dist, fx=fx, fy=fy, cx=cx, cy=cy)
            coords3d_pred.append((joint_type, x3d, y3d, z3d))

        # real 3D coordinates --> [association] --> list of poses
        poses = coords_to_poses(coords3d_pred, confs)

        # a solitary joint is a joint that has been excluded from the association
        # process since no valid connection could be found;
        # note that only solitary joints with a confidence value >0.6 are considered
        all_pose_joints = []
        for pose in poses:
            all_pose_joints += [(j.type, j.confidence, j.x3d, j.y3d, j.z3d) for j in pose]
        coords3d_pred_ = [(c[0], confs[k], c[1], c[2], c[3]) for k, c in enumerate(coords3d_pred)]
        solitary = [(s[0], s[2], s[3], s[4]) for s in (set(coords3d_pred_) - set(all_pose_joints)) if s[1] > 0.6]

        # list of poses --> [hole filler] --> refined list of poses
        refined_poses = []
        for person_id, pose in enumerate(poses):
            confidences = [j.confidence for j in pose]
            pose = [(joint.type, joint.x3d, joint.y3d, joint.z3d) for joint in pose]
            refined_pose = hole_filler.refine(pose=pose, hole_th=0.2, confidences=confidences, replace_th=1)
            refined_poses.append(refined_pose)

        # refined list of poses --> [something] --> refined_coords3d_pred
        refined_coords3d_pred = []
        for pose in refined_poses:
            refined_coords3d_pred += pose

        # compute metrics without refinement
        for th in THS:
            __m = joint_det_metrics(points_pred=coords3d_pred, points_true=coords3d_true, th=th)
            for key in ['pr', 're', 'f1']:
                metrics_dict[f'{key}@{th}'].append(__m[key])

        # compute metrics with refinement
        for th in THS:
            __m = joint_det_metrics(points_pred=refined_coords3d_pred + solitary, points_true=coords3d_true, th=th)
            for key in ['pr', 're', 'f1']:
                metrics_dict[f'{key}@{th}+'].append(__m[key])

        # print test progress
        print(f'\r>> processing test image {step} of {len(loader)}', end='')

    print('\r', end='')
    for th in THS:
        print(f'(PR, RE, F1)@{th}:'
              f'\tno_ref=('
              f'{np.mean(metrics_dict[f"pr@{th}"]) * 100:.2f}, '
              f'{np.mean(metrics_dict[f"re@{th}"]) * 100:.2f}, '
              f'{np.mean(metrics_dict[f"f1@{th}"]) * 100:.2f})'
              f'\twith_ref=('
              f'{np.mean(metrics_dict[f"pr@{th}+"]) * 100:.2f}, '
              f'{np.mean(metrics_dict[f"re@{th}+"]) * 100:.2f}, '
              f'{np.mean(metrics_dict[f"f1@{th}+"]) * 100:.2f}) '
              )


if __name__ == '__main__':
    compute()
