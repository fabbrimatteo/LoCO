# -*- coding: utf-8 -*-
# ---------------------

from copy import deepcopy
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch import nn

import utils
from models import BaseModel
from pose import Pose

JointList = Union[List[Tuple[int, float, float, float]], List[np.ndarray]]


class Refiner(BaseModel):

    def __init__(self, pretrained=True):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_features=39, out_features=128), nn.ReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(in_features=128, out_features=128), nn.ReLU(True))
        self.fc3 = nn.Sequential(nn.Linear(in_features=128, out_features=39))
        self.res_relu = nn.ReLU(True)

        if pretrained:
            self.load_w('models/weights/pose_refiner.pth')


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        fc1 = self.fc1(x.clone().view(x.shape[0], 39))
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        x = fc3 + self.res_relu(x.view(x.shape[0], 39))
        return x.view(x.shape[0], 13, 3)


    def refine(self, pose, confidences, hole_th, replace_th=0.4, head_pos3d=None):
        # type: (JointList, List[float], float, float, Tuple[float, float, float]) -> JointList
        """
        Refine `pose`, considering as missing all the joints whose confidence value
        is less than a given threshold `th`.
        :param pose: pose to refine
            >> it is list of 14 quadruples of type (jtype, x3d, y3d, z3d)
            >> the pose must always have 14 joints; to encode any "holes" use
            random coordinates with a confidence value <= 0
        :param confidences: confidence values of pose joints
            >> it is a list of 14 values such that `confidences[i]`
            is the confidence of the i-th joint of the pose
        :param hole_th: confidence threshold
        :param replace_th: replace a joint with its refined version only
            if its confidence is <= `replace_th`
        :param head_pos3d: 3D position of the head;
            >> if `None`, it is set to the first joint of `pose`
        :return: refined version of the input pose
        """
        from joint import Joint
        if head_pos3d is None:
            head_pos3d = pose[0][1:]

        input_pose = deepcopy(pose)

        # from coords to Pose
        joints = []
        for c in pose:
            jtype, x, y, z = c
            joint = np.array([-1, -1, jtype, -1, -1, x, y, z, -1, -1])
            joint = Joint(joint)
            joints.append(joint)
        pose = Pose(joints=joints)

        # from Pose to RR-Pose (root relative pose)
        rr_pose = deepcopy(pose).to_rr_pose(max_ls=np.array(utils.MAX_LS))
        for jtype in range(1, 14):
            # if the confidence of the joint is less then `th`
            # this joint is considered a hole
            if confidences[jtype] <= hole_th:
                rr_pose[jtype - 1] = np.array([-1, -1, -1])

        rr_pose = torch.tensor(rr_pose).unsqueeze(0)
        device = self.state_dict()['fc1.0.weight'].device
        rr_pose = rr_pose.to(device).float()

        # predict refined RR-Pose
        refined_rr_pose = self.forward(rr_pose)
        refined_rr_pose = refined_rr_pose.detach().cpu().numpy().squeeze()

        # from RR-Pose (with ref) to Pose (with ref)
        pose_ref = Pose.from_rr_pose(refined_rr_pose.copy(), head_pos3d=head_pos3d, max_ls=np.array(utils.MAX_LS))
        coords3d_pred_ref = []
        for jtype, c in enumerate(pose_ref):
            if confidences[jtype] > replace_th:
                coords3d_pred_ref.append(input_pose[jtype])
            else:
                coords3d_pred_ref.append((jtype, c[0], c[1], c[2]))

        return coords3d_pred_ref


# ---------

def main():
    batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Refiner(pretrained=False).to(device)

    print(model)
    print(f'* number of parameters: {model.n_param}')

    x = torch.rand((batch_size, 13, 3)).to(device)
    y = model.forward(x)

    print(f'* input shape: {tuple(x.shape)}')
    print(f'* output shape: {tuple(y.shape)}')


if __name__ == '__main__':
    main()
