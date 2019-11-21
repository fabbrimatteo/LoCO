# -*- coding: utf-8 -*-
# ---------------------

from typing import *

import numpy as np

import utils
from joint import Joint


class Pose(list):
    LIMBS = [
        (0, 1),  # head_top -> head_center
        (1, 2),  # head_center -> neck
        (2, 3),  # neck -> right_clavicle
        (3, 4),  # right_clavicle -> right_shoulder
        (4, 5),  # right_shoulder -> right_elbow
        (5, 6),  # right_elbow -> right_wrist
        (2, 7),  # neck -> left_clavicle
        (7, 8),  # left_clavicle -> left_shoulder
        (8, 9),  # left_shoulder -> left_elbow
        (9, 10),  # left_elbow -> left_wrist
        (2, 11),  # neck -> spine0
        (11, 12),  # spine0 -> spine1
        (12, 13),  # spine1 -> spine2
        (13, 14),  # spine2 -> spine3
        (14, 15),  # spine3 -> spine4
        (15, 16),  # spine4 -> right_hip
        (16, 17),  # right_hip -> right_knee
        (17, 18),  # right_knee -> right_ankle
        (15, 19),  # spine4 -> left_hip
        (19, 20),  # left_hip -> left_knee
        (20, 21)  # left_knee -> left_ankle
    ]

    LIMBS_14 = [
        (0, 1),  # head --> neck
        # ------
        (1, 2),  # neck --> right_shoulder
        (2, 3),  # right_shoulder --> right_elbow
        (3, 4),  # right_elbow --> right_wrist
        # ------
        (1, 5),  # neck --> left_shoulder
        (5, 6),  # left_shoulder --> left_elbow
        (6, 7),  # left_elbow --> left_wrist
        # ------
        (1, 8),  # neck --> right_hip
        (8, 9),  # right_hip --> right_knee
        (9, 10),  # right_knee --> right_ankle
        # ------
        (1, 11),  # neck --> left_hip
        (11, 12),  # left_hip --> left_knee
        (12, 13),  # left_knee --> left_ankle
        # ------
        (8, 11)  # right_hip --> left_hip
    ]


    def __init__(self, joints):
        # type: (List[Joint]) -> Pose
        super().__init__(joints)


    @property
    def invisible(self):
        # type: () -> bool
        """
        :return: True if all the joints of the pose are occluded, False otherwise
        """
        for j in self:
            if not j.occ:
                return False
        return True


    def to_rr_pose(self, max_ls=None):
        # type: (Optional[np.ndarray]) -> np.ndarray
        """
        Returns the root relative representation of the pose.
        If `max_ds` is specified, the values ​​are normalized to be in range [0, 1].
        :param max_ls: normalization array
        :return: root relative representation of the pose; shape: (13, 3)
        """
        rr_pose = np.array([j.pos3d for j in self])
        rr_pose = rr_pose - rr_pose[0, :]
        rr_pose = rr_pose[1:]
        if max_ls is not None:
            rr_pose = utils.normalize_rr_pose(rr_pose=rr_pose, max_ls=max_ls)
        return rr_pose


    @staticmethod
    def from_rr_pose(rr_pose, head_pos3d, max_ls):
        # type: (np.ndarray, Sequence[float], Optional[np.ndarray]) -> List[np.ndarray]
        if max_ls is not None:
            rr_pose = utils.denormalize_rr_pose(rr_pose=rr_pose, max_ls=max_ls)
        joints = [np.array(head_pos3d)]
        for f in rr_pose:
            joints.append(f + joints[0])
        return joints
