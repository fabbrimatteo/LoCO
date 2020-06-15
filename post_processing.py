# -*- coding: utf-8 -*-
# ---------------------

from typing import List
from typing import Sequence

import munkres
import numpy as np
import torch
from scipy import stats
from pose import Pose
from joint import Joint
from models.pose_refiner import Refiner
from typing import Optional

MAX_LS = np.array([0.27, 0.41, 0.67, 0.93, 0.41, 0.67, 0.92, 0.88, 1.28, 1.69, 0.88, 1.29, 1.70])


def hungarian(cost_matrix):
    # type: (np.ndarray) -> List[(int, int)]
    """
    Calculate the Hungarian solution to the classical assignment problem between two sets
    of elements (A) and (B).
     :param cost_matrix: cost matrix such that `cost_matrix [i, j]` represents the association
        cost of the i-th element in (A) with the j-th element in (B)
    :return: indexes for the lowest-cost pairings between rows and columns in `cost_matrix`
    """
    if cost_matrix.size > 0:
        if cost_matrix.shape[0] > cost_matrix.shape[1]:
            cost_matrix = cost_matrix.transpose((1, 0))
            indexes = np.array(munkres.Munkres().compute(cost_matrix.tolist()))[:, ::-1]
        else:
            indexes = np.array(munkres.Munkres().compute(cost_matrix.tolist()))
    else:
        indexes = []
    return indexes


def __torch_to_list(y):
    # type: (torch.Tensor) -> List[List[int]]
    if len(y.shape) == 3:
        y = y[0]
    return [list(point3d.cpu().numpy()) for point3d in y]


def joint_association(joints_pred):
    # type: (Sequence[Sequence[float]]) -> List[Sequence[Sequence[float]]]
    """
    :param joints_pred: predicted joint list
        >> each joint must be in the form [jtype, z3d, y3d, x3d]
    :return: poses list, where each pose is a list of joints
    """
    heads = [j for j in joints_pred if j[0] == 0]
    poses = [[h] for h in heads]
    for jtype in range(1, 14):
        other_joints = [j for j in joints_pred if j[0] == jtype]
        cost_matrix = np.zeros((len(heads), len(other_joints)))
        for row in range(len(heads)):
            for col in range(len(other_joints)):
                j1 = np.array(heads[row])
                j2 = np.array(other_joints[col])
                d = np.linalg.norm(j1[1:] - j2[1:])
                cost_matrix[row, col] = d
        associations = hungarian(cost_matrix)
        for a, b in associations:
            cost = cost_matrix[a, b]
            if cost < MAX_LS[other_joints[b][0] - 1] * 1.5:
                poses[a].append(other_joints[b])
    return poses


def count_people(joints):
    # type: (Sequence[Sequence[float]]) -> int
    """
    Given a list of joints belonging to an unknown number of people,        for jtype in range(14):
            if pose[jtype].visible:
                refined_pose_pred[jtype] = np.array(pose[jtype].pos3d)
    it returns the estimated number of such people.

    :param joints: list of joints, where each joint is in the form [jtype, z3d, y3d, x3d]
    :return: number of people
    """
    n_heads = len([j for j in joints if j[0] == 0])
    counts = [n_heads] if n_heads > 0 else []
    for jtype in range(1, 14):
        x = len([j for j in joints if j[0] == jtype])
        if x > 0:
            counts.append(x)
    mode = int(np.round(np.mean(stats.mode(counts)[0]), 0))
    return max(n_heads, mode)


def filter_joints(joints, duplicate_th):
    # type: (List[Sequence[float]], float) -> (Sequence[Sequence[float]])
    """
    Filter the list of input joints removing duplicates. Two joints (of the same type)
    are considered distinct only if they are more than `duplicate_th` meters apart;
    otherwise one of the two is considered as a duplicate of the other

    *** WARNING ***: inplace function! `joints` list will be modified!

    :param joints: sequence of joints where each joint is in the form [jtype, x3d, y3d, z3d]
    :return: filtered list of joints (without duplicates)
    """
    for jtype in range(14):

        # all joints of type `jtype`
        _joints = [j for j in joints if j[0] == jtype]

        # simmetric joint-to-joint distance matrix
        distance_matrix = np.zeros((len(_joints), len(_joints))) + -1
        for row in range(distance_matrix.shape[0]):
            for col in range(distance_matrix.shape[1]):
                if distance_matrix[row, col] == -1:
                    a = np.array(_joints[row][1:])
                    b = np.array(_joints[col][1:])
                    distance = np.linalg.norm(a - b)
                    distance_matrix[row, col] = distance
                    distance_matrix[col, row] = distance

        # find and remove duplicates from input list
        duplicates = np.argwhere(distance_matrix <= duplicate_th)
        duplicates = duplicates[duplicates[:, 0] > duplicates[:, 1]]
        for duplicate in duplicates:
            try:
                joints.remove(_joints[duplicate[-1]])
            except ValueError:
                pass


def refine_pose(pose, refiner):
    # type: (Sequence[Sequence[float]], Refiner) -> Optional[List[np.ndarray]]
    """
    :param pose: list of joints where each joint is in the form [jtype, x3d, y3d, z3d]
    :param refiner: pose refiner model
    :return: refined pose -> list of 14 ordered joints where each joint is in the form [x3d, y3d, z3d]
        >> see `Joint.NAMES` for joint order
    """

    # convert `pose` list into a `Pose` object
    joints = []
    for jtype in range(14):
        _joint = [j for j in pose if j[0] == jtype]
        if len(_joint) == 1:
            _, x, y, z = _joint[0][0], _joint[0][1], _joint[0][2], _joint[0][3]
            joint = np.array([-1, -1, jtype, -1, -1, x, y, z, 0, 0])
            joint = Joint(joint)
            joints.append(joint)
        else:
            joint = np.array([-1, -1, jtype, -1, -1, -1, -1, -1, 1, 1])
            joint = Joint(joint)
            joints.append(joint)
    pose = Pose(joints)

    # convert `Pose` object into a fountain
    rr_pose_pred = pose.to_rr_pose(MAX_LS)
    for jtype in range(1, 14):
        if not pose[jtype].visible:
            rr_pose_pred[jtype - 1] = np.array([-1, -1, -1])
    rr_pose_pred = torch.tensor(rr_pose_pred).unsqueeze(0).float()

    # refine fountain with `refiner` model
    refined_rr_pose_pred = refiner.forward(rr_pose_pred).numpy().squeeze()

    if pose[0].type == 0:
        refined_pose_pred = Pose.from_rr_pose(refined_rr_pose_pred, head_pos3d=pose[0].pos3d, max_ls=MAX_LS)
        return refined_pose_pred
    else:
        return None
