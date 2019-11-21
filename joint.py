# -*- coding: utf-8 -*-
# ---------------------

from typing import *

import numpy as np


class Joint(object):
    # list of joint names
    NAMES = [
        'head_top',
        'head_center',
        'neck',
        'right_clavicle',
        'right_shoulder',
        'right_elbow',
        'right_wrist',
        'left_clavicle',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'spine0',
        'spine1',
        'spine2',
        'spine3',
        'spine4',
        'right_hip',
        'right_knee',
        'right_ankle',
        'left_hip',
        'left_knee',
        'left_ankle',
    ]

    NAMES_14 = [
        'head_top',
        'neck',
        'right_shoulder',
        'right_elbow',
        'right_wrist',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'right_hip',
        'right_knee',
        'right_ankle',
        'left_hip',
        'left_knee',
        'left_ankle',
    ]


    def __init__(self, array, confidence=0):
        # type: (np.ndarray, float) -> Joint
        """
        :param array: array version of the joint
        """
        self.frame = int(array[0])
        self.person_id = int(array[1])
        self.type = int(array[2])
        self.x2d = int(array[3])
        self.y2d = int(array[4])
        self.x3d = array[5]
        self.y3d = array[6]
        self.z3d = array[7]
        self.occ = bool(array[8])  # is this joint occluded?
        self.soc = bool(array[9])  # is this joint self-occluded?
        self.confidence = confidence


    @property
    def cam_distance(self):
        # type: () -> float
        """
        :return: distance of the joint from the camera
        """
        # NOTE: camera coords = (0, 0, 0)
        return np.sqrt(self.x3d**2 + self.y3d**2 + self.z3d**2)


    @property
    def is_on_screen(self):
        # type: () -> bool
        """
        :return: True if the joint is on screen, False otherwise
        """
        return (0 <= self.x2d <= 1920) and (0 <= self.y2d <= 1080)


    @property
    def visible(self):
        # type: () -> bool
        """
        :return: True if the joint is visible, False otherwise
        """
        return not (self.occ or self.soc)


    @property
    def pos2d(self):
        # type: () -> Tuple[int, int]
        """
        :return: 2D coordinates of the joints [px]
        """
        return (self.x2d, self.y2d)


    @property
    def pos3d(self):
        # type: () -> Tuple[float, float, float]
        """
        :return: 3D coordinates of the joints [m]
        """
        return self.x3d, self.y3d, self.z3d


    @property
    def name(self):
        # type: () -> str
        """
        :return: name of the joint (eg: 'neck', 'left_elbow', ...)
        """
        return Joint.NAMES[self.type]


    def __str__(self):
        return f'{Joint.NAMES_14[self.type]}:({round(self.x3d, 3)},{round(self.y3d, 3)},{round(self.z3d, 3)})'


    __repr__ = __str__
