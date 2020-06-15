# -*- coding: utf-8 -*-
# ---------------------

import json
from typing import *

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils
from conf import Conf
from pose import Pose

# 14 useful joints for the JTA dataset
USEFUL_JOINTS = [0, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21]

# camera intrinsic parameters: fx, fy, cx, cy
CAMERA_PARAMS = 1158, 1158, 960, 540


class JTAValidationSet(Dataset):
    """
    Dataset composed of tuples (frame, gt_3d, fx, fy, cx, cy, frame_path) in which:
    * frame: RGB image, usually normalized with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    * gt_3d: 3D coordinates of all the joints; format: (jtype, x3d, y3d, z3d)
    * fx, fy, cx, cy: camera intrinsics
    * frame_path: path of `frame`
    """


    def __init__(self, cnf=None):
        # type: (Conf) -> None
        """
        :param cnf: configuration object
        """
        self.cnf = cnf
        self.validation_data_path = cnf.jta_path / 'poses' / 'val'
        self.sequences = self.validation_data_path.dirs()


    def __len__(self):
        # type: () -> int
        return self.cnf.test_set_len


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, str, float, float, float, float, str]

        # select sequence number
        sequence = i * 2 + 256 + 1

        # select frame number
        frame = 123

        # get corresponding data
        frame_path = self.cnf.jta_path / 'frames' / 'val' / f'seq_{sequence}/{frame}.jpg'
        data_path = self.validation_data_path / f'seq_{sequence}/{frame}.data'

        # read input frame
        frame = utils.imread(frame_path)
        frame = transforms.ToTensor()(frame)
        frame = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame)

        # read GT list of poses
        poses = torch.load(data_path)  # type: List[Pose]

        # create target
        target = []
        for jtype in USEFUL_JOINTS:
            for pose in poses:
                joint = pose[jtype]
                if not joint.is_on_screen:
                    continue
                target.append([USEFUL_JOINTS.index(jtype), joint.x3d, joint.y3d, joint.z3d])
        target = json.dumps(target)

        fx, fy, cx, cy = CAMERA_PARAMS

        return frame, target, fx, fy, cx, cy, frame_path


def main():
    ds = JTAValidationSet(cnf=Conf(exp_name='default'))

    for i in range(len(ds)):
        frame, gt_3d, _, _, _, _, frame_path = ds[i]
        gt_3d = json.loads(gt_3d)
        print(f'Example #{i}: frame.shape={tuple(frame.shape)}, gt_3d.len={len(gt_3d)}')
        print(f'\t>> {frame_path}')


if __name__ == '__main__':
    main()
