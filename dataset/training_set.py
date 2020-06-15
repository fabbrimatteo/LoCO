# -*- coding: utf-8 -*-
# ---------------------

from typing import *

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

import utils
from conf import Conf

# number of sequences
N_SEQUENCES = 256

# number frame in each sequence
N_FRAMES_IN_SEQ = 900

# number of frames used for training in each sequence
N_SELECTED_FRAMES = 180


class JTATrainingSet(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    * x: RGB image normalized with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    * y: GT code corresponding to image `x`
    """


    def __init__(self, cnf=None):
        # type: (Conf) -> None
        """
        :param cnf: configuration object
        """
        self.cnf = cnf


    def __len__(self):
        # type: () -> int
        return N_SEQUENCES * N_SELECTED_FRAMES


    def __getitem__(self, i):
        # type: (int) -> Tuple[Tensor, Tensor]

        # select sequence number
        sequence = i // N_SELECTED_FRAMES

        # select frame number
        frame_n = (i % N_SELECTED_FRAMES) * (N_FRAMES_IN_SEQ // N_SELECTED_FRAMES) + 1

        # read input frame
        frame_path = self.cnf.jta_path / 'frames' / 'train' / f'seq_{sequence}/{frame_n}.jpg'
        frame = utils.imread(frame_path)
        frame = transforms.ToTensor()(frame)
        frame = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame)

        # read GT code
        code_path = self.cnf.jta_path / 'codes' / f'{sequence}_{frame_n}.data'
        code = torch.load(code_path, map_location=torch.device('cpu'))

        return frame, code


def main():
    cnf = Conf(exp_name='default')
    ds = JTATrainingSet(cnf=cnf)

    for i in range(len(ds)):
        x, y = ds[i]
        print(f'Example #{i}: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}')


if __name__ == '__main__':
    main()
