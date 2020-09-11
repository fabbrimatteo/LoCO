# -*- coding: utf-8 -*-
# ---------------------

import random
import socket
from typing import Optional

import numpy as np
import torch
import yaml
import os
from path import Path


def set_seed(seed=None):
    # type: (Optional[int]) -> int
    """
    set the random seed using the input value (`seed`)
    or a random value if `seed` is `None`
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


class Conf(object):
    HOSTNAME = socket.gethostname()


    def __init__(self, conf_file_path=None, seed=None, exp_name=None, log=True):
        # type: (str, int, str, bool) -> None
        """
        :param conf_file_path: optional path of the configuration file
        :param seed: desired seed for the RNG; if `None`, it will be chosen randomly
        :param exp_name: name of the experiment
        :param log: `True` if you want to log each step; `False` otherwise
        """

        self.exp_name = exp_name
        self.log_each_step = log

        # print project name and host name
        self.project_name = Path(__file__).parent.parent.basename()
        m_str = f'┃ {self.project_name}@{Conf.HOSTNAME} ┃'
        u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
        b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
        print(u_str + '\n' + m_str + '\n' + b_str)

        # define output paths
        self.project_log_path = Path('log')
        self.exp_log_path = self.project_log_path / exp_name

        # set random seed
        self.seed = set_seed(seed)  # type: int

        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        tmp = Path(__file__).parent / (self.exp_name + '.yaml')
        if conf_file_path is None and tmp.exists():
            conf_file_path = tmp

        # read the YAML configuation file
        if conf_file_path is None:
            y = {}
        else:
            conf_file = open(conf_file_path, 'r')
            y = yaml.load(conf_file, Loader=yaml.FullLoader)

        # read configuration parameters from YAML file
        # or set their default value
        self.q = y.get('Q', 0.31746031746031744)  # type: float # --> quantization factor
        self.lr = y.get('LR', 0.0001)  # type: float # --> learning rate
        self.epochs = y.get('EPOCHS', 999)  # type: int
        self.det_th = y.get('DET_TH', 0.4)  # type: float # --> detection threshold for test metrics
        self.nms_th = y.get('NMS_TH', 0.1)  # type: float
        self.n_workers = y.get('N_WORKERS', 0)  # type: int
        self.batch_size = y.get('BATCH_SIZE', 1)  # type: int
        self.epoch_len = y.get('EPOCH_LEN', 4096)  # type: int
        self.jta_path = y.get('JTA_PATH', './jta')  # type: str
        self.test_set_len = y.get('TEST_SET_LEN', 128)  # type: int

        if y.get('DEVICE', None) is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(y.get('DEVICE').split(':')[1])
            self.device = 'cuda:0'
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.jta_path = Path(self.jta_path)
        assert self.jta_path.exists(), 'the specified directory for the JTA-Dataset does not exists'
