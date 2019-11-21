# -*- coding: utf-8 -*-
# ---------------------

import json
import math
from datetime import datetime
from time import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import utils
from conf import Conf
from dataset.validation_set import JTAValidationSet
from dataset.training_set import JTATrainingSet
from models import Autoencoder
from models import CodePredictor
from test_metrics import joint_det_metrics


class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> None
        self.cnf = cnf

        # init code predictor
        self.code_predictor = CodePredictor()
        self.code_predictor = self.code_predictor.to(cnf.device)

        # init volumetric heatmap autoencoder
        self.autoencoder = Autoencoder()
        self.autoencoder.eval()
        self.autoencoder.requires_grad(False)
        self.autoencoder = self.autoencoder.to(cnf.device)

        # init optimizer
        self.optimizer = optim.Adam(params=self.code_predictor.parameters(), lr=cnf.lr)

        # init dataset(s)
        training_set = JTATrainingSet(cnf)
        test_set = JTAValidationSet(cnf)

        # init train/test loader
        self.train_loader = DataLoader(training_set, cnf.batch_size, num_workers=cnf.n_workers, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=1, num_workers=cnf.n_workers, shuffle=False)

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        print(f'tensorboard --logdir={cnf.project_log_path.abspath()}\n')
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)

        # starting values values
        self.epoch = 0
        self.best_test_f1 = None

        # possibly load checkpoint
        self.load_ck()


    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path, map_location=torch.device('cpu'))
            print(f'[loading checkpoint \'{ck_path}\']')
            self.epoch = ck['epoch']
            self.code_predictor.load_state_dict(ck['model'])
            self.best_test_f1 = self.best_test_f1
            self.optimizer.load_state_dict(ck['optimizer'])


    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch': self.epoch,
            'model': self.code_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_test_loss': self.best_test_f1
        }
        torch.save(ck, self.log_path / 'training.ck')


    def train(self):
        """
        train model for one epoch on the Training-Set.
        """
        self.code_predictor.train()
        self.code_predictor.requires_grad(True)

        train_losses = []
        times = []
        start_time = time()
        t = time()
        for step, sample in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            x, y_true = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)

            y_pred = self.code_predictor.forward(x)
            loss = nn.MSELoss()(y_pred, y_true)
            loss.backward()
            train_losses.append(loss.item())

            self.optimizer.step(None)

            # print an incredible progress bar
            progress = (step + 1) / self.cnf.epoch_len
            progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
            times.append(time() - t)
            t = time()
            if self.cnf.log_each_step or (not self.cnf.log_each_step and progress == 1):
                print('\r[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}% │ Loss: {:.6f} │ ↯: {:5.2f} step/s'.format(
                    datetime.now().strftime("%m-%d@%H:%M"), self.epoch, step + 1,
                    progress_bar, 100 * progress,
                    np.mean(train_losses), 1 / np.mean(times),
                    e=math.ceil(math.log10(self.cnf.epochs)),
                    s=math.ceil(math.log10(self.cnf.epoch_len)),
                ), end='')

            if step >= self.cnf.epoch_len - 1:
                break

        # log average loss of this epoch
        mean_epoch_loss = np.mean(train_losses)  # type: float
        self.sw.add_scalar(tag='train/loss', scalar_value=mean_epoch_loss, global_step=self.epoch)

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')


    def test(self):
        """
        test model on the Validation-Set
        """

        self.code_predictor.eval()
        self.code_predictor.requires_grad(False)

        t = time()
        test_prs = []
        test_res = []
        test_f1s = []
        for step, sample in enumerate(self.test_loader):
            x, coords3d_true, fx, fy, cx, cy, _ = sample

            fx, fy, cx, cy = fx.item(), fy.item(), cx.item(), cy.item()
            x = x.to(self.cnf.device)
            coords3d_true = json.loads(coords3d_true[0])

            # image --> [code_predictor] --> code
            code_pred = self.code_predictor.forward(x).unsqueeze(0)

            # code --> [decode] --> hmap(s)
            hmap_pred = self.autoencoder.decode(code_pred).squeeze()

            # hmap --> [local_maxima_3d] --> rescaled pseudo-3D coordinates
            coords2d_pred = utils.local_maxima_3d(hmaps3d=hmap_pred, threshold=0.1, device=self.cnf.device)

            # rescaled pseudo-3D coordinates --> [to_3d] --> real 3D coordinates
            coords3d_pred = []
            for i in range(len(coords2d_pred)):
                joint_type, cam_dist, y2d, x2d = coords2d_pred[i]
                x2d, y2d, cam_dist = utils.rescale_to_real(x2d=x2d, y2d=y2d, cam_dist=cam_dist, q=self.cnf.q)
                x3d, y3d, z3d = utils.to3d(x2d=x2d, y2d=y2d, cam_dist=cam_dist, fx=fx, fy=fy, cx=cx, cy=cy)
                coords3d_pred.append((joint_type, x3d, y3d, z3d))

            # real 3D
            metrics = joint_det_metrics(points_pred=coords3d_pred, points_true=coords3d_true, th=self.cnf.det_th)
            pr, re, f1 = metrics['pr'], metrics['re'], metrics['f1']
            test_prs.append(pr)
            test_res.append(re)
            test_f1s.append(f1)

        # log average loss on test set
        mean_test_pr = float(np.mean(test_prs))
        mean_test_re = float(np.mean(test_res))
        mean_test_f1 = float(np.mean(test_f1s))

        # print test metrics
        print(
            f'\t● AVG (PR, RE, F1) on TEST-set: '
            f'({mean_test_pr * 100:.2f}, '
            f'{mean_test_re * 100:.2f}, '
            f'{mean_test_f1 * 100:.2f}) ',
            end=''
        )
        print(f'│ T: {time() - t:.2f} s')

        self.sw.add_scalar(tag='test/precision', scalar_value=mean_test_pr, global_step=self.epoch)
        self.sw.add_scalar(tag='test/recall', scalar_value=mean_test_re, global_step=self.epoch)
        self.sw.add_scalar(tag='test/f1', scalar_value=mean_test_f1, global_step=self.epoch)

        # save best model
        if self.best_test_f1 is None or mean_test_f1 >= self.best_test_f1:
            self.best_test_f1 = mean_test_f1
            torch.save(self.code_predictor.state_dict(), self.log_path / 'best.pth')


    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for _ in range(self.epoch, self.cnf.epochs):
            self.train()
            self.test()
            self.epoch += 1
            self.save_ck()
