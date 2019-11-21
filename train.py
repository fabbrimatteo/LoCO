# -*- coding: utf-8 -*-
# ---------------------

import click
import torch.backends.cudnn as cudnn

from conf import Conf
from trainer import Trainer

cudnn.benchmark = True


@click.command()
@click.option('--exp_name', type=str, default='default')
@click.option('--conf_file_path', type=str, default=None)
@click.option('--seed', type=int, default=None)
def main(exp_name, conf_file_path, seed):
    # type: (str, str, int) -> None

    # if `exp_name` contains a '@' character,
    # the number following '@' is considered as
    # the desired random seed for the experiment
    split = exp_name.split('@')
    if len(split) == 2:
        seed = int(split[1])
        exp_name = split[0]

    cnf = Conf(conf_file_path=conf_file_path, seed=seed, exp_name=exp_name)

    print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

    trainer = Trainer(cnf=cnf)
    trainer.run()


if __name__ == '__main__':
    main()
