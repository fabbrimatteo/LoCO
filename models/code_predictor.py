# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import nn

from models import BaseModel
from models.inception import Inception3


class BasicBlockTransposed(nn.Module):
    """
    Basic model block.
    """


    def __init__(self, in_channels, out_channels):
        # type: (int, int) -> None
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, padding=0, stride=2),
            nn.ReLU(True),
        )

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return self.main(x)

class BasicBlock(nn.Module):
    """
    Basic model block.
    """


    def __init__(self, in_channels, out_channels):
        # type: (int, int) -> None
        super().__init__()

        self.main = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=0),
            nn.ReLU(True),
        )


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return self.main(x)


class CodePredictor(BaseModel):

    def __init__(self):
        # type: () -> None
        super().__init__()

        self.f_extractor = Inception3(transform_input=True)
        self.main = nn.Sequential(
            BasicBlockTransposed(in_channels=2048, out_channels=1024),
            BasicBlock(in_channels=1024, out_channels=512),
            BasicBlock(in_channels=512, out_channels=256),
            nn.Conv2d(in_channels=256, out_channels=79, kernel_size=1, padding=0),
        )


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.f_extractor(x)
        x = self.main(x)
        return x


# ---------

def main():
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CodePredictor().to(device)
    print(model)

    print(f'\n* number of parameters: {model.n_param}')

    x = torch.rand((batch_size, 3, 1080, 1920)).to(device)
    y = model.forward(x)

    print(f'* input shape: {tuple(x.shape)}')
    print(f'* output shape: {tuple(y.shape)}')


if __name__ == '__main__':
    main()
