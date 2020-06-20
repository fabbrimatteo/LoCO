# -*- coding: utf-8 -*-
# ---------------------

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from models import BaseModel


class Upsample(nn.Module):
    """
    Upsamples a given tensor by (scale_factor)X.
    """


    def __init__(self, scale_factor=2, mode='trilinear'):
        # type: (int, str) -> Upsample
        """
        :param scale_factor: the multiplier for the image height / width
        :param mode: the upsampling algorithm - values in {'nearest', 'linear', 'bilinear', 'trilinear'}
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


    def extra_repr(self):
        return f'scale_factor={self.scale_factor}, mode={self.mode}'


# ---------------------

class Autoencoder(BaseModel):
    """
    VHA: (V)olumetric (H)eatmap (A)utoencoder
    """


    def __init__(self, hmap_d=316, pretrained=True):
        # type: (int, bool) -> None
        """
        :param hmap_d: number of input channels
        """

        super().__init__()

        self.fuser = nn.Sequential(
            nn.Conv3d(in_channels=14, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv3d(in_channels=4, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=hmap_d, out_channels=hmap_d // 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 2, out_channels=hmap_d // 4, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 4, out_channels=hmap_d // 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
        )

        # --------------

        self.defuser = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv3d(in_channels=4, out_channels=14, kernel_size=5, padding=2),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=hmap_d // 4, out_channels=hmap_d // 4, kernel_size=5, padding=2),
            Upsample(mode='bilinear'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 4, out_channels=hmap_d // 2, kernel_size=5, padding=2),
            Upsample(mode='bilinear'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 2, out_channels=hmap_d, kernel_size=5, padding=2),
            nn.ReLU(True)
        )

        if pretrained:
            self.load_w('models/weights/vha.pth')


    def encode(self, x):
        # type: (torch.Tensor) -> torch.Tensor

        x = self.encoder(torch.reshape(x, (x.shape[0] * 14, x.shape[2], x.shape[3], x.shape[4])).contiguous())
        x = torch.reshape(x, (x.shape[0] // 14, 14, x.shape[1], x.shape[2], x.shape[3])).contiguous()

        x = self.fuser(x)
        return x


    def decode(self, x):
        x = self.defuser(x)

        x = self.decoder(torch.reshape(x, (x.shape[0] * 14, x.shape[2], x.shape[3], x.shape[4])).contiguous())
        x = torch.reshape(x, (x.shape[0] // 14, 14, x.shape[1], x.shape[2], x.shape[3])).contiguous()
        return x


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.encode(x)
        x = self.decode(x)
        return x


# ---------------------


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1

    model = Autoencoder(pretrained=False).to(device)
    print(model)

    print(f'* number of parameters: {model.n_param}')

    print('\n--- ENCODER ---')
    x = torch.rand((batch_size, 14, 316, 1080 // 8, 1920 // 8)).to(device)
    y = model.encode(x)
    print(f'* input shape: {tuple(x.shape)}')
    print(f'* output shape: {tuple(y.shape)}')

    print('\n--- DECODER ---')
    xd = model.decode(y)
    print(f'* input shape: {tuple(y.shape)}')
    print(f'* output shape: {tuple(xd.shape)}')


if __name__ == '__main__':
    main()
