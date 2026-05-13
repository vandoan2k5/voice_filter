import torch
import torch.nn as nn


class GateConv2d(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k: tuple,
                 s: tuple):
        super(GateConv2d, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s

        self.conv = nn.Sequential(nn.ConstantPad2d((0, 0, k[0] - 1, 0), value=0.),
                                  nn.Conv2d(in_channels=cin,
                                            out_channels=cout * 2,
                                            kernel_size=k,
                                            stride=s))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)

        return outputs * gate.sigmoid()


class Conv2dUnit(nn.Module):
    def __init__(self, k: tuple, c: int):
        super(Conv2dUnit, self).__init__()
        self.k = k
        self.c = c
        self.conv = nn.Sequential(nn.Conv2d(c, c, k, (1, 2)),
                                  nn.BatchNorm2d(c),
                                  nn.PReLU(c))

    def forward(self, x):
        return self.conv(x)


class Deconv2dUnit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 expend_scale: int):
        super(Deconv2dUnit, self).__init__()
        self.k = k
        self.c = c
        self.expend_scale = expend_scale
        self.deconv = nn.Sequential(nn.ConvTranspose2d(c * expend_scale, c, k, (1, 2)),
                                    nn.BatchNorm2d(c),
                                    nn.PReLU(c))

    def forward(self, x):
        return self.deconv(x)

