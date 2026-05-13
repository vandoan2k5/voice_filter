import torch
import torch.nn as nn

from .conv import GateConv2d, Conv2dUnit, Deconv2dUnit


class EnUnetModule(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k1: tuple,
                 k2: tuple,
                 scale: int):
        super(EnUnetModule, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.cin = cin
        self.cout = cout
        self.scale = scale

        self.in_conv = nn.Sequential(GateConv2d(cin, cout, k1, (1, 2)),
                                     nn.BatchNorm2d(cout),
                                     nn.PReLU(cout))
        self.encoder = nn.ModuleList([Conv2dUnit(k2, cout) for _ in range(scale)])
        self.decoder = nn.ModuleList([Deconv2dUnit(k2, cout, 1)])
        for i in range(1, scale):
            self.decoder.append(Deconv2dUnit(k2, cout, 2))
        self.out_pool = nn.AvgPool2d((3, 1))

    def forward(self, x: torch.Tensor):
        x_resi = self.in_conv(x)
        x = x_resi
        x_list = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            x_list.append(x)

        x = self.decoder[0](x)
        for i in range(1, len(self.decoder)):
            x = self.decoder[i](torch.cat([x, x_list[-(i + 1)]], dim=1))
        x_resi = x_resi + x

        return self.out_pool(x_resi)

