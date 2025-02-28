import torch
import torch.nn as nn
from networks.FEM import FEM
from networks.SCRM import SCRM


class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


import torch.nn as nn
import torch.nn.init as init


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SCFG(nn.Module):
    def __init__(self, inp, out):
        super(SCFG, self).__init__()

        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(inp * 2, inp, 1),
            nn.GELU()
        )
        self.conv = FEM(in_channels=inp, out_channels=inp, kernel_size=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(inp * 2, out, kernel_size=1, padding=0),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
        self.fuse_siam = simam_module()
        self.out = nn.Sequential(
            nn.Conv2d(out, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
        # SCRM
        self.SCRM = SCRM(op_channel=inp)

    def forward(self, inp1, inp2):
        x = torch.cat([inp1, inp2], dim=1)
        x = self.conv_init(x)
        c1 = self.conv(x)
        c2 = self.SCRM(x)
        xcat = torch.cat([c1, c2], 1)
        fuse = self.fuse(xcat)
        out = self.fuse_siam(fuse + inp1 + inp2)
        return out


if __name__ == '__main__':
    input_channels = 64
    output_channels = 64
    batch_size = 1
    height, width = 16, 16
    sfg = SCFG(inp=input_channels, out=output_channels)
    inp1 = torch.rand(batch_size, input_channels, height, width)
    inp2 = torch.rand(batch_size, input_channels, height, width)
    output = sfg(inp1, inp2)
    print("inp1 shape:", inp1.shape)
    print("inp2 shape:", inp2.shape)
    print("Output shape:", output.shape)
