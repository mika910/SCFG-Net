import torch
import torch.nn as nn


class Conv2d1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.conv(x)


class LayerNorm4D(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer_norm = nn.GroupNorm(1, num_features)

    def forward(self, x):
        return self.layer_norm(x)


class MLKA(nn.Module):
    r"""Multi-scale Large Kernel Attention.

    Args:
        n_feats: Number of input channels

    """

    def __init__(self, n_feats: int) -> None:
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats
        self.norm = LayerNorm4D(n_feats)
        self.scale = nn.Parameter(torch.zeros(
            (1, n_feats, 1, 1)), requires_grad=True)
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7,
                      1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, 1, (9 // 2)
                      * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5,
                      1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, (7 // 2)
                      * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3,
                      3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, (5 // 2)
                      * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3,
                            3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5,
                            1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7,
                            1, 7 // 2, groups=n_feats // 3)

        self.proj_first = Conv2d1x1(n_feats, i_feats)
        self.proj_last = Conv2d1x1(n_feats, n_feats)

    def forward(self, x) -> torch.Tensor:
        shortcut = x.clone()

        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2)
                       * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)], dim=1)
        x = self.proj_last(x * a)

        return x * self.scale + shortcut
