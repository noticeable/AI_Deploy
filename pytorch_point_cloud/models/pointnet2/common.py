import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers.extend([
                nn.Conv1d(in_c, out_c, 1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP([in_channels, out_channels, out_channels])

    def forward(self, points):
        features = self.mlp(points)
        global_feature = torch.max(features, dim=2, keepdim=True)[0]
        return global_feature.repeat(1, 1, points.size(2)), features


class FeaturePropagation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP([in_channels, out_channels, out_channels])

    def forward(self, coarse, skip):
        if coarse.size(2) != skip.size(2):
            coarse = F.interpolate(coarse, size=skip.size(2), mode='nearest')
        return self.mlp(torch.cat([coarse, skip], dim=1))
