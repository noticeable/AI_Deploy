import torch.nn as nn
import torch.nn.functional as F

from .common import PointNetSetAbstraction


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.dataset.n_channels
        self.sa1 = PointNetSetAbstraction(in_channels, 64)
        self.sa2 = PointNetSetAbstraction(64, 128)
        self.sa3 = PointNetSetAbstraction(128, 256)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.pointnet2.dropout),
            nn.Linear(256, config.dataset.n_classes),
        )

    def forward(self, points):
        x = points.transpose(1, 2)
        x, _ = self.sa1(x)
        x, _ = self.sa2(x)
        x, _ = self.sa3(x)
        x = x[:, :, 0]
        logits = self.head(x)
        return {'logits': logits}
