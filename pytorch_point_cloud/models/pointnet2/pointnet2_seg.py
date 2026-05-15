import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import FeaturePropagation, PointNetSetAbstraction


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.dataset.n_channels
        self.num_cls = config.dataset.n_classes
        self.num_seg_classes = config.dataset.n_seg_classes
        self.sa1 = PointNetSetAbstraction(in_channels, 64)
        self.sa2 = PointNetSetAbstraction(64, 128)
        self.sa3 = PointNetSetAbstraction(128, 256)
        self.fp3 = FeaturePropagation(256 + 128, 256)
        self.fp2 = FeaturePropagation(256 + 64, 128)
        self.fp1 = FeaturePropagation(128 + in_channels + self.num_cls, 128)
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.num_seg_classes, 1),
        )

    def forward(self, points):
        x = points.transpose(1, 2)
        l1_global, l1_skip = self.sa1(x)
        l2_global, l2_skip = self.sa2(l1_skip)
        l3_global, _ = self.sa3(l2_skip)
        x = self.fp3(l3_global, l2_skip)
        x = self.fp2(x, l1_skip)
        cls_one_hot = torch.zeros(points.size(0), self.num_cls, points.size(1), device=points.device)
        x = self.fp1(x, torch.cat([points.transpose(1, 2), cls_one_hot], dim=1))
        seg_logits = self.classifier(x).transpose(1, 2)
        return {'seg_logits': seg_logits}
