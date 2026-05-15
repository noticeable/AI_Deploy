import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PointNetEncoder, feature_transform_regularizer


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reg_weight = config.model.pointnet.stn_regularization_weight
        self.num_seg_classes = config.dataset.n_seg_classes
        self.num_cls = config.dataset.n_classes
        self.encoder = PointNetEncoder(config.dataset.n_channels)
        self.conv1 = nn.Conv1d(1088 + self.num_cls, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.num_seg_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, points):
        x = points.transpose(1, 2)
        global_feature, point_feature, trans_feat = self.encoder(x)
        num_points = points.size(1)
        cls_one_hot = torch.zeros(points.size(0), self.num_cls, device=points.device)
        cls_feature = cls_one_hot.unsqueeze(-1).repeat(1, 1, num_points)
        global_feature = global_feature.unsqueeze(-1).repeat(1, 1, num_points)
        x = torch.cat([point_feature, global_feature, cls_feature], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        seg_logits = self.conv4(x).transpose(1, 2)
        regularization_loss = feature_transform_regularizer(trans_feat) * self.reg_weight
        return {
            'seg_logits': seg_logits,
            'regularization_loss': regularization_loss,
        }
