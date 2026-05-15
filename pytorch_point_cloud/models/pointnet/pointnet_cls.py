import torch.nn as nn
import torch.nn.functional as F

from .common import PointNetEncoder, feature_transform_regularizer


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reg_weight = config.model.pointnet.stn_regularization_weight
        self.encoder = PointNetEncoder(config.dataset.n_channels)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, config.dataset.n_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(config.model.pointnet.dropout)

    def forward(self, points):
        x = points.transpose(1, 2)
        global_feature, _, trans_feat = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(global_feature)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        logits = self.fc3(x)
        regularization_loss = feature_transform_regularizer(trans_feat) * self.reg_weight
        return {
            'logits': logits,
            'regularization_loss': regularization_loss,
        }
