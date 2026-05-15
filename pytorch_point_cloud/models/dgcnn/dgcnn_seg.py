import torch
import torch.nn as nn

from .common import DGCNNBackbone
from .utils import load_dgcnn_seg_pretrained


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        dgcnn_cfg = config.model.dgcnn
        edge_dims = list(dgcnn_cfg.edge_dims[:3])
        emb_dims = dgcnn_cfg.emb_dims
        self.num_cls = config.dataset.n_classes
        self.backbone = DGCNNBackbone(config.dataset.n_channels, edge_dims, emb_dims, dgcnn_cfg.k)
        seg_input_dim = sum(edge_dims) + emb_dims + self.num_cls
        self.seg_head = nn.Sequential(
            nn.Conv1d(seg_input_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dgcnn_cfg.dropout),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dgcnn_cfg.dropout),
            nn.Conv1d(256, config.dataset.n_seg_classes, kernel_size=1),
        )
        self.pretrained_load_info = load_dgcnn_seg_pretrained(self, config)

    def forward(self, points):
        features = self.backbone(points)
        num_points = points.size(1)
        local_features = torch.cat(features['stage_features'], dim=1)
        cls_one_hot = torch.zeros(points.size(0), self.num_cls, num_points, device=points.device)
        seg_features = torch.cat([local_features, features['fused_features'], cls_one_hot], dim=1)
        seg_logits = self.seg_head(seg_features).transpose(1, 2)
        return {'seg_logits': seg_logits}
