import torch.nn as nn

from .common import DGCNNBackbone
from .utils import load_dgcnn_cls_pretrained


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        dgcnn_cfg = config.model.dgcnn
        edge_dims = list(dgcnn_cfg.edge_dims)
        emb_dims = dgcnn_cfg.emb_dims
        self.backbone = DGCNNBackbone(config.dataset.n_channels, edge_dims, emb_dims, dgcnn_cfg.k)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dims, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dgcnn_cfg.dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dgcnn_cfg.dropout),
            nn.Linear(256, config.dataset.n_classes),
        )
        self.pretrained_load_info = load_dgcnn_cls_pretrained(self, config)

    def forward(self, points):
        features = self.backbone(points)
        logits = self.classifier(features['global_feature'])
        return {'logits': logits}
