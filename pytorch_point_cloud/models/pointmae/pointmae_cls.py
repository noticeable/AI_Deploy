import torch.nn as nn

from .common import PointMAEBackbone, PointMAEClassifier
from .utils import load_pointmae_cls_pretrained


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        pointmae_cfg = config.model.pointmae
        self.backbone = PointMAEBackbone(
            config.dataset.n_channels,
            pointmae_cfg.num_groups,
            pointmae_cfg.group_size,
            pointmae_cfg.embed_dim,
            pointmae_cfg.depth,
            pointmae_cfg.num_heads,
            pointmae_cfg.mlp_ratio,
            pointmae_cfg.dropout,
        )
        self.classifier = PointMAEClassifier(
            pointmae_cfg.embed_dim,
            config.dataset.n_classes,
            pointmae_cfg.dropout,
        )
        self.pretrained_load_info = load_pointmae_cls_pretrained(self, config)

    def forward(self, points):
        encoded = self.backbone(points)
        logits = self.classifier(encoded['global_feature'])
        return {
            'logits': logits,
            'global_feature': encoded['global_feature'],
            'patch_tokens': encoded['patch_tokens'],
        }
