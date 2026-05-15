import torch.nn as nn

from .common import PointBERTBackbone, PointBERTClassifier
from .utils import load_pointbert_cls_pretrained


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        pointbert_cfg = config.model.pointbert
        self.backbone = PointBERTBackbone(
            config.dataset.n_channels,
            pointbert_cfg.num_groups,
            pointbert_cfg.group_size,
            pointbert_cfg.embed_dim,
            pointbert_cfg.depth,
            pointbert_cfg.num_heads,
            pointbert_cfg.mlp_ratio,
            pointbert_cfg.dropout,
        )
        self.classifier = PointBERTClassifier(
            pointbert_cfg.embed_dim,
            config.dataset.n_classes,
            pointbert_cfg.dropout,
        )
        self.pretrained_load_info = load_pointbert_cls_pretrained(self, config)

    def forward(self, points):
        encoded = self.backbone(points)
        logits = self.classifier(encoded['cls_token'])
        return {
            'logits': logits,
            'cls_token': encoded['cls_token'],
            'patch_tokens': encoded['patch_tokens'],
        }
