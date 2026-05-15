import torch.nn as nn

from .common_v1 import ClassificationHead, PointTransformerSequence, TransitionDown
from .pointtransformer_utils import load_pointtransformer_v1_pretrained


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        pt_cfg = config.model.pointtransformer
        dims = pt_cfg.encoder_dims
        input_channels = config.dataset.n_channels
        num_classes = config.dataset.n_classes

        self.input_proj = nn.Linear(input_channels, dims[0])
        self.encoder_stages = nn.ModuleList([
            PointTransformerSequence(dims[0], pt_cfg.depth[0], pt_cfg.k,
                                     pt_cfg.mlp_ratio, pt_cfg.dropout,
                                     pt_cfg.drop_path),
            PointTransformerSequence(dims[1], pt_cfg.depth[1], pt_cfg.kv_k,
                                     pt_cfg.mlp_ratio, pt_cfg.dropout,
                                     pt_cfg.drop_path),
            PointTransformerSequence(dims[2], pt_cfg.depth[2], pt_cfg.kv_k,
                                     pt_cfg.mlp_ratio, pt_cfg.dropout,
                                     pt_cfg.drop_path),
        ])
        self.downsamples = nn.ModuleList([
            TransitionDown(dims[0], dims[1], pt_cfg.k, pt_cfg.sampling_ratio[1]),
            TransitionDown(dims[1], dims[2], pt_cfg.kv_k, pt_cfg.sampling_ratio[2]),
        ])
        self.head = ClassificationHead(dims[2], num_classes, pt_cfg.dropout)
        self.pretrained_load_info = load_pointtransformer_v1_pretrained(self, config)

    def forward(self, points):
        xyz = points[..., :3]
        features = self.input_proj(points)

        xyz, features = self.encoder_stages[0](xyz, features)
        xyz, features = self.downsamples[0](xyz, features)

        xyz, features = self.encoder_stages[1](xyz, features)
        xyz, features = self.downsamples[1](xyz, features)

        xyz, features = self.encoder_stages[2](xyz, features)
        logits = self.head(features)
        return {'logits': logits}
