import torch.nn as nn

from .common import PointTransformerStage, SegmentationHead, TransitionDown, TransitionUp
from .pointtransformer_utils import load_pointtransformer_v3_pretrained


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        pt_cfg = config.model.pointtransformer
        dims = pt_cfg.encoder_dims
        decoder_dims = pt_cfg.decoder_dims
        input_channels = config.dataset.n_channels
        num_seg_classes = config.dataset.n_seg_classes

        self.version_tag = 'v3'
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, dims[0]),
            nn.LayerNorm(dims[0]),
            nn.GELU(),
        )
        self.encoder_depths = [pt_cfg.depth[0] + 1, pt_cfg.depth[1] + 1, pt_cfg.depth[2] + 2]
        self.decoder_depths = [pt_cfg.depth[1], pt_cfg.depth[0]]

        self.encoder_stages = nn.ModuleList([
            PointTransformerStage(dims[0], self.encoder_depths[0], pt_cfg.num_heads[0], pt_cfg.k,
                                  pt_cfg.mlp_ratio, pt_cfg.dropout, pt_cfg.drop_path,
                                  pt_cfg.qkv_bias, pt_cfg.use_relative_position),
            PointTransformerStage(dims[1], self.encoder_depths[1], pt_cfg.num_heads[1], pt_cfg.kv_k,
                                  pt_cfg.mlp_ratio, pt_cfg.dropout, pt_cfg.drop_path,
                                  pt_cfg.qkv_bias, pt_cfg.use_relative_position),
            PointTransformerStage(dims[2], self.encoder_depths[2], pt_cfg.num_heads[2], pt_cfg.kv_k,
                                  pt_cfg.mlp_ratio, pt_cfg.dropout, pt_cfg.drop_path,
                                  pt_cfg.qkv_bias, pt_cfg.use_relative_position),
        ])
        self.downsamples = nn.ModuleList([
            TransitionDown(dims[0], dims[1], pt_cfg.k, pt_cfg.sampling_ratio[1]),
            TransitionDown(dims[1], dims[2], pt_cfg.kv_k, pt_cfg.sampling_ratio[2]),
        ])
        self.decoder_stages = nn.ModuleList([
            nn.ModuleDict({
                'up': TransitionUp(dims[2], dims[1], decoder_dims[1]),
                'blocks': PointTransformerStage(decoder_dims[1], self.decoder_depths[0], pt_cfg.num_heads[1],
                                                pt_cfg.kv_k, pt_cfg.mlp_ratio, pt_cfg.dropout,
                                                pt_cfg.drop_path, pt_cfg.qkv_bias,
                                                pt_cfg.use_relative_position),
            }),
            nn.ModuleDict({
                'up': TransitionUp(decoder_dims[1], dims[0], decoder_dims[2]),
                'blocks': PointTransformerStage(decoder_dims[2], self.decoder_depths[1], pt_cfg.num_heads[0],
                                                pt_cfg.k, pt_cfg.mlp_ratio, pt_cfg.dropout,
                                                pt_cfg.drop_path, pt_cfg.qkv_bias,
                                                pt_cfg.use_relative_position),
            }),
        ])
        self.head = SegmentationHead(decoder_dims[2], pt_cfg.decoder_dim,
                                     num_seg_classes, pt_cfg.dropout)
        self.pretrained_load_info = load_pointtransformer_v3_pretrained(self, config)

    def forward(self, points):
        xyz0 = points[..., :3]
        features0 = self.input_proj(points)
        xyz0, features0 = self.encoder_stages[0](xyz0, features0)

        xyz1, features1 = self.downsamples[0](xyz0, features0)
        xyz1, features1 = self.encoder_stages[1](xyz1, features1)

        xyz2, features2 = self.downsamples[1](xyz1, features1)
        xyz2, features2 = self.encoder_stages[2](xyz2, features2)

        xyz, features = self.decoder_stages[0]['up'](xyz2, features2, xyz1, features1)
        xyz, features = self.decoder_stages[0]['blocks'](xyz, features)
        xyz, features = self.decoder_stages[1]['up'](xyz, features, xyz0, features0)
        xyz, features = self.decoder_stages[1]['blocks'](xyz, features)

        seg_logits = self.head(features)
        return {'seg_logits': seg_logits}
