import torch
import torch.nn as nn

from .common import PointVoxelBackbone
from .utils import load_pointvoxel_seg_pretrained


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        pv_cfg = config.model.pointvoxel
        block_channels = list(pv_cfg.block_channels)
        self.backbone = PointVoxelBackbone(
            config.dataset.n_channels,
            block_channels,
            pv_cfg.voxel_resolution,
        )
        seg_input_dim = sum(block_channels) + block_channels[-1]
        self.seg_head = nn.Sequential(
            nn.Conv1d(seg_input_dim, pv_cfg.decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(pv_cfg.decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(pv_cfg.dropout),
            nn.Conv1d(pv_cfg.decoder_dim, pv_cfg.decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(pv_cfg.decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(pv_cfg.dropout),
            nn.Conv1d(pv_cfg.decoder_dim, config.dataset.n_seg_classes, kernel_size=1),
        )
        self.pretrained_load_info = load_pointvoxel_seg_pretrained(self, config)

    def forward(self, points):
        encoded = self.backbone(points)
        combined = list(encoded['stage_features'])
        combined.append(encoded['fused_features'])
        seg_logits = self.seg_head(torch.cat(combined, dim=1)).transpose(1, 2)
        return {'seg_logits': seg_logits}
