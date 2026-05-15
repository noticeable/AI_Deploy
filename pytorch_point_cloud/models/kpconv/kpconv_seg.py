import torch
import torch.nn as nn

from .common import KPConvEncoder
from .utils import load_kpconv_seg_pretrained


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        kpconv_cfg = config.model.kpconv
        encoder_dims = list(kpconv_cfg.encoder_dims)
        self.encoder = KPConvEncoder(
            config.dataset.n_channels,
            encoder_dims,
            kpconv_cfg.k,
            kpconv_cfg.kernel_points,
            kpconv_cfg.sigma,
        )
        seg_input_dim = sum(encoder_dims) + encoder_dims[-1]
        self.seg_head = nn.Sequential(
            nn.Conv1d(seg_input_dim, kpconv_cfg.decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(kpconv_cfg.decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(kpconv_cfg.dropout),
            nn.Conv1d(kpconv_cfg.decoder_dim, kpconv_cfg.decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(kpconv_cfg.decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(kpconv_cfg.dropout),
            nn.Conv1d(kpconv_cfg.decoder_dim, config.dataset.n_seg_classes, kernel_size=1),
        )
        self.pretrained_load_info = load_kpconv_seg_pretrained(self, config)

    def forward(self, points):
        encoded = self.encoder(points)
        combined = [feature.transpose(1, 2) for feature in encoded['skip_features']]
        combined.append(encoded['encoded_features'].transpose(1, 2))
        seg_logits = self.seg_head(torch.cat(combined, dim=1)).transpose(1, 2)
        return {'seg_logits': seg_logits}
