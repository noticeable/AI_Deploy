import math

import torch
import torch.nn as nn

from ..vit_utils import load_generic_pretrained


class LinearBlock(nn.Module):
    """Lightweight token-mixing block used by EfficientViT / EfficientViT 使用的轻量 token mixing 模块。"""

    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        """Apply token mixing and MLP residual updates / 依次应用 token mixing 与 MLP 残差更新。"""
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Network(nn.Module):
    """EfficientViT image classifier / EfficientViT 图像分类网络。"""

    def __init__(self, config):
        """Build patch embedding, lightweight blocks and classifier head / 构建 patch embedding、轻量块和分类头。"""
        super().__init__()
        model_config = config.model.efficientvit
        self.patch_embed = nn.Sequential(
            nn.Conv2d(config.dataset.n_channels,
                      model_config.emb_dim // 2,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.GELU(),
            nn.Conv2d(model_config.emb_dim // 2,
                      model_config.emb_dim,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.GELU())
        tokens = (config.dataset.image_size // 4)**2
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, tokens, model_config.emb_dim))
        self.blocks = nn.Sequential(*[
            LinearBlock(model_config.emb_dim, model_config.mlp_dim,
                        model_config.dropout)
            for _ in range(model_config.num_layers)
        ])
        self.norm = nn.LayerNorm(model_config.emb_dim)
        self.head = nn.Linear(model_config.emb_dim, config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'efficientvit')

    def _reset_parameters(self):
        """Initialize embeddings, convolutions, linear layers and head / 初始化位置编码、卷积层、线性层和分类头。"""
        nn.init.normal_(self.pos_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                fan_out //= module.groups
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x):
        """Encode the image into pooled token features / 将图像编码为池化后的 token 特征。"""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embedding[:, :x.shape[1]]
        x = self.blocks(x)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        return self.head(self.forward_features(x))
