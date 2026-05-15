import math

import torch
import torch.nn as nn

from ..vit_utils import load_generic_pretrained


class LocalAggregation(nn.Module):
    """Depthwise local mixing block used by EdgeViT / EdgeViT 使用的深度卷积局部聚合模块。"""

    def __init__(self, dim):
        super().__init__()
        self.depthwise = nn.Conv2d(dim,
                                   dim,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        """Aggregate nearby spatial features with convolutions / 用卷积聚合邻域空间特征。"""
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(x)


class GlobalSparseAttention(nn.Module):
    """Global attention branch used after local aggregation / 在局部聚合后使用的全局注意力分支。"""

    def __init__(self, dim, num_heads, qkv_bias, attention_dropout, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads)**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """Run multi-head attention on the token sequence / 在 token 序列上执行多头注意力。"""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads,
                                  c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EdgeVitBlock(nn.Module):
    """EdgeViT block combining local aggregation and global attention / 结合局部聚合与全局注意力的 EdgeViT 块。"""

    def __init__(self, dim, num_heads, mlp_dim, qkv_bias, attention_dropout,
                 dropout):
        super().__init__()
        self.local = LocalAggregation(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GlobalSparseAttention(dim, num_heads, qkv_bias,
                                          attention_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x, spatial_shape):
        """Apply local mixing, attention and MLP residual updates / 依次应用局部混合、注意力与 MLP 残差更新。"""
        b, n, c = x.shape
        h, w = spatial_shape
        x_2d = x.transpose(1, 2).reshape(b, c, h, w)
        x_2d = self.local(x_2d)
        x = x_2d.flatten(2).transpose(1, 2)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Network(nn.Module):
    """EdgeViT image classifier / EdgeViT 图像分类网络。"""

    def __init__(self, config):
        """Build patch embedding, EdgeViT blocks and classifier head / 构建 patch embedding、EdgeViT 块和分类头。"""
        super().__init__()
        model_config = config.model.edgevit
        self.patch_embed = nn.Conv2d(config.dataset.n_channels,
                                     model_config.emb_dim,
                                     kernel_size=model_config.patch_size,
                                     stride=model_config.patch_size)
        self.spatial_size = config.dataset.image_size // model_config.patch_size
        self.blocks = nn.ModuleList([
            EdgeVitBlock(model_config.emb_dim, model_config.num_heads,
                         model_config.mlp_dim, model_config.qkv_bias,
                         model_config.attention_dropout,
                         model_config.dropout)
            for _ in range(model_config.num_layers)
        ])
        self.norm = nn.LayerNorm(model_config.emb_dim)
        self.head = nn.Linear(model_config.emb_dim, config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'edgevit')

    def _reset_parameters(self):
        """Initialize patch embedding, blocks and classifier head / 初始化 patch embedding、主干块和分类头。"""
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
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
        """Encode the image into pooled EdgeViT features / 将图像编码为池化后的 EdgeViT 特征。"""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        spatial_shape = (self.spatial_size, self.spatial_size)
        for block in self.blocks:
            x = block(x, spatial_shape)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        return self.head(self.forward_features(x))
