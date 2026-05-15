import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vit_utils import load_generic_pretrained


def window_partition(x, window_size):
    """Split a feature map into flattened local windows / 将特征图切分为展平的局部窗口。"""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size,
               c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size,
                                                c)


def window_reverse(windows, window_size, h, w, c):
    """Merge flattened windows back to a feature map / 将展平窗口重新拼回特征图。"""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size,
                     window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)
    return x


class SqueezeExcite(nn.Module):
    """Channel recalibration block used in MaxViT / MaxViT 中使用的通道重标定模块。"""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // 4)
        self.fc2 = nn.Linear(dim // 4, dim)

    def forward(self, x):
        """Reweight token channels with global statistics / 用全局统计信息重标定 token 通道。"""
        scale = x.mean(dim=1)
        scale = F.gelu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale)).unsqueeze(1)
        return x * scale


class Attention(nn.Module):
    """Standard multi-head attention used inside MaxViT / MaxViT 内部使用的标准多头注意力。"""

    def __init__(self, dim, num_heads, qkv_bias, attention_dropout, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads)**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """Run self-attention on a token sequence / 在 token 序列上执行自注意力。"""
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


class MaxVitBlock(nn.Module):
    """MaxViT block combining block attention and grid attention / 结合 block attention 与 grid attention 的 MaxViT 块。"""

    def __init__(self, dim, num_heads, mlp_dim, qkv_bias, attention_dropout,
                 dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.block_attn = Attention(dim, num_heads, qkv_bias,
                                    attention_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.grid_attn = Attention(dim, num_heads, qkv_bias,
                                   attention_dropout, dropout)
        self.se = SqueezeExcite(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x, spatial_shape, window_size):
        """Apply block attention, grid attention and MLP updates / 依次应用 block attention、grid attention 与 MLP 更新。"""
        b, n, c = x.shape
        h, w = spatial_shape
        x_2d = x.view(b, h, w, c)
        windows = window_partition(x_2d, window_size)
        windows = windows + self.block_attn(self.norm1(windows))
        x_2d = window_reverse(windows, window_size, h, w, c)
        grid = x_2d.permute(0, 2, 1, 3)
        grid = window_partition(grid, window_size)
        grid = grid + self.grid_attn(self.norm2(grid))
        grid = window_reverse(grid, window_size, w, h, c).permute(0, 2, 1, 3)
        x = grid.reshape(b, n, c)
        x = self.se(x)
        x = x + self.mlp(self.norm3(x))
        return x


class Network(nn.Module):
    """MaxViT image classifier / MaxViT 图像分类网络。"""

    def __init__(self, config):
        """Build patch embedding, MaxViT blocks and classifier head / 构建 patch embedding、MaxViT 块和分类头。"""
        super().__init__()
        model_config = config.model.maxvit
        self.window_size = 7
        self.patch_embed = nn.Conv2d(config.dataset.n_channels,
                                     model_config.emb_dim,
                                     kernel_size=model_config.patch_size,
                                     stride=model_config.patch_size)
        tokens = config.dataset.image_size // model_config.patch_size
        self.blocks = nn.ModuleList([
            MaxVitBlock(model_config.emb_dim, model_config.num_heads,
                        model_config.mlp_dim, model_config.qkv_bias,
                        model_config.attention_dropout,
                        model_config.dropout)
            for _ in range(model_config.num_layers)
        ])
        self.norm = nn.LayerNorm(model_config.emb_dim)
        self.head = nn.Linear(model_config.emb_dim, config.dataset.n_classes)
        self.spatial_shape = (tokens, tokens)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'maxvit')

    def _reset_parameters(self):
        """Initialize embedding, transformer and head parameters / 初始化 embedding、Transformer 与分类头参数。"""
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x):
        """Encode the image into pooled MaxViT features / 将图像编码为池化后的 MaxViT 特征。"""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x, self.spatial_shape, self.window_size)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        return self.head(self.forward_features(x))
