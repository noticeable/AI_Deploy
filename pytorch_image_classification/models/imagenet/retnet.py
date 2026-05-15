import math

import torch
import torch.nn as nn

from ..vit_utils import load_generic_pretrained


class Retention(nn.Module):
    """Retention-style token mixing module / Retention 风格的 token mixing 模块。"""

    def __init__(self, emb_dim, num_heads, qkv_bias, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (emb_dim // num_heads)**-0.5
        self.q = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.k = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.v = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.gate = nn.Linear(emb_dim, emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Compute gated retention updates over the token sequence / 在 token 序列上计算带门控的 retention 更新。"""
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads)
        k = self.k(x).reshape(b, n, self.num_heads, c // self.num_heads)
        v = self.v(x).reshape(b, n, self.num_heads, c // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        retention = torch.softmax((q @ k.transpose(-2, -1)) * self.scale,
                                  dim=-1)
        x = (retention @ v).transpose(1, 2).reshape(b, n, c)
        gate = torch.sigmoid(self.gate(x))
        x = self.proj(x * gate)
        return self.dropout(x)


class Block(nn.Module):
    """RetNet transformer block / RetNet Transformer 块。"""

    def __init__(self, emb_dim, num_heads, mlp_dim, qkv_bias, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.retention = Retention(emb_dim, num_heads, qkv_bias, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, mlp_dim), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, emb_dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        """Apply retention and MLP residual branches / 依次应用 retention 与 MLP 残差分支。"""
        x = x + self.retention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Network(nn.Module):
    """RetNet image classifier / RetNet 图像分类网络。"""

    def __init__(self, config):
        """Build patch embedding, RetNet blocks and classifier head / 构建 patch embedding、RetNet 块和分类头。"""
        super().__init__()
        model_config = config.model.retnet
        self.patch_embed = nn.Conv2d(config.dataset.n_channels,
                                     model_config.emb_dim,
                                     kernel_size=model_config.patch_size,
                                     stride=model_config.patch_size)
        tokens = (config.dataset.image_size // model_config.patch_size)**2
        self.class_token = nn.Parameter(torch.zeros(1, 1, model_config.emb_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, tokens + 1, model_config.emb_dim))
        self.dropout = nn.Dropout(model_config.dropout)
        self.blocks = nn.Sequential(*[
            Block(model_config.emb_dim, model_config.num_heads,
                  model_config.mlp_dim, model_config.qkv_bias,
                  model_config.dropout)
            for _ in range(model_config.num_layers)
        ])
        self.norm = nn.LayerNorm(model_config.emb_dim)
        self.head = nn.Linear(model_config.emb_dim, config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'retnet')

    def _reset_parameters(self):
        """Initialize embeddings, RetNet blocks and classifier head / 初始化 embedding、RetNet 主干块和分类头。"""
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.zeros_(self.class_token)
        nn.init.normal_(self.pos_embedding, std=0.02)
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
        """Encode the image and return the class token feature / 编码图像并返回 class token 特征。"""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.pos_embedding[:, :x.shape[1]]
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        return self.head(self.forward_features(x))
