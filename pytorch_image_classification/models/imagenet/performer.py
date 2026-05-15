import math

import torch
import torch.nn as nn

from ..vit_utils import load_generic_pretrained


class DropPath(nn.Module):
    """Stochastic depth layer / 随机深度层。"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Randomly drop residual paths during training / 训练时随机丢弃残差路径。"""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape,
                                               dtype=x.dtype,
                                               device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """Feed-forward network inside Performer blocks / Performer 块中的前馈网络。"""

    def __init__(self, emb_dim, mlp_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Transform token features with an MLP / 用 MLP 变换 token 特征。"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LinearAttention(nn.Module):
    """Performer-style linear attention / Performer 风格的线性注意力。"""

    def __init__(self, emb_dim, num_heads, qkv_bias, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_dim, emb_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """Approximate self-attention with linear complexity / 以线性复杂度近似自注意力计算。"""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads,
                                  c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q = torch.softmax(qkv[0], dim=-1)
        k = torch.softmax(qkv[1], dim=-2)
        v = qkv[2]
        context = k.transpose(-2, -1) @ v
        x = (q @ context).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PerformerBlock(nn.Module):
    """Transformer block using linear attention / 使用线性注意力的 Transformer 块。"""

    def __init__(self, emb_dim, num_heads, mlp_dim, qkv_bias, dropout,
                 drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = LinearAttention(emb_dim, num_heads, qkv_bias, dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = Mlp(emb_dim, mlp_dim, dropout)

    def forward(self, x):
        """Apply attention and MLP residual branches / 依次应用注意力与 MLP 残差分支。"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Network(nn.Module):
    """Performer image classifier / Performer 图像分类网络。"""

    def __init__(self, config):
        """Build patch embedding, Performer blocks and classifier head / 构建 patch embedding、Performer 块和分类头。"""
        super().__init__()
        model_config = config.model.performer
        self.patch_embed = nn.Conv2d(config.dataset.n_channels,
                                     model_config.emb_dim,
                                     kernel_size=model_config.patch_size,
                                     stride=model_config.patch_size)
        tokens = (config.dataset.image_size // model_config.patch_size)**2
        self.class_token = nn.Parameter(torch.zeros(1, 1, model_config.emb_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, tokens + 1, model_config.emb_dim))
        self.dropout = nn.Dropout(model_config.dropout)
        dpr = torch.linspace(0, model_config.drop_path_rate,
                             model_config.num_layers).tolist()
        self.blocks = nn.Sequential(*[
            PerformerBlock(model_config.emb_dim, model_config.num_heads,
                           model_config.mlp_dim, model_config.qkv_bias,
                           model_config.dropout, dpr[i])
            for i in range(model_config.num_layers)
        ])
        self.norm = nn.LayerNorm(model_config.emb_dim)
        self.head = nn.Linear(model_config.emb_dim, config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'performer')

    def _reset_parameters(self):
        """Initialize embeddings, transformer blocks and head / 初始化 embedding、Transformer 块和分类头。"""
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
        """Encode the image and keep the class token / 编码图像 token，并保留 class token。"""
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
