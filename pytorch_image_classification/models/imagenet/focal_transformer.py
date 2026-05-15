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
    """Feed-forward network inside focal blocks / Focal 块中的前馈网络。"""

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


class FocalAttention(nn.Module):
    """Attention that mixes local tokens with pooled context / 融合局部 token 与聚合上下文的注意力。"""

    def __init__(self, emb_dim, num_heads, qkv_bias, attention_dropout,
                 dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (emb_dim // num_heads)**-0.5
        self.qkv = nn.Linear(emb_dim, emb_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, context):
        """Attend over tokens augmented with a context token / 在上下文 token 增强后执行注意力。"""
        b, n, c = x.shape
        qkv = self.qkv(torch.cat([x, context], dim=1)).reshape(
            b, n + context.shape[1], 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q = qkv[0][:, :, :n]
        k, v = qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FocalBlock(nn.Module):
    """Transformer block using focal attention / 使用 focal attention 的 Transformer 块。"""

    def __init__(self, emb_dim, num_heads, mlp_dim, dropout,
                 attention_dropout, drop_path, qkv_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = FocalAttention(emb_dim, num_heads, qkv_bias,
                                   attention_dropout, dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = Mlp(emb_dim, mlp_dim, dropout)

    def forward(self, x):
        """Apply focal attention and MLP residual branches / 依次应用 focal 注意力与 MLP 残差分支。"""
        context = x.mean(dim=1, keepdim=True)
        x = x + self.drop_path(self.attn(self.norm1(x), context))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Network(nn.Module):
    """Focal Transformer image classifier / Focal Transformer 图像分类网络。"""

    def __init__(self, config):
        """Build patch embedding, focal blocks and classifier head / 构建 patch embedding、focal 块和分类头。"""
        super().__init__()
        model_config = config.model.focal_transformer
        self.classifier = model_config.classifier
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
            FocalBlock(model_config.emb_dim, model_config.num_heads,
                       model_config.mlp_dim, model_config.dropout,
                       model_config.attention_dropout, dpr[i],
                       model_config.qkv_bias)
            for i in range(model_config.num_layers)
        ])
        self.norm = nn.LayerNorm(model_config.emb_dim)
        self.head = nn.Linear(model_config.emb_dim, config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'focal_transformer')

    def _reset_parameters(self):
        """Initialize embeddings, patch embed, transformer blocks and head / 初始化 embedding、patch embed、Transformer 块和分类头。"""
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.zeros_(self.class_token)
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
        """Encode the image into pooled token features / 将图像编码为聚合后的 token 特征。"""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.pos_embedding[:, :x.shape[1]]
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.classifier == 'token':
            return x[:, 0]
        return x[:, 1:].mean(dim=1)

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        return self.head(self.forward_features(x))
