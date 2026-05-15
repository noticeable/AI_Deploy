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
    """Feed-forward network inside CCT blocks / CCT 块中的前馈网络。"""

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


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention / 标准多头自注意力。"""

    def __init__(self, emb_dim, num_heads, qkv_bias, attention_dropout,
                 dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (emb_dim // num_heads)**-0.5
        self.qkv = nn.Linear(emb_dim, emb_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """Run scaled dot-product attention on tokens / 在 token 序列上执行缩放点积注意力。"""
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


class SequencePooling(nn.Module):
    """Attention-based token pooling / 基于注意力的 token 池化。"""

    def __init__(self, emb_dim):
        super().__init__()
        self.attention = nn.Linear(emb_dim, 1)

    def forward(self, x):
        """Aggregate a token sequence into one vector / 将 token 序列聚合为单个向量。"""
        attn = self.attention(x).softmax(dim=1)
        return (attn * x).sum(dim=1)


class EncoderBlock(nn.Module):
    """Transformer encoder block used by CCT / CCT 使用的 Transformer 编码块。"""

    def __init__(self, emb_dim, num_heads, mlp_dim, dropout,
                 attention_dropout, drop_path, qkv_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadSelfAttention(emb_dim, num_heads, qkv_bias,
                                           attention_dropout, dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = Mlp(emb_dim, mlp_dim, dropout)

    def forward(self, x):
        """Apply attention and MLP residual branches / 依次应用注意力与 MLP 残差分支。"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvTokenizer(nn.Module):
    """Tokenize an image with convolutions and pooling / 通过卷积与池化将图像 token 化。"""

    def __init__(self, in_channels, emb_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim // 2, kernel_size=7, stride=2,
                      padding=3),
            nn.GELU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(emb_dim // 2, emb_dim, kernel_size=3, stride=1,
                      padding=1), nn.GELU())

    def forward(self, x):
        """Convert an image to a token sequence / 将图像转换为 token 序列。"""
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Network(nn.Module):
    """Compact Convolutional Transformer classifier / 紧凑卷积 Transformer 分类网络。"""

    def __init__(self, config):
        """Build tokenizer, transformer encoder and classification head / 构建 tokenizer、Transformer 编码器和分类头。"""
        super().__init__()
        model_config = config.model.cct
        self.tokenizer = ConvTokenizer(config.dataset.n_channels,
                                       model_config.emb_dim)
        tokens = (config.dataset.image_size // 4)**2
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, tokens, model_config.emb_dim))
        self.dropout = nn.Dropout(model_config.dropout)
        dpr = torch.linspace(0, model_config.drop_path_rate,
                             model_config.num_layers).tolist()
        self.blocks = nn.Sequential(*[
            EncoderBlock(model_config.emb_dim, model_config.num_heads,
                         model_config.mlp_dim, model_config.dropout,
                         model_config.attention_dropout, dpr[i],
                         model_config.qkv_bias)
            for i in range(model_config.num_layers)
        ])
        self.norm = nn.LayerNorm(model_config.emb_dim)
        self.pool = SequencePooling(model_config.emb_dim)
        self.head = nn.Linear(model_config.emb_dim, config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'cct')

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
        """Encode the image into a pooled feature vector / 将图像编码为池化后的特征向量。"""
        x = self.tokenizer(x)
        x = x + self.pos_embedding[:, :x.shape[1]]
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pool(x)

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        return self.head(self.forward_features(x))
