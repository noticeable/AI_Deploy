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
    """Feed-forward network inside transformer blocks / Transformer 块中的前馈网络。"""

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


class Attention(nn.Module):
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


class Block(nn.Module):
    """Transformer block used after tokenization / token 化之后使用的 Transformer 块。"""

    def __init__(self, emb_dim, num_heads, mlp_dim, dropout,
                 attention_dropout, drop_path, qkv_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = Attention(emb_dim, num_heads, qkv_bias,
                              attention_dropout, dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = Mlp(emb_dim, mlp_dim, dropout)

    def forward(self, x):
        """Apply attention and MLP residual branches / 依次应用注意力与 MLP 残差分支。"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SoftSplit(nn.Module):
    """Extract overlapping image tokens with unfold / 用 unfold 提取带重叠的图像 token。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        self.proj = nn.Linear(in_channels * kernel_size * kernel_size,
                              out_channels)

    def forward(self, x):
        """Convert image patches into token embeddings / 将图像 patch 转成 token embedding。"""
        b, c, _, _ = x.shape
        x = self.unfold(x).transpose(1, 2)
        x = self.proj(x)
        return x


class TokenBackProjection(nn.Module):
    """Project tokens back to a feature map / 将 token 重新投影回特征图。"""

    def __init__(self, emb_dim, out_channels, spatial_size):
        super().__init__()
        self.proj = nn.Linear(emb_dim, out_channels)
        self.spatial_size = spatial_size

    def forward(self, x):
        """Restore 2D structure for the next soft split / 为下一次 soft split 恢复二维空间结构。"""
        b = x.shape[0]
        x = self.proj(x)
        h, w = self.spatial_size
        return x.transpose(1, 2).reshape(b, -1, h, w)


class Network(nn.Module):
    """Tokens-to-Token ViT classifier / Tokens-to-Token ViT 分类网络。"""

    def __init__(self, config):
        """Build progressive tokenization stages and transformer encoder / 构建渐进式 token 化阶段与 Transformer 编码器。"""
        super().__init__()
        model_config = config.model.t2t_vit
        self.soft_split0 = SoftSplit(config.dataset.n_channels,
                                     model_config.emb_dim // 4, 7, 4, 2)
        size0 = config.dataset.image_size // 4
        self.back0 = TokenBackProjection(model_config.emb_dim // 4,
                                         model_config.emb_dim // 4,
                                         (size0, size0))
        self.soft_split1 = SoftSplit(model_config.emb_dim // 4,
                                     model_config.emb_dim // 2, 3, 2, 1)
        size1 = size0 // 2
        self.back1 = TokenBackProjection(model_config.emb_dim // 2,
                                         model_config.emb_dim // 2,
                                         (size1, size1))
        self.soft_split2 = SoftSplit(model_config.emb_dim // 2,
                                     model_config.emb_dim, 3, 2, 1)
        size2 = size1 // 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, model_config.emb_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, size2 * size2 + 1, model_config.emb_dim))
        self.dropout = nn.Dropout(model_config.dropout)
        dpr = torch.linspace(0, model_config.drop_path_rate,
                             model_config.num_layers).tolist()
        self.blocks = nn.Sequential(*[
            Block(model_config.emb_dim, model_config.num_heads,
                  model_config.mlp_dim, model_config.dropout,
                  model_config.attention_dropout, dpr[i],
                  model_config.qkv_bias)
            for i in range(model_config.num_layers)
        ])
        self.norm = nn.LayerNorm(model_config.emb_dim)
        self.head = nn.Linear(model_config.emb_dim, config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 't2t_vit')

    def _reset_parameters(self):
        """Initialize class token, position embedding and head / 初始化 class token、位置编码和分类头。"""
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
        """Progressively tokenize the image and encode tokens / 渐进式 token 化图像并编码 token 序列。"""
        x = self.soft_split0(x)
        x = self.back0(x)
        x = self.soft_split1(x)
        x = self.back1(x)
        x = self.soft_split2(x)
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
