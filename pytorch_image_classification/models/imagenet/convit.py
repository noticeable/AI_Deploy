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
    """Feed-forward network inside ConViT blocks / ConViT 块中的前馈网络。"""

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


class GatedPositionalSelfAttention(nn.Module):
    """Attention with a locality bias / 带局部性偏置的自注意力。"""

    def __init__(self, emb_dim, num_heads, qkv_bias, attention_dropout,
                 dropout, locality_strength):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (emb_dim // num_heads)**-0.5
        self.qkv = nn.Linear(emb_dim, emb_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.locality_strength = locality_strength

    def forward(self, x):
        """Apply attention while encouraging nearby-token focus / 在注意力中显式加入邻近 token 偏置。"""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads,
                                  c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        local_bias = torch.eye(n, device=x.device, dtype=x.dtype)
        local_bias = local_bias.unsqueeze(0).unsqueeze(0)
        attn = attn + self.locality_strength * local_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard self-attention used in later ConViT layers / ConViT 后段使用的标准自注意力。"""

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
        """Run scaled dot-product self-attention / 执行标准缩放点积自注意力。"""
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


class ConvitBlock(nn.Module):
    """Transformer block that optionally uses GPSA / 可选 GPSA 的 ConViT Transformer 块。"""

    def __init__(self, emb_dim, num_heads, mlp_dim, dropout,
                 attention_dropout, drop_path, qkv_bias, use_gpsa,
                 locality_strength):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        if use_gpsa:
            self.attn = GatedPositionalSelfAttention(emb_dim, num_heads,
                                                     qkv_bias,
                                                     attention_dropout,
                                                     dropout,
                                                     locality_strength)
        else:
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


class Network(nn.Module):
    """ConViT image classifier / ConViT 图像分类网络。"""

    def __init__(self, config):
        """Build patch embedding, ConViT blocks and classifier head / 构建 patch embedding、ConViT 堆叠和分类头。"""
        super().__init__()
        model_config = config.model.convit
        image_size = config.dataset.image_size
        patch_size = model_config.patch_size
        if image_size % patch_size != 0:
            raise ValueError('dataset.image_size must be divisible by '
                             'model.convit.patch_size')

        self.emb_dim = model_config.emb_dim
        self.grid_size = image_size // patch_size
        self.seq_length = self.grid_size**2
        self.classifier = model_config.classifier
        self.conv_proj = nn.Conv2d(config.dataset.n_channels,
                                   self.emb_dim,
                                   kernel_size=patch_size,
                                   stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.seq_length + 1, self.emb_dim))
        self.dropout = nn.Dropout(model_config.dropout)

        dpr = torch.linspace(0, model_config.drop_path_rate,
                             model_config.num_layers).tolist()
        blocks = []
        for index in range(model_config.num_layers):
            blocks.append(
                ConvitBlock(self.emb_dim, model_config.num_heads,
                            model_config.mlp_dim, model_config.dropout,
                            model_config.attention_dropout, dpr[index],
                            model_config.qkv_bias,
                            index < model_config.local_layers,
                            model_config.locality_strength))
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(self.emb_dim)
        self.head = nn.Linear(self.emb_dim, config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'convit')

    def _reset_parameters(self):
        """Initialize embeddings, transformer blocks and head / 初始化 embedding、Transformer 块和分类头。"""
        nn.init.trunc_normal_(self.conv_proj.weight, std=0.02)
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
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
        """Encode an image into pooled token features / 将图像编码为聚合后的 token 特征。"""
        x = self.conv_proj(x).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.classifier == 'token':
            return x[:, 0]
        return x[:, 1:].mean(dim=1)

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        x = self.forward_features(x)
        return self.head(x)
