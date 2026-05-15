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
    """Feed-forward network inside MViT blocks / MViT 块中的前馈网络。"""

    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Transform token features with an MLP / 用 MLP 变换 token 特征。"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PoolAttention(nn.Module):
    """Attention that pools keys and values spatially / 对 key 和 value 做空间池化的注意力。"""

    def __init__(self, dim, num_heads, qkv_bias, attention_dropout, dropout,
                 pool_stride):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads)**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.pool_stride = pool_stride
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, spatial_shape):
        """Apply attention after pooling token maps / 先池化 token 对应特征图，再执行注意力。"""
        b, n, c = x.shape
        h, w = spatial_shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        pooled = x.transpose(1, 2).reshape(b, c, h, w)
        pooled = nn.functional.avg_pool2d(pooled,
                                          kernel_size=self.pool_stride,
                                          stride=self.pool_stride,
                                          ceil_mode=True)
        pooled = pooled.reshape(b, c, -1).transpose(1, 2)
        kv = self.kv(pooled).reshape(b, -1, 2, self.num_heads,
                                     c // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiScaleBlock(nn.Module):
    """Transformer block with pooled attention / 带池化注意力的多尺度 Transformer 块。"""

    def __init__(self, dim, num_heads, mlp_dim, dropout, attention_dropout,
                 drop_path, qkv_bias, pool_stride):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PoolAttention(dim, num_heads, qkv_bias,
                                  attention_dropout, dropout, pool_stride)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_dim, dropout)

    def forward(self, x, spatial_shape):
        """Apply attention and MLP residual branches / 依次应用注意力与 MLP 残差分支。"""
        x = x + self.drop_path(self.attn(self.norm1(x), spatial_shape))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Convert the input image into patch tokens / 将输入图像转换为 patch token。"""

    def __init__(self, in_channels, emb_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels,
                              emb_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        """Patchify the image and return spatial shape / 切分 patch，并返回对应空间尺寸。"""
        x = self.proj(x)
        h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        return x, (h, w)


class Network(nn.Module):
    """Multiscale Vision Transformer classifier / 多尺度视觉 Transformer 分类网络。"""

    def __init__(self, config):
        """Build patch embedding, multiscale blocks and classifier head / 构建 patch embedding、多尺度块和分类头。"""
        super().__init__()
        model_config = config.model.mvit
        self.patch_embed = PatchEmbed(config.dataset.n_channels,
                                      model_config.emb_dim,
                                      model_config.patch_size)
        tokens = (config.dataset.image_size // model_config.patch_size)**2
        self.class_token = nn.Parameter(torch.zeros(1, 1, model_config.emb_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, tokens + 1, model_config.emb_dim))
        self.dropout = nn.Dropout(model_config.dropout)
        dpr = torch.linspace(0, model_config.drop_path_rate,
                             model_config.num_layers).tolist()
        self.blocks = nn.ModuleList([
            MultiScaleBlock(model_config.emb_dim, model_config.num_heads,
                            model_config.mlp_dim, model_config.dropout,
                            model_config.attention_dropout, dpr[i],
                            model_config.qkv_bias, 2 if i % 2 == 0 else 1)
            for i in range(model_config.num_layers)
        ])
        self.norm = nn.LayerNorm(model_config.emb_dim)
        self.head = nn.Linear(model_config.emb_dim, config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'mvit')

    def _reset_parameters(self):
        """Initialize embeddings, blocks and classifier head / 初始化 embedding、主干块和分类头。"""
        nn.init.zeros_(self.class_token)
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
        """Encode the image and keep a class token / 编码图像 token，并保留 class token。"""
        x, spatial_shape = self.patch_embed(x)
        class_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.pos_embedding[:, :x.shape[1]]
        x = self.dropout(x)
        for block in self.blocks:
            x_cls, x_tokens = x[:, :1], x[:, 1:]
            x_tokens = block(x_tokens, spatial_shape)
            x = torch.cat((x_cls, x_tokens), dim=1)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        return self.head(self.forward_features(x))
