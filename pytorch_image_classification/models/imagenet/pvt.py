import math

import torch
import torch.nn as nn

from ..vit_utils import load_generic_pretrained


class Mlp(nn.Module):
    """Feed-forward block used inside PVT / PVT 中使用的前馈网络块。"""

    def __init__(self, in_features, hidden_features, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Project token features through the MLP / 对 token 特征执行前馈变换。"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PatchEmbed(nn.Module):
    """Convert feature maps to token sequences / 将特征图转换为 token 序列。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=kernel_size // 2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        """Patchify the input and return spatial shape / 切分 patch，并返回对应空间尺寸。"""
        x = self.proj(x)
        h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (h, w)


class DropPath(nn.Module):
    """Stochastic depth layer / 随机深度层。"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Randomly drop whole residual paths during training / 训练时随机丢弃整条残差分支。"""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape,
                                               dtype=x.dtype,
                                               device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PyramidAttention(nn.Module):
    """Spatial-reduction attention used by PVT / PVT 使用的空间降采样注意力。"""

    def __init__(self, dim, num_heads, sr_ratio, dropout, attention_dropout,
                 qkv_bias):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads)**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim,
                                dim,
                                kernel_size=sr_ratio,
                                stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, spatial_shape):
        """Attend over tokens with optional spatial reduction / 先可选降采样，再执行 token 注意力。"""
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        if self.sr is not None:
            h, w = spatial_shape
            pooled = x.transpose(1, 2).reshape(b, c, h, w)
            pooled = self.sr(pooled).reshape(b, c, -1).transpose(1, 2)
            pooled = self.norm(pooled)
        else:
            pooled = x

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


class PyramidBlock(nn.Module):
    """Single transformer block inside a PVT stage / PVT 某个 stage 内的单个 Transformer 块。"""

    def __init__(self, dim, num_heads, mlp_ratio, sr_ratio, dropout,
                 attention_dropout, drop_path, qkv_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PyramidAttention(dim, num_heads, sr_ratio, dropout,
                                     attention_dropout, qkv_bias)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, spatial_shape):
        """Apply attention and MLP residuals / 依次执行注意力与 MLP 残差更新。"""
        x = x + self.drop_path(self.attn(self.norm1(x), spatial_shape))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Network(nn.Module):
    """Pyramid Vision Transformer classifier / 金字塔视觉 Transformer 分类网络。"""

    def __init__(self, config):
        """Build four-stage PVT encoder and classifier head / 构建四阶段 PVT 编码器与分类头。"""
        super().__init__()
        model_config = config.model.pvt
        dims = model_config.stage_dims
        num_heads = model_config.num_heads
        depths = model_config.num_layers
        mlp_ratios = model_config.mlp_ratios
        sr_ratios = model_config.sr_ratios
        drop_path_rate = model_config.drop_path_rate
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        patch_sizes = [model_config.patch_size, 2, 2, 2]
        strides = [4, 2, 2, 2]
        in_channels = [config.dataset.n_channels] + dims[:-1]
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(in_channels[0], dims[0], patch_sizes[0], strides[0]),
            PatchEmbed(in_channels[1], dims[1], patch_sizes[1], strides[1]),
            PatchEmbed(in_channels[2], dims[2], patch_sizes[2], strides[2]),
            PatchEmbed(in_channels[3], dims[3], patch_sizes[3], strides[3]),
        ])

        block_index = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(
                    PyramidBlock(dims[i], num_heads[i], mlp_ratios[i],
                                 sr_ratios[i], model_config.dropout,
                                 model_config.attention_dropout,
                                 dpr[block_index], model_config.qkv_bias))
                block_index += 1
            self.stages.append(nn.ModuleList(blocks))

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'pvt')

    def _reset_parameters(self):
        """Initialize convolutional, linear and normalization layers / 初始化卷积、线性层和归一化层。"""
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
        """Encode the image through all pyramid stages / 经过全部金字塔 stage 提取图像特征。"""
        for patch_embed, blocks in zip(self.patch_embeds, self.stages):
            x, spatial_shape = patch_embed(x)
            for block in blocks:
                x = block(x, spatial_shape)
            h, w = spatial_shape
            x = x.transpose(1, 2).reshape(x.size(0), x.size(2), h, w)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        x = self.forward_features(x)
        return self.head(x)
