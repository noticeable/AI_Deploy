import math

import torch
import torch.nn as nn

from ..vit_utils import load_generic_pretrained


class LayerNorm2d(nn.Module):
    """Apply LayerNorm on channel-last view of a feature map / 在特征图的 channel-last 视图上应用 LayerNorm。"""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """Normalize a 2D feature map channel-wise / 对二维特征图按通道做归一化。"""
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvMlp(nn.Module):
    """Convolutional MLP used by CvT blocks / CvT 块中使用的卷积式 MLP。"""

    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Transform feature maps with pointwise convolutions / 用逐点卷积变换特征图。"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ConvAttention(nn.Module):
    """Attention implemented directly on feature maps / 直接在特征图上实现的注意力。"""

    def __init__(self, dim, num_heads, qkv_bias, attention_dropout, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads)**-0.5
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """Run multi-head attention on spatial tokens / 在空间 token 上执行多头注意力。"""
        b, c, h, w = x.shape
        q = self.q(x).reshape(b, self.num_heads, c // self.num_heads,
                              h * w).transpose(-2, -1)
        k = self.k(x).reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = self.v(x).reshape(b, self.num_heads, c // self.num_heads,
                              h * w).transpose(-2, -1)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-2, -1).reshape(b, c, h, w)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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


class CvtBlock(nn.Module):
    """CvT transformer block operating on feature maps / 作用在特征图上的 CvT Transformer 块。"""

    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, dropout,
                 attention_dropout, drop_path):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = ConvAttention(dim, num_heads, qkv_bias, attention_dropout,
                                  dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = ConvMlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        """Apply attention and MLP residual branches / 依次应用注意力与 MLP 残差分支。"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Downsample features into the next stage / 将特征下采样并映射到下一 stage。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=kernel_size // 2)
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x):
        """Project and normalize a feature map / 投影并归一化特征图。"""
        x = self.proj(x)
        x = self.norm(x)
        return x


class Network(nn.Module):
    """Convolutional vision transformer classifier / 卷积视觉 Transformer 分类网络。"""

    def __init__(self, config):
        """Build staged patch embeddings, CvT blocks and head / 构建分阶段 patch embedding、CvT 块和分类头。"""
        super().__init__()
        model_config = config.model.cvt
        dims = model_config.stage_dims
        depths = model_config.num_layers
        heads = model_config.num_heads
        kernels = model_config.stage_patch_sizes
        strides = model_config.stage_strides
        mlp_ratios = model_config.mlp_ratios
        dpr = torch.linspace(0, model_config.drop_path_rate, sum(depths)).tolist()

        in_channels = [config.dataset.n_channels] + dims[:-1]
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(in_channels[i], dims[i], kernels[i], strides[i])
            for i in range(len(dims))
        ])

        self.stages = nn.ModuleList()
        block_index = 0
        for stage_index, depth in enumerate(depths):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    CvtBlock(dims[stage_index], heads[stage_index],
                             mlp_ratios[stage_index], model_config.qkv_bias,
                             model_config.dropout,
                             model_config.attention_dropout,
                             dpr[block_index]))
                block_index += 1
            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'cvt')

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
        """Encode the image through all CvT stages / 经过全部 CvT stage 提取图像特征。"""
        for patch_embed, blocks in zip(self.patch_embeds, self.stages):
            x = patch_embed(x)
            x = blocks(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        x = self.forward_features(x)
        return self.head(x)
