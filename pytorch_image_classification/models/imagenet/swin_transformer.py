import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vit_utils import load_generic_pretrained


def window_partition(x, window_size):
    """Split a feature map into local windows / 将特征图切分为局部窗口。"""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size,
               c)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size,
                                                   window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """Merge local windows back to a feature map / 将局部窗口重新拼回特征图。"""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)
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


class Mlp(nn.Module):
    """Feed-forward network inside Swin blocks / Swin 块中的前馈网络。"""

    def __init__(self, dim, mlp_ratio, dropout):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
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


class WindowAttention(nn.Module):
    """Self-attention inside a local Swin window / Swin 局部窗口内的自注意力。"""

    def __init__(self, dim, num_heads, window_size, dropout,
                 attention_dropout, qkv_bias):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads)**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """Run attention on tokens within one window / 在单个窗口内对 token 执行注意力。"""
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


class SwinBlock(nn.Module):
    """Shifted-window transformer block / 带窗口平移的 Transformer 块。"""

    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio,
                 dropout, attention_dropout, drop_path, qkv_bias):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, dropout,
                                    attention_dropout, qkv_bias)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, dropout)

    def forward(self, x, spatial_shape):
        """Apply shifted-window attention then MLP / 先做平移窗口注意力，再做 MLP 更新。"""
        h, w = spatial_shape
        b, l, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            x = torch.roll(x,
                           shifts=(-self.shift_size, -self.shift_size),
                           dims=(1, 2))

        windows = window_partition(x, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size,
                                         c)
        x = window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))

        if pad_h > 0 or pad_w > 0:
            x = x[:, :h, :w, :]

        x = x.view(b, l, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Downsample tokens between Swin stages / 在 Swin stage 之间对 token 做下采样。"""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x, spatial_shape):
        """Merge each 2x2 neighborhood into one token / 将每个 2x2 邻域合并为一个 token。"""
        h, w = spatial_shape
        b, _, c = x.shape
        x = x.view(b, h, w, c)
        if h % 2 == 1 or w % 2 == 1:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            h, w = x.shape[1], x.shape[2]
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(b, -1, 4 * c)
        x = self.norm(x)
        x = self.reduction(x)
        return x, (h // 2, w // 2)


class PatchEmbed(nn.Module):
    """Convert the image into initial Swin tokens / 将图像转换为初始 Swin token。"""

    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Patchify and normalize the input image / 对输入图像做 patch 化并归一化。"""
        x = self.proj(x)
        h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (h, w)


class Network(nn.Module):
    """Swin Transformer classifier / Swin Transformer 分类网络。"""

    def __init__(self, config):
        """Build staged shifted-window blocks and classifier head / 构建分阶段平移窗口块与分类头。"""
        super().__init__()
        model_config = config.model.swin_transformer
        depths = model_config.num_layers
        heads = model_config.num_heads
        window_size = model_config.window_size
        mlp_ratios = model_config.mlp_ratios
        embed_dim = model_config.emb_dim
        drop_path_rate = model_config.drop_path_rate

        self.patch_embed = PatchEmbed(config.dataset.n_channels, embed_dim,
                                      model_config.patch_size)
        dims = [embed_dim * (2**i) for i in range(len(depths))]
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        block_index = 0
        for stage_index, depth in enumerate(depths):
            blocks = []
            for block_offset in range(depth):
                shift_size = 0 if block_offset % 2 == 0 else window_size // 2
                blocks.append(
                    SwinBlock(dims[stage_index], heads[stage_index],
                              window_size, shift_size,
                              mlp_ratios[stage_index], model_config.dropout,
                              model_config.attention_dropout,
                              dpr[block_index], model_config.qkv_bias))
                block_index += 1
            self.stages.append(nn.ModuleList(blocks))
            if stage_index < len(depths) - 1:
                self.downsamples.append(PatchMerging(dims[stage_index]))

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], config.dataset.n_classes)
        self._reset_parameters()
        load_generic_pretrained(self, config, 'swin_transformer')

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
        """Encode the image through all Swin stages / 经过全部 Swin stage 提取图像特征。"""
        x, spatial_shape = self.patch_embed(x)
        for stage_index, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, spatial_shape)
            if stage_index < len(self.downsamples):
                x, spatial_shape = self.downsamples[stage_index](x, spatial_shape)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        x = self.forward_features(x)
        return self.head(x)
