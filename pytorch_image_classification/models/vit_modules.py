import math
from collections import OrderedDict

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """Transformer MLP block / Transformer 前馈块。"""

    def __init__(self, emb_dim, mlp_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, emb_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """Project to MLP space and back / 先升维做非线性变换，再投影回 embedding 维度。"""
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class SelfAttention(nn.Module):
    """Explicit multi-head self-attention / 显式实现的多头自注意力。"""

    def __init__(self, emb_dim, num_heads, attention_dropout):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError('emb_dim must be divisible by num_heads')
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def _reshape_heads(self, x):
        """Split channel dim into heads / 将通道维拆分成多头布局。"""
        batch_size, seq_length, _ = x.shape
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Accept legacy packed QKV weights / 兼容旧版打包在一起的 QKV 权重。"""
        in_proj_weight_key = prefix + 'in_proj_weight'
        in_proj_bias_key = prefix + 'in_proj_bias'
        q_proj_weight_key = prefix + 'q_proj.weight'
        q_proj_bias_key = prefix + 'q_proj.bias'

        if (in_proj_weight_key in state_dict
                and q_proj_weight_key not in state_dict):
            q_weight, k_weight, v_weight = state_dict[in_proj_weight_key].chunk(3,
                                                                                dim=0)
            state_dict[q_proj_weight_key] = q_weight
            state_dict[prefix + 'k_proj.weight'] = k_weight
            state_dict[prefix + 'v_proj.weight'] = v_weight
        if in_proj_bias_key in state_dict and q_proj_bias_key not in state_dict:
            q_bias, k_bias, v_bias = state_dict[in_proj_bias_key].chunk(3, dim=0)
            state_dict[q_proj_bias_key] = q_bias
            state_dict[prefix + 'k_proj.bias'] = k_bias
            state_dict[prefix + 'v_proj.bias'] = v_bias

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        """Run scaled dot-product attention / 执行缩放点积注意力计算。"""
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        x = torch.matmul(attention, v)
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], self.emb_dim)
        x = self.out_proj(x)
        return x


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block / 预归一化 Transformer 编码块。"""

    def __init__(self, emb_dim, num_heads, mlp_dim, dropout,
                 attention_dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.self_attention = SelfAttention(emb_dim,
                                            num_heads,
                                            attention_dropout)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = MLPBlock(emb_dim, mlp_dim, dropout)

    def forward(self, x):
        """Apply attention and MLP residual branches / 依次执行注意力残差分支和 MLP 残差分支。"""
        y = self.ln1(x)
        y = self.self_attention(y)
        x = x + self.dropout(y)
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nn.Module):
    """Stacked transformer encoder / 多层堆叠的 Transformer 编码器。"""

    def __init__(self, seq_length, num_layers, emb_dim, num_heads, mlp_dim,
                 dropout, attention_dropout):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, emb_dim))
        self.dropout = nn.Dropout(dropout)
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f'encoder_layer_{i}'] = EncoderBlock(
                emb_dim, num_heads, mlp_dim, dropout, attention_dropout)
        self.layers = nn.Sequential(layers)
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x):
        """Add positional embedding and encode tokens / 加位置编码后对 token 序列编码。"""
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)
        return x


class VisionTransformer(nn.Module):
    """Shared ViT/DeiT backbone / ViT 与 DeiT 共用主干。"""

    def __init__(self, config, model_name='vit'):
        super().__init__()

        model_config = getattr(config.model, model_name)
        image_size = config.dataset.image_size
        patch_size = model_config.patch_size
        if image_size % patch_size != 0:
            raise ValueError('dataset.image_size must be divisible by '
                             f'model.{model_name}.patch_size')

        self.model_name = model_name
        self.classifier = model_config.classifier
        self.representation_size = model_config.representation_size
        self.emb_dim = model_config.emb_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_size = image_size // patch_size
        self.seq_length = self.grid_size**2

        self.conv_proj = nn.Conv2d(config.dataset.n_channels,
                                   self.emb_dim,
                                   kernel_size=patch_size,
                                   stride=patch_size)

        if self.classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
            encoder_seq_length = self.seq_length + 1
        elif self.classifier == 'gap':
            self.class_token = None
            encoder_seq_length = self.seq_length
        else:
            raise ValueError(f'model.{model_name}.classifier must be token or gap')

        self.encoder = Encoder(encoder_seq_length,
                               model_config.num_layers,
                               self.emb_dim,
                               model_config.num_heads,
                               model_config.mlp_dim,
                               model_config.dropout,
                               model_config.attention_dropout)

        if self.representation_size > 0:
            self.pre_logits = nn.Linear(self.emb_dim, self.representation_size)
            self.act = nn.Tanh()
            head_in_features = self.representation_size
        else:
            self.pre_logits = nn.Identity()
            self.act = nn.Identity()
            head_in_features = self.emb_dim

        self.head = nn.Linear(head_in_features, config.dataset.n_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize patch embed, encoder and head / 初始化 patch embedding、编码器和分类头。"""
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1.0 /
                                                                   self.conv_proj.weight[0].numel()))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
        if self.class_token is not None:
            nn.init.zeros_(self.class_token)
        nn.init.normal_(self.encoder.pos_embedding, std=0.02)

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

    def _process_input(self, x):
        """Convert an image to patch tokens / 将输入图像转换成 patch token 序列。"""
        n, _, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(f'Expected input size {(self.image_size, self.image_size)}, '
                             f'got {(h, w)}')
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def forward_features(self, x):
        """Produce pooled transformer features / 生成送入分类头前的聚合特征。"""
        x = self._process_input(x)
        if self.class_token is not None:
            class_token = self.class_token.expand(x.size(0), -1, -1)
            x = torch.cat((class_token, x), dim=1)
        x = self.encoder(x)
        if self.classifier == 'token':
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        x = self.pre_logits(x)
        x = self.act(x)
        return x

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向过程。"""
        x = self.forward_features(x)
        x = self.head(x)
        return x
