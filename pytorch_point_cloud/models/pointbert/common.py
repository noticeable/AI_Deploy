import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_point_cloud.models.pointtransformer.common import furthest_point_sample, index_points, knn


class PointPatchEmbedding(nn.Module):
    def __init__(self, input_channels, hidden_dim, embed_dim, k):
        super().__init__()
        self.k = k
        extra_channels = max(0, input_channels - 3)
        self.input_proj = nn.Sequential(
            nn.Linear(extra_channels + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.post_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, points, centers):
        center_xyz = index_points(points[..., :3], centers)
        neighbor_idx = knn(center_xyz, min(self.k, center_xyz.shape[1]))
        grouped_xyz = index_points(center_xyz, neighbor_idx)
        relative_xyz = grouped_xyz - center_xyz.unsqueeze(2)

        if points.shape[-1] > 3:
            center_features = index_points(points[..., 3:], centers)
            grouped_features = index_points(center_features, neighbor_idx)
            patch_inputs = torch.cat([relative_xyz, grouped_features], dim=-1)
        else:
            patch_inputs = relative_xyz

        embedded = self.input_proj(patch_inputs)
        embedded = embedded.max(dim=2)[0]
        return center_xyz, self.post_proj(embedded)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class PointBERTBackbone(nn.Module):
    def __init__(self, input_channels, num_groups, group_size, embed_dim, depth,
                 num_heads, mlp_ratio, dropout):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.patch_embed = PointPatchEmbedding(input_channels, embed_dim, embed_dim, group_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

    def forward(self, points):
        xyz = points[..., :3]
        num_groups = min(self.num_groups, xyz.shape[1])
        center_idx = furthest_point_sample(xyz, num_groups)
        center_xyz, patch_tokens = self.patch_embed(points, center_idx)

        pos_tokens = self.pos_embed(center_xyz)
        cls_token = self.cls_token.expand(points.shape[0], -1, -1)
        cls_pos = self.cls_pos.expand(points.shape[0], -1, -1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        pos = torch.cat([cls_pos, pos_tokens], dim=1)
        tokens = tokens + pos

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return {
            'cls_token': tokens[:, 0],
            'patch_tokens': tokens[:, 1:],
            'center_xyz': center_xyz,
        }


class PointBERTClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, features):
        return self.net(features)
