import math

import torch
import torch.nn as nn
import torch.nn.functional as F



def square_distance(src, dst):
    return torch.cdist(src, dst, p=2) ** 2



def knn(xyz, k):
    dist = square_distance(xyz, xyz)
    return dist.topk(k=k, dim=-1, largest=False)[1]



def index_points(points, idx):
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]



def furthest_point_sample(xyz, npoint):
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(batch_size, num_points, device=xyz.device) * 1e10
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class RelativePositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, relative_xyz):
        return self.net(relative_xyz)


class PointTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, k, mlp_ratio=4.0, dropout=0.0, drop_path=0.0,
                 qkv_bias=True, use_relative_position=True):
        super().__init__()
        self.k = k
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_relative_position = use_relative_position

        self.norm1 = nn.LayerNorm(dim)
        self.to_query = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_key = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_value = nn.Linear(dim, dim, bias=qkv_bias)
        self.position_encoding = RelativePositionEncoding(dim) if use_relative_position else None
        self.attention_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.output_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, xyz, features):
        shortcut = features
        normalized = self.norm1(features)
        idx = knn(xyz, self.k)
        grouped_xyz = index_points(xyz, idx)
        grouped_features = index_points(normalized, idx)
        relative_xyz = xyz.unsqueeze(2) - grouped_xyz

        query = self.to_query(normalized).view(normalized.shape[0], normalized.shape[1], self.num_heads, self.head_dim)
        key = self.to_key(grouped_features).view(normalized.shape[0], normalized.shape[1], self.k, self.num_heads, self.head_dim)
        value = self.to_value(grouped_features).view(normalized.shape[0], normalized.shape[1], self.k, self.num_heads, self.head_dim)

        attention_logits = query.unsqueeze(2) - key
        value_with_position = value
        if self.position_encoding is not None:
            position = self.position_encoding(relative_xyz).view(
                normalized.shape[0], normalized.shape[1], self.k, self.num_heads, self.head_dim)
            attention_logits = attention_logits + position
            value_with_position = value + position
        attention_logits = self.attention_mlp(
            attention_logits.reshape(normalized.shape[0], normalized.shape[1], self.k, self.dim))
        attention = attention_logits.view(
            normalized.shape[0], normalized.shape[1], self.k, self.num_heads, self.head_dim)
        attention = F.softmax(attention * self.scale, dim=2)
        aggregated = (attention * value_with_position).sum(dim=2).reshape(
            normalized.shape[0], normalized.shape[1], self.dim)
        features = shortcut + self.drop_path(self.output_proj(aggregated))
        features = features + self.drop_path(self.mlp(self.norm2(features)))
        return xyz, features


class PointTransformerStage(nn.Module):
    def __init__(self, dim, depth, num_heads, k, mlp_ratio, dropout, drop_path,
                 qkv_bias, use_relative_position):
        super().__init__()
        self.blocks = nn.ModuleList([
            PointTransformerLayer(dim, num_heads, k, mlp_ratio, dropout, drop_path,
                                  qkv_bias, use_relative_position)
            for _ in range(depth)
        ])

    def forward(self, xyz, features):
        for block in self.blocks:
            xyz, features = block(xyz, features)
        return xyz, features


class TransitionDown(nn.Module):
    def __init__(self, in_dim, out_dim, k, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.feature_proj = nn.Linear(in_dim, out_dim)
        self.position_encoding = RelativePositionEncoding(out_dim)
        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, xyz, features):
        num_sampled = max(1, int(xyz.shape[1] * self.ratio))
        fps_idx = furthest_point_sample(xyz, num_sampled)
        new_xyz = index_points(xyz, fps_idx)
        dist = square_distance(new_xyz, xyz)
        idx = dist.topk(k=min(self.k, xyz.shape[1]), dim=-1, largest=False)[1]
        grouped_features = index_points(features, idx)
        grouped_xyz = index_points(xyz, idx)
        relative_xyz = new_xyz.unsqueeze(2) - grouped_xyz

        projected_features = self.feature_proj(grouped_features)
        position = self.position_encoding(relative_xyz)
        fused = self.fusion(projected_features + position)
        new_features = fused.max(dim=2)[0]
        new_features = self.norm(new_features)
        return new_xyz, new_features


class TransitionUp(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.upsample_proj = nn.Linear(in_dim, out_dim)
        self.skip_proj = nn.Linear(skip_dim, out_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, xyz, features, skip_xyz, skip_features):
        if xyz.shape[1] != skip_xyz.shape[1]:
            interpolated = F.interpolate(features.transpose(1, 2), size=skip_xyz.shape[1], mode='nearest').transpose(1, 2)
        else:
            interpolated = features
        upsampled_features = self.upsample_proj(interpolated)
        skip_projected = self.skip_proj(skip_features)
        fused = upsampled_features + skip_projected
        return skip_xyz, self.fusion(fused)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, features):
        return self.head(features.mean(dim=1))


class SegmentationHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.head(features)
