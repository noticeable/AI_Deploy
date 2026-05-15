import torch
import torch.nn as nn
import torch.nn.functional as F



def square_distance(src, dst):
    return torch.cdist(src, dst, p=2) ** 2



def index_points(points, idx):
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]



def farthest_point_sample(xyz, npoint):
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch_size, num_points), 1e10, device=xyz.device)
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids



def knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    return dist.topk(k=min(k, xyz.shape[1]), dim=-1, largest=False)[1]


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class RelativePositionEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

    def forward(self, relative_xyz):
        batch_size, num_points, num_neighbors, channels = relative_xyz.shape[0], relative_xyz.shape[1], relative_xyz.shape[2], 3
        pos = self.net(relative_xyz.reshape(-1, channels))
        return pos.view(batch_size, num_points, num_neighbors, -1)


class PointTransformerBlock(nn.Module):
    def __init__(self, dim, k, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.k = k
        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.position_encoding = RelativePositionEncoding(dim)
        self.attention_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path)
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
        idx = knn_point(self.k, xyz, xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_features = index_points(normalized, idx)
        relative_xyz = xyz.unsqueeze(2) - grouped_xyz

        query = self.to_query(normalized).unsqueeze(2)
        key = self.to_key(grouped_features)
        value = self.to_value(grouped_features)
        position = self.position_encoding(relative_xyz)

        attention_logits = self.attention_mlp(query - key + position)
        attention = F.softmax(attention_logits, dim=2)
        value_with_position = value + position
        aggregated = torch.sum(attention * value_with_position, dim=2)

        features = shortcut + self.drop_path(self.output_proj(aggregated))
        features = features + self.drop_path(self.mlp(self.norm2(features)))
        return xyz, features


class PointTransformerSequence(nn.Module):
    def __init__(self, dim, depth, k, mlp_ratio, dropout, drop_path):
        super().__init__()
        self.blocks = nn.ModuleList([
            PointTransformerBlock(dim, k, mlp_ratio, dropout, drop_path)
            for _ in range(depth)
        ])

    def forward(self, xyz, features):
        for block in self.blocks:
            xyz, features = block(xyz, features)
        return xyz, features


class TransitionDown(nn.Module):
    def __init__(self, in_dim, out_dim, k, ratio):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.feature_proj = nn.Linear(in_dim, out_dim)
        self.position_encoding = RelativePositionEncoding(out_dim)
        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, xyz, features):
        npoint = max(1, int(xyz.shape[1] * self.ratio))
        fps_idx = farthest_point_sample(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = knn_point(self.k, xyz, new_xyz)
        grouped_features = index_points(features, idx)
        grouped_xyz = index_points(xyz, idx)
        relative_xyz = new_xyz.unsqueeze(2) - grouped_xyz

        projected_features = self.feature_proj(grouped_features)
        position = self.position_encoding(relative_xyz)
        fused = self.fusion(projected_features + position)
        new_features = fused.max(dim=2)[0]
        new_features = self.norm(new_features)
        return new_xyz, new_features


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, features):
        return self.net(features.mean(dim=1))
