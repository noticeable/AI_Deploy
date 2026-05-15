import torch
import torch.nn as nn



def pairwise_distance(xyz_a, xyz_b):
    return torch.cdist(xyz_a, xyz_b, p=2)



def gather_neighbors(features, idx):
    batch_size = features.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, device=features.device).view(view_shape).repeat(repeat_shape)
    return features[batch_indices, idx, :]



class KPConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_points, sigma):
        super().__init__()
        self.kernel_points = kernel_points
        self.sigma = sigma
        self.kernel_weights = nn.Parameter(torch.randn(kernel_points, in_channels, out_channels) * 0.01)
        self.center_mlp = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, xyz, features, neighbor_idx):
        neighbor_features = gather_neighbors(features, neighbor_idx)
        neighbor_xyz = gather_neighbors(xyz, neighbor_idx)
        relative_xyz = xyz.unsqueeze(2) - neighbor_xyz
        distance = relative_xyz.norm(dim=-1, keepdim=True)
        influence = torch.exp(-(distance ** 2) / max(self.sigma ** 2, 1e-6))

        weighted_kernels = self.kernel_weights.mean(dim=0)
        aggregated = torch.einsum('bnki,io->bnko', neighbor_features, weighted_kernels)
        aggregated = (aggregated * influence).sum(dim=2)
        aggregated = aggregated + self.center_mlp(relative_xyz.mean(dim=2))
        return self.activation(self.norm(aggregated))



class KPConvEncoder(nn.Module):
    def __init__(self, input_channels, encoder_dims, k, kernel_points, sigma):
        super().__init__()
        self.k = k
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, encoder_dims[0]),
            nn.LayerNorm(encoder_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        current_dim = encoder_dims[0]
        for out_dim in encoder_dims:
            self.blocks.append(KPConvBlock(current_dim, out_dim, kernel_points, sigma))
            current_dim = out_dim

    def _neighbors(self, xyz):
        dist = pairwise_distance(xyz, xyz)
        return dist.topk(k=min(self.k, xyz.shape[1]), dim=-1, largest=False)[1]

    def forward(self, points):
        xyz = points[..., :3]
        features = self.input_proj(points)
        skip_features = []
        for block in self.blocks:
            neighbor_idx = self._neighbors(xyz)
            features = block(xyz, features, neighbor_idx)
            skip_features.append(features)
        return {
            'xyz': xyz,
            'skip_features': skip_features,
            'encoded_features': features,
        }
