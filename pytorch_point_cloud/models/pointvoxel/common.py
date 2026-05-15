import torch
import torch.nn as nn



def build_voxel_grid(points, resolution):
    coords = points[..., :3]
    min_coords = coords.amin(dim=1, keepdim=True)
    max_coords = coords.amax(dim=1, keepdim=True)
    normalized = (coords - min_coords) / (max_coords - min_coords + 1e-6)
    voxel_coords = (normalized * (resolution - 1)).long().clamp(min=0, max=resolution - 1)
    return voxel_coords



def gather_voxel_features(voxel_features, voxel_coords):
    sampled = []
    for batch_index in range(voxel_features.shape[0]):
        coords = voxel_coords[batch_index]
        sampled.append(voxel_features[batch_index, :, coords[:, 0], coords[:, 1], coords[:, 2]])
    return torch.stack(sampled, dim=0)



class PointBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        return self.net(features)



class VoxelBranch(nn.Module):
    def __init__(self, in_channels, out_channels, resolution):
        super().__init__()
        self.resolution = resolution
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, features):
        batch_size = points.shape[0]
        voxel_coords = build_voxel_grid(points, self.resolution)
        voxel_grid = torch.zeros(
            batch_size,
            features.shape[1],
            self.resolution,
            self.resolution,
            self.resolution,
            device=features.device,
            dtype=features.dtype,
        )
        for batch_index in range(batch_size):
            coords = voxel_coords[batch_index]
            voxel_grid[batch_index, :, coords[:, 0], coords[:, 1], coords[:, 2]] += features[batch_index]
        voxel_features = self.net(voxel_grid)
        return gather_voxel_features(voxel_features, voxel_coords)



class PointVoxelFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, voxel_resolution):
        super().__init__()
        self.point_branch = PointBranch(in_channels, out_channels)
        self.voxel_branch = VoxelBranch(in_channels, out_channels, voxel_resolution)
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, features):
        point_features = self.point_branch(features)
        voxel_features = self.voxel_branch(points, features)
        return self.fusion(torch.cat([point_features, voxel_features], dim=1))



class PointVoxelBackbone(nn.Module):
    def __init__(self, input_channels, block_channels, voxel_resolution):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, block_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(block_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        current_dim = block_channels[0]
        for out_dim in block_channels:
            self.blocks.append(PointVoxelFusionBlock(current_dim, out_dim, voxel_resolution))
            current_dim = out_dim

    def forward(self, points):
        features = points.transpose(1, 2).contiguous()
        features = self.stem(features)
        stage_features = []
        for block in self.blocks:
            features = block(points, features)
            stage_features.append(features)
        return {
            'stage_features': stage_features,
            'fused_features': features,
        }
