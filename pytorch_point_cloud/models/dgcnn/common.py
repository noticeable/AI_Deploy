import torch
import torch.nn as nn
import torch.nn.functional as F



def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]



def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3)
    return feature.permute(0, 3, 1, 2).contiguous()



class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x, k):
        features = get_graph_feature(x, k=k)
        features = self.net(features)
        return features.max(dim=-1)[0]



class DGCNNBackbone(nn.Module):
    def __init__(self, input_channels, edge_dims, emb_dims, k):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        current_in = input_channels
        for out_channels in edge_dims:
            self.blocks.append(EdgeConvBlock(current_in, out_channels))
            current_in = out_channels

        self.fusion = nn.Sequential(
            nn.Conv1d(sum(edge_dims), emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, points):
        x = points.transpose(1, 2).contiguous()
        stage_features = []
        current = x
        for block in self.blocks:
            current = block(current, self.k)
            stage_features.append(current)
        fused = self.fusion(torch.cat(stage_features, dim=1))
        global_feature = F.adaptive_max_pool1d(fused, 1).view(points.size(0), -1)
        return {
            'stage_features': stage_features,
            'fused_features': fused,
            'global_feature': global_feature,
        }
