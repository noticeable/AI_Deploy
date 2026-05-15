import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        return x.view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_transform=True):
        super().__init__()
        self.input_transform = TNet(k=in_channels)
        self.feature_transform = TNet(k=64) if feature_transform else None
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        trans = self.input_transform(x)
        x = torch.bmm(trans, x)
        x = F.relu(self.bn1(self.conv1(x)))
        trans_feat = None
        if self.feature_transform is not None:
            trans_feat = self.feature_transform(x)
            x = torch.bmm(trans_feat, x)
        point_features = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        global_feature = torch.max(x, 2, keepdim=False)[0]
        return global_feature, point_features, trans_feat



def feature_transform_regularizer(trans):
    if trans is None:
        return 0.0
    d = trans.size(1)
    identity = torch.eye(d, device=trans.device).unsqueeze(0)
    diff = torch.bmm(trans, trans.transpose(2, 1)) - identity
    return torch.mean(torch.norm(diff, dim=(1, 2)))
