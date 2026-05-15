import pathlib
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from pytorch_point_cloud.transforms import create_transform


class ModelNet40Dataset(Dataset):
    def __init__(self, config, is_train: bool):
        split = 'train' if is_train else 'test'
        root = pathlib.Path(config.dataset.dataset_dir).expanduser() / 'modelnet40'
        self.points_path = root / f'{split}_points.npy'
        self.labels_path = root / f'{split}_labels.npy'
        self.transform = create_transform(config, is_train)
        self.num_points = config.dataset.num_points
        self.n_channels = config.dataset.n_channels

        if self.points_path.exists() and self.labels_path.exists():
            self.points = np.load(self.points_path)
            self.labels = np.load(self.labels_path)
        else:
            self.points = np.zeros((8, self.num_points, self.n_channels), dtype=np.float32)
            self.labels = np.zeros((8,), dtype=np.int64)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        points = self.transform(self.points[index].copy())
        label = torch.as_tensor(int(self.labels[index]), dtype=torch.long)
        return points, label


class ShapeNetPartDataset(Dataset):
    def __init__(self, config, is_train: bool):
        split = 'train' if is_train else 'test'
        root = pathlib.Path(config.dataset.dataset_dir).expanduser() / 'shapenet_part'
        self.points_path = root / f'{split}_points.npy'
        self.cls_path = root / f'{split}_cls.npy'
        self.seg_path = root / f'{split}_seg.npy'
        self.transform = create_transform(config, is_train)
        self.num_points = config.dataset.num_points
        self.n_channels = config.dataset.n_channels

        if self.points_path.exists() and self.cls_path.exists() and self.seg_path.exists():
            self.points = np.load(self.points_path)
            self.cls_labels = np.load(self.cls_path)
            self.seg_labels = np.load(self.seg_path)
        else:
            self.points = np.zeros((8, self.num_points, self.n_channels), dtype=np.float32)
            self.cls_labels = np.zeros((8,), dtype=np.int64)
            self.seg_labels = np.zeros((8, self.num_points), dtype=np.int64)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index: int):
        points = self.transform(self.points[index].copy())
        cls_label = torch.as_tensor(int(self.cls_labels[index]), dtype=torch.long)
        seg_label = torch.as_tensor(self.seg_labels[index], dtype=torch.long)
        return points, {
            'cls_label': cls_label,
            'seg_label': seg_label,
        }



def create_dataset(config, is_train):
    if config.dataset.name == 'ModelNet40':
        return ModelNet40Dataset(config, is_train)
    if config.dataset.name == 'ShapeNetPart':
        return ShapeNetPartDataset(config, is_train)
    raise ValueError(f'Unsupported dataset: {config.dataset.name}')
