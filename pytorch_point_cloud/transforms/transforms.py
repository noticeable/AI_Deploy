from typing import Callable, List

import numpy as np
import torch


class NormalizePointCloud:
    def __call__(self, points: np.ndarray) -> np.ndarray:
        centroid = np.mean(points[:, :3], axis=0, keepdims=True)
        points[:, :3] = points[:, :3] - centroid
        scale = np.max(np.linalg.norm(points[:, :3], axis=1))
        if scale > 0:
            points[:, :3] = points[:, :3] / scale
        return points


class RandomPointDropout:
    def __init__(self, max_dropout_ratio: float):
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points: np.ndarray) -> np.ndarray:
        dropout_ratio = np.random.random() * self.max_dropout_ratio
        drop_idx = np.where(np.random.random(points.shape[0]) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            points[drop_idx] = points[0]
        return points


class RandomScalePointCloud:
    def __init__(self, scale_low: float, scale_high: float):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, points: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.scale_low, self.scale_high)
        points[:, :3] *= scale
        return points


class RandomShiftPointCloud:
    def __init__(self, shift_range: float):
        self.shift_range = shift_range

    def __call__(self, points: np.ndarray) -> np.ndarray:
        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        points[:, :3] += shift
        return points


class RandomRotatePointCloud:
    def __call__(self, points: np.ndarray) -> np.ndarray:
        theta = np.random.uniform(0, 2 * np.pi)
        cosval = np.cos(theta)
        sinval = np.sin(theta)
        rotation = np.array([[cosval, 0, sinval],
                             [0, 1, 0],
                             [-sinval, 0, cosval]], dtype=np.float32)
        points[:, :3] = points[:, :3] @ rotation.T
        return points


class JitterPointCloud:
    def __init__(self, sigma: float, clip: float):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points: np.ndarray) -> np.ndarray:
        jitter = np.clip(self.sigma * np.random.randn(*points[:, :3].shape),
                         -self.clip,
                         self.clip)
        points[:, :3] += jitter
        return points


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, points: np.ndarray) -> torch.Tensor:
        for transform in self.transforms:
            points = transform(points)
        return torch.from_numpy(points.astype(np.float32))



def create_transform(config, is_train: bool):
    transforms = [NormalizePointCloud()]
    if is_train:
        pc_aug = config.augmentation.point_cloud
        transforms.extend([
            RandomPointDropout(pc_aug.dropout_ratio),
            RandomScalePointCloud(pc_aug.scale_low, pc_aug.scale_high),
            RandomShiftPointCloud(pc_aug.shift_range),
            RandomRotatePointCloud(),
            JitterPointCloud(pc_aug.jitter_sigma, pc_aug.jitter_clip),
        ])
    return Compose(transforms)
