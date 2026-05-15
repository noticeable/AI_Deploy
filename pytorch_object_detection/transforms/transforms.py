import math
import random

import torch
import torch.nn.functional as F


class DetectionTransform:
    def __init__(self, config, is_train):
        self.config = config
        self.is_train = is_train
        self.image_size = config.dataset.image_size

    def _resize_image(self, image):
        image = image.float()
        if image.dim() == 3:
            image = image.unsqueeze(0)
            image = F.interpolate(image,
                                  size=(self.image_size, self.image_size),
                                  mode='bilinear',
                                  align_corners=False)
            image = image.squeeze(0)
        return image

    def _normalize(self, image):
        if self.config.augmentation.normalize:
            image = image / 255.0
        return image

    def _flip(self, image, target):
        if not self.is_train or not self.config.augmentation.use_random_horizontal_flip:
            return image, target
        if random.random() >= self.config.augmentation.hflip_prob:
            return image, target
        image = torch.flip(image, dims=[2])
        boxes = target['boxes']
        if len(boxes) > 0:
            boxes = boxes.clone()
            width = image.shape[-1]
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = width - x2
            boxes[:, 2] = width - x1
            target['boxes'] = boxes
        return image, target

    def __call__(self, image, target):
        image = self._resize_image(image)
        target = _ensure_target_tensor_types(target)
        target['size'] = [self.image_size, self.image_size]
        image, target = self._flip(image, target)
        image = self._normalize(image)
        return image, target


class DetectionAugmentor:
    def __init__(self, config):
        self.config = config
        self.transform = DetectionTransform(config, is_train=True)
        self.image_size = config.dataset.image_size
        self.min_box_size = 2.0
        self.min_area_ratio = 0.05

    def apply(self, dataset, index, sample_loader):
        image, target = sample_loader(index)
        image, target = self.transform(image, target)
        if self.config.augmentation.use_mosaic and random.random() < self.config.augmentation.mosaic_prob:
            image, target = self._mosaic(dataset, index, sample_loader)
        if self.config.augmentation.use_mixup and random.random() < self.config.augmentation.mixup_prob:
            image, target = self._mixup(dataset, sample_loader, image, target)
        if self.config.augmentation.use_random_affine and random.random() < self.config.augmentation.affine_prob:
            image, target = self._random_projective_transform(image, target)
        target = _filter_invalid_boxes(target, self.image_size, self.min_box_size)
        return image, target

    def _mosaic(self, dataset, index, sample_loader):
        indices = [index] + [random.randrange(len(dataset)) for _ in range(3)]
        center_range = self.config.augmentation.mosaic_center_ratio_range
        cx = int(random.uniform(center_range[0], center_range[1]) * self.image_size)
        cy = int(random.uniform(center_range[0], center_range[1]) * self.image_size)
        canvas = torch.zeros((3, self.image_size * 2, self.image_size * 2),
                             dtype=torch.float32)
        placements = ((0, 0, cx, cy), (cx, 0, self.image_size * 2, cy),
                      (0, cy, cx, self.image_size * 2),
                      (cx, cy, self.image_size * 2, self.image_size * 2))
        merged_boxes = []
        merged_labels = []

        for sample_index, (x1a, y1a, x2a, y2a) in zip(indices, placements):
            mosaic_image, mosaic_target = sample_loader(sample_index)
            mosaic_image, mosaic_target = self.transform(mosaic_image,
                                                         mosaic_target)
            patch_h = y2a - y1a
            patch_w = x2a - x1a
            mosaic_patch = F.interpolate(mosaic_image.unsqueeze(0),
                                         size=(patch_h, patch_w),
                                         mode='bilinear',
                                         align_corners=False).squeeze(0)
            canvas[:, y1a:y2a, x1a:x2a] = mosaic_patch

            boxes = mosaic_target['boxes']
            if len(boxes) == 0:
                continue
            scale = torch.tensor([patch_w / self.image_size, patch_h / self.image_size,
                                  patch_w / self.image_size, patch_h / self.image_size],
                                 dtype=torch.float32)
            offset = torch.tensor([x1a, y1a, x1a, y1a], dtype=torch.float32)
            boxes = boxes * scale + offset
            merged_boxes.append(boxes)
            merged_labels.append(mosaic_target['labels'])

        image = F.interpolate(canvas.unsqueeze(0),
                              size=(self.image_size, self.image_size),
                              mode='bilinear',
                              align_corners=False).squeeze(0)
        target = {
            'image_id': index,
            'labels': torch.cat(merged_labels, dim=0)
            if merged_labels else torch.zeros((0,), dtype=torch.long),
            'boxes': torch.cat(merged_boxes, dim=0) * 0.5
            if merged_boxes else torch.zeros((0, 4), dtype=torch.float32),
            'area': torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((0,), dtype=torch.long),
            'orig_size': [self.image_size, self.image_size],
            'size': [self.image_size, self.image_size],
        }
        return image, _filter_invalid_boxes(target, self.image_size,
                                            self.min_box_size)

    def _mixup(self, dataset, sample_loader, image, target):
        mix_index = random.randrange(len(dataset))
        mix_image, mix_target = sample_loader(mix_index)
        mix_image, mix_target = self.transform(mix_image, mix_target)
        lam = random.betavariate(self.config.augmentation.mixup_alpha,
                                 self.config.augmentation.mixup_alpha)
        image = image * lam + mix_image * (1.0 - lam)
        boxes = [target['boxes']]
        labels = [target['labels']]
        if len(mix_target['boxes']) > 0:
            boxes.append(mix_target['boxes'])
            labels.append(mix_target['labels'])
        target = dict(target)
        target['boxes'] = torch.cat(boxes, dim=0) if boxes else torch.zeros(
            (0, 4), dtype=torch.float32)
        target['labels'] = torch.cat(labels, dim=0) if labels else torch.zeros(
            (0,), dtype=torch.long)
        target['area'] = _compute_box_areas(target['boxes'])
        target['iscrowd'] = torch.zeros((len(target['labels']),), dtype=torch.long)
        target['size'] = [self.image_size, self.image_size]
        return image, _filter_invalid_boxes(target, self.image_size,
                                            self.min_box_size)

    def _random_projective_transform(self, image, target):
        matrix = self._build_projective_matrix(image.device, image.dtype)
        image = self._warp_image(image, matrix)
        target = dict(target)
        boxes = target['boxes']
        if len(boxes) > 0:
            transformed_boxes = self._project_boxes(boxes.to(image.device), matrix)
            target['boxes'] = transformed_boxes.cpu()
        target['area'] = _compute_box_areas(target['boxes'])
        target['iscrowd'] = torch.zeros((len(target['labels']),), dtype=torch.long)
        return image, _filter_invalid_boxes(target,
                                            self.image_size,
                                            self.min_box_size,
                                            min_area_ratio=self.min_area_ratio,
                                            original_boxes=boxes.clone())

    def _build_projective_matrix(self, device, dtype):
        degrees = self.config.augmentation.affine_degrees
        translate = self.config.augmentation.affine_translate
        scale_range = self.config.augmentation.affine_scale_range
        shear = self.config.augmentation.affine_shear_degrees
        perspective = self.config.augmentation.affine_perspective

        angle = math.radians(random.uniform(-degrees, degrees))
        scale = random.uniform(scale_range[0], scale_range[1])
        tx = random.uniform(-translate, translate) * self.image_size
        ty = random.uniform(-translate, translate) * self.image_size
        shear_x = math.radians(random.uniform(-shear, shear))
        shear_y = math.radians(random.uniform(-shear, shear))
        p1 = random.uniform(-perspective, perspective)
        p2 = random.uniform(-perspective, perspective)

        c = math.cos(angle) * scale
        s = math.sin(angle) * scale

        center_to_origin = torch.tensor([[1.0, 0.0, -self.image_size / 2],
                                         [0.0, 1.0, -self.image_size / 2],
                                         [0.0, 0.0, 1.0]],
                                        device=device,
                                        dtype=dtype)
        rotation = torch.tensor([[c, -s, 0.0],
                                 [s, c, 0.0],
                                 [0.0, 0.0, 1.0]],
                                device=device,
                                dtype=dtype)
        shear_matrix = torch.tensor([[1.0, math.tan(shear_x), 0.0],
                                     [math.tan(shear_y), 1.0, 0.0],
                                     [0.0, 0.0, 1.0]],
                                    device=device,
                                    dtype=dtype)
        perspective_matrix = torch.tensor([[1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [p1, p2, 1.0]],
                                          device=device,
                                          dtype=dtype)
        translate_back = torch.tensor([[1.0, 0.0, self.image_size / 2 + tx],
                                       [0.0, 1.0, self.image_size / 2 + ty],
                                       [0.0, 0.0, 1.0]],
                                      device=device,
                                      dtype=dtype)
        return translate_back @ perspective_matrix @ shear_matrix @ rotation @ center_to_origin

    def _warp_image(self, image, matrix):
        height = self.image_size
        width = self.image_size
        yy, xx = torch.meshgrid(torch.arange(height, device=image.device, dtype=image.dtype),
                                torch.arange(width, device=image.device, dtype=image.dtype),
                                indexing='ij')
        ones = torch.ones_like(xx)
        dst = torch.stack((xx, yy, ones), dim=-1).reshape(-1, 3).T
        source = torch.linalg.inv(matrix) @ dst
        source_denominator = _stabilize_homogeneous_scale(source[2:])
        source_xy = source[:2] / source_denominator
        source_x = (source_xy[0].reshape(height, width) / max(width - 1, 1)) * 2 - 1
        source_y = (source_xy[1].reshape(height, width) / max(height - 1, 1)) * 2 - 1
        grid = torch.stack((source_x, source_y), dim=-1).unsqueeze(0)
        return F.grid_sample(image.unsqueeze(0),
                             grid,
                             mode='bilinear',
                             padding_mode='zeros',
                             align_corners=True).squeeze(0)

    def _project_boxes(self, boxes, matrix):
        corners = torch.stack([
            torch.stack([boxes[:, 0], boxes[:, 1]], dim=1),
            torch.stack([boxes[:, 2], boxes[:, 1]], dim=1),
            torch.stack([boxes[:, 2], boxes[:, 3]], dim=1),
            torch.stack([boxes[:, 0], boxes[:, 3]], dim=1),
        ], dim=1)
        ones = torch.ones((corners.shape[0], corners.shape[1], 1),
                          device=boxes.device,
                          dtype=boxes.dtype)
        homogenous = torch.cat([corners, ones], dim=2)
        warped = homogenous @ matrix.T
        warped_denominator = _stabilize_homogeneous_scale(warped[..., 2:])
        warped_xy = warped[..., :2] / warped_denominator
        x = warped_xy[..., 0]
        y = warped_xy[..., 1]
        return torch.stack([x.min(dim=1).values,
                            y.min(dim=1).values,
                            x.max(dim=1).values,
                            y.max(dim=1).values], dim=1)


def _ensure_target_tensor_types(target):
    target = dict(target)
    target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
    if target['boxes'].numel() == 0:
        target['boxes'] = target['boxes'].reshape(0, 4)
    target['labels'] = torch.as_tensor(target['labels'], dtype=torch.long)
    if 'area' in target:
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
    if 'iscrowd' in target:
        target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.long)
    return target


def _compute_box_areas(boxes):
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return widths * heights


def _stabilize_homogeneous_scale(scale):
    sign = torch.where(scale >= 0,
                       torch.ones_like(scale),
                       -torch.ones_like(scale))
    magnitude = scale.abs().clamp(min=1e-6)
    return sign * magnitude


def _filter_invalid_boxes(target,
                          image_size,
                          min_box_size,
                          min_area_ratio=0.0,
                          original_boxes=None):
    target = _ensure_target_tensor_types(target)
    boxes = target['boxes']
    labels = target['labels']
    area = target.get('area')
    iscrowd = target.get('iscrowd')
    if len(boxes) == 0:
        target['area'] = torch.zeros((0,), dtype=torch.float32)
        target['iscrowd'] = torch.zeros((0,), dtype=torch.long)
        return target
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_size)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_size)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths >= min_box_size) & (heights >= min_box_size)
    if original_boxes is not None and len(original_boxes) == len(boxes) and min_area_ratio > 0:
        original_boxes = torch.as_tensor(original_boxes, dtype=boxes.dtype, device=boxes.device)
        original_area = (original_boxes[:, 2] - original_boxes[:, 0]).clamp(min=1e-6) * \
                        (original_boxes[:, 3] - original_boxes[:, 1]).clamp(min=1e-6)
        new_area = widths * heights
        keep = keep & ((new_area / original_area) >= min_area_ratio)
    target['boxes'] = boxes[keep]
    target['labels'] = labels[keep]
    target['area'] = _compute_box_areas(target['boxes']) if area is None else area[keep]
    target['iscrowd'] = torch.zeros((len(target['labels']),), dtype=torch.long) if iscrowd is None else iscrowd[keep]
    return target


def create_transform(config, is_train):
    return DetectionTransform(config, is_train)


def create_augmentor(config):
    return DetectionAugmentor(config)
