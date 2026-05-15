import random

import torch
import torch.nn.functional as F


class DetSegTransform:
    def __init__(self, config, is_train):
        self.config = config
        self.is_train = is_train
        self.image_size = config.dataset.image_size

    def _resize_image(self, image):
        image = image.float()
        if image.dim() == 3:
            image = image.unsqueeze(0)
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            image = image.squeeze(0)
        return image

    def _resize_mask(self, mask):
        mask = mask.long().unsqueeze(0).unsqueeze(0).float()
        mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode='nearest')
        return mask.squeeze(0).squeeze(0).long()

    def _normalize(self, image):
        if self.config.augmentation.normalize:
            image = image / 255.0
        return image

    def _flip(self, image, target):
        if not self.is_train or not self.config.augmentation.use_random_horizontal_flip:
            return image, target
        if random.random() >= 0.5:
            return image, target
        image = torch.flip(image, dims=[2])
        target = dict(target)
        target['segmentation_mask'] = torch.flip(target['segmentation_mask'], dims=[1])
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
        target = dict(target)
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        if target['boxes'].numel() == 0:
            target['boxes'] = target['boxes'].reshape(0, 4)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.long)
        target['segmentation_mask'] = self._resize_mask(target['segmentation_mask'])
        target['size'] = [self.image_size, self.image_size]
        image, target = self._flip(image, target)
        image = self._normalize(image)
        return image, target



def create_transform(config, is_train):
    return DetSegTransform(config, is_train)
