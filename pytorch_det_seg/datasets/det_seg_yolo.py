import pathlib

from PIL import Image
import torch
from torch.utils.data import Dataset

from pytorch_det_seg.transforms import create_transform


class DetSegYOLODataset(Dataset):
    IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(self, config, is_train):
        self.config = config
        self.is_train = is_train
        self.root = pathlib.Path(config.dataset.dataset_dir)
        split_dir = self.root / ('train' if is_train else 'val')
        image_dir = split_dir / 'images'
        self.images = sorted(
            path for path in image_dir.rglob('*')
            if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES) if image_dir.exists() else []
        self.transform = create_transform(config, is_train)
        if not self.images:
            self.images = [None]

    def __len__(self):
        return len(self.images)

    def _resolve_label_path(self, image_path):
        split_root = image_path.parent.parent
        relative = image_path.relative_to(split_root / 'images')
        return split_root / 'labels' / relative.with_suffix('.txt')

    def _resolve_mask_path(self, image_path):
        split_root = image_path.parent.parent
        relative = image_path.relative_to(split_root / 'images')
        return split_root / 'masks' / relative.with_suffix('.png')

    def _load_sample(self, index):
        size = self.config.dataset.image_size
        image_path = self.images[index]
        if image_path is None:
            image = torch.zeros((3, size, size), dtype=torch.float32)
            mask = torch.zeros((size, size), dtype=torch.long)
            return image, {
                'image_id': index,
                'labels': [],
                'boxes': [],
                'orig_size': [size, size],
                'size': [size, size],
                'segmentation_mask': mask,
            }

        image_pil = Image.open(image_path).convert('RGB')
        width, height = image_pil.size
        image = torch.from_numpy(__import__('numpy').asarray(image_pil).transpose(2, 0, 1)).float()

        label_path = self._resolve_label_path(image_path)
        boxes = []
        labels = []
        if label_path.exists():
            for line in label_path.read_text(encoding='utf-8').splitlines():
                fields = line.strip().split()
                if len(fields) != 5:
                    continue
                class_id, cx, cy, bw, bh = map(float, fields)
                x1 = (cx - bw / 2.0) * width
                y1 = (cy - bh / 2.0) * height
                x2 = (cx + bw / 2.0) * width
                y2 = (cy + bh / 2.0) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id))

        mask_path = self._resolve_mask_path(image_path)
        if mask_path.exists():
            mask = torch.from_numpy(__import__('numpy').asarray(Image.open(mask_path), dtype='int64')).long()
        else:
            mask = torch.zeros((height, width), dtype=torch.long)

        return image, {
            'image_id': index,
            'labels': labels,
            'boxes': boxes,
            'orig_size': [height, width],
            'size': [height, width],
            'segmentation_mask': mask,
        }

    def __getitem__(self, index):
        image, target = self._load_sample(index)
        image, target = self.transform(image, target)
        return image, target
