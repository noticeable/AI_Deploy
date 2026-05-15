import json
import pathlib

from PIL import Image
import torch
from torch.utils.data import Dataset

from pytorch_object_detection.transforms import create_augmentor, create_transform


class COCODataset(Dataset):
    def __init__(self, config, is_train):
        self.config = config
        self.is_train = is_train
        self.root = pathlib.Path(config.dataset.dataset_dir)
        ann_file = config.dataset.train_ann if is_train else config.dataset.val_ann
        self.ann_path = pathlib.Path(ann_file) if ann_file else None
        self.transform = create_transform(config, is_train)
        self.augmentor = create_augmentor(config) if is_train else None
        self.images = []
        self.annotations = {}
        self.category_id_to_label = {}
        self.label_to_category_id = {}
        self.class_names = list(getattr(config.dataset, 'class_names', []))
        if self.ann_path and self.ann_path.exists():
            payload = json.loads(self.ann_path.read_text(encoding='utf-8'))
            self.images = payload.get('images', [])
            categories = sorted(payload.get('categories', []), key=lambda category: category['id'])
            if categories:
                self.category_id_to_label = {
                    int(category['id']): index for index, category in enumerate(categories)
                }
                self.label_to_category_id = {
                    index: int(category['id']) for index, category in enumerate(categories)
                }
                if not self.class_names:
                    self.class_names = [str(category.get('name', category['id'])) for category in categories]
            for annotation in payload.get('annotations', []):
                self.annotations.setdefault(annotation['image_id'], []).append(annotation)
        else:
            self.images = [{
                'id': 0,
                'file_name': '',
                'width': config.dataset.image_size,
                'height': config.dataset.image_size,
            }]

    def __len__(self):
        return len(self.images)

    def _resolve_image_path(self, image_info):
        file_name = image_info.get('file_name', '')
        if not file_name:
            return None
        direct_path = self.root / file_name
        if direct_path.exists():
            return direct_path
        basename_path = self.root / pathlib.Path(file_name).name
        if basename_path.exists():
            return basename_path
        return None

    def _load_image(self, image_info):
        image_path = self._resolve_image_path(image_info)
        if image_path is None:
            height = image_info.get('height', self.config.dataset.image_size)
            width = image_info.get('width', self.config.dataset.image_size)
            return torch.zeros((3, height, width), dtype=torch.float32), height, width
        image_pil = Image.open(image_path).convert('RGB')
        width, height = image_pil.size
        image = torch.from_numpy(__import__('numpy').asarray(image_pil).transpose(2, 0, 1)).float()
        return image, height, width

    def _load_sample(self, index):
        image_info = self.images[index]
        image, height, width = self._load_image(image_info)
        anns = self.annotations.get(image_info['id'], [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            category_id = int(ann['category_id'])
            labels.append(self.category_id_to_label.get(category_id, category_id))
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        return image, {
            'image_id': image_info['id'],
            'labels': labels,
            'boxes': boxes,
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.long),
            'orig_size': [height, width],
            'size': [height, width],
            'label_to_category_id': self.label_to_category_id,
        }

    def __getitem__(self, index):
        if self.augmentor is not None and (self.config.augmentation.use_mosaic or self.config.augmentation.use_mixup):
            return self.augmentor.apply(self, index, self._load_sample)
        image, target = self._load_sample(index)
        image, target = self.transform(image, target)
        return image, target
