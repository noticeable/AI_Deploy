from .coco import COCODataset
from .yolo import YOLODataset


def create_dataset(config, is_train):
    if config.dataset.format == 'coco':
        return COCODataset(config, is_train)
    if config.dataset.format == 'yolo':
        return YOLODataset(config, is_train)
    raise ValueError(f'Unsupported dataset format: {config.dataset.format}')
