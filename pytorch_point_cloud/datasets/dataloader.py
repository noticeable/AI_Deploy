import torch
from torch.utils.data import DataLoader

from .datasets import create_dataset



def classification_collate_fn(batch):
    points = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    return points, labels



def segmentation_collate_fn(batch):
    points = torch.stack([item[0] for item in batch], dim=0)
    cls_label = torch.stack([item[1]['cls_label'] for item in batch], dim=0)
    seg_label = torch.stack([item[1]['seg_label'] for item in batch], dim=0)
    return points, {
        'cls_label': cls_label,
        'seg_label': seg_label,
    }



def create_dataloader(config, is_train):
    dataset = create_dataset(config, is_train)
    if is_train:
        batch_size = config.train.batch_size
        loader_cfg = config.train.dataloader
    else:
        batch_size = config.test.batch_size
        loader_cfg = config.test.dataloader
    collate_fn = classification_collate_fn if config.task == 'classification' else segmentation_collate_fn
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=is_train,
                      num_workers=loader_cfg.num_workers,
                      drop_last=loader_cfg.drop_last if is_train else False,
                      pin_memory=loader_cfg.pin_memory,
                      collate_fn=collate_fn)
