from typing import Dict, List

import torch


def detection_collate_fn(batch: List):
    images = torch.stack([sample[0] for sample in batch])
    targets: List[Dict] = []
    for image_index, (_, target) in enumerate(batch):
        target = dict(target)
        target['image_id'] = torch.as_tensor(target['image_id'], dtype=torch.long)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.long)
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['orig_size'] = torch.as_tensor(target['orig_size'], dtype=torch.long)
        target['size'] = torch.as_tensor(target['size'], dtype=torch.long)
        target['batch_index'] = torch.full((len(target['labels']),),
                                           image_index,
                                           dtype=torch.long)
        targets.append(target)
    return images, targets


def create_dataloader(config, is_train):
    from torch.utils.data import DataLoader
    from pytorch_object_detection.datasets import create_dataset

    dataset = create_dataset(config, is_train)
    batch_size = config.train.batch_size if is_train else config.validation.batch_size
    dataloader_config = config.train.dataloader if is_train else config.validation.dataloader
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=is_train,
                      num_workers=dataloader_config.num_workers,
                      drop_last=dataloader_config.drop_last,
                      pin_memory=dataloader_config.pin_memory,
                      collate_fn=detection_collate_fn)
