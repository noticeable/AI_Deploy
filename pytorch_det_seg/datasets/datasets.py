from .det_seg_yolo import DetSegYOLODataset



def create_dataset(config, is_train):
    if config.dataset.format == 'det_seg_yolo':
        return DetSegYOLODataset(config, is_train)
    raise ValueError(f'Unsupported dataset format: {config.dataset.format}')
