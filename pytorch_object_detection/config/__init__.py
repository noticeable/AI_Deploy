from .defaults import get_default_config


def update_config(config):
    if config.dataset.class_names:
        config.dataset.n_classes = len(config.dataset.class_names)
    elif str(getattr(config.dataset, 'format', '')).lower() == 'coco':
        config.dataset.n_classes = max(int(getattr(config.dataset, 'n_classes', 80)), 1)
    else:
        config.dataset.n_classes = 0
    return config
