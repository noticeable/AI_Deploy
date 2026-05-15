from .defaults import get_default_config



def update_config(config):
    if not config.dataset.class_names:
        config.dataset.n_classes = 0
    else:
        config.dataset.n_classes = len(config.dataset.class_names)
    return config
