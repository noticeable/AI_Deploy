import importlib

import torch.nn as nn



def create_model(config):
    module = importlib.import_module(
        f'pytorch_det_seg.models.{config.model.meta_architecture}.{config.model.name}')
    model = getattr(module, 'Network')(config)
    model.to(config.device)
    return model



def apply_data_parallel_wrapper(config, model: nn.Module) -> nn.Module:
    model.to(config.device)
    return model
