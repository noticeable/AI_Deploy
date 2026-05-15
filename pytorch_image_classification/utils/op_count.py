from typing import Tuple
import thop
import torch
import torch.nn as nn
import yacs.config


def _remove_thop_buffers(module: nn.Module) -> None:
    for child in module.modules():
        buffers = getattr(child, '_buffers', None)
        if buffers is None:
            continue
        buffers.pop('total_ops', None)
        buffers.pop('total_params', None)


def count_op(config: yacs.config.CfgNode, model: nn.Module) -> Tuple[str, str]:
    data = torch.zeros((1, config.dataset.n_channels,
                        config.dataset.image_size, config.dataset.image_size),
                       dtype=torch.float32,
                       device=torch.device(config.device))
    result = thop.clever_format(thop.profile(model, (data, ), verbose=False))
    _remove_thop_buffers(model)
    return result
