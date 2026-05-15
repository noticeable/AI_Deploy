from typing import Callable

import torch.nn as nn


def create_initializer(mode: str) -> Callable:
    """Create a module initializer / 创建模块初始化器。

    The returned callable is applied to each submodule so model families can
    share one initialization policy selected from config.
    返回的可调用对象会作用于各个子模块，便于不同模型族按配置复用同一套初始化策略。
    """
    if mode in ['kaiming_fan_out', 'kaiming_fan_in']:
        mode = mode[8:]

        def initializer(module):
            """Apply Kaiming-style initialization / 应用 Kaiming 风格初始化。"""
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data,
                                        mode=mode,
                                        nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight.data)
                nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data,
                                        mode=mode,
                                        nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)
    elif mode == 'transformer':

        def initializer(module):
            """Apply transformer-friendly initialization / 应用适合 Transformer 的初始化。"""
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight.data, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight.data, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight.data)
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)
    else:
        raise ValueError()

    return initializer
