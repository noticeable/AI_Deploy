import torch


def create_optimizer(config, model):
    if config.train.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(),
                                 lr=config.train.base_lr,
                                 betas=config.optim.adamw.betas,
                                 weight_decay=config.train.weight_decay)
    if config.train.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(),
                                lr=config.train.base_lr,
                                betas=config.optim.adam.betas,
                                weight_decay=config.train.weight_decay)
    if config.train.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(),
                               lr=config.train.base_lr,
                               momentum=config.train.momentum,
                               nesterov=config.train.nesterov,
                               weight_decay=config.train.weight_decay)
    raise ValueError(f'Unsupported optimizer: {config.train.optimizer}')
