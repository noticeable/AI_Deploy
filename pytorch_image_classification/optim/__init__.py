import torch

from .adabound import AdaBound, AdaBoundW
from .lars import LARSOptimizer


_NO_WEIGHT_DECAY_KEYWORDS = (
    'bias',
    'bn',
    'norm',
    'pos_embedding',
    'pos_embed',
    'position_embedding',
    'position_embeddings',
    'cls_token',
    'class_token',
)


def get_param_list(config, model):
    if config.train.no_weight_decay_on_bn:
        decay_params = []
        no_decay_params = []
        for name, params in model.named_parameters():
            if not params.requires_grad:
                continue
            if params.ndim <= 1 or any(keyword in name
                                       for keyword in _NO_WEIGHT_DECAY_KEYWORDS):
                no_decay_params.append(params)
            else:
                decay_params.append(params)

        param_list = []
        if decay_params:
            param_list.append({
                'params': decay_params,
                'weight_decay': config.train.weight_decay,
            })
        if no_decay_params:
            param_list.append({
                'params': no_decay_params,
                'weight_decay': 0,
            })
    else:
        param_list = [{
            'params': list(model.parameters()),
            'weight_decay': config.train.weight_decay,
        }]
    return param_list


def create_optimizer(config, model):
    params = get_param_list(config, model)

    if config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=config.train.base_lr,
                                    momentum=config.train.momentum,
                                    nesterov=config.train.nesterov)
    elif config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=config.train.base_lr,
                                     betas=config.optim.adam.betas)
    elif config.train.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params,
                                      lr=config.train.base_lr,
                                      betas=config.optim.adam.betas)
    elif config.train.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(params,
                                     lr=config.train.base_lr,
                                     betas=config.optim.adam.betas,
                                     amsgrad=True)
    elif config.train.optimizer == 'adabound':
        optimizer = AdaBound(params,
                             lr=config.train.base_lr,
                             betas=config.optim.adabound.betas,
                             final_lr=config.optim.adabound.final_lr,
                             gamma=config.optim.adabound.gamma)
    elif config.train.optimizer == 'adaboundw':
        optimizer = AdaBoundW(params,
                              lr=config.train.base_lr,
                              betas=config.optim.adabound.betas,
                              final_lr=config.optim.adabound.final_lr,
                              gamma=config.optim.adabound.gamma)
    elif config.train.optimizer == 'lars':
        optimizer = LARSOptimizer(params,
                                  lr=config.train.base_lr,
                                  momentum=config.train.momentum,
                                  eps=config.optim.lars.eps,
                                  thresh=config.optim.lars.threshold)
    else:
        raise ValueError()
    return optimizer
