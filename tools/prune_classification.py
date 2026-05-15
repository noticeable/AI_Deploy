#!/usr/bin/env python

import argparse
import pathlib

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import create_model
from pytorch_image_classification.tasks.classification.entrypoints import load_classification_config

try:
    import torch_pruning as tp
except Exception:
    tp = None


SUPPORTED_TP_METHODS = {'tp_magnitude', 'structured_ln'}
SUPPORTED_CLASSIFICATION_BUILTIN_MODELS = {
    ('cifar', 'resnet'),
    ('imagenet', 'resnet'),
    ('cifar', 'vit'),
}
SUPPORTED_CLASSIFICATION_TP_MODELS = {
    ('cifar', 'vit'),
}


def is_supported_builtin_pruning_model(config):
    return (config.model.type, config.model.name) in SUPPORTED_CLASSIFICATION_BUILTIN_MODELS


def is_supported_torch_pruning_model(config):
    return (config.model.type, config.model.name) in SUPPORTED_CLASSIFICATION_TP_MODELS


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = load_classification_config(args.config, args.options)
    config.freeze()
    return config


def should_prune_module(module, enabled_modules):
    if isinstance(module, nn.Conv2d) and 'conv' in enabled_modules:
        return True
    if isinstance(module, nn.Linear) and 'linear' in enabled_modules:
        return True
    return False


def collect_parameters(model, enabled_modules):
    parameters_to_prune = []
    for _, module in model.named_modules():
        if should_prune_module(module, enabled_modules):
            parameters_to_prune.append((module, 'weight'))
    return parameters_to_prune


def collect_tp_root_modules(model, enabled_modules, target='model'):
    root_modules = []
    if target == 'head':
        head = getattr(model, 'head', None)
        if isinstance(head, nn.Linear) and 'linear' in enabled_modules:
            root_modules.append(head)
        return root_modules

    for name, module in model.named_modules():
        if should_prune_module(module, enabled_modules):
            if getattr(model, 'head', None) is module:
                continue
            root_modules.append(module)
    return root_modules


def collect_ignored_tp_layers(model, target):
    ignored_layers = []
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        ignored_layers.append(model.head)
    if target == 'head':
        encoder = getattr(model, 'encoder', None)
        if encoder is not None:
            ignored_layers.append(encoder)
        conv_proj = getattr(model, 'conv_proj', None)
        if conv_proj is not None:
            ignored_layers.append(conv_proj)
    return ignored_layers


def apply_builtin_pruning(model, prune_config):
    parameters_to_prune = collect_parameters(model, prune_config.modules)
    if not parameters_to_prune:
        raise RuntimeError(f'No parameters matched prune.modules={prune_config.modules}')

    method = prune_config.method
    if method == 'global_unstructured':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_config.amount,
        )
    elif method == 'structured_ln':
        for module, param_name in parameters_to_prune:
            prune.ln_structured(module,
                                name=param_name,
                                amount=prune_config.amount,
                                n=prune_config.n,
                                dim=prune_config.dim)
    else:
        raise ValueError(f'Unsupported builtin prune.method: {method}')

    if prune_config.remove_reparam:
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
    return False


def apply_torch_pruning(model, prune_config, config):
    if tp is None:
        raise RuntimeError('torch-pruning is not installed. Please install torch-pruning first.')
    method = prune_config.method
    if method not in SUPPORTED_TP_METHODS:
        raise ValueError(f'Unsupported torch-pruning prune.method: {method}')
    if not is_supported_torch_pruning_model(config):
        raise ValueError(f'torch-pruning is not supported for classification model {(config.model.type, config.model.name)}')

    device = next(model.parameters()).device
    example_inputs = torch.randn(
        int(prune_config.example_batch_size),
        config.dataset.n_channels,
        int(prune_config.example_image_size),
        int(prune_config.example_image_size),
        device=device,
    )
    target = getattr(prune_config, 'target', 'model')
    root_modules = collect_tp_root_modules(model, prune_config.modules, target=target)
    if not root_modules:
        raise RuntimeError(f'No root modules matched prune.modules={prune_config.modules} for torch-pruning')

    ignored_layers = collect_ignored_tp_layers(model, target)

    importance = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=float(prune_config.amount),
        root_module_types=[nn.Conv2d, nn.Linear],
        ignored_layers=ignored_layers,
    )
    pruner.step()
    return True


def apply_pruning(model, prune_config, config):
    if prune_config.backend == 'torch_pruning':
        return apply_torch_pruning(model, prune_config, config)
    if not is_supported_builtin_pruning_model(config):
        raise ValueError(f'builtin pruning is not supported for classification model {(config.model.type, config.model.name)}')
    return apply_builtin_pruning(model, prune_config)


def summarize_sparsity(model):
    total_params = 0
    zero_params = 0
    for _, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        total_params += param.numel()
        zero_params += int((param == 0).sum().item())
    ratio = 0.0 if total_params == 0 else zero_params / total_params
    return total_params, zero_params, ratio


def save_pruned_checkpoint(model, config, prune_config, checkpoint_path, output_path, rebuilt):
    checkpoint = {
        'model': model.state_dict(),
        'config': config.as_dict(),
        'prune': {
            'backend': prune_config.backend,
            'method': prune_config.method,
            'amount': prune_config.amount,
            'modules': list(prune_config.modules),
            'n': int(prune_config.n),
            'dim': int(prune_config.dim),
            'remove_reparam': bool(prune_config.remove_reparam),
            'rebuilt': bool(rebuilt),
            'target': getattr(prune_config, 'target', 'model'),
            'source_checkpoint': checkpoint_path.as_posix(),
            'zero_params': sum(int((p == 0).sum().item()) for p in model.parameters() if p.is_floating_point()),
            'total_params': sum(int(p.numel()) for p in model.parameters() if p.is_floating_point()),
        },
    }
    checkpoint['prune']['sparsity'] = 0.0 if checkpoint['prune']['total_params'] == 0 else checkpoint['prune']['zero_params'] / checkpoint['prune']['total_params']
    torch.save(checkpoint, output_path)


def main():
    config = load_config()
    prune_config = config.prune

    checkpoint_path = prune_config.checkpoint or config.train.checkpoint or config.test.checkpoint or config.export.checkpoint
    if not checkpoint_path:
        raise ValueError('No checkpoint specified. Set prune.checkpoint or train/test/export checkpoint.')

    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Prune checkpoint not found: {checkpoint_path}')

    model = create_model(config)
    model.config = config
    Checkpointer(model).load(checkpoint_path.as_posix())
    model.eval()

    rebuilt = apply_pruning(model, prune_config, config)
    total_params, zero_params, sparsity = summarize_sparsity(model)

    output_dir = pathlib.Path(prune_config.output_dir or checkpoint_path.parent)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / prune_config.save_name

    save_pruned_checkpoint(model, config, prune_config, checkpoint_path, output_path, rebuilt)
    print(f'OK | saved pruned classification checkpoint to {output_path}')
    print(f'INFO | backend={prune_config.backend} | method={prune_config.method} | amount={prune_config.amount} | modules={list(prune_config.modules)}')
    print(f'INFO | zero_params={zero_params} | total_params={total_params} | sparsity={sparsity:.6f} | rebuilt={rebuilt}')


if __name__ == '__main__':
    main()
