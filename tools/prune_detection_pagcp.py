#!/usr/bin/env python

import argparse
import pathlib

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from fvcore.common.checkpoint import Checkpointer

from pytorch_object_detection import create_model, get_default_config, update_config
from pytorch_object_detection.models.yolo.common import ConvBNAct

try:
    import torch_pruning as tp
except Exception:
    tp = None


SUPPORTED_TP_METHODS = {'tp_magnitude', 'structured_ln'}
SUPPORTED_PRUNE_TARGETS = {'backbone'}


def validate_prune_target(target):
    if target not in SUPPORTED_PRUNE_TARGETS:
        raise ValueError(
            'Unsupported detection prune.target: '
            f'{target}. The full YOLO FPN multi-head migration only supports prune.target=backbone.')


def collect_backbone_conv_root_modules(model, enabled_modules):
    root_modules = []
    for module in (getattr(model, 'stem', None),
                   getattr(model, 'stage2', None),
                   getattr(model, 'stage3', None),
                   getattr(model, 'stage4', None)):
        if module is None:
            continue
        for submodule in module.modules():
            if isinstance(submodule, nn.Conv2d) and 'conv' in enabled_modules:
                root_modules.append(submodule)
    return root_modules


def collect_model_root_modules(model, enabled_modules):
    root_modules = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv' in enabled_modules:
            root_modules.append(module)
    if isinstance(getattr(model, 'box_head', None), nn.Linear) and 'linear' in enabled_modules:
        root_modules.append(model.box_head)
    if isinstance(getattr(model, 'score_head', None), nn.Linear) and 'linear' in enabled_modules:
        root_modules.append(model.score_head)
    return root_modules


def collect_ignored_tp_layers(model, target):
    validate_prune_target(target)
    ignored_layers = []
    if target == 'backbone':
        if hasattr(model, 'box_head'):
            ignored_layers.append(model.box_head)
        if hasattr(model, 'score_head'):
            ignored_layers.append(model.score_head)
    return ignored_layers


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    config = update_config(config)
    config.freeze()
    return config


def should_prune_module(module_name, module, enabled_modules):
    if isinstance(module, nn.Conv2d) and 'conv' in enabled_modules:
        return True
    if isinstance(module, nn.Linear) and 'linear' in enabled_modules:
        return True
    return False


def collect_parameters(model, enabled_modules):
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if should_prune_module(module_name, module, enabled_modules):
            parameters_to_prune.append((module, 'weight'))
    return parameters_to_prune


def collect_tp_root_modules(model, enabled_modules, target='backbone'):
    validate_prune_target(target)
    if target == 'backbone':
        return collect_backbone_conv_root_modules(model, enabled_modules)
    return collect_model_root_modules(model, enabled_modules)


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

    device = next(model.parameters()).device
    example_inputs = torch.randn(
        int(prune_config.example_batch_size),
        config.dataset.n_channels,
        int(prune_config.example_image_size),
        int(prune_config.example_image_size),
        device=device,
    )
    target = getattr(prune_config, 'target', 'backbone')
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


def extract_tiny_yolo_channels(model):
    def extract_stage_out_channels(stage):
        if isinstance(stage, ConvBNAct):
            return int(stage.block[0].out_channels)
        blocks = getattr(stage, 'blocks', None)
        if blocks is not None and len(blocks) > 0:
            first_block = blocks[0]
            if hasattr(first_block, 'block') and len(first_block.block) > 0 and hasattr(first_block.block[0], 'out_channels'):
                return int(first_block.block[0].out_channels)
        return None

    stages = [
        getattr(model, 'stem', None),
        getattr(model, 'stage2', None),
        getattr(model, 'stage3', None),
        getattr(model, 'stage4', None),
    ]
    channels = []
    for stage in stages:
        out_channels = extract_stage_out_channels(stage)
        if out_channels is not None:
            channels.append(out_channels)
    return channels


def save_pruned_checkpoint(model, config, prune_config, checkpoint_path, output_path, rebuilt):
    saved_config = config.as_dict()
    saved_config.setdefault('model', {}).setdefault('yolo', {}).setdefault('dense_head', {})['type'] = 'fpn_multi'
    if prune_config.backend == 'torch_pruning':
        saved_channels = extract_tiny_yolo_channels(model)
        saved_config.setdefault('model', {}).setdefault('yolo', {})['channels'] = saved_channels
    checkpoint = {
        'model': model.state_dict(),
        'config': saved_config,
        'prune': {
            'backend': prune_config.backend,
            'method': prune_config.method,
            'amount': prune_config.amount,
            'modules': list(prune_config.modules),
            'n': int(prune_config.n),
            'dim': int(prune_config.dim),
            'remove_reparam': bool(prune_config.remove_reparam),
            'rebuilt': bool(rebuilt),
            'source_checkpoint': checkpoint_path.as_posix(),
            'zero_params': sum(int((p == 0).sum().item()) for p in model.parameters() if p.is_floating_point()),
            'total_params': sum(int(p.numel()) for p in model.parameters() if p.is_floating_point()),
        },
    }
    checkpoint['prune']['sparsity'] = 0.0 if checkpoint['prune']['total_params'] == 0 else checkpoint['prune']['zero_params'] / checkpoint['prune']['total_params']
    torch.save(checkpoint, output_path)


def _head_in_channels(module):
    if isinstance(module, nn.Conv2d):
        return int(module.in_channels)
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    raise TypeError(f'Unsupported detection head type: {type(module).__name__}')


def rebuild_tiny_yolo_channels(model):
    return model, False


def main():
    config = load_config()
    prune_config = config.prune

    checkpoint_path = prune_config.checkpoint or config.train.checkpoint or config.test.checkpoint or config.export.checkpoint
    if not checkpoint_path:
        raise ValueError('No checkpoint specified. Set prune.checkpoint or train/test/export checkpoint.')

    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Prune checkpoint not found: {checkpoint_path}')

    if prune_config.backend == 'builtin' and prune_config.method == 'structured_ln':
        raise ValueError(
            'Detection builtin structured_ln rebuild has not been aligned with the current FPN multi-head YOLO model. '
            'Use prune.backend=torch_pruning for structured channel pruning, or use builtin global_unstructured sparsification.')

    model = create_model(config)
    model.config = config
    Checkpointer(model).load(checkpoint_path.as_posix())
    model.eval()

    backend_rebuilt = apply_pruning(model, prune_config, config)

    rebuilt = False
    if prune_config.backend == 'builtin' and prune_config.method == 'structured_ln':
        model, rebuilt = rebuild_tiny_yolo_channels(model)
    elif prune_config.backend == 'torch_pruning':
        rebuilt = backend_rebuilt

    total_params, zero_params, sparsity = summarize_sparsity(model)

    output_dir = pathlib.Path(prune_config.output_dir or checkpoint_path.parent)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / prune_config.save_name

    save_pruned_checkpoint(model, config, prune_config, checkpoint_path, output_path, rebuilt)
    print(f'OK | saved pruned checkpoint to {output_path}')
    print(f'INFO | backend={prune_config.backend} | method={prune_config.method} | amount={prune_config.amount} | modules={list(prune_config.modules)}')
    print(f'INFO | zero_params={zero_params} | total_params={total_params} | sparsity={sparsity:.6f} | rebuilt={rebuilt}')


if __name__ == '__main__':
    main()
