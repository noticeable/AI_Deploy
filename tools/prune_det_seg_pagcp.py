#!/usr/bin/env python

import argparse
import pathlib

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from fvcore.common.checkpoint import Checkpointer

from pytorch_det_seg import create_model, get_default_config, update_config
from pytorch_det_seg.models.det_seg.shared_backbone_tiny import ConvBNAct

try:
    import torch_pruning as tp
except Exception:
    tp = None


SUPPORTED_PRUNE_TARGETS = {'backbone'}
SUPPORTED_TP_METHODS = {'tp_magnitude', 'structured_ln'}


def validate_prune_target(target):
    if target not in SUPPORTED_PRUNE_TARGETS:
        raise ValueError(f"Unsupported det-seg prune.target: {target}. Only 'backbone' is currently supported.")


def collect_backbone_conv_root_modules(model, enabled_modules):
    root_modules = []
    backbone = getattr(model, 'backbone', None)
    if backbone is None:
        return root_modules
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d) and 'conv' in enabled_modules:
            root_modules.append(module)
    return root_modules


def collect_ignored_tp_layers(model):
    ignored_layers = []
    if hasattr(model, 'det_head'):
        ignored_layers.append(model.det_head)
    if hasattr(model, 'seg_head'):
        ignored_layers.append(model.seg_head)
    if hasattr(model, 'seg_neck'):
        ignored_layers.append(model.seg_neck)
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
    return collect_backbone_conv_root_modules(model, enabled_modules)


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

    ignored_layers = collect_ignored_tp_layers(model)

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


def extract_det_seg_backbone_channels(model):
    if not hasattr(model, 'backbone') or not isinstance(model.backbone, nn.Sequential):
        return []
    channels = []
    for module in model.backbone:
        if isinstance(module, ConvBNAct):
            channels.append(int(module.block[0].out_channels))
    return channels


def _copy_bn_stats(new_bn, old_bn, out_indices):
    for attr in ('weight', 'bias', 'running_mean', 'running_var'):
        getattr(new_bn, attr).data.zero_()
        getattr(new_bn, attr).data[:len(out_indices)] = getattr(old_bn, attr).data[out_indices]



def _copy_conv_bn_block(new_block, old_block, out_keep, in_keep):
    new_conv = new_block.block[0]
    old_conv = old_block.block[0]
    out_indices = torch.arange(out_keep)
    in_indices = torch.arange(in_keep)
    new_conv.weight.data.zero_()
    new_conv.weight.data[:out_keep, :in_keep] = old_conv.weight.data[out_indices][:, in_indices]
    _copy_bn_stats(new_block.block[1], old_block.block[1], out_indices)



def _infer_backbone_keep_counts(model):
    conv_blocks = [module for module in model.backbone if isinstance(module, ConvBNAct)]
    original_widths = []
    inferred_widths = []
    for block in conv_blocks:
        conv = block.block[0]
        original_widths.append(conv.out_channels)
        keep_mask = conv.weight.detach().abs().sum(dim=(1, 2, 3)) > 0
        keep_count = int(keep_mask.sum().item())
        inferred_widths.append(max(1, keep_count))
    return conv_blocks, original_widths, inferred_widths



def _copy_detection_neck(new_model, old_model, last_keep):
    old_neck = old_model.det_neck
    new_neck = new_model.det_neck
    old_conv = old_neck.block.block[0]
    new_conv = new_neck.block.block[0]
    out_keep = min(new_conv.out_channels, old_conv.out_channels)
    _copy_conv_bn_block(new_neck.block, old_neck.block, out_keep=out_keep, in_keep=min(last_keep, old_conv.in_channels))



def _copy_segmentation_neck(new_model, old_model, last_keep):
    old_neck = old_model.seg_neck
    new_neck = new_model.seg_neck
    first_old = old_neck.block[0]
    first_new = new_neck.block[0]
    first_out_keep = min(first_new.block[0].out_channels, first_old.block[0].out_channels)
    _copy_conv_bn_block(first_new, first_old, out_keep=first_out_keep, in_keep=min(last_keep, first_old.block[0].in_channels))

    second_old = old_neck.block[2]
    second_new = new_neck.block[2]
    second_out_keep = min(second_new.block[0].out_channels, second_old.block[0].out_channels)
    _copy_conv_bn_block(second_new, second_old, out_keep=second_out_keep, in_keep=min(second_new.block[0].in_channels, second_old.block[0].in_channels))



def _copy_detection_head(new_model, old_model):
    old_box = old_model.det_head.box_head
    old_score = old_model.det_head.score_head
    new_box = new_model.det_head.box_head
    new_score = new_model.det_head.score_head
    keep = min(new_box.in_features, old_box.in_features)
    indices = torch.arange(keep)
    new_box.weight.data.zero_()
    new_box.weight.data[:, :keep] = old_box.weight.data[:, indices]
    new_box.bias.data.copy_(old_box.bias.data)
    new_score.weight.data.zero_()
    new_score.weight.data[:, :keep] = old_score.weight.data[:, indices]
    new_score.bias.data.copy_(old_score.bias.data)



def _copy_segmentation_head(new_model, old_model):
    old_head0 = old_model.seg_head.block[0]
    new_head0 = new_model.seg_head.block[0]
    out_keep = min(new_head0.block[0].out_channels, old_head0.block[0].out_channels)
    in_keep = min(new_head0.block[0].in_channels, old_head0.block[0].in_channels)
    _copy_conv_bn_block(new_head0, old_head0, out_keep=out_keep, in_keep=in_keep)

    new_conv = new_model.seg_head.block[1]
    old_conv = old_model.seg_head.block[1]
    keep_in = min(new_conv.in_channels, old_conv.in_channels)
    new_conv.weight.data.zero_()
    new_conv.weight.data[:, :keep_in] = old_conv.weight.data[:, :keep_in]
    if new_conv.bias is not None and old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)


def rebuild_det_seg_backbone_branches(model):
    if not hasattr(model, 'backbone') or not isinstance(model.backbone, nn.Sequential):
        return model, False

    conv_blocks, original_widths, inferred_widths = _infer_backbone_keep_counts(model)
    if len(conv_blocks) != len(model.backbone):
        return model, False

    config = model.config
    rebuilt = type(model)(config)
    rebuilt.config = config
    new_blocks = [module for module in rebuilt.backbone if isinstance(module, ConvBNAct)]
    if len(new_blocks) != len(conv_blocks):
        return model, False

    prev_keep = config.dataset.n_channels
    for new_block, old_block, out_keep in zip(new_blocks, conv_blocks, inferred_widths):
        old_conv = old_block.block[0]
        if out_keep > old_conv.out_channels:
            return model, False
        _copy_conv_bn_block(new_block, old_block, out_keep=out_keep, in_keep=min(prev_keep, old_conv.in_channels))
        prev_keep = out_keep

    last_keep = inferred_widths[-1]
    _copy_detection_neck(rebuilt, model, last_keep)
    _copy_segmentation_neck(rebuilt, model, last_keep)
    _copy_detection_head(rebuilt, model)
    _copy_segmentation_head(rebuilt, model)

    rebuilt.to(next(model.parameters()).device)
    return rebuilt, inferred_widths != original_widths



def save_pruned_checkpoint(model, config, prune_config, checkpoint_path, output_path, rebuilt):
    saved_config = config.as_dict()
    saved_channels = extract_det_seg_backbone_channels(model)
    if saved_channels:
        saved_config.setdefault('model', {}).setdefault('backbone', {})['channels'] = saved_channels
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
            'target': str(prune_config.target),
            'remove_reparam': bool(prune_config.remove_reparam),
            'rebuilt': bool(rebuilt),
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

    backend_rebuilt = apply_pruning(model, prune_config, config)

    rebuilt = False
    if prune_config.backend == 'builtin' and prune_config.method == 'structured_ln':
        model, rebuilt = rebuild_det_seg_backbone_branches(model)
    elif prune_config.backend == 'torch_pruning':
        model, rebuilt = rebuild_det_seg_backbone_branches(model)
        rebuilt = bool(rebuilt or backend_rebuilt)

    total_params, zero_params, sparsity = summarize_sparsity(model)

    output_dir = pathlib.Path(prune_config.output_dir or checkpoint_path.parent)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / prune_config.save_name

    save_pruned_checkpoint(model, config, prune_config, checkpoint_path, output_path, rebuilt)
    print(f'OK | saved pruned det-seg checkpoint to {output_path}')
    print(f'INFO | backend={prune_config.backend} | method={prune_config.method} | amount={prune_config.amount} | modules={list(prune_config.modules)} | target={prune_config.target}')
    print(f'INFO | zero_params={zero_params} | total_params={total_params} | sparsity={sparsity:.6f} | rebuilt={rebuilt}')


if __name__ == '__main__':
    main()
