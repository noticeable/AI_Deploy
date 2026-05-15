import argparse
import pathlib

import yaml
import torch
from yacs.config import CfgNode

from pytorch_image_classification import get_default_config, update_config


def _load_config_file(config, config_path, loaded_paths=None):
    config_path = pathlib.Path(config_path).resolve()
    if loaded_paths is None:
        loaded_paths = []
    if config_path in loaded_paths:
        cycle = ' -> '.join(path.as_posix() for path in [*loaded_paths, config_path])
        raise ValueError(f'Circular _base_ config reference detected: {cycle}')

    with config_path.open('r', encoding='utf-8') as handle:
        config_data = yaml.safe_load(handle) or {}

    base_configs = config_data.pop('_base_', [])
    if isinstance(base_configs, str):
        base_configs = [base_configs]
    elif not isinstance(base_configs, list):
        raise TypeError(f'_base_ must be a string or list in {config_path.as_posix()}')

    for base_config in base_configs:
        base_path = (config_path.parent / base_config).resolve()
        _load_config_file(config, base_path, [*loaded_paths, config_path])

    config.merge_from_other_cfg(CfgNode(config_data))


def load_classification_config(config_path, options=None):
    config = get_default_config()
    if config_path is not None:
        _load_config_file(config, config_path)
    config.merge_from_list(options or [])
    return update_config(config)


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def parse_export_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_train_config(args):
    config = load_classification_config(args.config, args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config = load_classification_config(config_path)
        config.merge_from_list(['train.resume', True])
    config.merge_from_list(['train.dist.local_rank', args.local_rank])
    config = update_config(config)
    config.freeze()
    return config


def load_export_config(args):
    config = load_classification_config(args.config, args.options)
    config.freeze()
    return config


def infer_task_family(config):
    if config.model.type == 'imagenet' or config.dataset.name == 'ImageNet':
        return 'imagenet'
    if config.model.type == 'cifar' or config.dataset.name in [
            'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'KMNIST'
    ]:
        return 'cifar'
    raise ValueError(
        f'Unsupported classification task for config: '
        f'model.type={config.model.type}, dataset.name={config.dataset.name}')


def ensure_task_family(config, expected_family):
    actual_family = infer_task_family(config)
    if actual_family != expected_family:
        raise ValueError(
            f'This entrypoint expects {expected_family} config, got '
            f'{actual_family} (model.type={config.model.type}, '
            f'dataset.name={config.dataset.name})')
    return actual_family


def resolve_export_output_dir(config):
    if config.test.output_dir:
        output_dir = pathlib.Path(config.test.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir
    return pathlib.Path(config.test.checkpoint).parent
