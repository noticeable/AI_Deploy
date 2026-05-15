#!/usr/bin/env python

import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = pathlib.Path.home() / '.conda' / 'envs' / 'pytorch310' / ('python.exe' if os.name == 'nt' else 'bin/python')
PYTHON = str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else pathlib.Path(sys.executable))
PRUNE_CASES = [
    {
        'name': 'resnet_builtin',
        'config': (ROOT / 'configs' / 'classification' / 'presets' / 'cifar10' / 'resnet.yaml').as_posix(),
        'prune_options': [
            'prune.backend', 'builtin',
            'prune.method', 'global_unstructured',
            'prune.modules', "('conv','linear')",
            'prune.save_name', 'resnet_builtin_pruned.pth',
        ],
    },
    {
        'name': 'vit_builtin',
        'config': (ROOT / 'configs' / 'classification' / 'presets' / 'cifar10' / 'vit.yaml').as_posix(),
        'prune_options': [
            'prune.backend', 'builtin',
            'prune.method', 'global_unstructured',
            'prune.modules', "('linear',)",
            'prune.save_name', 'vit_builtin_pruned.pth',
        ],
    },
]


def run_command(command):
    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH', '')
    root_str = str(ROOT)
    env['PYTHONPATH'] = root_str if not existing_pythonpath else root_str + os.pathsep + existing_pythonpath
    return subprocess.run(command,
                          cwd=ROOT,
                          env=env,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          text=True)


def main():
    with tempfile.TemporaryDirectory(prefix='prune_cls_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        ok = 0
        total = 0

        for case in PRUNE_CASES:
            total += 2
            checkpoint_dir = tmp / case['name'] / 'train_run'
            checkpoint_path = checkpoint_dir / 'checkpoint_00001.pth'
            pruned_path = tmp / case['name'] / case['prune_options'][-1]
            pruned_path.parent.mkdir(parents=True, exist_ok=True)

            train_command = [
                PYTHON,
                'train.py',
                '--config',
                case['config'],
                'device', 'cpu',
                'train.output_dir', str(checkpoint_dir),
                'train.batch_size', '4',
                'validation.batch_size', '4',
                'train.dataloader.num_workers', '0',
                'validation.dataloader.num_workers', '0',
                'scheduler.epochs', '1',
                'train.checkpoint_period', '1',
                'train.log_period', '1',
                'train.use_tensorboard', 'False',
                'train.val_period', '0',
                'dataset.name', 'FakeData',
                'dataset.image_size', '32',
                'dataset.n_channels', '3',
                'dataset.n_classes', '10',
            ]
            train_result = run_command(train_command)
            if train_result.returncode != 0:
                print(f"FAIL | classification prune smoke | train failed | model={case['name']}")
                print(train_result.stdout)
                sys.exit(1)
            if not checkpoint_path.exists():
                print(f"FAIL | classification prune smoke | checkpoint not found | model={case['name']} | path={checkpoint_path}")
                sys.exit(1)

            prune_command = [
                PYTHON,
                'tools/prune_classification.py',
                '--config',
                case['config'],
                'device', 'cpu',
                'dataset.name', 'FakeData',
                'dataset.image_size', '32',
                'dataset.n_channels', '3',
                'dataset.n_classes', '10',
                'prune.checkpoint', str(checkpoint_path),
                'prune.output_dir', str(pruned_path.parent),
                'prune.amount', '0.2',
            ] + case['prune_options']
            prune_result = run_command(prune_command)
            if prune_result.returncode != 0:
                print(f"FAIL | classification prune smoke | prune failed | model={case['name']}")
                print(prune_result.stdout)
                sys.exit(1)
            if not pruned_path.exists():
                print(f"FAIL | classification prune smoke | pruned checkpoint not found | model={case['name']} | path={pruned_path}")
                sys.exit(1)
            print(prune_result.stdout.strip())
            ok += 1

            verify_command = [
                PYTHON,
                '-c',
                (
                    'import torch; '
                    'from pytorch_image_classification import create_model; '
                    'from pytorch_image_classification.tasks.classification.entrypoints import load_classification_config; '
                    'from fvcore.common.checkpoint import Checkpointer; '
                    f"config=load_classification_config(r'{case['config']}', ['device','cpu','dataset.name','FakeData','dataset.image_size','32','dataset.n_channels','3','dataset.n_classes','10']); "
                    'model=create_model(config); '
                    f"Checkpointer(model).load(r'{pruned_path.as_posix()}'); "
                    'x=torch.randn(1, config.dataset.n_channels, config.dataset.image_size, config.dataset.image_size); '
                    'y=model(x); '
                    f"print('OK | classification prune verify | model=', '{case['name']}', 'output_shape=', tuple(y.shape))"
                )
            ]
            verify_result = run_command(verify_command)
            if verify_result.returncode != 0:
                print(f"FAIL | classification prune smoke | verify failed | model={case['name']}")
                print(verify_result.stdout)
                sys.exit(1)
            print(verify_result.stdout.strip())
            ok += 1

        print(f'SUMMARY | ok={ok} | skip=0 | fail=0 | total={total}')


if __name__ == '__main__':
    main()
