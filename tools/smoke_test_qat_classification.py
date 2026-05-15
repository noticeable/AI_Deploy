#!/usr/bin/env python

import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = pathlib.Path.home() / '.conda' / 'envs' / 'pytorch310' / ('python.exe' if os.name == 'nt' else 'bin/python')
PYTHON = str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else pathlib.Path(sys.executable))
QAT_CASES = [
    {
        'name': 'resnet',
        'config': (ROOT / 'configs' / 'classification' / 'presets' / 'cifar10' / 'resnet.yaml').as_posix(),
        'output_shape': '(1, 10)',
    },
    {
        'name': 'vit',
        'config': (ROOT / 'configs' / 'classification' / 'presets' / 'cifar10' / 'vit.yaml').as_posix(),
        'output_shape': '(1, 10)',
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
    with tempfile.TemporaryDirectory(prefix='qat_cls_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        ok = 0
        total = 0

        for case in QAT_CASES:
            total += 2
            output_dir = tmp / case['name'] / 'qat_run'
            checkpoint_dir = output_dir / 'artifacts'
            checkpoint_path = checkpoint_dir / 'checkpoint_00001.pth'

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
                'qat.enabled', 'True',
                'qat.freeze_bn_epoch', '1',
                'qat.disable_observer_epoch', '1',
            ]
            result = run_command(train_command)
            if result.returncode != 0:
                print(f"FAIL | qat classification smoke | train command failed | model={case['name']}")
                print(result.stdout)
                sys.exit(1)

            if not checkpoint_path.exists():
                print(f"FAIL | qat classification smoke | checkpoint not found | model={case['name']} | path={checkpoint_path}")
                sys.exit(1)

            verify_command = [
                PYTHON,
                '-c',
                (
                    'import torch; '
                    'from pytorch_image_classification import get_default_config, create_model, update_config; '
                    'from pytorch_image_classification.models.qat import prepare_model_for_qat, convert_qat_model; '
                    'from pytorch_image_classification.tasks.classification.entrypoints import _load_config_file; '
                    'from fvcore.common.checkpoint import Checkpointer; '
                    'config=get_default_config(); '
                    f"_load_config_file(config, r'{case['config']}'); "
                    "config.merge_from_list(['device','cpu','dataset.name','FakeData','dataset.image_size','32','dataset.n_channels','3','dataset.n_classes','10','qat.enabled','True']); "
                    'config=update_config(config); '
                    'model=create_model(config); '
                    'model,_=prepare_model_for_qat(config, model); '
                    f"Checkpointer(model).load(r'{checkpoint_path.as_posix()}'); "
                    'quantized=convert_qat_model(config, model); '
                    'x=torch.randn(1, config.dataset.n_channels, config.dataset.image_size, config.dataset.image_size); '
                    'y=quantized(x); '
                    f"shape=tuple(y.shape); assert str(shape) == '{case['output_shape']}', 'expected {case['output_shape']}, got %s' % (shape,); "
                    f"print('OK | qat convert | model=', '{case['name']}', 'output_shape=', shape)"
                )
            ]
            result = run_command(verify_command)
            if result.returncode != 0:
                print(f"FAIL | qat classification smoke | convert verification failed | model={case['name']}")
                print(result.stdout)
                sys.exit(1)
            print(result.stdout.strip())
            ok += 2

        print(f'SUMMARY | ok={ok} | skip=0 | fail=0 | total={total}')


if __name__ == '__main__':
    main()
