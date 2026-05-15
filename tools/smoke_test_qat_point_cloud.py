#!/usr/bin/env python

import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = pathlib.Path.home() / '.conda' / 'envs' / 'pytorch310' / ('python.exe' if os.name == 'nt' else 'bin/python')
PYTHON = str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else pathlib.Path(sys.executable))


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
    with tempfile.TemporaryDirectory(prefix='qat_point_cloud_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        output_dir = tmp / 'qat_run'
        checkpoint_dir = output_dir / 'artifacts'
        checkpoint_path = checkpoint_dir / 'checkpoint_00001.pth'

        train_command = [
            PYTHON,
            'scripts/point_cloud/train.py',
            '--config',
            'configs/point_cloud/classification/pointnet_cls_modelnet40.yaml',
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
            'dataset.name', 'ModelNet40',
            'dataset.dataset_dir', str(tmp),
            'dataset.num_points', '32',
            'dataset.n_classes', '40',
            'qat.enabled', 'True',
            'qat.freeze_bn_epoch', '1',
            'qat.disable_observer_epoch', '1',
        ]
        result = run_command(train_command)
        if result.returncode != 0:
            print('FAIL | point-cloud qat smoke | train command failed')
            print(result.stdout)
            sys.exit(1)

        if not checkpoint_path.exists():
            print(f'FAIL | point-cloud qat smoke | checkpoint not found: {checkpoint_path}')
            sys.exit(1)

        verify_command = [
            PYTHON,
            '-c',
            (
                'import torch; '
                'from pytorch_point_cloud import get_default_config, create_model, update_config; '
                'from pytorch_point_cloud.models.qat import prepare_model_for_qat, convert_qat_model; '
                'from fvcore.common.checkpoint import Checkpointer; '
                'config=get_default_config(); '
                "config.merge_from_file('configs/point_cloud/classification/pointnet_cls_modelnet40.yaml'); "
                f"config.merge_from_list(['device','cpu','dataset.dataset_dir',r'{tmp.as_posix()}','dataset.num_points','32','qat.enabled','True']); "
                'config=update_config(config); '
                'model=create_model(config); '
                'model,_=prepare_model_for_qat(config, model); '
                f"Checkpointer(model).load(r'{checkpoint_path.as_posix()}'); "
                'quantized=convert_qat_model(config, model); '
                'x=torch.randn(1, config.dataset.num_points, config.dataset.n_channels); '
                'y=quantized(x); '
                "print('OK | point-cloud qat convert | output_shape=', tuple(y['logits'].shape) if isinstance(y, dict) else tuple(y.shape))"
            )
        ]
        result = run_command(verify_command)
        if result.returncode != 0:
            print('FAIL | point-cloud qat smoke | convert verification failed')
            print(result.stdout)
            sys.exit(1)
        print(result.stdout.strip())
        print('SUMMARY | ok=2 | skip=0 | fail=0 | total=2')


if __name__ == '__main__':
    main()
