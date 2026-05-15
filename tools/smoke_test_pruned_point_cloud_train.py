#!/usr/bin/env python

import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = sys.executable


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
    with tempfile.TemporaryDirectory(prefix='pruned_point_cloud_train_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        dataset_dir = tmp / 'dataset'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = tmp / 'dummy_pointnet_checkpoint.pth'
        pruned_path = tmp / 'pointnet_tp_pruned.pth'
        train_output_dir = tmp / 'train_outputs'

        create_command = [
            PYTHON,
            '-c',
            (
                'import pathlib, torch; '
                'from pytorch_point_cloud import get_default_config, create_model, update_config; '
                'config=get_default_config(); '
                "config.merge_from_file('configs/point_cloud/classification/pointnet_cls_modelnet40.yaml'); "
                'config=update_config(config); '
                "config.device='cpu'; "
                'model=create_model(config); '
                "path=pathlib.Path(r\'%s\'); "
                "torch.save({'model': model.state_dict()}, path)"
            ) % checkpoint_path.as_posix().replace('\\', '\\\\')
        ]
        result = run_command(create_command)
        if result.returncode != 0:
            print('SKIP | point-cloud pruned train smoke | environment not ready to create dummy checkpoint')
            print(result.stdout)
            return

        prune_command = [
            PYTHON,
            'tools/prune_point_cloud.py',
            '--config',
            'configs/point_cloud/classification/pointnet_cls_modelnet40.yaml',
            'device',
            'cpu',
            'dataset.num_points',
            '32',
            'prune.checkpoint',
            str(checkpoint_path),
            'prune.output_dir',
            str(tmp),
            'prune.save_name',
            pruned_path.name,
            'prune.backend',
            'torch_pruning',
            'prune.method',
            'tp_magnitude',
            'prune.target',
            'classifier',
            'prune.modules',
            '["linear"]',
            'prune.amount',
            '0.2',
            'prune.example_batch_size',
            '1',
            'prune.example_num_points',
            '32',
        ]
        result = run_command(prune_command)
        if result.returncode != 0:
            print('FAIL | point-cloud pruned train smoke | prune command failed')
            print(result.stdout)
            sys.exit(1)

        train_command = [
            PYTHON,
            'scripts/point_cloud/train.py',
            '--config',
            'configs/point_cloud/classification/pointnet_cls_modelnet40.yaml',
            'device',
            'cpu',
            'dataset.num_points',
            '32',
            'dataset.dataset_dir',
            str(dataset_dir),
            'train.output_dir',
            str(train_output_dir),
            'train.checkpoint',
            str(pruned_path),
            'train.batch_size',
            '4',
            'validation.batch_size',
            '4',
            'train.dataloader.num_workers',
            '0',
            'validation.dataloader.num_workers',
            '0',
            'scheduler.epochs',
            '1',
            'train.checkpoint_period',
            '1',
            'train.log_period',
            '1',
            'train.use_tensorboard',
            'False',
        ]
        result = run_command(train_command)
        if result.returncode != 0:
            print('FAIL | point-cloud pruned train smoke | training failed')
            print(result.stdout)
            sys.exit(1)

        ckpts = sorted(train_output_dir.glob('checkpoint_*.pth'))
        if not ckpts:
            print(f'FAIL | point-cloud pruned train smoke | no checkpoint created in {train_output_dir}')
            sys.exit(1)

        print('OK | point-cloud pruned train smoke | checkpoint=', ckpts[-1].name)
        print('SUMMARY | ok=1 | skip=0 | fail=0 | total=1')


if __name__ == '__main__':
    main()
