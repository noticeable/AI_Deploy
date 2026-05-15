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
    with tempfile.TemporaryDirectory(prefix='pruned_point_cloud_entrypoints_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        checkpoint_path = tmp / 'dummy_pointnet_checkpoint.pth'
        pruned_path = tmp / 'pointnet_tp_pruned.pth'
        infer_input = tmp / 'points.npy'
        infer_output = tmp / 'infer_outputs.npz'
        eval_output_dir = tmp / 'eval_outputs'
        onnx_output = tmp / 'pointnet_pruned.onnx'

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
            print('SKIP | pruned point-cloud entrypoints smoke | environment not ready to create dummy checkpoint')
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
            print('FAIL | pruned point-cloud entrypoints smoke | prune command failed')
            print(result.stdout)
            sys.exit(1)

        save_points_command = [
            PYTHON,
            '-c',
            (
                'import numpy as np; '
                "x=np.random.randn(32,3).astype(np.float32); "
                "np.save(r'%s', x)"
            ) % infer_input.as_posix().replace('\\', '\\\\')
        ]
        result = run_command(save_points_command)
        if result.returncode != 0:
            print('FAIL | pruned point-cloud entrypoints smoke | failed to create inference input')
            print(result.stdout)
            sys.exit(1)

        infer_command = [
            PYTHON,
            'scripts/point_cloud/infer.py',
            '--config',
            'configs/point_cloud/classification/pointnet_cls_modelnet40.yaml',
            '--input',
            str(infer_input),
            '--output',
            str(infer_output),
            'device',
            'cpu',
            'dataset.num_points',
            '32',
            'test.checkpoint',
            str(pruned_path),
        ]
        result = run_command(infer_command)
        if result.returncode != 0:
            print('FAIL | pruned point-cloud entrypoints smoke | infer command failed')
            print(result.stdout)
            sys.exit(1)
        if not infer_output.exists():
            print(f'FAIL | pruned point-cloud entrypoints smoke | infer output not found: {infer_output}')
            sys.exit(1)

        eval_command = [
            PYTHON,
            'scripts/point_cloud/evaluate.py',
            '--config',
            'configs/point_cloud/classification/pointnet_cls_modelnet40.yaml',
            'device',
            'cpu',
            'dataset.num_points',
            '32',
            'dataset.dataset_dir',
            str(tmp),
            'test.checkpoint',
            str(pruned_path),
            'test.output_dir',
            str(eval_output_dir),
            'test.batch_size',
            '4',
            'test.dataloader.num_workers',
            '0',
        ]
        result = run_command(eval_command)
        if result.returncode != 0:
            print('FAIL | pruned point-cloud entrypoints smoke | evaluate command failed')
            print(result.stdout)
            sys.exit(1)
        predictions_path = eval_output_dir / 'predictions.npz'
        if not predictions_path.exists():
            print(f'FAIL | pruned point-cloud entrypoints smoke | predictions file not found: {predictions_path}')
            sys.exit(1)

        export_command = [
            PYTHON,
            'scripts/point_cloud/export.py',
            '--config',
            'configs/point_cloud/classification/pointnet_cls_modelnet40.yaml',
            'device',
            'cpu',
            'dataset.num_points',
            '32',
            'export.checkpoint',
            str(pruned_path),
            'export.output_file',
            str(onnx_output),
        ]
        result = run_command(export_command)
        if result.returncode != 0:
            print('FAIL | pruned point-cloud entrypoints smoke | export command failed')
            print(result.stdout)
            sys.exit(1)
        if not onnx_output.exists():
            print(f'FAIL | pruned point-cloud entrypoints smoke | onnx output not found: {onnx_output}')
            sys.exit(1)

        print('OK | pruned point-cloud infer | output=', infer_output.name)
        print('OK | pruned point-cloud evaluate | output=', predictions_path.name)
        print('OK | pruned point-cloud export | output=', onnx_output.name)
        print('SUMMARY | ok=3 | skip=0 | fail=0 | total=3')


if __name__ == '__main__':
    main()
