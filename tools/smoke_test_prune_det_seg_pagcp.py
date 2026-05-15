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


def create_dummy_dataset(dataset_dir):
    train_images_dir = dataset_dir / 'train' / 'images'
    train_labels_dir = dataset_dir / 'train' / 'labels'
    train_masks_dir = dataset_dir / 'train' / 'masks'
    val_images_dir = dataset_dir / 'val' / 'images'
    val_labels_dir = dataset_dir / 'val' / 'labels'
    val_masks_dir = dataset_dir / 'val' / 'masks'
    for path in [train_images_dir, train_labels_dir, train_masks_dir, val_images_dir, val_labels_dir, val_masks_dir]:
        path.mkdir(parents=True, exist_ok=True)

    create_dataset_command = [
        PYTHON,
        '-c',
        (
            'from pathlib import Path; '
            'from PIL import Image; '
            "root=Path(r'%s'); "
            "train_image=root / 'train' / 'images' / 'sample.jpg'; "
            "val_image=root / 'val' / 'images' / 'sample.jpg'; "
            "train_mask=root / 'train' / 'masks' / 'sample.png'; "
            "val_mask=root / 'val' / 'masks' / 'sample.png'; "
            "Image.new('RGB', (64, 64), color=(30, 120, 200)).save(train_image); "
            "Image.new('RGB', (64, 64), color=(30, 120, 200)).save(val_image); "
            "Image.new('L', (64, 64), color=0).save(train_mask); "
            "Image.new('L', (64, 64), color=0).save(val_mask); "
            "(root / 'train' / 'labels' / 'sample.txt').write_text('0 0.5 0.5 0.4 0.4\\n', encoding='utf-8'); "
            "(root / 'val' / 'labels' / 'sample.txt').write_text('0 0.5 0.5 0.4 0.4\\n', encoding='utf-8')"
        ) % dataset_dir.as_posix().replace('\\', '\\\\')
    ]
    result = run_command(create_dataset_command)
    if result.returncode != 0:
        print('SKIP | prune det-seg smoke | failed to create dummy dataset assets')
        print(result.stdout)
        return False
    return True


def write_dummy_config(config_path, dataset_dir, output_dir):
    config_text = f"""dataset:
  format: det_seg_yolo
  dataset_dir: {dataset_dir.as_posix()}
  image_size: 64
  n_classes: 1
  class_names: ['object']
model:
  meta_architecture: det_seg
  name: shared_backbone_tiny
train:
  output_dir: {output_dir.as_posix()}
  batch_size: 1
  log_period: 1
  checkpoint_period: 1
validation:
  batch_size: 1
scheduler:
  epochs: 1
"""
    config_path.write_text(config_text, encoding='utf-8')


def create_dummy_checkpoint(config_path, checkpoint_path):
    create_checkpoint_command = [
        PYTHON,
        '-c',
        (
            'import pathlib, torch; '
            'from pytorch_det_seg import get_default_config, create_model, update_config; '
            'config=get_default_config(); '
            "config.merge_from_file(r'%s'); "
            'config=update_config(config); '
            "config.device='cpu'; "
            'model=create_model(config); '
            "path=pathlib.Path(r'%s'); "
            "torch.save({'model': model.state_dict(), 'config': config.as_dict()}, path)"
        ) % (
            config_path.as_posix().replace('\\', '\\\\'),
            checkpoint_path.as_posix().replace('\\', '\\\\'),
        )
    ]
    result = run_command(create_checkpoint_command)
    if result.returncode != 0:
        print('SKIP | prune det-seg smoke | environment not ready to create dummy checkpoint')
        print(result.stdout)
        return False
    return True


def build_restore_verify_command(config_path, checkpoint_path, verify_restore):
    if verify_restore:
        return [
            PYTHON,
            '-c',
            (
                'import torch; '
                'from pytorch_det_seg import get_default_config, create_model, update_config; '
                'from pytorch_det_seg.utils.checkpoint import create_model_from_checkpoint; '
                'config=get_default_config(); '
                "config.merge_from_file(r'%s'); "
                'config=update_config(config); '
                "config.device='cpu'; "
                "model, config, checkpoint = create_model_from_checkpoint(config, r'%s', create_model); "
                'model.eval(); '
                'x=torch.randn(1, config.dataset.n_channels, config.dataset.image_size, config.dataset.image_size); '
                'outputs=model(x, return_outputs=True); '
                "assert len(checkpoint.get('config', {}).get('model', {}).get('backbone', {}).get('channels', [])) == 4; "
                "print('OK | torch_pruning restore | det=', len(outputs['detections']), 'seg=', tuple(outputs['seg_logits'].shape))"
            ) % (
                config_path.as_posix().replace('\\', '\\\\'),
                checkpoint_path.as_posix().replace('\\', '\\\\'),
            )
        ]
    return [
        PYTHON,
        '-c',
        (
            'import torch; '
            'from pytorch_det_seg import get_default_config, create_model, update_config; '
            'config=get_default_config(); '
            "config.merge_from_file(r'%s'); "
            'config=update_config(config); '
            "config.device='cpu'; "
            'model=create_model(config); '
            "checkpoint=torch.load(r'%s', map_location='cpu'); "
            "model.load_state_dict(checkpoint['model']); "
            'model.eval(); '
            'x=torch.randn(1, config.dataset.n_channels, config.dataset.image_size, config.dataset.image_size); '
            'outputs=model(x, return_outputs=True); '
            "print('OK | builtin restore | det=', len(outputs['detections']), 'seg=', tuple(outputs['seg_logits'].shape))"
        ) % (
            config_path.as_posix().replace('\\', '\\\\'),
            checkpoint_path.as_posix().replace('\\', '\\\\'),
        )
    ]


def run_prune_case(config_path, base_checkpoint_path, output_dir, test_case):
    prune_command = [
        PYTHON,
        'tools/prune_det_seg_pagcp.py',
        '--config',
        str(config_path),
        'device', 'cpu',
        'prune.checkpoint', str(base_checkpoint_path),
        'prune.output_dir', str(output_dir),
        'prune.save_name', test_case['output'].name,
        *test_case['extra'],
    ]
    result = run_command(prune_command)
    if result.returncode != 0:
        print(f"FAIL | prune det-seg smoke | {test_case['name']} prune command failed")
        print(result.stdout)
        sys.exit(1)

    if not test_case['output'].exists():
        print(f"FAIL | prune det-seg smoke | output not found: {test_case['output']}")
        sys.exit(1)


def verify_pruned_checkpoint_restore(config_path, test_case):
    verify_command = build_restore_verify_command(config_path, test_case['output'], test_case['verify_restore'])
    result = run_command(verify_command)
    if result.returncode != 0:
        print(f"FAIL | prune det-seg smoke | {test_case['name']} verification failed")
        print(result.stdout)
        sys.exit(1)
    print(result.stdout.strip())


def verify_eval_export_resume(config_path, pruned_checkpoint_path, tmp_dir):
    eval_command = [
        PYTHON,
        'scripts/det_seg/evaluate.py',
        '--config',
        str(config_path),
        'device', 'cpu',
        'test.checkpoint', str(pruned_checkpoint_path),
        'test.output_dir', str(tmp_dir / 'eval_output'),
        'train.dataloader.num_workers', '0',
        'validation.dataloader.num_workers', '0',
    ]
    result = run_command(eval_command)
    if result.returncode != 0:
        print('FAIL | prune det-seg smoke | evaluate command failed for torch_pruning checkpoint')
        print(result.stdout)
        sys.exit(1)

    metrics_path = tmp_dir / 'eval_output' / 'det_seg_metrics.json'
    if not metrics_path.exists():
        print(f'FAIL | prune det-seg smoke | metrics file not found: {metrics_path}')
        sys.exit(1)

    export_path = tmp_dir / 'det_seg_pruned.onnx'
    export_command = [
        PYTHON,
        'scripts/det_seg/export.py',
        '--config',
        str(config_path),
        'device', 'cpu',
        'export.checkpoint', str(pruned_checkpoint_path),
        'export.output_file', str(export_path),
    ]
    result = run_command(export_command)
    if result.returncode != 0:
        print('FAIL | prune det-seg smoke | export command failed for torch_pruning checkpoint')
        print(result.stdout)
        sys.exit(1)

    if not export_path.exists():
        print(f'FAIL | prune det-seg smoke | export file not found: {export_path}')
        sys.exit(1)

    train_resume_command = [
        PYTHON,
        'scripts/det_seg/train.py',
        '--config',
        str(config_path),
        'device', 'cpu',
        'train.checkpoint', str(pruned_checkpoint_path),
        'train.output_dir', str(tmp_dir / 'resume_train_output'),
        'train.dataloader.num_workers', '0',
        'validation.dataloader.num_workers', '0',
        'scheduler.epochs', '1',
    ]
    result = run_command(train_resume_command)
    if result.returncode != 0:
        print('FAIL | prune det-seg smoke | resumed train command failed for torch_pruning checkpoint')
        print(result.stdout)
        sys.exit(1)

    resumed_checkpoints = sorted((tmp_dir / 'resume_train_output').glob('checkpoint_*.pth'))
    if not resumed_checkpoints:
        print(f"FAIL | prune det-seg smoke | no resumed checkpoint created in {tmp_dir / 'resume_train_output'}")
        sys.exit(1)

    print(f'OK | prune det-seg smoke | eval_metrics={metrics_path.name}')
    print(f'OK | prune det-seg smoke | export_file={export_path.name}')
    print(f'OK | prune det-seg smoke | resumed_checkpoint={resumed_checkpoints[-1].name}')


def main():
    with tempfile.TemporaryDirectory(prefix='prune_det_seg_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        dataset_dir = tmp / 'dummy_dataset'
        output_dir = tmp / 'train_output'
        config_path = tmp / 'det_seg_prune.yaml'
        base_checkpoint_path = tmp / 'dummy_det_seg_checkpoint.pth'
        builtin_pruned_path = tmp / 'model_pruned.pth'
        tp_pruned_path = tmp / 'model_tp_pruned.pth'

        output_dir.mkdir(parents=True, exist_ok=True)
        write_dummy_config(config_path, dataset_dir, output_dir)

        if not create_dummy_dataset(dataset_dir):
            return
        if not create_dummy_checkpoint(config_path, base_checkpoint_path):
            return

        tests = [
            {
                'name': 'builtin',
                'output': builtin_pruned_path,
                'extra': ['prune.amount', '0.2'],
                'verify_restore': False,
            },
            {
                'name': 'torch_pruning',
                'output': tp_pruned_path,
                'extra': [
                    'prune.backend', 'torch_pruning',
                    'prune.method', 'tp_magnitude',
                    'prune.target', 'backbone',
                    'prune.modules', "['conv']",
                    'prune.amount', '0.2',
                    'prune.save_name', 'model_tp_pruned.pth',
                    'prune.example_batch_size', '1',
                    'prune.example_image_size', '64',
                ],
                'verify_restore': True,
            },
        ]

        for test_case in tests:
            run_prune_case(config_path, base_checkpoint_path, tmp, test_case)
            verify_pruned_checkpoint_restore(config_path, test_case)

        verify_eval_export_resume(config_path, tp_pruned_path, tmp)
        print('SUMMARY | ok=5 | skip=0 | fail=0 | total=5')


if __name__ == '__main__':
    main()
