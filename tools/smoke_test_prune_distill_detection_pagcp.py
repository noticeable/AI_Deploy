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
    with tempfile.TemporaryDirectory(prefix='prune_distill_detection_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        teacher_checkpoint_path = tmp / 'teacher_checkpoint.pth'
        student_checkpoint_path = tmp / 'student_checkpoint.pth'
        pruned_student_path = tmp / 'student_pruned.pth'
        train_config_path = tmp / 'dummy_prune_distill_train.yaml'
        output_dir = tmp / 'train_output'
        dataset_dir = tmp / 'dummy_dataset'

        create_command = [
            PYTHON,
            '-c',
            (
                'import pathlib, torch; '
                'from pytorch_object_detection import get_default_config, create_model, update_config; '
                'config=get_default_config(); '
                "config.merge_from_file('configs/detection/yolo/yolov8_n.yaml'); "
                "config.dataset.class_names=['object']; "
                'config=update_config(config); '
                "config.device='cpu'; "
                'teacher=create_model(config); '
                'student=create_model(config); '
                "teacher_path=pathlib.Path(r'%s'); "
                "student_path=pathlib.Path(r'%s'); "
                "torch.save({'model': teacher.state_dict()}, teacher_path); "
                "torch.save({'model': student.state_dict()}, student_path)"
            ) % (
                teacher_checkpoint_path.as_posix().replace('\\', '\\\\'),
                student_checkpoint_path.as_posix().replace('\\', '\\\\'),
            )
        ]
        result = run_command(create_command)
        if result.returncode != 0:
            print('SKIP | prune+distill smoke | environment not ready to create dummy checkpoints')
            print(result.stdout)
            return

        prune_command = [
            PYTHON,
            'tools/prune_detection_pagcp.py',
            '--config',
            'configs/detection/yolo/yolov8_n.yaml',
            'device', 'cpu',
            'dataset.class_names', "['object']",
            'prune.backend', 'torch_pruning',
            'prune.method', 'tp_magnitude',
            'prune.target', 'backbone',
            'prune.checkpoint', str(student_checkpoint_path),
            'prune.output_dir', str(tmp),
            'prune.save_name', pruned_student_path.name,
            'prune.amount', '0.2',
            'prune.example_batch_size', '1',
            'prune.example_image_size', '64',
        ]
        result = run_command(prune_command)
        if result.returncode != 0:
            print('FAIL | prune+distill smoke | prune command failed')
            print(result.stdout)
            sys.exit(1)

        if not pruned_student_path.exists():
            print(f'FAIL | prune+distill smoke | pruned checkpoint not found: {pruned_student_path}')
            sys.exit(1)

        train_config = f"""dataset:
  format: yolo
  dataset_dir: {dataset_dir.as_posix()}
  image_size: 64
  n_classes: 1
  class_names: ['object']
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
        train_images_dir = dataset_dir / 'train' / 'images'
        train_labels_dir = dataset_dir / 'train' / 'labels'
        val_images_dir = dataset_dir / 'val' / 'images'
        val_labels_dir = dataset_dir / 'val' / 'labels'
        output_dir.mkdir(parents=True, exist_ok=True)
        train_images_dir.mkdir(parents=True, exist_ok=True)
        train_labels_dir.mkdir(parents=True, exist_ok=True)
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)
        train_config_path.write_text(train_config, encoding='utf-8')

        create_dataset_command = [
            PYTHON,
            '-c',
            (
                'from pathlib import Path; '
                'from PIL import Image; '
                "root=Path(r'%s'); "
                "train_image=root / 'train' / 'images' / 'sample.jpg'; "
                "val_image=root / 'val' / 'images' / 'sample.jpg'; "
                "Image.new('RGB', (64, 64), color=(30, 120, 200)).save(train_image); "
                "Image.new('RGB', (64, 64), color=(30, 120, 200)).save(val_image); "
                "(root / 'train' / 'labels' / 'sample.txt').write_text('0 0.5 0.5 0.4 0.4\\n', encoding='utf-8'); "
                "(root / 'val' / 'labels' / 'sample.txt').write_text('0 0.5 0.5 0.4 0.4\\n', encoding='utf-8')"
            ) % dataset_dir.as_posix().replace('\\', '\\\\')
        ]
        result = run_command(create_dataset_command)
        if result.returncode != 0:
            print('SKIP | prune+distill smoke | failed to create dummy dataset assets')
            print(result.stdout)
            return

        train_command = [
            PYTHON,
            'scripts/detection/train.py',
            '--config',
            str(train_config_path),
            'task', 'detection',
            'device', 'cpu',
            'train.checkpoint', str(pruned_student_path),
            'distill.enabled', 'True',
            'distill.teacher_checkpoint', str(teacher_checkpoint_path),
            'distill.temperature', '2.0',
            'distill.cls_weight', '1.0',
            'distill.box_weight', '1.0',
            'distill.hard_loss_weight', '1.0',
            'distill.soft_loss_weight', '1.0',
            'train.dataloader.num_workers', '0',
            'validation.dataloader.num_workers', '0',
        ]
        result = run_command(train_command)
        if result.returncode != 0:
            print('FAIL | prune+distill smoke | distill training command failed')
            print(result.stdout)
            sys.exit(1)

        checkpoint_files = sorted(output_dir.glob('checkpoint_*.pth'))
        if not checkpoint_files:
            print(f'FAIL | prune+distill smoke | no distilled checkpoint created in {output_dir}')
            sys.exit(1)

        print('OK | prune+distill smoke | pruned_student=', pruned_student_path.name)
        print('OK | prune+distill smoke | distilled_checkpoint=', checkpoint_files[-1].name)
        print('SUMMARY | ok=2 | skip=0 | fail=0 | total=2')


if __name__ == '__main__':
    main()
