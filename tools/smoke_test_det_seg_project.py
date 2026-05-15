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
    with tempfile.TemporaryDirectory(prefix='det_seg_smoke_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        dataset_dir = tmp / 'dummy_dataset'
        output_dir = tmp / 'train_output'
        config_path = tmp / 'det_seg_train.yaml'

        train_images_dir = dataset_dir / 'train' / 'images'
        train_labels_dir = dataset_dir / 'train' / 'labels'
        train_masks_dir = dataset_dir / 'train' / 'masks'
        val_images_dir = dataset_dir / 'val' / 'images'
        val_labels_dir = dataset_dir / 'val' / 'labels'
        val_masks_dir = dataset_dir / 'val' / 'masks'
        for path in [train_images_dir, train_labels_dir, train_masks_dir, val_images_dir, val_labels_dir, val_masks_dir, output_dir]:
            path.mkdir(parents=True, exist_ok=True)

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
            print('SKIP | det-seg smoke | failed to create dummy dataset assets')
            print(result.stdout)
            return

        strategies = ('EW', 'GradNorm', 'MGDA', 'PCGrad', 'DWA', 'CAGrad', 'IMTL', 'GradVac', 'RLW', 'UW', 'GLS', 'GradDrop')
        for strategy in strategies:
            train_command = [
                PYTHON,
                'scripts/det_seg/train.py',
                '--config',
                str(config_path),
                'device', 'cpu',
                'train.output_dir', str(output_dir / strategy.lower()),
                'train.dataloader.num_workers', '0',
                'validation.dataloader.num_workers', '0',
                'limtl.enabled', 'True',
                'limtl.strategy', strategy,
            ]
            result = run_command(train_command)
            if result.returncode != 0:
                print(f'FAIL | det-seg smoke | train command failed for {strategy}')
                print(result.stdout)
                sys.exit(1)

            checkpoint_files = sorted((output_dir / strategy.lower()).glob('checkpoint_*.pth'))
            if not checkpoint_files:
                print(f'FAIL | det-seg smoke | no checkpoint created for {strategy} in {output_dir / strategy.lower()}')
                sys.exit(1)

            eval_command = [
                PYTHON,
                'scripts/det_seg/evaluate.py',
                '--config',
                str(config_path),
                'device', 'cpu',
                'test.checkpoint', str(checkpoint_files[-1]),
                'train.dataloader.num_workers', '0',
                'validation.dataloader.num_workers', '0',
            ]
            result = run_command(eval_command)
            if result.returncode != 0:
                print(f'FAIL | det-seg smoke | evaluate command failed for {strategy}')
                print(result.stdout)
                sys.exit(1)

            export_command = [
                PYTHON,
                'scripts/det_seg/export.py',
                '--config',
                str(config_path),
                'device', 'cpu',
                'export.checkpoint', str(checkpoint_files[-1]),
            ]
            result = run_command(export_command)
            if result.returncode != 0:
                print(f'FAIL | det-seg smoke | export command failed for {strategy}')
                print(result.stdout)
                sys.exit(1)
            print(f'OK | det-seg smoke | strategy={strategy} checkpoint={checkpoint_files[-1].name}')

        print(f'SUMMARY | ok={len(strategies) * 3} | skip=0 | fail=0 | total={len(strategies) * 3}')


if __name__ == '__main__':
    main()
