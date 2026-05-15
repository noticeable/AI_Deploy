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
    with tempfile.TemporaryDirectory(prefix='qat_det_seg_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        dataset_dir = tmp / 'dummy_dataset'
        output_dir = tmp / 'qat_det_seg_run'
        checkpoint_dir = output_dir / 'artifacts'
        checkpoint_path = checkpoint_dir / 'checkpoint_00001.pth'
        config_path = tmp / 'det_seg_qat.yaml'

        train_images_dir = dataset_dir / 'train' / 'images'
        train_labels_dir = dataset_dir / 'train' / 'labels'
        train_masks_dir = dataset_dir / 'train' / 'masks'
        val_images_dir = dataset_dir / 'val' / 'images'
        val_labels_dir = dataset_dir / 'val' / 'labels'
        val_masks_dir = dataset_dir / 'val' / 'masks'
        for path in [train_images_dir, train_labels_dir, train_masks_dir, val_images_dir, val_labels_dir, val_masks_dir, checkpoint_dir]:
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
  output_dir: {checkpoint_dir.as_posix()}
  batch_size: 1
  log_period: 1
  checkpoint_period: 1
validation:
  batch_size: 1
scheduler:
  epochs: 1
qat:
  enabled: true
  backend: fbgemm
  freeze_bn_epoch: 1
  disable_observer_epoch: 1
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
            print('FAIL | qat det-seg smoke | failed to create dummy dataset assets')
            print(result.stdout)
            sys.exit(1)

        train_command = [
            PYTHON,
            'scripts/det_seg/train.py',
            '--config',
            str(config_path),
            'device', 'cpu',
            'train.output_dir', str(checkpoint_dir),
            'train.dataloader.num_workers', '0',
            'validation.dataloader.num_workers', '0',
            'limtl.enabled', 'False',
        ]
        result = run_command(train_command)
        if result.returncode != 0:
            print('FAIL | qat det-seg smoke | train command failed')
            print(result.stdout)
            sys.exit(1)

        if not checkpoint_path.exists():
            print(f'FAIL | qat det-seg smoke | checkpoint not found: {checkpoint_path}')
            sys.exit(1)

        verify_command = [
            PYTHON,
            '-c',
            (
                'import torch; '
                'import torch.ao.quantization as quantization; '
                "torch.backends.quantized.engine='fbgemm'; "
                'from pytorch_det_seg import get_default_config, create_model, update_config; '
                'from pytorch_det_seg.models.qat import prepare_model_for_qat, convert_qat_model; '
                'from fvcore.common.checkpoint import Checkpointer; '
                'config=get_default_config(); '
                f"config.merge_from_file(r'{config_path.as_posix()}'); "
                f"config.merge_from_list(['device','cpu','dataset.dataset_dir',r'{dataset_dir.as_posix()}','qat.enabled','True','qat.backend','fbgemm']); "
                'config=update_config(config); '
                'model=create_model(config); '
                'model,_=prepare_model_for_qat(config, model); '
                f"Checkpointer(model).load(r'{checkpoint_path.as_posix()}'); "
                'quantized=convert_qat_model(config, model); '
                'x=torch.randn(1, config.dataset.n_channels, config.dataset.image_size, config.dataset.image_size); '
                'y=quantized(x); '
                "print('OK | det-seg qat convert | dets=', len(y['detections']), 'seg_shape=', tuple(y['segmentation'].shape))"
            )
        ]
        result = run_command(verify_command)
        if result.returncode != 0:
            print('FAIL | qat det-seg smoke | convert verification failed')
            print(result.stdout)
            sys.exit(1)

        print(result.stdout.strip())
        print('SUMMARY | ok=2 | skip=0 | fail=0 | total=2')


if __name__ == '__main__':
    main()
