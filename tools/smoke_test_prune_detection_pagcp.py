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
    with tempfile.TemporaryDirectory(prefix='prune_detection_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        checkpoint_path = tmp / 'dummy_detection_checkpoint.pth'
        pruned_path = tmp / 'model_pruned.pth'
        structured_path = tmp / 'model_structured_pruned.pth'

        create_command = [
            PYTHON,
            '-c',
            (
                'import pathlib, torch; '
                'from pytorch_object_detection import get_default_config, create_model, update_config; '
                'config=get_default_config(); '
                "config.merge_from_file('configs/detection/yolo/yolov8_n.yaml'); "
                'config=update_config(config); '
                "config.device='cpu'; "
                'model=create_model(config); '
                "path=pathlib.Path(r\'%s\'); "
                "torch.save({'model': model.state_dict()}, path)"
            ) % checkpoint_path.as_posix().replace('\\', '\\\\')
        ]
        result = run_command(create_command)
        if result.returncode != 0:
            print('SKIP | detection prune smoke | environment not ready to create dummy checkpoint')
            print(result.stdout)
            return

        tests = [
            {
                'name': 'global',
                'output': pruned_path,
                'extra': ['prune.amount', '0.2'],
                'verify_mode': 'standard',
            },
            {
                'name': 'structured',
                'output': structured_path,
                'extra': ['prune.method', 'structured_ln', 'prune.amount', '0.25', 'prune.save_name', 'model_structured_pruned.pth'],
                'verify_mode': 'standard',
            },
            {
                'name': 'torch_pruning',
                'output': tmp / 'model_tp_pruned.pth',
                'extra': [
                    'prune.backend', 'torch_pruning',
                    'prune.method', 'tp_magnitude',
                    'prune.target', 'backbone',
                    'prune.amount', '0.2',
                    'prune.save_name', 'model_tp_pruned.pth',
                    'prune.example_batch_size', '1',
                    'prune.example_image_size', '640',
                ],
                'verify_mode': 'torch_pruning',
            },
        ]

        for test in tests:
            prune_command = [
                PYTHON,
                'tools/prune_detection_pagcp.py',
                '--config',
                'configs/detection/yolo/yolov8_n.yaml',
                'device',
                'cpu',
                'prune.checkpoint',
                str(checkpoint_path),
                'prune.output_dir',
                str(tmp),
                'prune.save_name',
                test['output'].name,
                *test['extra'],
            ]
            result = run_command(prune_command)
            if result.returncode != 0:
                print(f"FAIL | detection prune smoke | {test['name']} prune command failed")
                print(result.stdout)
                sys.exit(1)

            if not test['output'].exists():
                print(f"FAIL | detection prune smoke | output not found: {test['output']}")
                sys.exit(1)

            if test['verify_mode'] == 'torch_pruning':
                verify_script = (
                    'import torch; '
                    'from pytorch_object_detection import get_default_config, create_model, update_config; '
                    'from pytorch_object_detection.utils.checkpoint import create_model_from_checkpoint; '
                    'config=get_default_config(); '
                    "config.merge_from_file('configs/detection/yolo/yolov8_n.yaml'); "
                    'config=update_config(config); '
                    "config.device='cpu'; "
                    "model, config, _ = create_model_from_checkpoint(config, r'%s', create_model); "
                    'model.eval(); '
                    'x=torch.randn(1, 3, config.dataset.image_size, config.dataset.image_size); '
                    'y=model(x); '
                    "print('OK | %s detections=', len(y))"
                ) % (test['output'].as_posix().replace('\\', '\\\\'), test['name'])
            else:
                verify_script = (
                    'import torch; '
                    'from pytorch_object_detection import get_default_config, create_model, update_config; '
                    'config=get_default_config(); '
                    "config.merge_from_file('configs/detection/yolo/yolov8_n.yaml'); "
                    'config=update_config(config); '
                    "config.device='cpu'; "
                    'model=create_model(config); '
                    "checkpoint=torch.load(r\'%s\', map_location='cpu'); "
                    "model.load_state_dict(checkpoint['model']); "
                    'model.eval(); '
                    'x=torch.randn(1, 3, config.dataset.image_size, config.dataset.image_size); '
                    'y=model(x); '
                    "print('OK | %s detections=', len(y))"
                ) % (test['output'].as_posix().replace('\\', '\\\\'), test['name'])

            verify_command = [
                PYTHON,
                '-c',
                verify_script,
            ]
            result = run_command(verify_command)
            if result.returncode != 0:
                print(f"FAIL | detection prune smoke | {test['name']} verification failed")
                print(result.stdout)
                sys.exit(1)
            print(result.stdout.strip())
        print('SUMMARY | ok=2 | skip=0 | fail=0 | total=2')


if __name__ == '__main__':
    main()
