#!/usr/bin/env python

import os
import pathlib
import subprocess
import sys
import tempfile

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = sys.executable

TESTS = [
    {
        'name': 'image_classification',
        'command': [
            PYTHON,
            'export.py',
            '--config',
            'configs/classification/presets/cifar10/resnet.yaml',
            'export.output_file',
            '',
        ],
        'output_name': 'image_classification.onnx',
    },
    {
        'name': 'point_cloud',
        'command': [
            PYTHON,
            'scripts/point_cloud/export.py',
            '--config',
            'configs/point_cloud/classification/pointnet_cls_modelnet40.yaml',
            'export.output_file',
            '',
        ],
        'output_name': 'point_cloud.onnx',
    },
    {
        'name': 'detection',
        'command': [
            PYTHON,
            'scripts/detection/export.py',
            '--config',
            'configs/detection/yolo/yolov8_n.yaml',
            'export.output_file',
            '',
        ],
        'output_name': 'detection.onnx',
    },
]

SKIP_HINTS = [
    'Failed to import torch',
    'Failed to import fvcore.common.checkpoint.Checkpointer',
    'No module named \'fvcore\'',
    'No module named \'onnx\'',
    'Module onnx is not installed!',
    'No module named \'pytorch_point_cloud\'',
    'No module named \'pytorch_object_detection\'',
    'Error loading',
    'DLL',
]


def should_skip(output):
    return any(hint in output for hint in SKIP_HINTS)


def validate_onnx_output(test_name, output_path):
    try:
        model = onnx.load(output_path.as_posix())
        onnx.checker.check_model(model)
    except Exception as exc:
        print(f"FAIL | {test_name} | invalid onnx: {exc}")
        return 'fail'

    input_names = [value.name for value in model.graph.input]
    output_names = [value.name for value in model.graph.output]
    node_count = len(model.graph.node)
    print(
        f"OK | {test_name} | {output_path} | inputs={input_names} | outputs={output_names} | nodes={node_count}")
    return 'ok'


def run_test(test, output_dir):
    output_path = output_dir / test['output_name']
    command = list(test['command'])
    command[-1] = str(output_path)
    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH', '')
    root_str = str(ROOT)
    env['PYTHONPATH'] = root_str if not existing_pythonpath else root_str + os.pathsep + existing_pythonpath
    result = subprocess.run(command,
                            cwd=ROOT,
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)
    if result.returncode != 0:
        if should_skip(result.stdout):
            print(f"SKIP | {test['name']} | environment not ready")
            print(result.stdout)
            return 'skip'
        print(f"FAIL | {test['name']} | returncode={result.returncode}")
        print(result.stdout)
        return 'fail'
    if not output_path.exists():
        print(f"FAIL | {test['name']} | output not found: {output_path}")
        return 'fail'
    return validate_onnx_output(test['name'], output_path)


def main():
    with tempfile.TemporaryDirectory(prefix='onnx_smoke_') as tmp_dir:
        output_dir = pathlib.Path(tmp_dir)
        results = [run_test(test, output_dir) for test in TESTS]

    ok_count = sum(result == 'ok' for result in results)
    skip_count = sum(result == 'skip' for result in results)
    fail_count = sum(result == 'fail' for result in results)
    print(
        f'SUMMARY | ok={ok_count} | skip={skip_count} | fail={fail_count} | total={len(results)}')
    if fail_count:
        sys.exit(1)


if __name__ == '__main__':
    main()
