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
    with tempfile.TemporaryDirectory(prefix='qat_det_export_onnx_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        output_dir = tmp / 'qat_det_run'
        checkpoint_dir = output_dir / 'artifacts'
        checkpoint_path = checkpoint_dir / 'checkpoint_00001.pth'
        onnx_path = tmp / 'quantized_detection.onnx'
        float_onnx_path = tmp / 'quantized_detection.float.onnx'

        train_command = [
            PYTHON,
            'scripts/detection/train.py',
            '--config',
            'configs/detection/yolo/yolov8_n.yaml',
            'device', 'cpu',
            'train.output_dir', str(checkpoint_dir),
            'train.batch_size', '2',
            'train.dataloader.num_workers', '0',
            'validation.batch_size', '2',
            'validation.dataloader.num_workers', '0',
            'scheduler.epochs', '1',
            'train.checkpoint_period', '1',
            'train.log_period', '1',
            'dataset.name', 'SmokeCOCO',
            'dataset.n_classes', '3',
            'dataset.image_size', '64',
            'dataset.dataset_dir', str(tmp),
            'dataset.train_ann', '',
            'dataset.val_ann', '',
            'model.yolo.dense_head.type', 'fpn_multi',
            'qat.enabled', 'True',
            'qat.freeze_bn_epoch', '1',
            'qat.disable_observer_epoch', '1',
        ]
        train_result = run_command(train_command)
        if train_result.returncode != 0:
            print('FAIL | detection quantized onnx smoke | training failed')
            print(train_result.stdout)
            sys.exit(1)

        if not checkpoint_path.exists():
            print(f'FAIL | detection quantized onnx smoke | checkpoint not found: {checkpoint_path}')
            sys.exit(1)

        export_command = [
            PYTHON,
            'scripts/detection/export.py',
            '--config',
            'configs/detection/yolo/yolov8_n.yaml',
            'device', 'cpu',
            'dataset.name', 'SmokeCOCO',
            'dataset.n_classes', '3',
            'dataset.image_size', '64',
            'dataset.dataset_dir', str(tmp),
            'dataset.train_ann', '',
            'dataset.val_ann', '',
            'export.checkpoint', str(checkpoint_path),
            'export.output_file', str(onnx_path),
            'export.quantized_onnx', 'True',
            'export.quantized_onnx_backend', 'onnxruntime_dynamic',
            'model.yolo.dense_head.type', 'fpn_multi',
            'qat.enabled', 'True',
        ]
        export_result = run_command(export_command)
        if export_result.returncode != 0:
            print('FAIL | detection quantized onnx smoke | export failed')
            print(export_result.stdout)
            sys.exit(1)

        if not onnx_path.exists():
            print(f'FAIL | detection quantized onnx smoke | quantized onnx file not found: {onnx_path}')
            sys.exit(1)
        if not float_onnx_path.exists():
            print(f'FAIL | detection quantized onnx smoke | float onnx file not found: {float_onnx_path}')
            sys.exit(1)

        verify_command = [
            PYTHON,
            '-c',
            (
                'import numpy as np; '
                'import onnx; '
                'import onnxruntime as ort; '
                f"model=onnx.load(r'{onnx_path.as_posix()}'); "
                'node_types={node.op_type for node in model.graph.node}; '
                f"session=ort.InferenceSession(r'{onnx_path.as_posix()}', providers=['CPUExecutionProvider']); "
                'input_meta=session.get_inputs()[0]; '
                "x=np.random.randn(1, 3, 64, 64).astype(np.float32); "
                'outputs=session.run(None, {input_meta.name: x}); '
                "print('OK | detection quantized onnx export | nodes=', len(model.graph.node), 'has_quant_ops=', any(op in node_types for op in ('MatMulInteger','DynamicQuantizeLinear','QuantizeLinear','DequantizeLinear')), 'output_shape=', tuple(outputs[0].shape))"
            )
        ]
        verify_result = run_command(verify_command)
        if verify_result.returncode != 0:
            print('FAIL | detection quantized onnx smoke | onnx verification failed')
            print(verify_result.stdout)
            sys.exit(1)

        print(verify_result.stdout.strip())
        print('SUMMARY | ok=2 | skip=0 | fail=0 | total=2')


if __name__ == '__main__':
    main()
