#!/usr/bin/env python

import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = pathlib.Path.home() / '.conda' / 'envs' / 'pytorch310' / ('python.exe' if os.name == 'nt' else 'bin/python')
PYTHON = str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else pathlib.Path(sys.executable))
QAT_EXPORT_CASES = [
    {
        'name': 'resnet',
        'config': (ROOT / 'configs' / 'classification' / 'presets' / 'cifar10' / 'resnet.yaml').as_posix(),
        'onnx_name': 'quantized_resnet.onnx',
        'output_shape': (1, 10),
    },
    {
        'name': 'vit',
        'config': (ROOT / 'configs' / 'classification' / 'presets' / 'cifar10' / 'vit.yaml').as_posix(),
        'onnx_name': 'quantized_vit.onnx',
        'output_shape': (1, 10),
    },
]


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
    with tempfile.TemporaryDirectory(prefix='qat_export_onnx_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        ok = 0
        total = 0

        for case in QAT_EXPORT_CASES:
            total += 2
            train_output_dir = tmp / case['name'] / 'train_run'
            checkpoint_path = train_output_dir / 'checkpoint_00001.pth'
            onnx_path = tmp / case['name'] / case['onnx_name']
            float_onnx_path = onnx_path.with_suffix('.float.onnx')
            onnx_path.parent.mkdir(parents=True, exist_ok=True)

            train_command = [
                PYTHON,
                'train.py',
                '--config',
                case['config'],
                'device', 'cpu',
                'train.output_dir', str(train_output_dir),
                'train.batch_size', '4',
                'validation.batch_size', '4',
                'train.dataloader.num_workers', '0',
                'validation.dataloader.num_workers', '0',
                'test.dataloader.num_workers', '0',
                'scheduler.epochs', '1',
                'train.checkpoint_period', '1',
                'train.log_period', '1',
                'train.use_tensorboard', 'False',
                'train.val_period', '0',
                'dataset.name', 'FakeData',
                'dataset.image_size', '32',
                'dataset.n_channels', '3',
                'dataset.n_classes', '10',
                'qat.enabled', 'True',
                'qat.backend', 'fbgemm',
                'qat.freeze_bn_epoch', '1',
                'qat.disable_observer_epoch', '1',
            ]
            train_result = run_command(train_command)
            if train_result.returncode != 0:
                print(f"FAIL | quantized onnx smoke | training failed | model={case['name']}")
                print(train_result.stdout)
                sys.exit(1)

            if not checkpoint_path.exists():
                print(f"FAIL | quantized onnx smoke | checkpoint not found | model={case['name']} | path={checkpoint_path}")
                sys.exit(1)
            ok += 1

            export_command = [
                PYTHON,
                'export.py',
                '--config',
                case['config'],
                'device', 'cpu',
                'dataset.name', 'FakeData',
                'dataset.image_size', '32',
                'dataset.n_channels', '3',
                'dataset.n_classes', '10',
                'export.checkpoint', str(checkpoint_path),
                'export.output_file', str(onnx_path),
                'export.quantized_onnx', 'True',
                'export.quantized_onnx_backend', 'onnxruntime_dynamic',
                'qat.enabled', 'True',
                'export.opset', '13',
            ]
            export_result = run_command(export_command)
            if export_result.returncode != 0:
                print(f"FAIL | quantized onnx smoke | export failed | model={case['name']}")
                print(export_result.stdout)
                sys.exit(1)

            if not onnx_path.exists():
                print(f"FAIL | quantized onnx smoke | quantized onnx file not found | model={case['name']} | path={onnx_path}")
                sys.exit(1)
            if not float_onnx_path.exists():
                print(f"FAIL | quantized onnx smoke | float onnx file not found | model={case['name']} | path={float_onnx_path}")
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
                    "has_quant_ops=any(op in node_types for op in ('MatMulInteger','DynamicQuantizeLinear','QuantizeLinear','DequantizeLinear')); "
                    'assert has_quant_ops, f\'missing quant ops: {sorted(node_types)}\'; '
                    f"session=ort.InferenceSession(r'{onnx_path.as_posix()}', providers=['CPUExecutionProvider']); "
                    'input_meta=session.get_inputs()[0]; '
                    'input_shape=[dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_meta.shape]; '
                    'x=np.random.randn(*input_shape).astype(np.float32); '
                    'outputs=session.run(None, {input_meta.name: x}); '
                    f"expected_shape={case['output_shape']!r}; "
                    'assert tuple(outputs[0].shape) == expected_shape, f"unexpected output shape: {tuple(outputs[0].shape)}"; '
                    f"print('OK | quantized onnx export | model=', '{case['name']}', 'nodes=', len(model.graph.node), 'has_quant_ops=', has_quant_ops, 'inputs=', len(session.get_inputs()), 'outputs=', len(session.get_outputs()), 'output_shape=', tuple(outputs[0].shape))"
                )
            ]
            verify_result = run_command(verify_command)
            if verify_result.returncode != 0:
                print(f"FAIL | quantized onnx smoke | onnx verification failed | model={case['name']}")
                print(verify_result.stdout)
                sys.exit(1)

            print(verify_result.stdout.strip())
            ok += 1

        print(f'SUMMARY | ok={ok} | skip=0 | fail=0 | total={total}')


if __name__ == '__main__':
    main()
