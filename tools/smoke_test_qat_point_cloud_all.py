#!/usr/bin/env python

import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = pathlib.Path.home() / '.conda' / 'envs' / 'pytorch310' / ('python.exe' if os.name == 'nt' else 'bin/python')
PYTHON = str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else pathlib.Path(sys.executable))

CASES = [
    {
        'name': 'pointnet_cls',
        'config': 'configs/point_cloud/classification/pointnet_cls_modelnet40.yaml',
        'task': 'classification',
        'onnx_name': 'quantized_pointnet_cls.onnx',
        'float_onnx_name': 'quantized_pointnet_cls.float.onnx',
        'extra_options': ['dataset.n_classes', '40'],
        'expected_shape': '(1, 40)',
        'expect_quant_ops': True,
        'expect_qdq': False,
    },
    {
        'name': 'dgcnn_cls',
        'config': 'configs/point_cloud/classification/dgcnn_cls_modelnet40.yaml',
        'task': 'classification',
        'onnx_name': 'quantized_dgcnn_cls.onnx',
        'float_onnx_name': 'quantized_dgcnn_cls.float.onnx',
        'extra_options': ['dataset.n_classes', '40'],
        'expected_shape': '(1, 256)',
        'expect_quant_ops': True,
        'expect_qdq': False,
    },
    {
        'name': 'pointnet_seg',
        'config': 'configs/point_cloud/segmentation/pointnet_seg_shapenetpart.yaml',
        'task': 'segmentation',
        'onnx_name': 'quantized_pointnet_seg.onnx',
        'float_onnx_name': 'quantized_pointnet_seg.float.onnx',
        'extra_options': ['dataset.n_classes', '16', 'dataset.n_seg_classes', '50'],
        'expected_shape': '(1, 32, 50)',
        'expect_quant_ops': True,
        'expect_qdq': False,
    },
    {
        'name': 'dgcnn_seg',
        'config': 'configs/point_cloud/segmentation/dgcnn_seg_shapenetpart.yaml',
        'task': 'segmentation',
        'onnx_name': 'quantized_dgcnn_seg.onnx',
        'float_onnx_name': 'quantized_dgcnn_seg.float.onnx',
        'extra_options': ['dataset.n_classes', '16', 'dataset.n_seg_classes', '50'],
        'expected_shape': '(1, 32, 50)',
        'expect_quant_ops': True,
        'expect_qdq': True,
    },
    {
        'name': 'pointnet2_cls',
        'config': 'configs/point_cloud/classification/pointnet2_cls_modelnet40.yaml',
        'task': 'classification',
        'onnx_name': 'quantized_pointnet2_cls.onnx',
        'float_onnx_name': 'quantized_pointnet2_cls.float.onnx',
        'extra_options': ['dataset.n_classes', '40'],
        'expected_shape': '(1, 40)',
        'expect_quant_ops': True,
        'expect_qdq': False,
    },
    {
        'name': 'pointnet2_seg',
        'config': 'configs/point_cloud/segmentation/pointnet2_seg_shapenetpart.yaml',
        'task': 'segmentation',
        'onnx_name': 'quantized_pointnet2_seg.onnx',
        'float_onnx_name': 'quantized_pointnet2_seg.float.onnx',
        'extra_options': ['dataset.n_classes', '16', 'dataset.n_seg_classes', '50'],
        'expected_shape': '(1, 32, 50)',
        'expect_quant_ops': True,
        'expect_qdq': True,
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


def verify_convert(tmp, checkpoint_path, case):
    if case['task'] == 'classification':
        options = [
            'device', 'cpu',
            'dataset.dataset_dir', str(tmp),
            'dataset.num_points', '32',
            'qat.enabled', 'True',
        ] + case['extra_options']
        output_expr = "tuple(y.shape)"
        label = 'point-cloud qat convert'
    else:
        options = [
            'device', 'cpu',
            'dataset.dataset_dir', str(tmp),
            'dataset.num_points', '32',
            'qat.enabled', 'True',
        ] + case['extra_options']
        output_expr = "tuple(y['seg_logits'].shape) if isinstance(y, dict) else tuple(y.shape)"
        label = 'point-cloud seg qat convert'

    verify_command = [
        PYTHON,
        '-c',
        (
            'import torch; '
            'from pytorch_point_cloud import get_default_config, create_model, update_config; '
            'from pytorch_point_cloud.models.qat import prepare_model_for_qat, convert_qat_model; '
            'from fvcore.common.checkpoint import Checkpointer; '
            'config=get_default_config(); '
            f"config.merge_from_file('{case['config']}'); "
            f"config.merge_from_list({options!r}); "
            'config=update_config(config); '
            'model=create_model(config); '
            'model,_=prepare_model_for_qat(config, model); '
            f"Checkpointer(model).load(r'{checkpoint_path.as_posix()}'); "
            'quantized=convert_qat_model(config, model); '
            'x=torch.randn(1, config.dataset.num_points, config.dataset.n_channels); '
            'y=quantized(x); '
            f"shape={output_expr}; assert str(shape) == '{case['expected_shape']}', 'expected {case['expected_shape']}, got %s' % (shape,); "
            f"print('OK | {label} | model=', '{case['name']}', 'output_shape=', shape)"
        )
    ]
    return run_command(verify_command)


def verify_onnx(onnx_path, case):
    verify_command = [
        PYTHON,
        '-c',
        (
            'import numpy as np; '
            'import onnx; '
            'import onnxruntime as ort; '
            f"model=onnx.load(r'{onnx_path.as_posix()}'); "
            'node_types={node.op_type for node in model.graph.node}; '
            "has_quant_ops=any(op in node_types for op in ('MatMulInteger','DynamicQuantizeLinear','QuantizeLinear','DequantizeLinear','ConvInteger','QLinearConv')); "
            "has_qdq=('QuantizeLinear' in node_types and 'DequantizeLinear' in node_types); "
            f"assert has_quant_ops == {case['expect_quant_ops']!r}, 'expected has_quant_ops={case['expect_quant_ops']}, got %s' % (has_quant_ops,); "
            f"assert has_qdq == {case['expect_qdq']!r}, 'expected has_qdq={case['expect_qdq']}, got %s' % (has_qdq,); "
            f"session=ort.InferenceSession(r'{onnx_path.as_posix()}', providers=['CPUExecutionProvider']); "
            'input_meta=session.get_inputs()[0]; '
            "x=np.random.randn(1, 32, 3).astype(np.float32); "
            'outputs=session.run(None, {input_meta.name: x}); '
            f"print('OK | point-cloud quantized onnx export | model=', '{case['name']}', 'nodes=', len(model.graph.node), 'has_quant_ops=', has_quant_ops, 'has_qdq=', has_qdq, 'output_shape=', tuple(outputs[0].shape))"
        )
    ]
    return run_command(verify_command)


def main():
    with tempfile.TemporaryDirectory(prefix='qat_point_cloud_multi_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        ok = 0
        total = 0

        for case in CASES:
            total += 2
            output_dir = tmp / case['name'] / 'qat_run'
            checkpoint_dir = output_dir / 'artifacts'
            checkpoint_path = checkpoint_dir / 'checkpoint_00001.pth'
            onnx_path = tmp / case['name'] / case['onnx_name']
            float_onnx_path = tmp / case['name'] / case['float_onnx_name']
            onnx_path.parent.mkdir(parents=True, exist_ok=True)

            train_command = [
                PYTHON,
                'scripts/point_cloud/train.py',
                '--config',
                case['config'],
                'device', 'cpu',
                'train.output_dir', str(checkpoint_dir),
                'train.batch_size', '4',
                'validation.batch_size', '4',
                'train.dataloader.num_workers', '0',
                'validation.dataloader.num_workers', '0',
                'scheduler.epochs', '1',
                'train.checkpoint_period', '1',
                'train.log_period', '1',
                'train.use_tensorboard', 'False',
                'dataset.dataset_dir', str(tmp),
                'dataset.num_points', '32',
                'qat.enabled', 'True',
                'qat.freeze_bn_epoch', '1',
                'qat.disable_observer_epoch', '1',
            ] + case['extra_options']
            train_result = run_command(train_command)
            if train_result.returncode != 0:
                print(f"FAIL | {case['name']} point-cloud qat smoke | train command failed")
                print(train_result.stdout)
                sys.exit(1)
            if not checkpoint_path.exists():
                print(f'FAIL | {case["name"]} point-cloud qat smoke | checkpoint not found: {checkpoint_path}')
                sys.exit(1)

            convert_result = verify_convert(tmp, checkpoint_path, case)
            if convert_result.returncode != 0:
                print(f"FAIL | {case['name']} point-cloud qat smoke | convert verification failed")
                print(convert_result.stdout)
                sys.exit(1)
            print(convert_result.stdout.strip())
            ok += 1

            export_command = [
                PYTHON,
                'scripts/point_cloud/export.py',
                '--config',
                case['config'],
                'device', 'cpu',
                'dataset.dataset_dir', str(tmp),
                'dataset.num_points', '32',
                'export.checkpoint', str(checkpoint_path),
                'export.output_file', str(onnx_path),
                'export.quantized_onnx', 'True',
                'export.quantized_onnx_backend', 'onnxruntime_dynamic',
                'qat.enabled', 'True',
            ] + case['extra_options']
            export_result = run_command(export_command)
            if export_result.returncode != 0:
                print(f"FAIL | {case['name']} point-cloud quantized onnx smoke | export failed")
                print(export_result.stdout)
                sys.exit(1)
            if not onnx_path.exists():
                print(f'FAIL | {case["name"]} point-cloud quantized onnx smoke | quantized onnx file not found: {onnx_path}')
                sys.exit(1)
            if not float_onnx_path.exists():
                print(f'FAIL | {case["name"]} point-cloud quantized onnx smoke | float onnx file not found: {float_onnx_path}')
                sys.exit(1)

            onnx_verify_result = verify_onnx(onnx_path, case)
            if onnx_verify_result.returncode != 0:
                print(f"FAIL | {case['name']} point-cloud quantized onnx smoke | onnx verification failed")
                print(onnx_verify_result.stdout)
                sys.exit(1)
            print(onnx_verify_result.stdout.strip())
            ok += 1

        print(f'SUMMARY | ok={ok} | skip=0 | fail=0 | total={total}')


if __name__ == '__main__':
    main()
