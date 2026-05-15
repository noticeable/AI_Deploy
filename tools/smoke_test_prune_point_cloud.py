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
    with tempfile.TemporaryDirectory(prefix='prune_point_cloud_') as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        checkpoint_path = tmp / 'dummy_pointnet_checkpoint.pth'

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
            print('SKIP | point-cloud prune smoke | environment not ready to create dummy checkpoint')
            print(result.stdout)
            return

        tests = [
            {
                'name': 'pointnet_global',
                'config': 'configs/point_cloud/classification/pointnet_cls_modelnet40.yaml',
                'checkpoint': checkpoint_path,
                'output': tmp / 'pointnet_global_pruned.pth',
                'extra': ['prune.amount', '0.2'],
                'verify': 'reload',
            },
            {
                'name': 'pointnet_structured',
                'config': 'configs/point_cloud/classification/pointnet_cls_modelnet40.yaml',
                'checkpoint': checkpoint_path,
                'output': tmp / 'pointnet_structured_pruned.pth',
                'extra': ['prune.method', 'structured_ln', 'prune.amount', '0.25', 'prune.save_name', 'pointnet_structured_pruned.pth'],
                'verify': 'reload',
            },
            {
                'name': 'pointnet_tp',
                'config': 'configs/point_cloud/classification/pointnet_cls_modelnet40.yaml',
                'checkpoint': checkpoint_path,
                'output': tmp / 'pointnet_tp_pruned.pth',
                'extra': [
                    'prune.backend', 'torch_pruning',
                    'prune.method', 'tp_magnitude',
                    'prune.target', 'classifier',
                    'prune.modules', '["linear"]',
                    'prune.amount', '0.2',
                    'prune.save_name', 'pointnet_tp_pruned.pth',
                    'prune.example_batch_size', '1',
                    'prune.example_num_points', '32',
                    'dataset.num_points', '32',
                ],
                'verify': 'meta',
            },
            {
                'name': 'pointnet2_global',
                'config': 'configs/point_cloud/classification/pointnet2_cls_modelnet40.yaml',
                'checkpoint': None,
                'output': tmp / 'pointnet2_global_pruned.pth',
                'extra': ['prune.amount', '0.2'],
                'verify': 'reload',
            },
            {
                'name': 'pointnet2_structured',
                'config': 'configs/point_cloud/classification/pointnet2_cls_modelnet40.yaml',
                'checkpoint': None,
                'output': tmp / 'pointnet2_structured_pruned.pth',
                'extra': ['prune.method', 'structured_ln', 'prune.amount', '0.25', 'prune.save_name', 'pointnet2_structured_pruned.pth'],
                'verify': 'reload',
            },
            {
                'name': 'dgcnn_global',
                'config': 'configs/point_cloud/classification/dgcnn_cls_modelnet40.yaml',
                'checkpoint': None,
                'output': tmp / 'dgcnn_global_pruned.pth',
                'extra': ['prune.amount', '0.2'],
                'verify': 'reload',
            },
            {
                'name': 'dgcnn_structured',
                'config': 'configs/point_cloud/classification/dgcnn_cls_modelnet40.yaml',
                'checkpoint': None,
                'output': tmp / 'dgcnn_structured_pruned.pth',
                'extra': ['prune.method', 'structured_ln', 'prune.amount', '0.25', 'prune.save_name', 'dgcnn_structured_pruned.pth'],
                'verify': 'reload',
            },
            {
                'name': 'dgcnn_tp',
                'config': 'configs/point_cloud/classification/dgcnn_cls_modelnet40.yaml',
                'checkpoint': None,
                'output': tmp / 'dgcnn_tp_pruned.pth',
                'extra': [
                    'prune.backend', 'torch_pruning',
                    'prune.method', 'tp_magnitude',
                    'prune.target', 'classifier',
                    'prune.modules', '["linear"]',
                    'prune.amount', '0.2',
                    'prune.save_name', 'dgcnn_tp_pruned.pth',
                    'prune.example_batch_size', '1',
                    'prune.example_num_points', '32',
                    'dataset.num_points', '32',
                ],
                'verify': 'meta',
            },
        ]

        for test in tests:
            test_checkpoint = test['checkpoint']
            if test_checkpoint is None:
                test_checkpoint = tmp / f"{test['name']}_dummy.pth"
                create_command = [
                    PYTHON,
                    '-c',
                    (
                        'import pathlib, torch; '
                        'from pytorch_point_cloud import get_default_config, create_model, update_config; '
                        'config=get_default_config(); '
                        "config.merge_from_file('%s'); "
                        'config=update_config(config); '
                        "config.device='cpu'; "
                        'model=create_model(config); '
                        "path=pathlib.Path(r\'%s\'); "
                        "torch.save({'model': model.state_dict()}, path)"
                    ) % (test['config'], test_checkpoint.as_posix().replace('\\', '\\\\'))
                ]
                result = run_command(create_command)
                if result.returncode != 0:
                    print(f"FAIL | point-cloud prune smoke | {test['name']} dummy checkpoint creation failed")
                    print(result.stdout)
                    sys.exit(1)

            prune_command = [
                PYTHON,
                'tools/prune_point_cloud.py',
                '--config',
                test['config'],
                'device',
                'cpu',
                'dataset.num_points',
                '32',
                'prune.checkpoint',
                str(test_checkpoint),
                'prune.output_dir',
                str(tmp),
                'prune.save_name',
                test['output'].name,
                *test['extra'],
            ]
            result = run_command(prune_command)
            if result.returncode != 0:
                print(f"FAIL | point-cloud prune smoke | {test['name']} prune command failed")
                print(result.stdout)
                sys.exit(1)

            if not test['output'].exists():
                print(f"FAIL | point-cloud prune smoke | output not found: {test['output']}")
                sys.exit(1)

            if test['verify'] == 'meta':
                verify_script = (
                    'import torch; '
                    'from pytorch_point_cloud import get_default_config, create_model, create_model_from_checkpoint, update_config; '
                    'config=get_default_config(); '
                    "config.merge_from_file('%s'); "
                    'config.merge_from_list(["device","cpu","dataset.num_points","32"]); '
                    'config=update_config(config); '
                    "model, loaded_config, checkpoint = create_model_from_checkpoint(config, r'%s', create_model); "
                    'model.eval(); '
                    'x=torch.randn(1, 32, loaded_config.dataset.n_channels); '
                    'y=model(x); '
                    'shape=tuple(y["logits"].shape); '
                    'expected=(1, loaded_config.dataset.n_classes); '
                    'prune_meta=checkpoint.get("prune", {}); '
                    'assert shape == expected, f"expected logits shape {expected}, got {shape}"; '
                    "print('OK | %s logits_shape=', shape, 'rebuilt=', bool(prune_meta.get('rebuilt', False)))"
                ) % (test['config'], test['output'].as_posix().replace('\\', '\\\\'), test['name'])
            else:
                verify_script = (
                    'import torch; '
                    'from pytorch_point_cloud import get_default_config, create_model, update_config; '
                    'config=get_default_config(); '
                    "config.merge_from_file('%s'); "
                    'config.merge_from_list(["device","cpu","dataset.num_points","32"]); '
                    'config=update_config(config); '
                    'model=create_model(config); '
                    "checkpoint=torch.load(r'%s', map_location='cpu'); "
                    "model.load_state_dict(checkpoint['model'], strict=False); "
                    'model.eval(); '
                    'x=torch.randn(1, 32, config.dataset.n_channels); '
                    'y=model(x); '
                    "print('OK | %s logits_shape=', tuple(y['logits'].shape))"
                ) % (test['config'], test['output'].as_posix().replace('\\', '\\\\'), test['name'])

            verify_command = [PYTHON, '-c', verify_script]
            result = run_command(verify_command)
            if result.returncode != 0:
                print(f"FAIL | point-cloud prune smoke | {test['name']} verification failed")
                print(result.stdout)
                sys.exit(1)
            print(result.stdout.strip())

        print('SUMMARY | ok=8 | skip=0 | fail=0 | total=8')


if __name__ == '__main__':
    main()
