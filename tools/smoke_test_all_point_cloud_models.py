import sys
import importlib
import traceback
import torch

sys.path.insert(0, r'D:\claude_project\pytorch_image_classification')

from pytorch_point_cloud.config import get_default_config, update_config

TESTS = [
    ('classification', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\classification\pointnet_cls_modelnet40.yaml'),
    ('classification', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\classification\pointnet2_cls_modelnet40.yaml'),
    ('classification', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\classification\pointtransformer_v1_modelnet40.yaml'),
    ('classification', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\classification\pointbert_cls_modelnet40.yaml'),
    ('classification', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\classification\pointmae_cls_modelnet40.yaml'),
    ('classification', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\classification\dgcnn_cls_modelnet40.yaml'),
    ('segmentation', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\segmentation\pointnet_seg_shapenetpart.yaml'),
    ('segmentation', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\segmentation\pointnet2_seg_shapenetpart.yaml'),
    ('segmentation', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\segmentation\pointtransformer_v2_shapenetpart.yaml'),
    ('segmentation', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\segmentation\pointtransformer_v3_shapenetpart.yaml'),
    ('segmentation', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\segmentation\dgcnn_seg_shapenetpart.yaml'),
    ('segmentation', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\segmentation\kpconv_seg_shapenetpart.yaml'),
    ('segmentation', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\segmentation\pvcnn_seg_shapenetpart.yaml'),
    ('segmentation', r'D:\claude_project\pytorch_image_classification\configs\point_cloud\segmentation\pointvoxel_seg_shapenetpart.yaml'),
]

torch.set_grad_enabled(False)
results = []

for task, cfg_path in TESTS:
    try:
        cfg = get_default_config()
        cfg.merge_from_file(cfg_path)
        cfg = update_config(cfg)
        module_name = f'pytorch_point_cloud.models.{cfg.model.type}.{cfg.model.name}'
        module = importlib.import_module(module_name)
        model = module.Network(cfg).eval()
        batch_size = 2
        num_points = cfg.dataset.num_points
        channels = cfg.dataset.n_channels
        inputs = torch.randn(batch_size, num_points, channels)
        outputs = model(inputs)
        shapes = {k: tuple(v.shape) for k, v in outputs.items() if hasattr(v, 'shape')}
        line = f"OK | {cfg.model.type} | {cfg.model.name} | {task} | {sorted(outputs.keys())} | {shapes}"
        print(line)
        results.append((True, line))
    except Exception as exc:
        line = f"FAIL | {cfg_path} | {type(exc).__name__}: {exc}"
        print(line)
        traceback.print_exc()
        results.append((False, line))

ok_count = sum(1 for ok, _ in results if ok)
fail_count = len(results) - ok_count
print(f'SUMMARY | ok={ok_count} | fail={fail_count} | total={len(results)}')
if fail_count:
    sys.exit(1)
