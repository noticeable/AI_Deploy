# Detection + Segmentation

检测任务代码位于 `pytorch_object_detection/`，脚本入口位于 `scripts/detection/`。

## Image Classification

分类任务现在使用组合式配置：
- `configs/classification/datasets/`：数据集基础配置
- `configs/classification/models/cifar/`：CIFAR 系列模型基础配置
- `configs/classification/models/imagenet/`：ImageNet 系列模型基础配置
- `configs/classification/augmentations/`：增强 overlay
- `configs/classification/presets/`：可直接传给入口脚本的组合 preset

推荐直接使用 `presets` 路径，例如：

```bash
python train.py --config configs/classification/presets/cifar10/resnet.yaml
python train_imagenet.py --config configs/classification/presets/imagenet/resnet18.yaml
python evaluate.py --config configs/classification/presets/imagenet/resnet18.yaml \
    test.checkpoint path/to/checkpoint.pth
python export.py --config configs/classification/presets/cifar10/resnet.yaml \
    export.checkpoint path/to/checkpoint.pth
```

旧的 `configs/cifar/*.yaml` 和 `configs/imagenet/*.yaml` 仍然保留为兼容入口，但推荐新用法统一切到 `configs/classification/presets/...`。

## Detection + Segmentation 并网项目

仓库现已新增一个独立的检测+分割并网 project：
- package: `pytorch_det_seg/`
- scripts: `scripts/det_seg/`
- config: `configs/det_seg/shared_backbone_tiny.yaml`

设计约束：
- 共 backbone
- 不共 detection / segmentation neck
- 不共 detection / segmentation head

数据目录约定：
- `train/images`
- `train/labels`
- `train/masks`
- `val/images`
- `val/labels`
- `val/masks`

其中：
- `labels` 仍采用 YOLO txt 标注格式
- `masks` 采用与图像同名的 png 掩码图

训练：

```bash
python scripts/det_seg/train.py --config configs/det_seg/shared_backbone_tiny.yaml
```

评估：

```bash
python scripts/det_seg/evaluate.py --config configs/det_seg/shared_backbone_tiny.yaml \
    test.checkpoint path/to/checkpoint.pth
```

导出：

```bash
python scripts/det_seg/export.py --config configs/det_seg/shared_backbone_tiny.yaml \
    export.checkpoint path/to/checkpoint.pth
```

det-seg 剪枝：

```bash
python tools/prune_det_seg_pagcp.py --config configs/det_seg/shared_backbone_tiny.yaml \
    device cpu \
    prune.backend torch_pruning \
    prune.method tp_magnitude \
    prune.target backbone \
    prune.checkpoint path/to/det_seg_checkpoint.pth \
    prune.output_dir outputs/prune_det_seg \
    prune.save_name model_tp_pruned.pth \
    prune.amount 0.2 \
    prune.example_batch_size 1 \
    prune.example_image_size 640
```

说明：
- 当前 det-seg torch-pruning 默认只支持 `prune.target=backbone`
- 保存前会重建 shared-backbone / det-branch / seg-branch 结构，保证 pruned checkpoint 可直接恢复
- pruned checkpoint 会保存真实 `config.model.backbone.channels`
- pruned checkpoint 已验证 restore / train / evaluate / export 闭环

loss 配置：
- `loss.det_weight`
- `loss.seg_weight`

limtl 兼容入口：
- `limtl.enabled`
- `limtl.strategy`

当前第一版默认使用静态 loss 权重；当 `limtl.enabled=True` 时，会接入 LibMTL weighting 策略。当前已完成真实 LibMTL 兼容，并已验证 `EW`、`GradNorm`、`MGDA`、`PCGrad`、`DWA`、`CAGrad`、`IMTL`、`GradVac`、`RLW`、`UW`、`GLS`、`GradDrop` 十二种策略可在该 det-seg 项目中训练运行，并通过 smoke test。`GradDrop` 当前走 shared representation gradient 路径，默认使用 `limtl.graddrop.leak=0.0`。

当前 det-seg evaluator 已输出更正式的分割指标：
- `seg_acc`
- `seg_mIoU`
- `seg_mDice`
- `seg_fwIoU`
- `seg_class_iou`
- `seg_class_dice`

当前 det-seg 的 checkpoint / evaluate / export 已形成完整闭环：
- evaluate 可从 checkpoint 加载并输出 `det_seg_metrics.json`
- export 可从 checkpoint 加载并导出 ONNX
- ONNX 当前输出两个张量：`detections` 与 `segmentation`

Smoke test：

```bash
python tools/smoke_test_det_seg_project.py
python tools/smoke_test_prune_det_seg_pagcp.py
```

## 常用入口

训练 YOLO：

```bash
python scripts/detection/train.py --config configs/detection/yolo/yolov8_n.yaml
```

当前默认 YOLO 检测路径已切到 dense candidate head：
- 默认不再走 `first_box`
- 默认使用 `assignment.yolo.name=dynamic_k`
- 默认 `eval.nms_type=soft`，Soft-NMS 现在会在真实候选集合上生效
- 若需要导出 ONNX，请显式覆盖为 `eval.nms_type hard`

评估 YOLO：

```bash
python scripts/detection/evaluate.py --config configs/detection/yolo/yolov8_n.yaml \
    test.checkpoint path/to/checkpoint.pth
```

训练 DETR：

```bash
python scripts/detection/train.py --config configs/detection/detr/detr_r50.yaml
```

推理：

```bash
python scripts/detection/infer.py --config configs/detection/detr/detr_r50.yaml \
    --input path/to/image.jpg --output outputs/detection.jpg
```

导出 ONNX：

```bash
python scripts/detection/export.py --config configs/detection/yolo/yolov8_n.yaml \
    eval.nms_type hard \
    export.output_file outputs/yolov8_n.onnx
```

若 `export.checkpoint` 指向 pruned detection checkpoint，导出入口会先按 checkpoint 中保存的结构自动重建模型，再加载权重导出。

## 剪枝与蒸馏
