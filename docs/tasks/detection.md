# Detection

检测任务代码位于 `pytorch_object_detection/`，脚本入口位于 `scripts/detection/`。

## 常用入口

训练 YOLO：

```bash
python scripts/detection/train.py --config configs/detection/yolo/yolov8_n.yaml
```

也可以直接使用动态分配配置：

```bash
python scripts/detection/train.py --config configs/detection/yolo/yolov8_n_dynamic_k.yaml
```

也可以直接开启保守版 label smoothing：

```bash
python scripts/detection/train.py --config configs/detection/yolo/yolov8_n.yaml \
    augmentation.use_label_smoothing True \
    augmentation.label_smoothing.epsilon 0.05
```

也可以在评估 / 推理时切到 Soft-NMS：

```bash
python scripts/detection/evaluate.py --config configs/detection/yolo/yolov8_n.yaml \
    test.checkpoint path/to/checkpoint.pth \
    eval.nms_type soft
```

```bash
python scripts/detection/infer.py --config configs/detection/yolo/yolov8_n.yaml \
    --input path/to/image.jpg --output outputs/detection.jpg \
    eval.nms_type soft
```

说明：
- Soft-NMS 当前已接到 YOLO / QAT YOLO 的推理输出后处理路径
- 当前 TinyYOLO 默认每张图候选数很少，因此 Soft-NMS 多数情况下是保守 no-op，但接口已预留给后续更 dense 的 head
- ONNX 导出当前不支持 `eval.nms_type=soft`，导出前请切回 `hard`

`model.yolo.block` 当前支持以下可选轻量 backbone block：
- `conv`（默认）
- `dconv`
- `dsconv`
- `ghost`
- `gsconv`
- `pconv`

示例：

```bash
python scripts/detection/train.py --config configs/detection/yolo/yolov8_n.yaml \
    model.yolo.block ghost
```

也可以直接使用现成配置：
- `configs/detection/yolo/yolov8_n_dconv.yaml`
- `configs/detection/yolo/yolov8_n_dsconv.yaml`
- `configs/detection/yolo/yolov8_n_dsconv_giou.yaml`
- `configs/detection/yolo/yolov8_n_dsconv_ciou.yaml`
- `configs/detection/yolo/yolov8_n_ghost.yaml`
- `configs/detection/yolo/yolov8_n_gsconv.yaml`
- `configs/detection/yolo/yolov8_n_gsconv_giou.yaml`
- `configs/detection/yolo/yolov8_n_gsconv_ciou.yaml`
- `configs/detection/yolo/yolov8_n_pconv.yaml`
- `configs/detection/yolo/yolov8_n_pconv_giou.yaml`
- `configs/detection/yolo/yolov8_n_pconv_ciou.yaml`

说明：
- `block_kernel_size` 默认是 `3`
- 这些轻量 block 当前主要面向普通 train / evaluate / export 流程
- QAT 目前仅保证默认 `conv` backbone
- `gsconv` 当前实现保留了原始 GSConv 的 channel shuffle 思路，并显式让 `stride` 生效
- `pconv` 当前是面向 TinyYOLO 的保守适配版：使用 `1x1 project -> partial conv -> BN -> ReLU`，不是源脚本里的纯 `PartialConv` 直连写法

### YOLO block variants workflow

现成配置与输出目录：
- `configs/detection/yolo/yolov8_n_dconv.yaml` -> `dconv` -> `experiments/detection/yolo_dconv/exp00`
- `configs/detection/yolo/yolov8_n_dsconv.yaml` -> `dsconv` -> `experiments/detection/yolo_dsconv/exp00`
- `configs/detection/yolo/yolov8_n_dsconv_giou.yaml` -> `dsconv + giou` -> `experiments/detection/yolo_dsconv_giou/exp00`
- `configs/detection/yolo/yolov8_n_dsconv_ciou.yaml` -> `dsconv + ciou` -> `experiments/detection/yolo_dsconv_ciou/exp00`
- `configs/detection/yolo/yolov8_n_ghost.yaml` -> `ghost` -> `experiments/detection/yolo_ghost/exp00`
- `configs/detection/yolo/yolov8_n_gsconv.yaml` -> `gsconv` -> `experiments/detection/yolo_gsconv/exp00`
- `configs/detection/yolo/yolov8_n_gsconv_giou.yaml` -> `gsconv + giou` -> `experiments/detection/yolo_gsconv_giou/exp00`
- `configs/detection/yolo/yolov8_n_gsconv_ciou.yaml` -> `gsconv + ciou` -> `experiments/detection/yolo_gsconv_ciou/exp00`
- `configs/detection/yolo/yolov8_n_pconv.yaml` -> `pconv` -> `experiments/detection/yolo_pconv/exp00`
- `configs/detection/yolo/yolov8_n_pconv_giou.yaml` -> `pconv + giou` -> `experiments/detection/yolo_pconv_giou/exp00`
- `configs/detection/yolo/yolov8_n_pconv_ciou.yaml` -> `pconv + ciou` -> `experiments/detection/yolo_pconv_ciou/exp00`

端到端示例：

```bash
python scripts/detection/train.py --config configs/detection/yolo/yolov8_n_ghost.yaml
python scripts/detection/evaluate.py --config configs/detection/yolo/yolov8_n_ghost.yaml \
    test.checkpoint experiments/detection/yolo_ghost/exp00/checkpoint_best.pth
python scripts/detection/export.py --config configs/detection/yolo/yolov8_n_ghost.yaml \
    export.checkpoint experiments/detection/yolo_ghost/exp00/checkpoint_best.pth \
    export.output_file outputs/yolov8_n_ghost.onnx
```

不改配置文件直接切换 block：

```bash
python scripts/detection/train.py --config configs/detection/yolo/yolov8_n.yaml \
    model.yolo.block gsconv
```

补充说明：
- 当前已提供 11 份 block / box-loss 组合配置
- 另提供 1 份动态分配示例配置：`configs/detection/yolo/yolov8_n_dynamic_k.yaml`
- `dynamic_k` 是保守迁移版 OTA-style assigner：仅保留 pairwise IoU + classification cost + dynamic-k matching，不包含 YOLOv5 原始的 anchor/grid/objectness 多层候选筛选
- evaluate 后仍会产出 `detection_predictions.json` 和 `predictions_yolo/*.txt`
- 若要继续做误检/漏检可视化，可直接配合 `scripts/detection/visualize_errors.py`

YOLO 转 COCO：

```bash
python scripts/detection/convert_yolo_to_coco.py --root-dir data/my_yolo --output annotations/instances_all.json
```

VOC XML 转 COCO：

```bash
python scripts/detection/convert_voc_to_coco.py --xml-dir data/voc/Annotations --output data/voc/annotations/train.json
```

离线增强 YOLO 数据集：

```bash
python scripts/detection/offline_augment_yolo.py --input-root data/my_yolo/train --output-root data/my_yolo_aug/train --loops 1
```

可视化 GT / 预测误检漏检：

```bash
python scripts/detection/visualize_errors.py --image-dir data/my_eval/images --label-dir data/my_eval/labels --prediction-dir outputs/preds --output-dir outputs/vis_errors
```

评估 YOLO：

```bash
python scripts/detection/evaluate.py --config configs/detection/yolo/yolov8_n.yaml \
    test.checkpoint path/to/checkpoint.pth
```

评估后会额外产出：
- `detection_predictions.json`
- `predictions_yolo/*.txt`

可直接配合 `scripts/detection/visualize_errors.py` 做误检/漏检可视化。

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
    export.output_file outputs/yolov8_n.onnx
```

若 `export.checkpoint` 指向 pruned detection checkpoint，导出入口会先按 checkpoint 中保存的结构自动重建模型，再加载权重导出。

## 剪枝与蒸馏

检测剪枝：

```bash
python tools/prune_detection_pagcp.py --config configs/detection/yolo/yolov8_n.yaml \
    device cpu \
    prune.backend builtin \
    prune.checkpoint path/to/detection_checkpoint.pth \
    prune.output_dir outputs/prune \
    prune.save_name model_pruned.pth \
    prune.amount 0.2
```

使用 `torch-pruning`：

```bash
python tools/prune_detection_pagcp.py --config configs/detection/yolo/yolov8_n.yaml \
    device cpu \
    prune.backend torch_pruning \
    prune.method tp_magnitude \
    prune.target backbone \
    prune.checkpoint path/to/detection_checkpoint.pth \
    prune.output_dir outputs/prune \
    prune.save_name model_tp_pruned.pth \
    prune.amount 0.2 \
    prune.example_batch_size 1 \
    prune.example_image_size 640
```

说明：
- 默认优先裁剪 TinyYOLO backbone
- 会自动同步 `box_head` / `score_head` 的输入维度，不直接裁掉检测输出维度
- `torch-pruning` 产出的 pruned checkpoint 会保存实际的 TinyYOLO channel layout，后续可用于自动结构重建

检测蒸馏微调：

```bash
python scripts/detection/train.py --config configs/detection/yolo/yolov8_n.yaml \
    device cpu \
    train.checkpoint outputs/prune/model_pruned.pth \
    distill.enabled True \
    distill.teacher_checkpoint path/to/teacher_checkpoint.pth \
    distill.temperature 2.0 \
    distill.cls_weight 1.0 \
    distill.box_weight 1.0 \
    distill.hard_loss_weight 1.0 \
    distill.soft_loss_weight 1.0
```

说明：
- student 可以直接使用普通 checkpoint 或 torch-pruning 产出的 pruned checkpoint
- teacher 也可以是普通 checkpoint 或 pruned checkpoint
- 训练入口会分别按各自 checkpoint 中保存的结构恢复模型，再加载权重

推荐 smoke test：

```bash
python tools/smoke_test_prune_detection_pagcp.py
python tools/smoke_test_prune_distill_detection_pagcp.py
```

## Pruned checkpoint 自动重建

对 TinyYOLO 而言，`torch-pruning` 之后的通道数可能不再是 `width_mult` 能表达的规则倍率，例如 `[12, 25, 51, 102]`。因此 pruned checkpoint 会额外保存真实结构信息，加载流程会优先读取这些信息来重建模型。

当前以下入口都支持这种自动重建：
- `scripts/detection/train.py`
- `scripts/detection/evaluate.py`
- `scripts/detection/export.py`
- `pytorch_object_detection.utils.checkpoint.create_model_from_checkpoint(...)`

这意味着：
- pruned checkpoint 可以直接继续训练
- pruned checkpoint 可以直接评估
- pruned checkpoint 可以直接导出 ONNX
- prune -> distill 流程不需要手动改模型定义

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
- 当前默认会在保存前重建 shared-backbone / det-branch / seg-branch 结构，保证 pruned checkpoint 可直接恢复
- pruned checkpoint 会保存真实 `config.model.backbone.channels`
- pruned checkpoint 已验证 restore / train / evaluate / export 闭环

loss 配置：
- `loss.det_weight`
- `loss.seg_weight`

limtl / LibMTL 兼容入口：
- `limtl.enabled`
- `limtl.strategy`

当前默认仍支持静态 loss 权重；当 `limtl.enabled=True` 时，会接入 LibMTL weighting 策略。

当前 det-seg evaluator 已输出更正式的分割指标：
- `seg_acc`
- `seg_mIoU`
- `seg_mDice`
- `seg_fwIoU`
- `seg_class_iou`
- `seg_class_dice`

当前 det-seg 的 checkpoint / evaluate / export 已形成完整闭环：
- `pytorch_det_seg.utils.checkpoint.create_model_from_checkpoint(...)` 可统一从 checkpoint 恢复 config 与 model
- `scripts/det_seg/evaluate.py` 会从 checkpoint 加载模型并输出 `det_seg_metrics.json`
- `scripts/det_seg/export.py` 会从 checkpoint 加载模型并导出 ONNX

当前已验证通过的策略：
- `EW`
- `GradNorm`
- `MGDA`
- `PCGrad`
- `DWA`
- `CAGrad`
- `IMTL`
- `GradVac`
- `RLW`
- `UW`
- `GLS`
- `GradDrop`

说明：
- `EW` 可直接作为基础动态权重策略使用
- `GradNorm` 当前在训练循环中已补齐 `epoch` 与 `train_loss_buffer` 生命周期
- `MGDA` 当前已补齐 `mgda_gn='none'` 默认参数，并通过 safe gradient reset 兼容当前 shared-backbone 结构
- `PCGrad` 当前已通过 safe gradient reset 兼容当前 shared-backbone 结构
- `DWA` 已补齐 `T=2.0` 默认参数并通过 smoke test
- `CAGrad` 当前已补齐 `calpha=0.5`、`rescale=1` 默认参数，并修正为适配双任务场景的优化目标形状
- `IMTL` 当前已通过 safe gradient reset 兼容当前 shared-backbone 结构
- `GradVac` 当前已补齐 `beta=0.5` 默认参数，并通过 safe gradient reset 兼容当前 shared-backbone 结构
- `RLW` 当前已补齐 `dist='Normal'` 默认参数并通过 smoke test
- `UW` 使用 LibMTL 原生不确定性加权逻辑即可通过当前 det-seg smoke test
- `GLS` 使用 LibMTL 原生几何损失策略即可通过当前 det-seg smoke test
- `GradDrop` 当前已接通 shared representation gradient 路径，默认使用 `limtl.graddrop.leak=0.0`，并通过 smoke test

Smoke test：

```bash
python tools/smoke_test_det_seg_project.py
python tools/smoke_test_prune_det_seg_pagcp.py
```

## 导出输出格式

导出的 ONNX 主输出名为 `detections`，张量格式为 `[batch_size, num_detections, 6]`，最后一维依次表示：
- `x1`
- `y1`
- `x2`
- `y2`
- `score`
- `label`

同时还会输出 `segmentation` 张量，格式为 `[batch_size, n_classes, height, width]`。

说明：
- 当前导出包装会将每张图的检测结果整理为单个 `detections` 张量
- 若同一 batch 内不同图片的检测数量不同，导出包装会自动补零到相同长度，再按 batch 维拼接
- `segmentation` 输出保留分割 logits，便于后处理阶段自行做 argmax / softmax

## 配置目录

- `configs/detection/datasets/`: 数据集配置
- `configs/detection/augmentations/`: 增强策略
- `configs/detection/yolo/`: YOLO 配置
- `configs/detection/detr/`: DETR 配置

## 相关代码

- `pytorch_object_detection/`: 检测模型、数据、增强、评估与导出实现
- `scripts/detection/`: 检测脚本入口
- `tools/prune_detection_pagcp.py`: 检测剪枝入口
- `tools/smoke_test_prune_detection_pagcp.py`: 检测剪枝 smoke test
- `tools/smoke_test_prune_distill_detection_pagcp.py`: 检测剪枝 + 蒸馏 smoke test
- `tools/prune_det_seg_pagcp.py`: det-seg 剪枝入口
- `tools/smoke_test_prune_det_seg_pagcp.py`: det-seg 剪枝 smoke test

## 量化说明

- 当前仓库不内置检测任务的量化/PTQ 流程。
- 如需量化，请在完成普通 PyTorch 训练或 ONNX 导出后，结合第三方工具链自行处理。
- 建议将本仓库产出的 checkpoint 或 ONNX 模型作为外部量化流程的输入，而不是在仓库内继续维护检测专用量化脚本。
