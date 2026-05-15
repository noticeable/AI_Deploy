# Point Cloud Support

新增点云任务支持：

- 分类：ModelNet40
- 分割：ShapeNet Part
- 模型：PointNet、PointNet++、DGCNN、KPConv、PVCNN、PointVoxel、PointTransformer V1/V2/V3
- PointTransformer 支持更贴近论文的分层结构与官方原生权重导入
- PointTransformer V1 已拆分为独立 strict-style 实现路径（`common_v1.py` + V1 专属权重映射）

## 目录约定

- `~/datasets/modelnet40/train_points.npy`
- `~/datasets/modelnet40/train_labels.npy`
- `~/datasets/modelnet40/test_points.npy`
- `~/datasets/modelnet40/test_labels.npy`
- `~/datasets/shapenet_part/train_points.npy`
- `~/datasets/shapenet_part/train_cls.npy`
- `~/datasets/shapenet_part/train_seg.npy`
- `~/datasets/shapenet_part/test_points.npy`
- `~/datasets/shapenet_part/test_cls.npy`
- `~/datasets/shapenet_part/test_seg.npy`

## 配置示例

分类：
- `configs/point_cloud/classification/pointnet_cls_modelnet40.yaml`
- `configs/point_cloud/classification/pointnet2_cls_modelnet40.yaml`
- `configs/point_cloud/classification/dgcnn_cls_modelnet40.yaml`
- `configs/point_cloud/classification/pointtransformer_v1_modelnet40.yaml`

分割：
- `configs/point_cloud/segmentation/pointnet_seg_shapenetpart.yaml`
- `configs/point_cloud/segmentation/pointnet2_seg_shapenetpart.yaml`
- `configs/point_cloud/segmentation/dgcnn_seg_shapenetpart.yaml`
- `configs/point_cloud/segmentation/kpconv_seg_shapenetpart.yaml`
- `configs/point_cloud/segmentation/pvcnn_seg_shapenetpart.yaml`
- `configs/point_cloud/segmentation/pointvoxel_seg_shapenetpart.yaml`
- `configs/point_cloud/segmentation/pointtransformer_v2_shapenetpart.yaml`
- `configs/point_cloud/segmentation/pointtransformer_v3_shapenetpart.yaml`

## DGCNN 预训练权重

配置字段：
- `model.dgcnn.pretrained`
- `model.dgcnn.pretrained_path`
- `model.dgcnn.pretrained_source`
- `model.dgcnn.strict_pretrained_load`
- `model.dgcnn.reset_classifier`

当前支持：
- 官方原生 checkpoint 风格
- 常见前缀清理：`module.` / `model.` / `backbone.`
- 分类/分割 head 重置
- 分类 / 分割任务分开的 key 映射
- rename / dropped head / missing / unexpected / shape mismatch 诊断返回

示例：

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/classification/dgcnn_cls_modelnet40.yaml \
    model.dgcnn.pretrained True \
    model.dgcnn.pretrained_source official \
    model.dgcnn.pretrained_path path/to/dgcnn_checkpoint.pth \
    model.dgcnn.reset_classifier True
```

## KPConv 预训练权重

配置字段：
- `model.kpconv.pretrained`
- `model.kpconv.pretrained_path`
- `model.kpconv.pretrained_source`
- `model.kpconv.strict_pretrained_load`
- `model.kpconv.reset_classifier`

当前支持：
- 官方原生 checkpoint 风格
- 常见前缀清理：`module.` / `model.` / `backbone.`
- 分割 head 重置
- segmentation key 映射
- rename / dropped head / missing / unexpected / shape mismatch 诊断返回

示例：

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/segmentation/kpconv_seg_shapenetpart.yaml \
    model.kpconv.pretrained True \
    model.kpconv.pretrained_source official \
    model.kpconv.pretrained_path path/to/kpconv_checkpoint.pth \
    model.kpconv.reset_classifier True
```

## PVCNN 预训练权重

配置字段：
- `model.pvcnn.pretrained`
- `model.pvcnn.pretrained_path`
- `model.pvcnn.pretrained_source`
- `model.pvcnn.strict_pretrained_load`
- `model.pvcnn.reset_classifier`

当前支持：
- 官方原生 checkpoint 风格
- 常见前缀清理：`module.` / `model.` / `backbone.`
- 分割 head 重置
- point / voxel / fusion key 映射
- rename / dropped head / missing / unexpected / shape mismatch 诊断返回

示例：

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/segmentation/pvcnn_seg_shapenetpart.yaml \
    model.pvcnn.pretrained True \
    model.pvcnn.pretrained_source official \
    model.pvcnn.pretrained_path path/to/pvcnn_checkpoint.pth \
    model.pvcnn.reset_classifier True
```

## PointVoxel 预训练权重

配置字段：
- `model.pointvoxel.pretrained`
- `model.pointvoxel.pretrained_path`
- `model.pointvoxel.pretrained_source`
- `model.pointvoxel.strict_pretrained_load`
- `model.pointvoxel.reset_classifier`

当前支持：
- 官方原生 checkpoint 风格
- 常见前缀清理：`module.` / `model.` / `backbone.`
- 分割 head 重置
- point / voxel / fusion key 映射
- rename / dropped head / missing / unexpected / shape mismatch 诊断返回

示例：

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/segmentation/pointvoxel_seg_shapenetpart.yaml \
    model.pointvoxel.pretrained True \
    model.pointvoxel.pretrained_source official \
    model.pointvoxel.pretrained_path path/to/pointvoxel_checkpoint.pth \
    model.pointvoxel.reset_classifier True
```

## PointTransformer 预训练权重

配置字段：
- `model.pointtransformer.pretrained`
- `model.pointtransformer.pretrained_path`
- `model.pointtransformer.pretrained_source`
- `model.pointtransformer.strict_pretrained_load`
- `model.pointtransformer.reset_classifier`

当前支持：
- 官方原生 checkpoint 风格
- 常见前缀清理：`module.` / `model.` / `backbone.`
- 分类/分割 head 重置
- V1/V2/V3 版本化 key 映射
- rename / dropped head / missing / unexpected / shape mismatch 诊断返回
- 未命中 key 分类统计

示例：

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/classification/pointtransformer_v1_modelnet40.yaml \
    model.pointtransformer.pretrained True \
    model.pointtransformer.pretrained_source official \
    model.pointtransformer.pretrained_path path/to/official_checkpoint.pth \
    model.pointtransformer.reset_classifier True
```

## Point-cloud checkpoint diagnose 脚本

可使用独立脚本直接检查不同点云模型族的 checkpoint key 映射结果，而不必先跑训练或建模加载流程：

- 脚本：`tools/diagnose_point_cloud_checkpoint.py`
- 兼容入口：`tools/diagnose_pointtransformer_checkpoint.py`

支持参数：
- `--config`
- `--checkpoint`
- `--version {v1,v2,v3}`：PointTransformer 可选
- `--task {classification,segmentation}`：DGCNN 可选
- `--pretrained-source`
- `--reset-classifier`
- `--no-reset-classifier`
- `--show-keys-limit`
- `--json`

示例：

```bash
python tools/diagnose_point_cloud_checkpoint.py \
    --config configs/point_cloud/classification/dgcnn_cls_modelnet40.yaml \
    --checkpoint path/to/checkpoint.pth
```

```bash
python tools/diagnose_point_cloud_checkpoint.py \
    --config configs/point_cloud/segmentation/pvcnn_seg_shapenetpart.yaml \
    --checkpoint path/to/checkpoint.pth \
    --json
```

输出内容包括：
- 文本模式：
  - basic info
  - mapping summary
  - unmatched category counts
  - renamed pairs
  - dropped classifier keys
  - unmatched input keys
  - 各类未命中 key 明细
- JSON 模式：
  - `basic_info`
  - `mapping_summary`
  - `renamed_pairs`
  - `dropped_classifier_keys`
  - `mapped_keys`

## 评估

```bash
python scripts/point_cloud/evaluate.py --config configs/point_cloud/classification/pointnet_cls_modelnet40.yaml
```

## 导出 ONNX

```bash
python scripts/point_cloud/export.py --config configs/point_cloud/classification/pointnet_cls_modelnet40.yaml \
    export.checkpoint path/to/checkpoint.pth \
    export.output_file outputs/pointnet_cls.onnx
```

## 量化与剪枝说明

当前仓库对点云任务的压缩能力是**部分支持**，不是全模型族全覆盖：

- 已支持：`PointNet classification + ModelNet40`
  - QAT 训练
  - quantized ONNX 导出
  - 点云剪枝脚本（builtin pruning / torch-pruning 最小闭环）
- 已支持：`DGCNN classification + ModelNet40`
  - QAT 训练
  - quantized ONNX 导出
- 已支持：`PointNet2 classification + ModelNet40`
  - QAT 训练
  - quantized ONNX 导出
- 已支持：`PointNet2 segmentation + ShapeNetPart`
  - QAT 训练
  - quantized ONNX 导出
- 当前 segmentation 量化 ONNX 现状：
  - `PointNet segmentation`：导出图中可见显式量化算子
  - `DGCNN segmentation`：导出图中可见显式量化算子
  - `PointNet2 segmentation`：导出图中可见显式量化算子
- 暂未纳入仓内闭环：KPConv、PVCNN、PointVoxel、PointTransformer 的量化/剪枝

### PointNet classification QAT 训练

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/classification/pointnet_cls_modelnet40.yaml \
    device cpu \
    qat.enabled True
```

### PointNet classification quantized ONNX 导出

```bash
python scripts/point_cloud/export.py --config configs/point_cloud/classification/pointnet_cls_modelnet40.yaml \
    device cpu \
    export.checkpoint path/to/checkpoint.pth \
    export.output_file outputs/pointnet_cls_quant.onnx \
    export.quantized_onnx True \
    qat.enabled True
```

### DGCNN classification QAT 训练

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/classification/dgcnn_cls_modelnet40.yaml \
    device cpu \
    qat.enabled True
```

### DGCNN classification quantized ONNX 导出

```bash
python scripts/point_cloud/export.py --config configs/point_cloud/classification/dgcnn_cls_modelnet40.yaml \
    device cpu \
    export.checkpoint path/to/checkpoint.pth \
    export.output_file outputs/dgcnn_cls_quant.onnx \
    export.quantized_onnx True \
    qat.enabled True
```

### PointNet segmentation QAT 训练

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/segmentation/pointnet_seg_shapenetpart.yaml \
    device cpu \
    qat.enabled True
```

### PointNet segmentation quantized ONNX 导出

```bash
python scripts/point_cloud/export.py --config configs/point_cloud/segmentation/pointnet_seg_shapenetpart.yaml \
    device cpu \
    export.checkpoint path/to/checkpoint.pth \
    export.output_file outputs/pointnet_seg_quant.onnx \
    export.quantized_onnx True \
    qat.enabled True
```

### DGCNN segmentation QAT 训练

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/segmentation/dgcnn_seg_shapenetpart.yaml \
    device cpu \
    qat.enabled True
```

### DGCNN segmentation quantized ONNX 导出

```bash
python scripts/point_cloud/export.py --config configs/point_cloud/segmentation/dgcnn_seg_shapenetpart.yaml \
    device cpu \
    export.checkpoint path/to/checkpoint.pth \
    export.output_file outputs/dgcnn_seg_quant.onnx \
    export.quantized_onnx True \
    qat.enabled True
```

### PointNet2 classification QAT 训练

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/classification/pointnet2_cls_modelnet40.yaml \
    device cpu \
    qat.enabled True
```

### PointNet2 classification quantized ONNX 导出

```bash
python scripts/point_cloud/export.py --config configs/point_cloud/classification/pointnet2_cls_modelnet40.yaml \
    device cpu \
    export.checkpoint path/to/checkpoint.pth \
    export.output_file outputs/pointnet2_cls_quant.onnx \
    export.quantized_onnx True \
    qat.enabled True
```

### PointNet2 segmentation QAT 训练

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/segmentation/pointnet2_seg_shapenetpart.yaml \
    device cpu \
    qat.enabled True
```

### PointNet2 segmentation quantized ONNX 导出

```bash
python scripts/point_cloud/export.py --config configs/point_cloud/segmentation/pointnet2_seg_shapenetpart.yaml \
    device cpu \
    export.checkpoint path/to/checkpoint.pth \
    export.output_file outputs/pointnet2_seg_quant.onnx \
    export.quantized_onnx True \
    qat.enabled True
```

### PointNet classification 剪枝

builtin pruning 示例：

```bash
python tools/prune_point_cloud.py --config configs/point_cloud/classification/pointnet_cls_modelnet40.yaml \
    device cpu \
    prune.checkpoint path/to/checkpoint.pth \
    prune.output_dir outputs/point_cloud_prune \
    prune.save_name pointnet_global_pruned.pth \
    prune.amount 0.2
```

torch-pruning 示例：

```bash
python tools/prune_point_cloud.py --config configs/point_cloud/classification/pointnet_cls_modelnet40.yaml \
    device cpu \
    dataset.num_points 32 \
    prune.checkpoint path/to/checkpoint.pth \
    prune.output_dir outputs/point_cloud_prune \
    prune.save_name pointnet_tp_pruned.pth \
    prune.backend torch_pruning \
    prune.method tp_magnitude \
    prune.target classifier \
    prune.modules [linear] \
    prune.example_batch_size 1 \
    prune.example_num_points 32 \
    prune.amount 0.2
```

### 当前点云 pruning 支持范围

- builtin pruning：
  - `PointNet classification`
  - `PointNet2 classification`
  - `DGCNN classification`
- torch-pruning：
  - `PointNet classification`
  - `DGCNN classification`
- 当前已支持仓内“重建后加载 + 前向验证”的 torch-pruning checkpoint：
  - `PointNet classification`
  - `DGCNN classification`
- 对于上述 rebuilt pruned checkpoint，当前可直接用于正式入口：
  - `scripts/point_cloud/train.py`（通过 `train.checkpoint` 作为 finetune / 初始化权重入口）
  - `scripts/point_cloud/evaluate.py`
  - `scripts/point_cloud/infer.py`
  - `scripts/point_cloud/export.py`（普通 ONNX 导出路径）
- `train.resume` 与 `train.checkpoint` 语义不同：
  - `train.resume=True` 用于恢复完整训练状态（optimizer / scheduler / epoch）
  - rebuilt pruned checkpoint 更适合作为 `train.checkpoint=path/to/pruned.pth` 继续训练/finetune，而不是替代 `train.resume`
- 当前 torch-pruning 的 classifier pruning 只裁中间 hidden layers，保留最终分类输出层维度不变（与 `dataset.n_classes` 一致）
- 暂未纳入当前 torch-pruning 白名单：
  - `PointBERT`
  - `PointMAE`
  - `PointTransformer`
  - 点云 segmentation 模型

```bash
python scripts/point_cloud/train.py --config configs/point_cloud/classification/pointnet_cls_modelnet40.yaml \
    device cpu \
    train.checkpoint path/to/pruned.pth \
    train.output_dir outputs/pointnet_pruned_finetune
```

PointNet2 / DGCNN 也可沿用相同脚本：

```bash
python tools/prune_point_cloud.py --config configs/point_cloud/classification/pointnet2_cls_modelnet40.yaml \
    device cpu \
    prune.checkpoint path/to/checkpoint.pth \
    prune.output_dir outputs/point_cloud_prune \
    prune.save_name pointnet2_global_pruned.pth \
    prune.amount 0.2
```

```bash
python tools/prune_point_cloud.py --config configs/point_cloud/classification/dgcnn_cls_modelnet40.yaml \
    device cpu \
    prune.checkpoint path/to/checkpoint.pth \
    prune.output_dir outputs/point_cloud_prune \
    prune.save_name dgcnn_global_pruned.pth \
    prune.amount 0.2
```

DGCNN torch-pruning classifier 示例：

```bash
python tools/prune_point_cloud.py --config configs/point_cloud/classification/dgcnn_cls_modelnet40.yaml \
    device cpu \
    dataset.num_points 32 \
    prune.checkpoint path/to/checkpoint.pth \
    prune.output_dir outputs/point_cloud_prune \
    prune.save_name dgcnn_tp_pruned.pth \
    prune.backend torch_pruning \
    prune.method tp_magnitude \
    prune.target classifier \
    prune.modules [linear] \
    prune.example_batch_size 1 \
    prune.example_num_points 32 \
    prune.amount 0.2
```

## 相关脚本

- `scripts/point_cloud/train.py`: 训练入口
- `scripts/point_cloud/evaluate.py`: 评估入口
- `scripts/point_cloud/infer.py`: 推理入口
- `scripts/point_cloud/export.py`: 导出 ONNX
- `tools/diagnose_point_cloud_checkpoint.py`: checkpoint 映射诊断
- `tools/smoke_test_all_point_cloud_models.py`: 批量 smoke test
