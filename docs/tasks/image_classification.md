# Image Classification

本项目的图像分类主线仍然使用根目录入口脚本与 `pytorch_image_classification/` 包。

## 常用入口

- 训练：`python train.py --config configs/cifar/resnet_preact.yaml`
- 评估：`python evaluate.py --config configs/cifar/resnet.yaml test.checkpoint path/to/checkpoint.pth`
- 导出 ONNX：`python export.py --config configs/cifar/resnet.yaml export.checkpoint path/to/checkpoint.pth export.output_file outputs/resnet.onnx`

## 配置目录

- `configs/cifar/`: CIFAR 系列配置
- `configs/imagenet/`: ImageNet 与 ViT/Transformer 系列配置
- `configs/datasets/`: 数据集基础配置
- `configs/augmentations/`: 增强策略配置

## 模型范围

- CNN 系列：ResNet、WRN、DenseNet、PyramidNet、ResNeXt、VGG
- Transformer 系列：ViT、DeiT、Swin Transformer、PVT、CvT、ConViT、RetNet 等

## 量化说明

- 当前仓库不内置通用 PyTorch PTQ 流程。
- 若部署目标是 Jetson Orin Nano，推荐使用 TensorRT 作为量化与部署工具链。
- 建议流程：先用 `export.py` 导出 ONNX，再用 TensorRT 构建 FP16 或 INT8 engine。

当前 classification 侧已验证的最小闭环：
- `QAT smoke`：已覆盖 `CIFAR ResNet` 与 `CIFAR ViT`
- `quantized ONNX smoke`：已覆盖 `CIFAR ResNet` 与 `CIFAR ViT`
- `pruning smoke`：classification builtin pruning 已覆盖 `CIFAR ResNet` 与 `CIFAR ViT`

当前支持边界说明：
- 上述 QAT / quantized ONNX / pruning 闭环当前优先保证 `CIFAR ViT`
- `ImageNet ViT`、`ImageNet DeiT`、`T2T-ViT`、`ConViT`、`PVT`、`MViT`、`CvT`、`Swin Transformer` 已验证最小 `float ONNX export + ONNX Runtime` 闭环
- 但这批 ImageNet / Transformer 变体目前**仅确认 float ONNX 导出可用**，并不表示已具备与 `CIFAR ViT` 相同的 QAT / quantized ONNX / pruning smoke 覆盖
- 若要继续扩到 quantized ONNX 或 QAT，建议逐模型补 smoke 后再更新支持矩阵

Jetson 环境建议：
- 优先使用 JetPack 自带 TensorRT
- 若使用 Python 脚本构建 engine，需确保环境可导入 `tensorrt`、`pycuda`、`Pillow`
- 可先运行 `python tools/check_tensorrt_env.py` 检查 TensorRT / CUDA / `trtexec` / Python 依赖是否齐全

分类 TensorRT 脚本：
- 通用 engine 构建脚本：`tools/build_tensorrt_engine.py`
- 分类一键导出+构建脚本：`tools/build_tensorrt_classification.py`

FP16 示例：
- `python tools/build_tensorrt_classification.py --config configs/cifar/resnet.yaml --checkpoint path/to/checkpoint.pth --onnx outputs/resnet.onnx --engine outputs/resnet_fp16.engine --precision fp16`

INT8 示例：
- `python tools/build_tensorrt_classification.py --config configs/cifar/resnet.yaml --checkpoint path/to/checkpoint.pth --onnx outputs/resnet.onnx --engine outputs/resnet_int8.engine --precision int8 --calib-dir /path/to/calibration_images --calib-cache outputs/resnet_int8.cache --batch-size 8`

QAT / quantized ONNX / pruning 示例：
- QAT 训练：`python train.py --config configs/classification/presets/cifar10/vit.yaml device cpu dataset.name FakeData dataset.image_size 32 dataset.n_channels 3 dataset.n_classes 10 qat.enabled True`
- quantized ONNX 导出：`python export.py --config configs/classification/presets/cifar10/vit.yaml export.checkpoint path/to/checkpoint.pth export.output_file outputs/quantized_vit.onnx export.quantized_onnx True export.quantized_onnx_backend onnxruntime_dynamic qat.enabled True`
- classification pruning：`python tools/prune_classification.py --config configs/classification/presets/cifar10/vit.yaml prune.checkpoint path/to/checkpoint.pth prune.backend builtin prune.method global_unstructured prune.modules "('linear',)" prune.output_dir outputs/pruned prune.save_name vit_pruned.pth`

说明：
- INT8 校准脚本当前按“图片目录”读取校准样本，适合分类模型
- 默认假设 ONNX 输入名为 `images`，布局为 `CHW`
- 如果 ONNX 含动态输入 shape，请额外传 `--min-shape / --opt-shape / --max-shape`

## 相关代码

- `train.py`: 分类训练入口
- `evaluate.py`: 分类评估入口
- `export.py`: 分类 ONNX 导出入口
- `pytorch_image_classification/`: 分类模型、优化器、调度器、数据与工具函数
