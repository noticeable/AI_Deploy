import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as quantization

from pytorch_image_classification.models.qat_common import (
    QATStateController,
    apply_qat_epoch_controls,
    create_default_qat_qconfig,
    ensure_cpu_float_tensor,
    is_qat_enabled,
    quantize_input_tensor,
)
from pytorch_object_detection.losses import compute_box_loss

SUPPORTED_DETECTION_QAT_MODELS = {
    ('yolo', 'yolov8_n'),
}
SUPPORTED_TINYYOLO_QAT_BLOCKS = {'conv'}
SUPPORTED_TINYYOLO_DENSE_HEAD_TYPES = {'fpn_multi'}


class QuantizableConvBN(nn.Module):
    def __init__(self, conv, bn):
        super().__init__()
        self.conv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              dilation=conv.dilation,
                              groups=conv.groups,
                              bias=False,
                              padding_mode=conv.padding_mode)
        self.bn = nn.BatchNorm2d(bn.num_features,
                                 eps=bn.eps,
                                 momentum=bn.momentum,
                                 affine=bn.affine,
                                 track_running_stats=bn.track_running_stats)
        self.relu = nn.ReLU(inplace=False)
        self.conv.load_state_dict(conv.state_dict(), strict=True)
        self.bn.load_state_dict(bn.state_dict(), strict=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x)

    def fuse_model(self):
        quantization.fuse_modules(self, [['conv', 'bn']], inplace=True)


class QuantizedDetectionBackbone(nn.Module):
    def __init__(self, quant, backbone, dequant):
        super().__init__()
        self.quant = quant
        self.backbone = backbone
        self.dequant = dequant

    def forward(self, images):
        if getattr(self, '_manual_quantized_inference', False):
            x = quantize_input_tensor(images)
        else:
            x = self.quant(images)
        x = self.backbone(x)
        if getattr(self, '_manual_quantized_inference', False):
            x = x.dequantize()
        else:
            x = self.dequant(x)
        x = ensure_cpu_float_tensor(x)
        return x


class QuantizableTinyYOLO(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config
        self.n_classes = base_model.n_classes
        self.assigner = base_model.assigner
        self.grid_sizes = list(getattr(base_model, 'grid_sizes', []))
        self.per_cell_boxes = base_model.per_cell_boxes
        self.num_candidates = base_model.num_candidates
        self.min_box_size = base_model.min_box_size
        self.use_objectness = base_model.use_objectness
        self.strides = list(getattr(base_model, 'strides', [8, 16, 32]))
        self.quant = quantization.QuantStub()
        self.stem = copy.deepcopy(base_model.stem)
        self.stage2 = copy.deepcopy(base_model.stage2)
        self.stage3 = copy.deepcopy(base_model.stage3)
        self.stage4 = copy.deepcopy(base_model.stage4)
        self.quantized_backbone = QuantizedDetectionBackbone(
            self.quant,
            nn.Sequential(self.stem, self.stage2, self.stage3, self.stage4),
            quantization.DeQuantStub(),
        )
        self.pool = None
        self.dense_neck = copy.deepcopy(base_model.dense_neck)
        self.neck = self.dense_neck
        self.box_head = self.neck.heads[-1].box_head
        self.score_head = self.neck.heads[-1].score_head
        self.objectness_head = self.neck.heads[-1].objectness_head
        self._quantized_inference_ready = False

    _reshape_dense_predictions = staticmethod(lambda *args, **kwargs: None)
    _decode_dense_boxes = staticmethod(lambda *args, **kwargs: None)
    _build_detections = staticmethod(lambda *args, **kwargs: None)
    _forward_backbone = staticmethod(lambda *args, **kwargs: None)

    def fuse_model(self):
        for module in self.modules():
            if hasattr(module, 'fuse_model'):
                module.fuse_model()

    def _run_heads(self, features, image_size, targets=None, return_outputs=False):
        prediction_maps = self.neck(features)
        raw_pred_boxes, pred_logits, objectness = self._reshape_dense_predictions(prediction_maps)
        pred_boxes = self._decode_dense_boxes(raw_pred_boxes)
        if pred_logits.shape[1] > self.num_candidates:
            pred_logits = pred_logits[:, :self.num_candidates]
            objectness = objectness[:, :self.num_candidates]
        detections = self._build_detections(pred_boxes, pred_logits, objectness, image_size)
        if self.training and targets is not None:
            assignments = self.assigner.assign(pred_boxes, pred_logits, targets)
            loss_box = pred_boxes.sum() * 0.0
            loss_cls = pred_logits.sum() * 0.0
            box_loss_type = getattr(self.config.assignment, 'box_loss', 'l1')
            iou_variant = getattr(self.config.assignment, 'iou_variant', 'iou')
            box_loss_weight = getattr(self.config.assignment, 'box_loss_weight', 1.0)
            use_label_smoothing = getattr(self.config.augmentation, 'use_label_smoothing', False)
            label_smoothing = getattr(getattr(self.config.augmentation, 'label_smoothing', None),
                                      'epsilon',
                                      0.0) if use_label_smoothing else 0.0
            for sample_boxes, sample_logits, assignment in zip(pred_boxes, pred_logits, assignments):
                if assignment['positive_mask'].any():
                    positive_indices = assignment['matched_pred_indices']
                    target_boxes = assignment['target_boxes'][positive_indices]
                    loss_box = loss_box + box_loss_weight * compute_box_loss(
                        sample_boxes[positive_indices],
                        target_boxes,
                        loss_type=box_loss_type,
                        iou_variant=iou_variant,
                    )
                loss_cls = loss_cls + F.cross_entropy(sample_logits,
                                                      assignment['target_labels'],
                                                      reduction='sum',
                                                      label_smoothing=label_smoothing)
            batch_size = max(len(assignments), 1)
            losses = {
                'loss_box': loss_box / batch_size,
                'loss_cls': loss_cls / batch_size,
            }
            if self.objectness_head is not None:
                losses['loss_obj'] = objectness.mean() * 0.0
            if return_outputs:
                return {
                    'losses': losses,
                    'pred_boxes': pred_boxes,
                    'pred_logits': pred_logits,
                    'objectness': objectness,
                    'detections': detections,
                }
            return losses
        if return_outputs:
            return {
                'detections': detections,
                'pred_boxes': pred_boxes,
                'pred_logits': pred_logits,
                'objectness': objectness,
            }
        return detections

    def forward_train_qat(self, images, targets=None, return_outputs=False):
        features = self._forward_backbone(images)
        return self._run_heads(features, images.shape[-1], targets=targets, return_outputs=return_outputs)

    def forward_quantized_infer(self, images, targets=None, return_outputs=False):
        if getattr(self, '_manual_quantized_inference', False):
            x = quantize_input_tensor(images)
        else:
            x = self.quant(images)
        x1 = self.stem(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        if getattr(self, '_manual_quantized_inference', False):
            x2 = x2.dequantize()
            x3 = x3.dequantize()
            x4 = x4.dequantize()
        else:
            dequant = self.quantized_backbone.dequant
            x2 = dequant(x2)
            x3 = dequant(x3)
            x4 = dequant(x4)
        features = tuple(ensure_cpu_float_tensor(x) for x in (x2, x3, x4))
        return self._run_heads(features, images.shape[-1], targets=targets, return_outputs=return_outputs)

    def forward(self, images, targets=None, return_outputs=False):
        if getattr(self, '_quantized_inference_ready', False):
            return self.forward_quantized_infer(images, targets=targets, return_outputs=return_outputs)
        return self.forward_train_qat(images, targets=targets, return_outputs=return_outputs)


def _configure_backbone_only_qat(model, qconfig, excluded_modules):
    model.quantized_backbone.quant.qconfig = qconfig
    backbone_modules = [model.stem, model.stage2, model.stage3, model.stage4]
    for backbone in backbone_modules:
        for module in backbone.modules():
            module.qconfig = qconfig
    for module in excluded_modules:
        if module is not None:
            module.qconfig = None


def is_qat_supported_model(config):
    return (config.model.meta_architecture, config.model.name) in SUPPORTED_DETECTION_QAT_MODELS


def _validate_tinyyolo_qat_block(config):
    block_name = str(getattr(config.model.yolo, 'block', 'conv')).lower()
    if block_name not in SUPPORTED_TINYYOLO_QAT_BLOCKS:
        raise ValueError(
            f'Detection QAT currently supports TinyYOLO block types {sorted(SUPPORTED_TINYYOLO_QAT_BLOCKS)}, got {block_name}')


def _validate_tinyyolo_qat_dense_head(config):
    dense_head_cfg = getattr(config.model.yolo, 'dense_head', None)
    dense_head_type = str(getattr(dense_head_cfg, 'type', 'fpn_multi')).lower()
    if dense_head_type not in SUPPORTED_TINYYOLO_DENSE_HEAD_TYPES:
        raise ValueError(
            'Detection QAT currently supports full TinyYOLO dense-head types '
            f'{sorted(SUPPORTED_TINYYOLO_DENSE_HEAD_TYPES)}, got {dense_head_type}')


def convert_model_to_qat_compatible(config, model):
    if not is_qat_supported_model(config):
        raise ValueError(
            f'QAT is not supported for detection model family {(config.model.meta_architecture, config.model.name)}')
    _validate_tinyyolo_qat_block(config)
    _validate_tinyyolo_qat_dense_head(config)
    if getattr(model, '_is_qat_compatible', False):
        return model
    model_copy = copy.deepcopy(model)
    qat_model = QuantizableTinyYOLO(model_copy)
    qat_model._reshape_dense_predictions = model_copy._reshape_dense_predictions
    qat_model._decode_dense_boxes = model_copy._decode_dense_boxes
    qat_model._build_detections = model_copy._build_detections
    qat_model._base_forward_from_dense_features = model_copy.forward
    qat_model._is_qat_compatible = True
    return qat_model


def prepare_model_for_qat(config, model):
    model = convert_model_to_qat_compatible(config, model)
    model.eval()
    model.fuse_model()
    model.train()
    qconfig = create_default_qat_qconfig(config)
    _configure_backbone_only_qat(model,
                                 qconfig,
                                 excluded_modules=[
                                     model.quantized_backbone.dequant,
                                 ])
    quantization.prepare_qat(model.quantized_backbone, inplace=True)
    with torch.no_grad():
        device = next(model.parameters()).device
        dummy = torch.randn(1,
                            int(config.dataset.n_channels),
                            int(config.dataset.image_size),
                            int(config.dataset.image_size),
                            device=device)
        model(dummy)
    return model, QATStateController()


def convert_qat_model(config, model):
    if not is_qat_enabled(config):
        return model
    model_to_convert = copy.deepcopy(model).cpu().eval()
    model_to_convert.quantized_backbone = quantization.convert(model_to_convert.quantized_backbone,
                                                              inplace=True)
    model_to_convert.quantized_backbone._manual_quantized_inference = True
    model_to_convert._quantized_inference_ready = True
    return model_to_convert
