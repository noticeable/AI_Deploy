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

SUPPORTED_DETECTION_QAT_MODELS = {
    ('yolo', 'yolov8_n'),
}
SUPPORTED_TINYYOLO_QAT_BLOCKS = {'conv'}


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
        self.n_classes = base_model.n_classes
        self.assigner = base_model.assigner
        self.quant = quantization.QuantStub()
        self.backbone = nn.Sequential(*[
            QuantizableConvBN(block.conv, block.bn)
            for block in base_model.backbone
        ])
        self.quantized_backbone = QuantizedDetectionBackbone(self.quant,
                                                             self.backbone,
                                                             quantization.DeQuantStub())
        self.float_pool = nn.AdaptiveAvgPool2d(1)
        self.box_head = copy.deepcopy(base_model.box_head)
        self.score_head = copy.deepcopy(base_model.score_head)
        self._quantized_inference_ready = False

    def fuse_model(self):
        for module in self.backbone.modules():
            if hasattr(module, 'fuse_model'):
                module.fuse_model()

    def _run_heads(self, x, image_size, targets=None, return_outputs=False):
        x = self.float_pool(x)
        raw_pred_boxes, pred_logits, objectness = self._reshape_dense_predictions(x)
        pred_boxes = self._decode_dense_boxes(raw_pred_boxes)
        if pred_logits.shape[1] > self.num_candidates:
            pred_logits = pred_logits[:, :self.num_candidates]
            objectness = objectness[:, :self.num_candidates]
        detections = self._build_detections(pred_boxes, pred_logits, objectness, image_size)
        if self.training and targets is not None:
            assignments = self.assigner.assign(pred_boxes, pred_logits, targets)
            loss_box = pred_boxes.sum() * 0.0
            loss_cls = pred_logits.sum() * 0.0
            use_label_smoothing = getattr(self.assigner.config.augmentation, 'use_label_smoothing', False)
            label_smoothing = getattr(getattr(self.assigner.config.augmentation, 'label_smoothing', None),
                                      'epsilon',
                                      0.0) if use_label_smoothing else 0.0
            for sample_boxes, sample_logits, assignment in zip(pred_boxes, pred_logits, assignments):
                if assignment['positive_mask'].any():
                    positive_indices = assignment['matched_pred_indices']
                    target_boxes = assignment['target_boxes'][positive_indices]
                    loss_box = loss_box + F.l1_loss(sample_boxes[positive_indices],
                                                    target_boxes,
                                                    reduction='sum')
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
        x = self.backbone(images)
        return self._run_heads(x, images.shape[-1], targets=targets, return_outputs=return_outputs)

    def forward_quantized_infer(self, images, targets=None, return_outputs=False):
        x = self.quantized_backbone(images)
        return self._run_heads(x, images.shape[-1], targets=targets, return_outputs=return_outputs)

    def forward(self, images, targets=None, return_outputs=False):
        if getattr(self, '_quantized_inference_ready', False):
            return self.forward_quantized_infer(images, targets=targets, return_outputs=return_outputs)
        return self.forward_train_qat(images, targets=targets, return_outputs=return_outputs)


def _configure_backbone_only_qat(model, qconfig, excluded_modules):
    model.quantized_backbone.quant.qconfig = qconfig
    for module in model.quantized_backbone.backbone.modules():
        module.qconfig = qconfig
    for module in excluded_modules:
        module.qconfig = None


def is_qat_supported_model(config):
    return (config.model.meta_architecture, config.model.name) in SUPPORTED_DETECTION_QAT_MODELS


def _validate_tinyyolo_qat_block(config):
    block_name = str(getattr(config.model.yolo, 'block', 'conv')).lower()
    if block_name not in SUPPORTED_TINYYOLO_QAT_BLOCKS:
        raise ValueError(
            f'Detection QAT currently supports TinyYOLO block types {sorted(SUPPORTED_TINYYOLO_QAT_BLOCKS)}, got {block_name}')


def convert_model_to_qat_compatible(config, model):
    if not is_qat_supported_model(config):
        raise ValueError(
            f'QAT is not supported for detection model family {(config.model.meta_architecture, config.model.name)}')
    _validate_tinyyolo_qat_block(config)
    if getattr(model, '_is_qat_compatible', False):
        return model
    model_copy = copy.deepcopy(model)
    qat_model = QuantizableTinyYOLO(model_copy)
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
                                     model.quant,
                                     model.backbone,
                                     model.quantized_backbone.dequant,
                                     model.float_pool,
                                     model.box_head,
                                     model.score_head,
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
