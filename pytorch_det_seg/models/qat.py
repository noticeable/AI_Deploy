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

SUPPORTED_DET_SEG_QAT_MODELS = {
    ('det_seg', 'shared_backbone_tiny'),
}


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


class QuantizedSharedBackbone(nn.Module):
    def __init__(self, quant, backbone, dequant):
        super().__init__()
        self.quant = quant
        self.backbone = backbone
        self.dequant = dequant

    def forward(self, images):
        if getattr(self, '_manual_quantized_inference', False):
            x = quantize_input_tensor(images)
            x = self.backbone(x)
            x = x.dequantize()
            return ensure_cpu_float_tensor(x)

        x = self.quant(images)
        x = self.backbone(x)
        x = self.dequant(x)
        return x


class QuantizableSharedBackboneDetSeg(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.config = copy.deepcopy(base_model.config)
        self.n_classes = base_model.n_classes
        self.quant = quantization.QuantStub()
        self.backbone = nn.Sequential(*[
            QuantizableConvBN(block.conv, block.bn)
            for block in base_model.backbone
        ])
        self.quantized_backbone = QuantizedSharedBackbone(self.quant,
                                                          self.backbone,
                                                          quantization.DeQuantStub())
        self.det_neck = copy.deepcopy(base_model.det_neck)
        self.seg_neck = copy.deepcopy(base_model.seg_neck)
        self.det_head = copy.deepcopy(base_model.det_head)
        self.seg_head = copy.deepcopy(base_model.seg_head)
        self._quantized_inference_ready = False

    def fuse_model(self):
        for module in self.backbone.modules():
            if hasattr(module, 'fuse_model'):
                module.fuse_model()

    def _forward_shared(self, features, image_size, targets=None, return_outputs=False):
        rep_features = features
        det_features = self.det_neck(features)
        seg_features = self.seg_neck(features)
        pred_boxes, pred_logits = self.det_head(det_features)
        seg_logits = self.seg_head(seg_features, image_size)

        detections = []
        scores = torch.softmax(pred_logits.squeeze(1), dim=-1)
        for box, score in zip(pred_boxes.squeeze(1), scores):
            cls_scores = score[:-1]
            if cls_scores.numel() == 0:
                cls_score = score.new_zeros(())
                cls_label = torch.zeros((), dtype=torch.long, device=score.device)
            else:
                cls_score, cls_label = cls_scores.max(dim=0)
            xyxy = box.clone()
            xyxy[2:] = torch.maximum(xyxy[:2] + 0.1, xyxy[2:])
            xyxy = xyxy * image_size
            detections.append({
                'boxes': xyxy.unsqueeze(0),
                'scores': cls_score.unsqueeze(0),
                'labels': cls_label.unsqueeze(0),
            })

        if self.training and targets is not None:
            rep_features.retain_grad()
            loss_box = pred_boxes.sum() * 0.0
            loss_cls = pred_logits.sum() * 0.0
            for sample_boxes, sample_logits, target in zip(pred_boxes, pred_logits, targets):
                target_boxes = target['boxes'].to(sample_boxes.device)
                target_labels = target['labels'].to(sample_logits.device)
                if len(target_boxes) > 0:
                    loss_box = loss_box + F.l1_loss(sample_boxes[0], target_boxes[0], reduction='sum')
                    loss_cls = loss_cls + F.cross_entropy(sample_logits, target_labels[:1], reduction='sum')
                else:
                    background = torch.full((sample_logits.shape[0],), self.n_classes, dtype=torch.long, device=sample_logits.device)
                    loss_cls = loss_cls + F.cross_entropy(sample_logits, background, reduction='sum')
            seg_targets = torch.stack([target['segmentation_mask'].to(seg_logits.device) for target in targets])
            loss_seg = F.cross_entropy(seg_logits, seg_targets, ignore_index=self.config.dataset.ignore_index)
            losses = {
                'loss_box': loss_box / max(len(targets), 1),
                'loss_cls': loss_cls / max(len(targets), 1),
                'loss_seg': loss_seg,
            }
            losses['det_loss'] = losses['loss_box'] + losses['loss_cls']
            losses['seg_loss'] = losses['loss_seg']
            if return_outputs:
                shared_rep = rep_features
                return {
                    'losses': losses,
                    'pred_boxes': pred_boxes,
                    'pred_logits': pred_logits,
                    'seg_logits': seg_logits,
                    'shared_rep': shared_rep,
                    'rep_tasks': {
                        'det_loss': shared_rep,
                        'seg_loss': shared_rep,
                    },
                }
            return losses

        if return_outputs:
            return {
                'detections': detections,
                'pred_boxes': pred_boxes,
                'pred_logits': pred_logits,
                'seg_logits': seg_logits,
            }
        return {
            'detections': detections,
            'segmentation': seg_logits.argmax(dim=1),
        }

    def forward_train_qat(self, images, targets=None, return_outputs=False):
        features = self.quantized_backbone(images)
        return self._forward_shared(features, images.shape[-1], targets=targets, return_outputs=return_outputs)

    def forward_quantized_infer(self, images, targets=None, return_outputs=False):
        features = self.quantized_backbone(images)
        return self._forward_shared(features, images.shape[-1], targets=targets, return_outputs=return_outputs)

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
    return (config.model.meta_architecture, config.model.name) in SUPPORTED_DET_SEG_QAT_MODELS


def convert_model_to_qat_compatible(config, model):
    if not is_qat_supported_model(config):
        raise ValueError(f'QAT is not supported for det-seg model family {(config.model.meta_architecture, config.model.name)}')
    if getattr(model, '_is_qat_compatible', False):
        return model
    model_copy = copy.deepcopy(model)
    qat_model = QuantizableSharedBackboneDetSeg(model_copy)
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
                                     model.det_neck,
                                     model.seg_neck,
                                     model.det_head,
                                     model.seg_head,
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
