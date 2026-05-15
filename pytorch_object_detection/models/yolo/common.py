import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_object_detection.losses import compute_box_loss, create_assigner
from pytorch_object_detection.models.postprocess import postprocess_detections


def autopad(kernel_size, padding=None, dilation=1):
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1 if isinstance(kernel_size, int) else [
            dilation * (value - 1) + 1 for value in kernel_size
        ]
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [value // 2 for value in kernel_size]
    return padding


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = autopad(kernel_size)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)
        self.block = nn.Sequential(self.conv, self.bn, self.act)

    def forward(self, x):
        return self.block(x)

    def fuse_model(self):
        torch.ao.quantization.fuse_modules(self, [['conv', 'bn', 'act']], inplace=True)


class DepthwiseConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.pointwise = ConvBNAct(in_channels, out_channels, 1, 1)
        self.depthwise = ConvBNAct(out_channels, out_channels, kernel_size, stride)
        self.depthwise.conv = nn.Conv2d(out_channels,
                                        out_channels,
                                        kernel_size,
                                        stride=stride,
                                        padding=autopad(kernel_size),
                                        groups=out_channels,
                                        bias=False)
        self.depthwise.block = nn.Sequential(self.depthwise.conv, self.depthwise.bn, self.depthwise.act)

    def forward(self, x):
        return self.depthwise(self.pointwise(x))


class GhostConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, ratio=2, dw_size=3):
        super().__init__()
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        self.out_channels = out_channels
        self.primary_conv = ConvBNAct(in_channels, init_channels, kernel_size, stride)
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_channels,
                      new_channels,
                      dw_size,
                      stride=1,
                      padding=autopad(dw_size),
                      groups=init_channels,
                      bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        cheap = self.cheap_conv(primary)
        return torch.cat([primary, cheap], dim=1)[:, :self.out_channels]


class PartialConv3(nn.Module):
    def __init__(self, channels, kernel_size, n_div=4):
        super().__init__()
        self.partial_channels = channels // n_div
        self.remaining_channels = channels - self.partial_channels
        self.partial_conv = nn.Conv2d(self.partial_channels,
                                      self.partial_channels,
                                      kernel_size,
                                      stride=1,
                                      padding=autopad(kernel_size),
                                      bias=False)

    def forward(self, x):
        x_partial, x_remaining = torch.split(x, [self.partial_channels, self.remaining_channels], dim=1)
        x_partial = self.partial_conv(x_partial)
        return torch.cat((x_partial, x_remaining), dim=1)


class PConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.project = ConvBNAct(in_channels, out_channels, 1, stride)
        self.partial = PartialConv3(out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.project(x)
        x = self.partial(x)
        x = self.bn(x)
        return self.act(x)


class GSConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        hidden_channels = out_channels // 2
        self.primary_conv = ConvBNAct(in_channels, hidden_channels, kernel_size, stride)
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(hidden_channels,
                      hidden_channels,
                      5,
                      stride=1,
                      padding=autopad(5),
                      groups=hidden_channels,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        merged = torch.cat((primary, self.cheap_conv(primary)), dim=1)
        batch_size, channels, height, width = merged.shape
        merged = merged.reshape(batch_size * channels // 2, 2, height * width)
        merged = merged.permute(1, 0, 2)
        merged = merged.reshape(2, -1, channels // 2, height, width)
        return torch.cat((merged[0], merged[1]), dim=1)


class DSConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size,
                      stride=stride,
                      padding=autopad(kernel_size),
                      groups=in_channels,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
        )
        self.pointwise = ConvBNAct(in_channels, out_channels, 1, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def create_yolo_block(block_name, in_channels, out_channels, kernel_size, stride):
    block_name = str(block_name).lower()
    if block_name == 'conv':
        return ConvBNAct(in_channels, out_channels, kernel_size, stride)
    if block_name == 'dconv':
        return DepthwiseConvBNAct(in_channels, out_channels, kernel_size, stride)
    if block_name == 'ghost':
        return GhostConvBNAct(in_channels, out_channels, kernel_size, stride)
    if block_name == 'gsconv':
        return GSConvBNAct(in_channels, out_channels, kernel_size, stride)
    if block_name == 'pconv':
        return PConvBNAct(in_channels, out_channels, kernel_size, stride)
    if block_name == 'dsconv':
        return DSConvBNAct(in_channels, out_channels, kernel_size, stride)
    raise ValueError(f'Unsupported config.model.yolo.block: {block_name}')


class TinyYOLO(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = list(getattr(config.model.yolo, 'channels', []))
        if channels:
            if len(channels) != 4:
                raise ValueError(f'config.model.yolo.channels must contain 4 entries, got {channels}')
        else:
            base_channels = max(16, int(64 * config.model.yolo.width_mult))
            channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.config = config
        self.n_classes = config.dataset.n_classes
        self.assigner = create_assigner(config, 'yolo')
        block_name = getattr(config.model.yolo, 'block', 'conv')
        kernel_size = getattr(config.model.yolo, 'block_kernel_size', 3)
        self.backbone = nn.Sequential(
            create_yolo_block(block_name, config.dataset.n_channels, channels[0], kernel_size, 2),
            create_yolo_block(block_name, channels[0], channels[1], kernel_size, 2),
            create_yolo_block(block_name, channels[1], channels[2], kernel_size, 2),
            create_yolo_block(block_name, channels[2], channels[3], kernel_size, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.box_head = nn.Linear(channels[-1], 4)
        self.score_head = nn.Linear(channels[-1], config.dataset.n_classes + 1)

    def _sync_heads_with_backbone(self):
        last_block = self.backbone[-1]
        if hasattr(last_block, 'block'):
            last_channels = last_block.block[0].out_channels
        elif hasattr(last_block, 'primary_conv'):
            last_channels = last_block.primary_conv.block[0].out_channels * 2
        elif hasattr(last_block, 'project'):
            last_channels = last_block.project.block[0].out_channels
        elif hasattr(last_block, 'pointwise'):
            last_channels = last_block.pointwise.block[0].out_channels
        else:
            raise ValueError('Unable to infer TinyYOLO backbone output channels.')
        if self.box_head.in_features == last_channels and self.score_head.in_features == last_channels:
            return
        box_head = nn.Linear(last_channels, self.box_head.out_features)
        score_head = nn.Linear(last_channels, self.score_head.out_features)
        box_in = min(last_channels, self.box_head.in_features)
        score_in = min(last_channels, self.score_head.in_features)
        with torch.no_grad():
            box_head.weight.zero_()
            box_head.weight[:, :box_in] = self.box_head.weight[:, :box_in]
            box_head.bias.copy_(self.box_head.bias)
            score_head.weight.zero_()
            score_head.weight[:, :score_in] = self.score_head.weight[:, :score_in]
            score_head.bias.copy_(self.score_head.bias)
        box_head = box_head.to(device=self.box_head.weight.device, dtype=self.box_head.weight.dtype)
        score_head = score_head.to(device=self.score_head.weight.device, dtype=self.score_head.weight.dtype)
        self.box_head = box_head
        self.score_head = score_head

    def forward(self, images, targets=None, return_outputs=False):
        self._sync_heads_with_backbone()
        x = self.backbone(images)
        x = self.pool(x).flatten(1)
        pred_boxes = torch.sigmoid(self.box_head(x)).unsqueeze(1)
        pred_logits = self.score_head(x).unsqueeze(1)
        scores = torch.softmax(pred_logits.squeeze(1), dim=-1)
        detections = []
        image_size = images.shape[-1]
        nms_type = getattr(self.config.eval, 'nms_type', 'hard')
        conf_threshold = getattr(self.config.eval, 'conf_threshold', 0.25)
        nms_threshold = getattr(self.config.eval, 'nms_threshold', 0.45)
        max_detections = getattr(self.config.eval, 'max_detections', 300)
        soft_nms_sigma = getattr(self.config.eval, 'soft_nms_sigma', 0.5)
        soft_nms_score_threshold = getattr(self.config.eval, 'soft_nms_score_threshold', 1e-3)
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
            boxes, det_scores, labels = postprocess_detections(
                xyxy.unsqueeze(0),
                cls_score.unsqueeze(0),
                cls_label.unsqueeze(0),
                nms_type=nms_type,
                score_threshold=conf_threshold,
                iou_threshold=nms_threshold,
                max_detections=max_detections,
                soft_nms_sigma=soft_nms_sigma,
                soft_nms_score_threshold=soft_nms_score_threshold,
            )
            detections.append({
                'boxes': boxes,
                'scores': det_scores,
                'labels': labels,
            })
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
            if return_outputs:
                return {
                    'losses': losses,
                    'pred_boxes': pred_boxes,
                    'pred_logits': pred_logits,
                }
            return losses
        if return_outputs:
            return {
                'detections': detections,
                'pred_boxes': pred_boxes,
                'pred_logits': pred_logits,
            }
        return detections
