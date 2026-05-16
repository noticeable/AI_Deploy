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


class YOLOStage(nn.Module):
    def __init__(self, block_name, kernel_size, in_channels, out_channels, stride, num_blocks=1):
        super().__init__()
        layers = [create_yolo_block(block_name, in_channels, out_channels, kernel_size, stride)]
        for _ in range(max(int(num_blocks), 1) - 1):
            layers.append(create_yolo_block(block_name, out_channels, out_channels, kernel_size, 1))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class YOLOUpsampleMerge(nn.Module):
    def __init__(self, block_name, kernel_size, low_res_channels, skip_channels, out_channels):
        super().__init__()
        self.low_proj = ConvBNAct(low_res_channels, out_channels, 1, 1)
        self.skip_proj = ConvBNAct(skip_channels, out_channels, 1, 1)
        self.fuse = YOLOStage(block_name, kernel_size, out_channels * 2, out_channels, 3, num_blocks=2)

    def forward(self, low_res, skip):
        low_res = self.low_proj(low_res)
        low_res = F.interpolate(low_res, size=skip.shape[-2:], mode='nearest')
        skip = self.skip_proj(skip)
        return self.fuse(torch.cat((low_res, skip), dim=1))


class YOLODownsampleMerge(nn.Module):
    def __init__(self, block_name, kernel_size, high_res_channels, skip_channels, out_channels):
        super().__init__()
        self.down = create_yolo_block(block_name, high_res_channels, out_channels, kernel_size, 2)
        self.skip_proj = ConvBNAct(skip_channels, out_channels, 1, 1)
        self.fuse = YOLOStage(block_name, kernel_size, out_channels * 2, out_channels, 3, num_blocks=2)

    def forward(self, high_res, skip):
        high_res = self.down(high_res)
        skip = self.skip_proj(skip)
        return self.fuse(torch.cat((high_res, skip), dim=1))


class YOLODetectionHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, per_cell_boxes, use_objectness):
        super().__init__()
        hidden_channels = max(int(hidden_channels), int(in_channels))
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, 3, 1),
            ConvBNAct(hidden_channels, hidden_channels, 3, 1),
        )
        self.box_head = nn.Conv2d(hidden_channels, per_cell_boxes * 4, kernel_size=1)
        self.score_head = nn.Conv2d(hidden_channels, per_cell_boxes * (n_classes + 1), kernel_size=1)
        self.objectness_head = nn.Conv2d(hidden_channels, per_cell_boxes, kernel_size=1) if use_objectness else None

    def forward(self, x):
        x = self.stem(x)
        pred_boxes = torch.sigmoid(self.box_head(x))
        pred_logits = self.score_head(x)
        if self.objectness_head is not None:
            objectness = torch.sigmoid(self.objectness_head(x))
        else:
            objectness = None
        return pred_boxes, pred_logits, objectness


class TinyYOLOFPNMultiHead(nn.Module):
    def __init__(self, block_name, kernel_size, channels, head_channels, n_classes, per_cell_boxes, use_objectness):
        super().__init__()
        c3, c4, c5 = channels
        self.p5_reduce = ConvBNAct(c5, head_channels, 1, 1)
        self.p4_fuse = YOLOUpsampleMerge(block_name, kernel_size, head_channels, c4, head_channels)
        self.p3_fuse = YOLOUpsampleMerge(block_name, kernel_size, head_channels, c3, head_channels)
        self.n4_fuse = YOLODownsampleMerge(block_name, kernel_size, head_channels, head_channels, head_channels)
        self.n5_fuse = YOLODownsampleMerge(block_name, kernel_size, head_channels, head_channels, head_channels)
        self.heads = nn.ModuleList([
            YOLODetectionHead(head_channels, head_channels, n_classes, per_cell_boxes, use_objectness),
            YOLODetectionHead(head_channels, head_channels, n_classes, per_cell_boxes, use_objectness),
            YOLODetectionHead(head_channels, head_channels, n_classes, per_cell_boxes, use_objectness),
        ])

    def forward(self, features):
        c3, c4, c5 = features
        p5 = self.p5_reduce(c5)
        p4 = self.p4_fuse(p5, c4)
        p3 = self.p3_fuse(p4, c3)
        n4 = self.n4_fuse(p3, p4)
        n5 = self.n5_fuse(n4, p5)
        outputs = []
        for feature, head in zip((p3, n4, n5), self.heads):
            outputs.append(head(feature))
        return outputs


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
        self.channels = channels
        block_name = getattr(config.model.yolo, 'block', 'conv')
        kernel_size = getattr(config.model.yolo, 'block_kernel_size', 3)
        stage_blocks = max(1, int(round(getattr(config.model.yolo, 'depth_mult', 1.0) * 3)))
        self.stem = create_yolo_block(block_name, config.dataset.n_channels, channels[0], kernel_size, 2)
        self.stage2 = YOLOStage(block_name, kernel_size, channels[0], channels[1], 2, num_blocks=stage_blocks)
        self.stage3 = YOLOStage(block_name, kernel_size, channels[1], channels[2], 2, num_blocks=stage_blocks)
        self.stage4 = YOLOStage(block_name, kernel_size, channels[2], channels[3], 2, num_blocks=stage_blocks)
        dense_head_cfg = getattr(config.model.yolo, 'dense_head', None)
        dense_enabled = bool(getattr(dense_head_cfg, 'enabled', False))
        dense_head_type = str(getattr(dense_head_cfg, 'type', 'fpn_multi')).lower()
        if not dense_enabled:
            raise ValueError('TinyYOLO now requires config.model.yolo.dense_head.enabled=True for multi-scale detection')
        if dense_head_type != 'fpn_multi':
            raise ValueError(f'TinyYOLO only supports dense head type "fpn_multi", got {dense_head_type}')
        per_cell_boxes = int(getattr(dense_head_cfg, 'per_cell_boxes', 1))
        self.per_cell_boxes = max(per_cell_boxes, 1)
        self.min_box_size = float(getattr(config.model.yolo, 'min_box_size', 0.02))
        self.use_objectness = bool(getattr(dense_head_cfg, 'use_objectness', True))
        head_channels = int(getattr(dense_head_cfg, 'neck_channels', channels[2]))
        self.strides = list(getattr(config.model.yolo, 'strides', [8, 16, 32]))
        if len(self.strides) != 3:
            raise ValueError(f'config.model.yolo.strides must contain 3 entries, got {self.strides}')
        self.neck = TinyYOLOFPNMultiHead(block_name,
                                         kernel_size,
                                         channels[1:],
                                         head_channels,
                                         self.n_classes,
                                         self.per_cell_boxes,
                                         self.use_objectness)
        self.grid_sizes = []
        default_candidates = sum(self.per_cell_boxes * max(config.dataset.image_size // int(stride), 1) ** 2
                                 for stride in self.strides)
        configured_candidates = int(getattr(config.model.yolo, 'num_candidates', default_candidates))
        self.num_candidates = max(1, configured_candidates)
        self.pool = None
        self.pool_kernel_size = None
        self.dense_neck = self.neck
        self.box_head = self.neck.heads[-1].box_head
        self.score_head = self.neck.heads[-1].score_head
        self.objectness_head = self.neck.heads[-1].objectness_head

    def _forward_backbone(self, images):
        x1 = self.stem(images)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x2, x3, x4

    def _reshape_prediction_map(self, pred_boxes, pred_logits, objectness, grid_h, grid_w):
        batch_size = pred_boxes.shape[0]
        pred_boxes = pred_boxes.view(batch_size, self.per_cell_boxes, 4, grid_h, grid_w)
        pred_boxes = pred_boxes.permute(0, 3, 4, 1, 2).reshape(batch_size, -1, 4)
        pred_logits = pred_logits.view(batch_size,
                                       self.per_cell_boxes,
                                       self.n_classes + 1,
                                       grid_h,
                                       grid_w)
        pred_logits = pred_logits.permute(0, 3, 4, 1, 2).reshape(batch_size, -1, self.n_classes + 1)
        if objectness is not None:
            objectness = objectness.view(batch_size, self.per_cell_boxes, 1, grid_h, grid_w)
            objectness = objectness.permute(0, 3, 4, 1, 2).reshape(batch_size, -1)
        else:
            objectness = pred_logits.new_ones((batch_size, pred_logits.shape[1]))
        return pred_boxes, pred_logits, objectness

    def _reshape_dense_predictions(self, prediction_maps):
        all_boxes = []
        all_logits = []
        all_objectness = []
        self.grid_sizes = []
        for pred_boxes, pred_logits, objectness in prediction_maps:
            grid_h = int(pred_boxes.shape[-2])
            grid_w = int(pred_boxes.shape[-1])
            self.grid_sizes.append((grid_h, grid_w))
            reshaped_boxes, reshaped_logits, reshaped_objectness = self._reshape_prediction_map(
                pred_boxes,
                pred_logits,
                objectness,
                grid_h,
                grid_w,
            )
            all_boxes.append(reshaped_boxes)
            all_logits.append(reshaped_logits)
            all_objectness.append(reshaped_objectness)
        return torch.cat(all_boxes, dim=1), torch.cat(all_logits, dim=1), torch.cat(all_objectness, dim=1)

    def _decode_level_boxes(self, pred_boxes, grid_h, grid_w, stride):
        device = pred_boxes.device
        dtype = pred_boxes.dtype
        grid_y = torch.arange(grid_h, device=device, dtype=dtype)
        grid_x = torch.arange(grid_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing='ij')
        centers = torch.stack((xx, yy), dim=-1).reshape(1, grid_h * grid_w, 1, 2)
        centers = centers.repeat(pred_boxes.shape[0], 1, self.per_cell_boxes, 1).reshape(pred_boxes.shape[0], -1, 2)
        stride_x = 1.0 / float(grid_w)
        stride_y = 1.0 / float(grid_h)
        center_x = (centers[..., 0:1] + pred_boxes[..., 0:1]) * stride_x
        center_y = (centers[..., 1:2] + pred_boxes[..., 1:2]) * stride_y
        center_xy = torch.cat((center_x, center_y), dim=-1)
        stride_scale = max(float(stride), 1.0) / max(float(self.config.dataset.image_size), 1.0)
        wh = torch.clamp(pred_boxes[..., 2:], min=self.min_box_size) * stride_scale
        top_left = torch.clamp(center_xy - 0.5 * wh, min=0.0, max=1.0)
        bottom_right = torch.clamp(center_xy + 0.5 * wh, min=0.0, max=1.0)
        return torch.cat([top_left, torch.maximum(top_left + 1e-4, bottom_right)], dim=-1)

    def _decode_dense_boxes(self, pred_boxes):
        decoded_levels = []
        start = 0
        for (grid_h, grid_w), stride in zip(self.grid_sizes, self.strides):
            num_locations = grid_h * grid_w * self.per_cell_boxes
            level_boxes = pred_boxes[:, start:start + num_locations]
            decoded_levels.append(self._decode_level_boxes(level_boxes, grid_h, grid_w, stride))
            start += num_locations
        decoded = torch.cat(decoded_levels, dim=1)
        if decoded.shape[1] > self.num_candidates:
            decoded = decoded[:, :self.num_candidates]
        return decoded

    def _build_detections(self, pred_boxes, pred_logits, objectness, image_size):
        scores = torch.softmax(pred_logits, dim=-1)
        detections = []
        nms_type = getattr(self.config.eval, 'nms_type', 'hard')
        conf_threshold = getattr(self.config.eval, 'conf_threshold', 0.25)
        nms_threshold = getattr(self.config.eval, 'nms_threshold', 0.45)
        max_detections = getattr(self.config.eval, 'max_detections', 300)
        soft_nms_sigma = getattr(self.config.eval, 'soft_nms_sigma', 0.5)
        soft_nms_score_threshold = getattr(self.config.eval, 'soft_nms_score_threshold', 1e-3)
        for sample_boxes, sample_scores, sample_objectness in zip(pred_boxes, scores, objectness):
            cls_scores = sample_scores[..., :-1]
            if cls_scores.numel() == 0:
                candidate_scores = sample_scores.new_zeros((0,))
                candidate_labels = torch.zeros((0,), dtype=torch.long, device=sample_scores.device)
                candidate_boxes = sample_boxes[:0]
            else:
                candidate_scores, candidate_labels = cls_scores.max(dim=-1)
                candidate_scores = candidate_scores * sample_objectness
                candidate_boxes = sample_boxes
            result = postprocess_detections(
                candidate_boxes * image_size,
                candidate_scores,
                candidate_labels,
                nms_type=nms_type,
                score_threshold=conf_threshold,
                iou_threshold=nms_threshold,
                max_detections=max_detections,
                soft_nms_sigma=soft_nms_sigma,
                soft_nms_score_threshold=soft_nms_score_threshold,
                return_candidates=True,
            )
            detections.append({
                'boxes': result.boxes,
                'scores': result.scores,
                'labels': result.labels,
                'candidate_boxes': result.candidate_boxes,
                'candidate_scores': result.candidate_scores,
                'candidate_labels': result.candidate_labels,
            })
        return detections

    def forward(self, images, targets=None, return_outputs=False):
        features = self._forward_backbone(images)
        prediction_maps = self.neck(features)
        raw_pred_boxes, pred_logits, objectness = self._reshape_dense_predictions(prediction_maps)
        pred_boxes = self._decode_dense_boxes(raw_pred_boxes)
        if pred_logits.shape[1] > self.num_candidates:
            pred_logits = pred_logits[:, :self.num_candidates]
            objectness = objectness[:, :self.num_candidates]
        detections = self._build_detections(pred_boxes, pred_logits, objectness, images.shape[-1])
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
            if self.use_objectness:
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
