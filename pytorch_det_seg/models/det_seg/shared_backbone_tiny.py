import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)
        self.block = nn.Sequential(self.conv, self.bn, self.act)

    def forward(self, x):
        return self.block(x)

    def fuse_model(self):
        torch.ao.quantization.fuse_modules(self, [['conv', 'bn']], inplace=True)


class DetectionNeck(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.block = ConvBNAct(in_channels, hidden_channels, 3, 1)

    def forward(self, x):
        return self.block(x)


class SegmentationNeck(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, 3, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNAct(hidden_channels, hidden_channels, 3, 1),
        )

    def forward(self, x):
        return self.block(x)


class DetectionHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.box_head = nn.Linear(in_channels, 4)
        self.score_head = nn.Linear(in_channels, n_classes + 1)

    def forward(self, x):
        pooled = self.pool(x).flatten(1)
        pred_boxes = torch.sigmoid(self.box_head(pooled)).unsqueeze(1)
        pred_logits = self.score_head(pooled).unsqueeze(1)
        return pred_boxes, pred_logits


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3, 1),
            nn.Conv2d(in_channels, n_classes, kernel_size=1),
        )

    def forward(self, x, image_size):
        logits = self.block(x)
        return F.interpolate(logits, size=(image_size, image_size), mode='bilinear', align_corners=False)


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = list(getattr(config.model.backbone, 'channels', []))
        if channels:
            if len(channels) != 4:
                raise ValueError(f'config.model.backbone.channels must contain 4 entries, got {channels}')
        else:
            base_channels = max(16, int(64 * config.model.backbone.width_mult))
            channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.config = config
        self.n_classes = config.dataset.n_classes
        self.backbone = nn.Sequential(
            ConvBNAct(config.dataset.n_channels, channels[0], 3, 2),
            ConvBNAct(channels[0], channels[1], 3, 2),
            ConvBNAct(channels[1], channels[2], 3, 2),
            ConvBNAct(channels[2], channels[3], 3, 2),
        )
        self.det_neck = DetectionNeck(channels[-1], config.model.det_neck.hidden_channels)
        self.seg_neck = SegmentationNeck(channels[-1], config.model.seg_neck.hidden_channels)
        self.det_head = DetectionHead(config.model.det_neck.hidden_channels, config.dataset.n_classes)
        self.seg_head = SegmentationHead(config.model.seg_neck.hidden_channels, config.dataset.n_classes)

    def forward(self, images, targets=None, return_outputs=False):
        features = self.backbone(images)
        rep_features = features
        det_features = self.det_neck(features)
        seg_features = self.seg_neck(features)
        pred_boxes, pred_logits = self.det_head(det_features)
        seg_logits = self.seg_head(seg_features, images.shape[-1])

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
            xyxy = xyxy * images.shape[-1]
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
