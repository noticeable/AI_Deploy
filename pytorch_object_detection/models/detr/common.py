import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_object_detection.losses import create_assigner


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, x):
        h, w = x.shape[-2:]
        y_embed = torch.linspace(0, 1, h, device=x.device)
        x_embed = torch.linspace(0, 1, w, device=x.device)
        yy, xx = torch.meshgrid(y_embed, x_embed, indexing='ij')
        pos = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return pos.repeat(x.shape[0], self.num_pos_feats, 1, 1)


class TinyDETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_classes = config.dataset.n_classes
        self.assigner = create_assigner(config, 'detr')
        hidden_dim = config.model.detr.hidden_dim
        self.num_queries = config.model.detr.num_queries
        self.backbone = nn.Sequential(
            nn.Conv2d(config.dataset.n_channels, hidden_dim // 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.position = PositionEmbeddingSine(hidden_dim // 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.model.detr.nheads,
            dim_feedforward=config.model.detr.dim_feedforward,
            dropout=config.model.detr.dropout,
            batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=config.model.detr.nheads,
            dim_feedforward=config.model.detr.dim_feedforward,
            dropout=config.model.detr.dropout,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=config.model.detr.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=config.model.detr.num_decoder_layers)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, config.dataset.n_classes + 1)
        self.box_embed = nn.Linear(hidden_dim, 4)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        pos = self.position(features)
        src = (features + pos).flatten(2).transpose(1, 2)
        memory = self.encoder(src)
        query = self.query_embed.weight.unsqueeze(0).repeat(images.size(0), 1, 1)
        hs = self.decoder(query, memory)
        logits = self.class_embed(hs)
        boxes = torch.sigmoid(self.box_embed(hs))
        if self.training and targets is not None:
            assignments = self.assigner.assign(boxes, logits, targets)
            loss_bbox = boxes.sum() * 0.0
            loss_cls = logits.sum() * 0.0
            for sample_boxes, sample_logits, assignment in zip(boxes, logits, assignments):
                if assignment['positive_mask'].any():
                    positive_indices = assignment['matched_pred_indices']
                    target_boxes = assignment['target_boxes'][positive_indices]
                    loss_bbox = loss_bbox + F.l1_loss(sample_boxes[positive_indices],
                                                      target_boxes,
                                                      reduction='sum')
                loss_cls = loss_cls + F.cross_entropy(sample_logits,
                                                      assignment['target_labels'],
                                                      reduction='sum')
            batch_size = max(len(assignments), 1)
            return {
                'loss_bbox': loss_bbox / batch_size,
                'loss_cls': loss_cls / batch_size,
            }
        scores = torch.softmax(logits, dim=-1)
        detections = []
        image_size = images.shape[-1]
        for pred_boxes, pred_scores in zip(boxes, scores):
            cls_scores, cls_labels = pred_scores[..., :-1].max(dim=-1)
            top_idx = cls_scores.argmax()
            xyxy = pred_boxes[top_idx].clone()
            xyxy[2:] = torch.maximum(xyxy[:2] + 0.1, xyxy[2:])
            detections.append({
                'boxes': (xyxy.unsqueeze(0) * image_size),
                'scores': cls_scores[top_idx].unsqueeze(0),
                'labels': cls_labels[top_idx].unsqueeze(0),
            })
        return detections
