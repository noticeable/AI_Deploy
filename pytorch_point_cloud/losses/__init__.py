import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    def forward(self, outputs, targets):
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        regularization_loss = outputs.get('regularization_loss', 0.0) if isinstance(outputs, dict) else 0.0
        return F.cross_entropy(logits, targets) + regularization_loss


class SegmentationLoss(nn.Module):
    def forward(self, outputs, targets):
        seg_logits = outputs['seg_logits']
        seg_targets = targets['seg_label']
        regularization_loss = outputs.get('regularization_loss', 0.0)
        return F.cross_entropy(seg_logits.reshape(-1, seg_logits.size(-1)),
                               seg_targets.reshape(-1)) + regularization_loss



def create_loss(config):
    train_loss = ClassificationLoss() if config.task == 'classification' else SegmentationLoss()
    val_loss = train_loss
    return train_loss, val_loss
