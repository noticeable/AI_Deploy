import torch.nn as nn


class DetSegLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.det_weight = float(config.loss.det_weight)
        self.seg_weight = float(config.loss.seg_weight)

    def forward(self, loss_dict):
        total = self.det_weight * loss_dict['det_loss'] + self.seg_weight * loss_dict['loss_seg']
        merged = dict(loss_dict)
        merged['loss_total'] = total
        return merged



def create_loss(config):
    return DetSegLoss(config)
