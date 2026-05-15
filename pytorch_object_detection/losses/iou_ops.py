import math

import torch


class WIoUScale:
    iou_mean = 1.0
    monotonous = False
    momentum = 1 - 0.5 ** (1 / 7000)
    is_train = True

    def __init__(self, iou_loss):
        self.iou_loss = iou_loss
        self._update_running_mean()

    def _update_running_mean(self):
        if self.is_train:
            self.__class__.iou_mean = ((1 - self.momentum) * self.__class__.iou_mean +
                                       self.momentum * self.iou_loss.detach().mean().item())

    def scaled_loss(self, gamma=1.9, delta=3.0):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou_loss.detach() / self.iou_mean).sqrt()
            beta = self.iou_loss.detach() / self.iou_mean
            alpha = delta * torch.pow(torch.as_tensor(gamma, device=beta.device, dtype=beta.dtype), beta - delta)
            return beta / alpha
        return 1.0


def bbox_iou(box1,
             box2,
             xywh=False,
             giou=False,
             diou=False,
             ciou=False,
             siou=False,
             eiou=False,
             wiou=False,
             focal=False,
             alpha=1.0,
             gamma=0.5,
             scale=False,
             eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_half, h1_half, w2_half, h2_half = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_half, x1 + w1_half
        b1_y1, b1_y2 = y1 - h1_half, y1 + h1_half
        b2_x1, b2_x2 = x2 - w2_half, x2 + w2_half
        b2_y1, b2_y2 = y2 - h2_half, y2 + h2_half
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1 = b1_x2 - b1_x1
        h1 = (b1_y2 - b1_y1).clamp(min=eps)
        w2 = b2_x2 - b2_x1
        h2 = (b2_y2 - b2_y1).clamp(min=eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = torch.pow(inter / (union + eps), alpha)
    wiou_scale = WIoUScale(1 - inter / union) if scale else None

    if not (giou or diou or ciou or siou or eiou or wiou):
        return (iou, torch.pow(inter / (union + eps), gamma)) if focal else iou

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    if ciou or diou or eiou or siou or wiou:
        c2 = (cw ** 2 + ch ** 2) ** alpha + eps
        rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                 (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha
        if ciou:
            v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
            with torch.no_grad():
                alpha_ciou = v / (v - iou + (1 + eps))
            penalty = rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)
            return (iou - penalty, torch.pow(inter / (union + eps), gamma)) if focal else iou - penalty
        if eiou:
            rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
            rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
            cw2 = torch.pow(cw ** 2 + eps, alpha)
            ch2 = torch.pow(ch ** 2 + eps, alpha)
            penalty = rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2
            return (iou - penalty, torch.pow(inter / (union + eps), gamma)) if focal else iou - penalty
        if siou:
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma_siou = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma_siou * rho_x) - torch.exp(gamma_siou * rho_y)
            omega_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omega_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-omega_w), 4) + torch.pow(1 - torch.exp(-omega_h), 4)
            penalty = torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha)
            return (iou - penalty, torch.pow(inter / (union + eps), gamma)) if focal else iou - penalty
        if wiou:
            if focal:
                raise RuntimeError('WIoU does not support focal mode.')
            if scale:
                return wiou_scale.scaled_loss(), (1 - iou) * torch.exp(rho2 / c2), iou
            return iou, torch.exp(rho2 / c2)
        return (iou - rho2 / c2, torch.pow(inter / (union + eps), gamma)) if focal else iou - rho2 / c2

    c_area = cw * ch + eps
    penalty = torch.pow((c_area - union) / c_area + eps, alpha)
    return (iou - penalty, torch.pow(inter / (union + eps), gamma)) if focal else iou - penalty


def box_iou_xyxy(boxes1, boxes2, eps=1e-6):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32, device=boxes1.device)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=eps)


def compute_box_loss(pred_boxes, target_boxes, loss_type='l1', iou_variant='iou'):
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0
    if loss_type == 'l1':
        return torch.nn.functional.l1_loss(pred_boxes, target_boxes, reduction='sum')
    if loss_type == 'iou':
        iou_kwargs = {
            'giou': iou_variant == 'giou',
            'diou': iou_variant == 'diou',
            'ciou': iou_variant == 'ciou',
            'siou': iou_variant == 'siou',
            'eiou': iou_variant == 'eiou',
            'wiou': iou_variant == 'wiou',
        }
        iou = bbox_iou(pred_boxes, target_boxes, xywh=False, **iou_kwargs)
        if isinstance(iou, tuple):
            primary = iou[0]
            if len(iou) == 2:
                return ((1.0 - primary) * iou[1].detach()).sum()
            return (iou[0] * iou[1]).sum()
        return (1.0 - iou).sum()
    raise ValueError(f'Unsupported box loss type: {loss_type}')
