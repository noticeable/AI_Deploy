from .assigners import create_assigner
from .iou_ops import bbox_iou, box_iou_xyxy, compute_box_loss


def create_loss(config):
    return None, None
