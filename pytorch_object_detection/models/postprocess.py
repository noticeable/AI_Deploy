import torch


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-12)


def _hard_nms_indices(boxes, scores, iou_threshold, max_detections):
    keep = []
    order = scores.argsort(descending=True)
    while order.numel() > 0 and len(keep) < max_detections:
        current = order[0]
        keep.append(int(current.item()))
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = box_iou(boxes[current].unsqueeze(0), boxes[remaining]).squeeze(0)
        order = remaining[ious <= iou_threshold]
    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)


def _soft_nms_single_class(boxes, scores, iou_threshold, sigma, score_threshold, max_detections):
    boxes = boxes.clone()
    scores = scores.clone()
    keep = []
    while scores.numel() > 0 and len(keep) < max_detections:
        best_index = int(scores.argmax().item())
        best_score = float(scores[best_index].item())
        if best_score < score_threshold:
            break
        keep.append(best_index)
        if scores.numel() == 1:
            break
        best_box = boxes[best_index].unsqueeze(0)
        ious = box_iou(best_box, boxes).squeeze(0)
        decay = torch.exp(-(ious * ious) / max(sigma, 1e-12))
        suppress_mask = ious > iou_threshold
        scores = torch.where(suppress_mask, scores * decay, scores)
        scores[best_index] = -1.0
    keep = [index for index in keep if index >= 0]
    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)


def postprocess_detections(boxes,
                           scores,
                           labels,
                           nms_type='hard',
                           score_threshold=0.25,
                           iou_threshold=0.45,
                           max_detections=300,
                           soft_nms_sigma=0.5,
                           soft_nms_score_threshold=1e-3):
    if boxes.numel() == 0 or scores.numel() == 0 or labels.numel() == 0:
        return boxes, scores, labels
    keep_mask = scores >= score_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]
    if boxes.numel() == 0:
        return boxes, scores, labels
    if boxes.shape[0] <= 1:
        return boxes[:max_detections], scores[:max_detections], labels[:max_detections]

    keep_indices = []
    for label in labels.unique(sorted=False):
        class_mask = labels == label
        class_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]
        if str(nms_type).lower() == 'soft':
            selected = _soft_nms_single_class(class_boxes,
                                              class_scores,
                                              iou_threshold,
                                              soft_nms_sigma,
                                              soft_nms_score_threshold,
                                              max_detections)
        else:
            selected = _hard_nms_indices(class_boxes,
                                         class_scores,
                                         iou_threshold,
                                         max_detections)
        if selected.numel() > 0:
            keep_indices.append(class_indices[selected])

    if not keep_indices:
        empty_boxes = boxes[:0]
        empty_scores = scores[:0]
        empty_labels = labels[:0]
        return empty_boxes, empty_scores, empty_labels

    keep_indices = torch.cat(keep_indices, dim=0)
    keep_scores = scores[keep_indices]
    order = keep_scores.argsort(descending=True)
    keep_indices = keep_indices[order][:max_detections]
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]
