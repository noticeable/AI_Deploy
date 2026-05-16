import torch


class DetectionPostprocessResult(tuple):
    __slots__ = ()

    def __new__(cls, boxes, scores, labels, candidate_boxes, candidate_scores, candidate_labels):
        return super().__new__(cls, (boxes, scores, labels, candidate_boxes, candidate_scores, candidate_labels))

    @property
    def boxes(self):
        return self[0]

    @property
    def scores(self):
        return self[1]

    @property
    def labels(self):
        return self[2]

    @property
    def candidate_boxes(self):
        return self[3]

    @property
    def candidate_scores(self):
        return self[4]

    @property
    def candidate_labels(self):
        return self[5]


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
    active = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    keep = []
    kept_scores = []
    while active.any() and len(keep) < max_detections:
        active_scores = scores.clone()
        active_scores[~active] = -1.0
        best_index = int(active_scores.argmax().item())
        best_score = float(active_scores[best_index].item())
        if best_score < score_threshold:
            break
        keep.append(best_index)
        kept_scores.append(scores[best_index].clone())
        active[best_index] = False
        if not active.any():
            break
        remaining_indices = torch.nonzero(active, as_tuple=False).flatten()
        ious = box_iou(boxes[best_index].unsqueeze(0), boxes[remaining_indices]).squeeze(0)
        decay = torch.exp(-(ious * ious) / max(float(sigma), 1e-12))
        suppress_mask = ious > iou_threshold
        decayed_scores = torch.where(suppress_mask, scores[remaining_indices] * decay, scores[remaining_indices])
        active[remaining_indices] = decayed_scores >= score_threshold
        scores[remaining_indices] = decayed_scores
    keep_tensor = torch.as_tensor(keep, dtype=torch.long, device=boxes.device)
    if kept_scores:
        score_tensor = torch.stack(kept_scores).to(dtype=scores.dtype, device=scores.device)
    else:
        score_tensor = scores.new_zeros((0,))
    return keep_tensor, score_tensor


def _empty_result(boxes, scores, labels):
    return DetectionPostprocessResult(boxes[:0], scores[:0], labels[:0], boxes[:0], scores[:0], labels[:0])


def postprocess_detections(boxes,
                           scores,
                           labels,
                           nms_type='hard',
                           score_threshold=0.25,
                           iou_threshold=0.45,
                           max_detections=300,
                           soft_nms_sigma=0.5,
                           soft_nms_score_threshold=1e-3,
                           return_candidates=False):
    if boxes.ndim != 2:
        boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if boxes.numel() == 0 or scores.numel() == 0 or labels.numel() == 0:
        return _empty_result(boxes, scores, labels) if return_candidates else (boxes, scores, labels)

    keep_mask = scores >= score_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]
    if boxes.numel() == 0:
        return _empty_result(boxes, scores, labels) if return_candidates else (boxes, scores, labels)

    candidate_order = scores.argsort(descending=True)
    candidate_boxes = boxes[candidate_order][:max_detections]
    candidate_scores = scores[candidate_order][:max_detections]
    candidate_labels = labels[candidate_order][:max_detections]

    if boxes.shape[0] <= 1:
        result = DetectionPostprocessResult(candidate_boxes,
                                            candidate_scores,
                                            candidate_labels,
                                            candidate_boxes,
                                            candidate_scores,
                                            candidate_labels)
        return result if return_candidates else result[:3]

    keep_indices = []
    keep_scores = []
    nms_type = str(nms_type).lower()
    per_class_limit = max(int(max_detections), 1)
    for label in labels.unique(sorted=False):
        class_mask = labels == label
        class_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]
        if nms_type == 'soft':
            selected, selected_scores = _soft_nms_single_class(class_boxes,
                                                               class_scores,
                                                               iou_threshold,
                                                               soft_nms_sigma,
                                                               soft_nms_score_threshold,
                                                               per_class_limit)
        else:
            selected = _hard_nms_indices(class_boxes,
                                         class_scores,
                                         iou_threshold,
                                         per_class_limit)
            selected_scores = class_scores[selected]
        if selected.numel() > 0:
            keep_indices.append(class_indices[selected])
            keep_scores.append(selected_scores)

    if not keep_indices:
        return _empty_result(boxes, scores, labels) if return_candidates else (boxes[:0], scores[:0], labels[:0])

    keep_indices = torch.cat(keep_indices, dim=0)
    updated_scores = torch.cat(keep_scores, dim=0)
    order = updated_scores.argsort(descending=True)
    keep_indices = keep_indices[order][:max_detections]
    updated_scores = updated_scores[order][:max_detections]
    result = DetectionPostprocessResult(boxes[keep_indices],
                                        updated_scores,
                                        labels[keep_indices],
                                        candidate_boxes,
                                        candidate_scores,
                                        candidate_labels)
    return result if return_candidates else result[:3]
