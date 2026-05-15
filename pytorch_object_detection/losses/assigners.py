import torch

from .iou_ops import box_iou_xyxy


class BaseAssigner:
    def __init__(self, config, meta_architecture):
        self.config = config
        self.meta_architecture = meta_architecture
        self.assignment_config = self._resolve_assignment_config(config, meta_architecture)
        self.n_classes = config.dataset.n_classes
        self.background_index = self.n_classes

    def _resolve_assignment_config(self, config, meta_architecture):
        assignment_config = config.assignment.clone()
        if hasattr(config.assignment, meta_architecture):
            assignment_config.merge_from_other_cfg(getattr(config.assignment, meta_architecture))
        return assignment_config

    def _normalize_targets(self, targets, device):
        normalized = []
        for target in targets:
            boxes = torch.as_tensor(target['boxes'], dtype=torch.float32, device=device)
            if boxes.numel() == 0:
                boxes = boxes.reshape(0, 4)
            labels = torch.as_tensor(target['labels'], dtype=torch.long, device=device)
            size_h = target['size'][0]
            size_w = target['size'][1] if len(target['size']) > 1 else target['size'][0]
            image_height = size_h.item() if hasattr(size_h, 'item') else size_h
            image_width = size_w.item() if hasattr(size_w, 'item') else size_w
            scale = torch.tensor([
                max(float(image_width), 1.0),
                max(float(image_height), 1.0),
                max(float(image_width), 1.0),
                max(float(image_height), 1.0),
            ], dtype=torch.float32, device=device)
            normalized.append({
                'boxes': boxes / scale,
                'labels': labels,
            })
        return normalized

    def _empty_assignment(self, num_predictions, device):
        return {
            'matched_pred_indices': torch.zeros((0,), dtype=torch.long, device=device),
            'matched_target_indices': torch.zeros((0,), dtype=torch.long, device=device),
            'positive_mask': torch.zeros((num_predictions,), dtype=torch.bool, device=device),
            'background_mask': torch.ones((num_predictions,), dtype=torch.bool, device=device),
            'target_boxes': torch.zeros((num_predictions, 4), dtype=torch.float32, device=device),
            'target_labels': torch.full((num_predictions,),
                                        self.background_index,
                                        dtype=torch.long,
                                        device=device),
        }

    def _build_assignment(self, pred_boxes, target_boxes, target_labels, matches):
        device = pred_boxes.device
        num_predictions = pred_boxes.shape[0]
        assignment = self._empty_assignment(num_predictions, device)
        if not matches:
            return assignment
        matched_pred_indices = torch.as_tensor([pair[0] for pair in matches],
                                               dtype=torch.long,
                                               device=device)
        matched_target_indices = torch.as_tensor([pair[1] for pair in matches],
                                                 dtype=torch.long,
                                                 device=device)
        assignment['matched_pred_indices'] = matched_pred_indices
        assignment['matched_target_indices'] = matched_target_indices
        assignment['positive_mask'][matched_pred_indices] = True
        assignment['background_mask'][matched_pred_indices] = False
        assignment['target_boxes'][matched_pred_indices] = target_boxes[matched_target_indices]
        assignment['target_labels'][matched_pred_indices] = target_labels[matched_target_indices]
        return assignment

    def assign(self, pred_boxes, pred_logits, targets):
        normalized_targets = self._normalize_targets(targets, pred_boxes.device)
        return [self.assign_single(boxes, logits, target)
                for boxes, logits, target in zip(pred_boxes, pred_logits, normalized_targets)]


class FirstBoxAssigner(BaseAssigner):
    def assign_single(self, pred_boxes, pred_logits, target):
        if target['boxes'].numel() == 0 or pred_boxes.shape[0] == 0:
            return self._empty_assignment(pred_boxes.shape[0], pred_boxes.device)
        return self._build_assignment(pred_boxes,
                                      target['boxes'],
                                      target['labels'],
                                      [(0, 0)])


class IoUAssigner(BaseAssigner):
    def assign_single(self, pred_boxes, pred_logits, target):
        if target['boxes'].numel() == 0 or pred_boxes.shape[0] == 0:
            return self._empty_assignment(pred_boxes.shape[0], pred_boxes.device)
        iou_matrix = box_iou_xyxy(pred_boxes, target['boxes'])
        max_ious, matched_targets = iou_matrix.max(dim=1)
        candidate_indices = torch.nonzero(max_ious >= self.assignment_config.iou_threshold,
                                          as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            return self._empty_assignment(pred_boxes.shape[0], pred_boxes.device)
        if self.assignment_config.topk > 0 and candidate_indices.numel() > self.assignment_config.topk:
            top_values, top_order = torch.topk(max_ious[candidate_indices],
                                               k=self.assignment_config.topk)
            del top_values
            candidate_indices = candidate_indices[top_order]
        used_targets = set()
        matches = []
        for pred_index in candidate_indices.tolist():
            target_index = int(matched_targets[pred_index].item())
            if target_index in used_targets and self.assignment_config.max_matches_per_target == 1:
                continue
            used_targets.add(target_index)
            matches.append((pred_index, target_index))
        return self._build_assignment(pred_boxes,
                                      target['boxes'],
                                      target['labels'],
                                      matches)


class DynamicKAssigner(BaseAssigner):
    def assign_single(self, pred_boxes, pred_logits, target):
        if target['boxes'].numel() == 0 or pred_boxes.shape[0] == 0:
            return self._empty_assignment(pred_boxes.shape[0], pred_boxes.device)
        iou_matrix = box_iou_xyxy(pred_boxes, target['boxes']).transpose(0, 1)
        eps = float(getattr(self.assignment_config, 'eps', 1e-8))
        pair_wise_iou_loss = -(iou_matrix + eps).log()

        target_labels = target['labels']
        cls_prob = pred_logits.softmax(dim=-1)[:, target_labels].transpose(0, 1)
        cls_cost = -(cls_prob + eps).log()
        total_cost = (float(getattr(self.assignment_config, 'cls_cost', 1.0)) * cls_cost +
                      float(getattr(self.assignment_config, 'iou_cost', 3.0)) * pair_wise_iou_loss)

        num_targets, num_predictions = total_cost.shape
        matching_matrix = torch.zeros_like(total_cost, dtype=torch.bool)
        topk_limit = min(int(getattr(self.assignment_config, 'dynamic_k_topk', 10)), num_predictions)
        if topk_limit <= 0:
            topk_limit = 1
        topk_ious, _ = torch.topk(iou_matrix, k=topk_limit, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(dim=1).int(), min=1)

        for target_index in range(num_targets):
            _, pred_indices = torch.topk(total_cost[target_index],
                                         k=min(dynamic_ks[target_index].item(), num_predictions),
                                         largest=False)
            matching_matrix[target_index, pred_indices] = True

        matched_counts = matching_matrix.sum(dim=0)
        if (matched_counts > 1).any():
            multi_match_indices = torch.nonzero(matched_counts > 1, as_tuple=False).flatten()
            for pred_index in multi_match_indices.tolist():
                target_indices = torch.nonzero(matching_matrix[:, pred_index], as_tuple=False).flatten()
                best_target_offset = total_cost[target_indices, pred_index].argmin()
                best_target = target_indices[best_target_offset]
                matching_matrix[target_indices, pred_index] = False
                matching_matrix[best_target, pred_index] = True

        matched_target_indices, matched_pred_indices = torch.nonzero(matching_matrix,
                                                                     as_tuple=True)
        matches = list(zip(matched_pred_indices.tolist(), matched_target_indices.tolist()))
        return self._build_assignment(pred_boxes,
                                      target['boxes'],
                                      target['labels'],
                                      matches)


class HungarianAssigner(BaseAssigner):
    def assign_single(self, pred_boxes, pred_logits, target):
        if target['boxes'].numel() == 0 or pred_boxes.shape[0] == 0:
            return self._empty_assignment(pred_boxes.shape[0], pred_boxes.device)
        cls_prob = pred_logits.softmax(dim=-1)
        target_labels = target['labels']
        cls_cost = -cls_prob[:, target_labels]
        box_cost = torch.cdist(pred_boxes, target['boxes'], p=1)
        cost_matrix = (self.assignment_config.cls_cost * cls_cost +
                       self.assignment_config.box_cost * box_cost)
        matches = []
        remaining_predictions = list(range(pred_boxes.shape[0]))
        remaining_targets = list(range(target['boxes'].shape[0]))
        while remaining_predictions and remaining_targets:
            submatrix = cost_matrix[remaining_predictions][:, remaining_targets]
            flat_index = submatrix.argmin().item()
            pred_offset = flat_index // submatrix.shape[1]
            target_offset = flat_index % submatrix.shape[1]
            pred_index = remaining_predictions.pop(pred_offset)
            target_index = remaining_targets.pop(target_offset)
            matches.append((pred_index, target_index))
        return self._build_assignment(pred_boxes,
                                      target['boxes'],
                                      target['labels'],
                                      matches)


def create_assigner(config, meta_architecture):
    assignment_config = config.assignment
    name = assignment_config.name
    if hasattr(assignment_config, meta_architecture):
        arch_config = getattr(assignment_config, meta_architecture)
        if hasattr(arch_config, 'name'):
            name = arch_config.name
    if name == 'first_box':
        return FirstBoxAssigner(config, meta_architecture)
    if name == 'iou':
        return IoUAssigner(config, meta_architecture)
    if name == 'dynamic_k':
        return DynamicKAssigner(config, meta_architecture)
    if name == 'hungarian':
        return HungarianAssigner(config, meta_architecture)
    raise ValueError(f'Unsupported assignment strategy: {name}')
