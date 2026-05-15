import json

import torch



def _compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


class DetSegEvaluator:
    def __init__(self, config):
        self.config = config

    def _compute_segmentation_metrics(self, seg_pred, seg_target):
        ignore_index = self.config.dataset.ignore_index
        num_classes = int(self.config.dataset.n_classes)
        valid = seg_target != ignore_index
        total_seg_correct = int(((seg_pred == seg_target) & valid).sum().item())
        total_seg_pixels = int(valid.sum().item())
        seg_acc = total_seg_correct / total_seg_pixels if total_seg_pixels else 0.0

        intersection_sum = torch.zeros(num_classes, dtype=torch.float64)
        union_sum = torch.zeros(num_classes, dtype=torch.float64)
        target_sum = torch.zeros(num_classes, dtype=torch.float64)
        pred_sum = torch.zeros(num_classes, dtype=torch.float64)

        for class_index in range(num_classes):
            pred_mask = (seg_pred == class_index) & valid
            target_mask = (seg_target == class_index) & valid
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            target_pixels = target_mask.sum().item()
            pred_pixels = pred_mask.sum().item()
            intersection_sum[class_index] += intersection
            union_sum[class_index] += union
            target_sum[class_index] += target_pixels
            pred_sum[class_index] += pred_pixels

        class_iou = []
        class_dice = []
        valid_iou = []
        valid_dice = []
        for class_index in range(num_classes):
            union = float(union_sum[class_index].item())
            intersection = float(intersection_sum[class_index].item())
            target_pixels = float(target_sum[class_index].item())
            pred_pixels = float(pred_sum[class_index].item())
            iou = intersection / union if union > 0 else None
            denom = pred_pixels + target_pixels
            dice = (2.0 * intersection) / denom if denom > 0 else None
            class_iou.append(iou)
            class_dice.append(dice)
            if iou is not None:
                valid_iou.append(iou)
            if dice is not None:
                valid_dice.append(dice)

        miou = sum(valid_iou) / len(valid_iou) if valid_iou else 0.0
        mdice = sum(valid_dice) / len(valid_dice) if valid_dice else 0.0
        freq_weighted_iou = 0.0
        total_target_pixels = float(target_sum.sum().item())
        if total_target_pixels > 0:
            for class_index, iou in enumerate(class_iou):
                if iou is None:
                    continue
                freq_weighted_iou += (float(target_sum[class_index].item()) / total_target_pixels) * iou

        return {
            'seg_acc': seg_acc,
            'seg_mIoU': miou,
            'seg_mDice': mdice,
            'seg_fwIoU': freq_weighted_iou,
            'seg_class_iou': class_iou,
            'seg_class_dice': class_dice,
        }

    def evaluate(self, predictions, targets):
        matched = 0
        total_pred = 0
        total_gt = 0
        serializable_predictions = []
        for prediction, target in zip(predictions['detections'], targets):
            pred_boxes = prediction['boxes'].detach().cpu().tolist()
            gt_boxes = target['boxes'].detach().cpu().tolist()
            total_pred += len(pred_boxes)
            total_gt += len(gt_boxes)
            used = set()
            for pred_box in pred_boxes:
                for index, gt_box in enumerate(gt_boxes):
                    if index in used:
                        continue
                    if _compute_iou(pred_box, gt_box) >= self.config.eval.iou_threshold:
                        matched += 1
                        used.add(index)
                        break
            serializable_predictions.append({
                'image_id': int(target['image_id'].item() if hasattr(target['image_id'], 'item') else target['image_id']),
                'boxes': pred_boxes,
                'scores': prediction['scores'].detach().cpu().tolist(),
                'labels': prediction['labels'].detach().cpu().tolist(),
            })
        seg_pred = predictions['segmentation'].detach().cpu()
        seg_target = torch.stack([target['segmentation_mask'].detach().cpu() for target in targets])
        precision = matched / total_pred if total_pred else 0.0
        recall = matched / total_gt if total_gt else 0.0
        ap50 = precision * recall
        seg_metrics = self._compute_segmentation_metrics(seg_pred, seg_target)
        return {
            'AP': ap50,
            'AP50': ap50,
            'AP75': 0.0,
            'precision': precision,
            'recall': recall,
            **seg_metrics,
            'predictions_json': json.dumps(serializable_predictions, ensure_ascii=False),
        }



def create_evaluator(config):
    return DetSegEvaluator(config)
