import json


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


class DetectionEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, predictions, targets):
        matched = 0
        total_pred = 0
        total_gt = 0
        serializable_predictions = []
        yolo_prediction_lines = {}
        for prediction, target in zip(predictions, targets):
            image_id = int(target['image_id'].item() if hasattr(target['image_id'], 'item') else target['image_id'])
            pred_boxes = prediction['boxes'].detach().cpu().tolist()
            pred_scores = prediction['scores'].detach().cpu().tolist()
            pred_labels = prediction['labels'].detach().cpu().tolist()
            label_to_category_id = target.get('label_to_category_id', {})
            if hasattr(label_to_category_id, 'items'):
                label_to_category_id = {
                    int(key): int(value) for key, value in label_to_category_id.items()
                }
            gt_boxes = target['boxes'].detach().cpu().tolist()
            total_pred += len(pred_boxes)
            total_gt += len(gt_boxes)
            used = set()
            for pred_box in pred_boxes:
                for index, gt_box in enumerate(gt_boxes):
                    if index in used:
                        continue
                    if _compute_iou(pred_box, gt_box) >= 0.5:
                        matched += 1
                        used.add(index)
                        break
            serializable_predictions.append({
                'image_id': image_id,
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': [label_to_category_id.get(int(label), int(label)) for label in pred_labels],
            })
            size_value = target.get('orig_size', target.get('size', [1, 1]))
            height = float(size_value[0].item() if hasattr(size_value[0], 'item') else size_value[0])
            width = float(size_value[1].item() if hasattr(size_value[1], 'item') else size_value[1])
            lines = []
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                x1, y1, x2, y2 = box
                box_w = max(0.0, x2 - x1)
                box_h = max(0.0, y2 - y1)
                cx = x1 + box_w / 2.0
                cy = y1 + box_h / 2.0
                export_label = label_to_category_id.get(int(label), int(label))
                lines.append(
                    f'{export_label} {cx / max(width, 1.0):.6f} {cy / max(height, 1.0):.6f} '
                    f'{box_w / max(width, 1.0):.6f} {box_h / max(height, 1.0):.6f} {float(score):.6f}')
            yolo_prediction_lines[str(image_id)] = lines
        precision = matched / total_pred if total_pred else 0.0
        recall = matched / total_gt if total_gt else 0.0
        ap50 = precision * recall
        return {
            'AP': ap50,
            'AP50': ap50,
            'AP75': 0.0,
            'precision': precision,
            'recall': recall,
            'predictions': serializable_predictions,
            'predictions_json': json.dumps(serializable_predictions, ensure_ascii=False),
            'predictions_yolo': yolo_prediction_lines,
        }


def create_evaluator(config):
    return DetectionEvaluator(config)
