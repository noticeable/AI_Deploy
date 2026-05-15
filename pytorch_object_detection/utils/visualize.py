import json
import pathlib

from PIL import Image, ImageDraw
import numpy as np


def draw_box_with_label(draw, box, color, label_text=None, width=2):
    draw.rectangle(box, outline=color, width=width)
    if label_text:
        draw.text((box[0], box[1]), label_text, fill=color)


def draw_detections(image_path, detections, output_path, class_names=None):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    boxes = detections.get('boxes', [])
    scores = detections.get('scores', [])
    labels = detections.get('labels', [])
    for box, score, label in zip(boxes, scores, labels):
        label_name = _resolve_label_name(label, class_names)
        draw_box_with_label(draw,
                            box,
                            color='red',
                            label_text=f'{label_name}:{score:.2f}')
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    image.save(output_path)
    return output_path


def xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    return boxes


def compute_iou_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))
    area_inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    area_1 = np.maximum(0, x12 - x11) * np.maximum(0, y12 - y11)
    area_2 = np.maximum(0, x22 - x21) * np.maximum(0, y22 - y21)
    area_union = area_1 + np.transpose(area_2) - area_inter
    return area_inter / np.maximum(area_union, 1e-6)


def load_yolo_annotations(label_path, width, height):
    label_path = pathlib.Path(label_path)
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding='utf-8').splitlines():
        fields = line.strip().split()
        if len(fields) < 5:
            continue
        row = np.asarray(fields[:5], dtype=np.float32)
        rows.append(row)
    if not rows:
        return []
    labels = np.stack(rows, axis=0)
    boxes = xywh_to_xyxy(labels[:, 1:5])
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    return [{'label': int(label), 'box': box.tolist()} for label, box in zip(labels[:, 0], boxes)]


def load_prediction_entries(prediction_json_path):
    prediction_json_path = pathlib.Path(prediction_json_path)
    if not prediction_json_path.exists():
        return {}
    payload = json.loads(prediction_json_path.read_text(encoding='utf-8'))
    return {str(item['image_id']): item for item in payload}


def prediction_entry_to_annotations(prediction_entry):
    if not prediction_entry:
        return []
    boxes = prediction_entry.get('boxes', [])
    labels = prediction_entry.get('labels', [])
    scores = prediction_entry.get('scores', [])
    results = []
    for box, label, score in zip(boxes, labels, scores):
        results.append({
            'label': int(label),
            'box': list(box),
            'score': float(score),
        })
    return results


def match_detections_to_targets(targets, predictions, iou_threshold=0.45):
    target_boxes = np.asarray([item['box'] for item in targets], dtype=np.float32).reshape(-1, 4)
    pred_boxes = np.asarray([item['box'] for item in predictions], dtype=np.float32).reshape(-1, 4)
    iou_matrix = compute_iou_matrix(target_boxes, pred_boxes)
    matches = []
    used_predictions = set()
    for target_index, target in enumerate(targets):
        if len(predictions) == 0:
            break
        ious = iou_matrix[target_index] if iou_matrix.size else np.zeros((0,), dtype=np.float32)
        for prediction_index in np.argsort(ious)[::-1]:
            if prediction_index in used_predictions:
                continue
            if ious[prediction_index] < iou_threshold:
                break
            if predictions[prediction_index]['label'] != target['label']:
                continue
            used_predictions.add(int(prediction_index))
            matches.append((target_index, int(prediction_index)))
            break
    return matches, used_predictions


def draw_detection_errors(image_path,
                          targets,
                          predictions,
                          output_path,
                          class_names=None,
                          iou_threshold=0.45,
                          colors=None):
    palette = colors or {
        'matched': 'green',
        'missing': 'red',
        'error': 'blue',
    }
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    matches, used_predictions = match_detections_to_targets(targets, predictions, iou_threshold=iou_threshold)
    matched_targets = {target_index for target_index, _ in matches}
    for target_index, prediction_index in matches:
        prediction = predictions[prediction_index]
        label_name = _resolve_label_name(prediction['label'], class_names)
        draw_box_with_label(draw, prediction['box'], palette['matched'], label_text=label_name)
    for target_index, target in enumerate(targets):
        if target_index in matched_targets:
            continue
        label_name = _resolve_label_name(target['label'], class_names)
        draw_box_with_label(draw, target['box'], palette['missing'], label_text=label_name)
    for prediction_index, prediction in enumerate(predictions):
        if prediction_index in used_predictions:
            continue
        label_name = _resolve_label_name(prediction['label'], class_names)
        draw_box_with_label(draw, prediction['box'], palette['error'], label_text=label_name)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    image.save(output_path)
    return output_path


def _resolve_label_name(label, class_names=None):
    if class_names and 0 <= int(label) < len(class_names):
        return class_names[int(label)]
    return str(label)
