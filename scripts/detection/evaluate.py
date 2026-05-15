#!/usr/bin/env python

import argparse
import json
import pathlib

import torch
from PIL import Image

from pytorch_object_detection import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_evaluator,
    create_model,
    get_default_config,
    update_config,
)
from pytorch_object_detection.utils import create_logger
from pytorch_object_detection.utils.checkpoint import create_model_from_checkpoint
from pytorch_object_detection.utils.visualize import (
    draw_detection_errors,
    load_yolo_annotations,
    prediction_entry_to_annotations,
)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--visualize-errors', action='store_true')
    parser.add_argument('--image-dir', type=str, default='')
    parser.add_argument('--label-dir', type=str, default='')
    parser.add_argument('--image-suffix', type=str, default='jpg')
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    config = update_config(config)
    config.freeze()
    return config, args


def _write_prediction_artifacts(output_dir, metrics):
    prediction_entries = metrics.get('predictions', [])
    prediction_json_path = output_dir / 'detection_predictions.json'
    prediction_json_path.write_text(
        json.dumps(prediction_entries, indent=2, ensure_ascii=False),
        encoding='utf-8')
    yolo_dir = output_dir / 'predictions_yolo'
    yolo_dir.mkdir(parents=True, exist_ok=True)
    for image_id, lines in metrics.get('predictions_yolo', {}).items():
        (yolo_dir / f'{image_id}.txt').write_text('\n'.join(lines), encoding='utf-8')
    return prediction_json_path, yolo_dir


def _render_error_visualizations(output_dir, metrics, args, class_names):
    if not args.visualize_errors or not args.image_dir or not args.label_dir:
        return
    image_dir = pathlib.Path(args.image_dir)
    label_dir = pathlib.Path(args.label_dir)
    vis_dir = output_dir / 'visualize_errors'
    vis_dir.mkdir(parents=True, exist_ok=True)
    prediction_map = {str(item['image_id']): item for item in metrics.get('predictions', [])}
    for label_path in sorted(label_dir.glob('*.txt')):
        image_path = image_dir / f'{label_path.stem}.{args.image_suffix}'
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        targets = load_yolo_annotations(label_path, width, height)
        predictions = prediction_entry_to_annotations(prediction_map.get(label_path.stem))
        draw_detection_errors(image_path,
                              targets,
                              predictions,
                              vis_dir / image_path.name,
                              class_names=class_names)


def main():
    config, args = load_config()
    output_dir = pathlib.Path(config.test.output_dir or config.train.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = create_logger(__name__, distributed_rank=0, output_dir=output_dir)

    dataloader = create_dataloader(config, is_train=False)
    if config.test.checkpoint:
        model, config, _ = create_model_from_checkpoint(config, config.test.checkpoint, create_model)
    else:
        model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)
    model.eval()

    predictions = []
    targets_all = []
    for images, targets in dataloader:
        images = images.to(config.device)
        outputs = model(images)
        predictions.extend(outputs)
        targets_all.extend(targets)

    evaluator = create_evaluator(config)
    metrics = evaluator.evaluate(predictions, targets_all)
    _write_prediction_artifacts(output_dir, metrics)
    _render_error_visualizations(output_dir, metrics, args, config.dataset.class_names)
    result_path = output_dir / 'detection_metrics.json'
    metrics_to_save = dict(metrics)
    metrics_to_save.pop('predictions', None)
    metrics_to_save.pop('predictions_yolo', None)
    result_path.write_text(json.dumps(metrics_to_save, indent=2), encoding='utf-8')
    logger.info(metrics_to_save)


if __name__ == '__main__':
    main()
