#!/usr/bin/env python

import argparse
import pathlib

from PIL import Image

from pytorch_object_detection.utils.visualize import (
    draw_detection_errors,
    load_prediction_entries,
    load_yolo_annotations,
    prediction_entry_to_annotations,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True, type=str)
    parser.add_argument('--label-dir', required=True, type=str)
    parser.add_argument('--prediction-dir', type=str, default='')
    parser.add_argument('--prediction-json', type=str, default='')
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--class-names', nargs='*', default=None)
    parser.add_argument('--image-suffix', type=str, default='jpg')
    parser.add_argument('--iou-threshold', type=float, default=0.45)
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = pathlib.Path(args.image_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_entries = load_prediction_entries(args.prediction_json) if args.prediction_json else {}
    prediction_dir = pathlib.Path(args.prediction_dir) if args.prediction_dir else None

    for label_path in sorted(pathlib.Path(args.label_dir).glob('*.txt')):
        image_path = image_dir / f'{label_path.stem}.{args.image_suffix}'
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        labels = load_yolo_annotations(label_path, width, height)
        if prediction_entries:
            predictions = prediction_entry_to_annotations(prediction_entries.get(label_path.stem))
        elif prediction_dir is not None:
            predictions = load_yolo_annotations(prediction_dir / label_path.name, width, height)
        else:
            predictions = []
        draw_detection_errors(image_path,
                              labels,
                              predictions,
                              output_dir / image_path.name,
                              class_names=args.class_names,
                              iou_threshold=args.iou_threshold)


if __name__ == '__main__':
    main()
