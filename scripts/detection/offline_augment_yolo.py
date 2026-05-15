#!/usr/bin/env python

import argparse
import pathlib
import shutil

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from pytorch_object_detection.utils.visualize import load_yolo_annotations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True, type=str)
    parser.add_argument('--output-root', required=True, type=str)
    parser.add_argument('--loops', type=int, default=1)
    parser.add_argument('--brightness', type=float, default=0.2)
    parser.add_argument('--contrast', type=float, default=0.2)
    parser.add_argument('--grayscale-prob', type=float, default=0.1)
    parser.add_argument('--flip-prob', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def adjust_boxes_for_horizontal_flip(boxes, width):
    adjusted = []
    for item in boxes:
        x1, y1, x2, y2 = item['box']
        adjusted.append({
            'label': item['label'],
            'box': [width - x2, y1, width - x1, y2],
        })
    return adjusted


def to_yolo_line(item, width, height):
    x1, y1, x2, y2 = item['box']
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    return f"{item['label']} {cx / width:.6f} {cy / height:.6f} {box_w / width:.6f} {box_h / height:.6f}"


def augment_image(image, boxes, args, rng):
    width, height = image.size
    if rng.random() < args.flip_prob:
        image = ImageOps.mirror(image)
        boxes = adjust_boxes_for_horizontal_flip(boxes, width)
    brightness_factor = 1.0 + rng.uniform(-args.brightness, args.brightness)
    contrast_factor = 1.0 + rng.uniform(-args.contrast, args.contrast)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    if rng.random() < args.grayscale_prob:
        image = ImageOps.grayscale(image).convert('RGB')
    return image, boxes


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    input_root = pathlib.Path(args.input_root)
    output_root = pathlib.Path(args.output_root)
    image_dir = input_root / 'images'
    label_dir = input_root / 'labels'
    out_image_dir = output_root / 'images'
    out_label_dir = output_root / 'labels'
    if output_root.exists():
        shutil.rmtree(output_root)
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(image_dir.rglob('*')):
        if not image_path.is_file() or image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            continue
        label_path = label_dir / image_path.relative_to(image_dir).with_suffix('.txt')
        with Image.open(image_path).convert('RGB') as image:
            width, height = image.size
            boxes = load_yolo_annotations(label_path, width, height)
            for loop_index in range(args.loops):
                augmented_image, augmented_boxes = augment_image(image.copy(), [dict(item) for item in boxes], args, rng)
                out_name = f'{image_path.stem}_{loop_index:03d}{image_path.suffix}'
                augmented_image.save(out_image_dir / out_name)
                lines = [to_yolo_line(item, width, height) for item in augmented_boxes]
                (out_label_dir / f'{image_path.stem}_{loop_index:03d}.txt').write_text(
                    '\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
