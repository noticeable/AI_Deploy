#!/usr/bin/env python

import argparse

from pytorch_object_detection.datasets.converters import convert_yolo_to_coco, discover_images, split_paths, split_paths_from_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', required=True, type=str)
    parser.add_argument('--output', type=str, default='annotations/instances_all.json')
    parser.add_argument('--classes', type=str, default='')
    parser.add_argument('--split-mode', choices=['all', 'random', 'files'], default='all')
    parser.add_argument('--train-output', type=str, default='annotations/train.json')
    parser.add_argument('--val-output', type=str, default='annotations/val.json')
    parser.add_argument('--test-output', type=str, default='annotations/test.json')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-file', type=str, default='')
    parser.add_argument('--val-file', type=str, default='')
    parser.add_argument('--test-file', type=str, default='')
    return parser.parse_args()


def main():
    args = parse_args()
    classes_path = args.classes or None
    if args.split_mode == 'all':
        output_path = convert_yolo_to_coco(args.root_dir, args.output, classes_path=classes_path)
        print(output_path)
        return

    image_paths = discover_images(f'{args.root_dir}/images')
    if args.split_mode == 'random':
        train_images, val_images, test_images = split_paths(
            image_paths,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        train_names, val_names, test_names = split_paths_from_files(
            args.root_dir,
            train_file=args.train_file or None,
            val_file=args.val_file or None,
            test_file=args.test_file or None,
        )
        train_images = [path for path in image_paths if path.name in train_names]
        val_images = [path for path in image_paths if path.name in val_names]
        test_images = [path for path in image_paths if path.name in test_names]

    for subset, output in ((train_images, args.train_output),
                           (val_images, args.val_output),
                           (test_images, args.test_output)):
        output_path = convert_yolo_to_coco(args.root_dir,
                                           output,
                                           classes_path=classes_path,
                                           image_paths=subset)
        print(output_path)


if __name__ == '__main__':
    main()
