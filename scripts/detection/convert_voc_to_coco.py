#!/usr/bin/env python

import argparse
import pathlib

from pytorch_object_detection.datasets.converters import convert_voc_to_coco, discover_voc_classes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml-dir', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--image-suffix', type=str, default='jpg')
    parser.add_argument('--classes', nargs='*', default=None)
    parser.add_argument('--skip-unknown', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    xml_paths = sorted(pathlib.Path(args.xml_dir).glob('*.xml'))
    class_names = args.classes or discover_voc_classes(args.xml_dir)
    output_path = convert_voc_to_coco(xml_paths,
                                      args.output,
                                      class_names=class_names,
                                      image_suffix=args.image_suffix,
                                      skip_unknown=args.skip_unknown)
    print(output_path)


if __name__ == '__main__':
    main()
