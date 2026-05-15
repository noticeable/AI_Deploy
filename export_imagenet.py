#!/usr/bin/env python

from evaluate import run_export
from pytorch_image_classification.tasks.classification import (
    ensure_task_family,
    load_export_config,
    parse_export_args,
)


def main():
    args = parse_export_args()
    config = load_export_config(args)
    ensure_task_family(config, 'imagenet')
    run_export(config)


if __name__ == '__main__':
    main()
