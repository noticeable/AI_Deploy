#!/usr/bin/env python

from pytorch_image_classification.tasks.classification import (
    ensure_task_family,
    load_train_config,
    parse_train_args,
)
from train import run_training


def main():
    args = parse_train_args()
    config = load_train_config(args)
    ensure_task_family(config, 'imagenet')
    run_training(config)


if __name__ == '__main__':
    main()
