#!/usr/bin/env python

import argparse
import json
import pathlib

import torch

from pytorch_det_seg import apply_data_parallel_wrapper, create_dataloader, create_evaluator, create_model, get_default_config, update_config
from pytorch_det_seg.utils import create_logger, create_model_from_checkpoint



def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    config = update_config(config)
    config.freeze()
    return config



def main():
    config = load_config()
    dataloader = create_dataloader(config, is_train=False)
    if config.test.checkpoint:
        model, config, _ = create_model_from_checkpoint(config, config.test.checkpoint, create_model)
    else:
        model = create_model(config)
    output_dir = pathlib.Path(config.test.output_dir or config.train.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = create_logger(__name__, distributed_rank=0, output_dir=output_dir)
    model = apply_data_parallel_wrapper(config, model)
    model.eval()
    device = torch.device(config.device)
    evaluator = create_evaluator(config)
    all_predictions = None
    all_targets = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            predictions = model(images)
            if all_predictions is None:
                all_predictions = {'detections': [], 'segmentation': []}
            all_predictions['detections'].extend(predictions['detections'])
            all_predictions['segmentation'].append(predictions['segmentation'].cpu())
            all_targets.extend(targets)
    if all_predictions is None:
        all_predictions = {'detections': [], 'segmentation': torch.zeros((0,), dtype=torch.long)}
    else:
        all_predictions['segmentation'] = torch.cat(all_predictions['segmentation'], dim=0)
    metrics = evaluator.evaluate(all_predictions, all_targets)
    result_path = output_dir / 'det_seg_metrics.json'
    result_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    logger.info(metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
