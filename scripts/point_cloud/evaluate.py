#!/usr/bin/env python

import argparse
import pathlib
import time

import numpy as np
import torch

from fvcore.common.checkpoint import Checkpointer

from pytorch_point_cloud import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    create_model_from_checkpoint,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import AverageMeter, create_logger



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



def evaluate(config, model, test_loader, loss_func, logger):
    model.eval()
    device = torch.device(config.device)
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()
    logits_all = []
    start = time.time()

    with torch.no_grad():
        for points, targets in test_loader:
            points = points.to(device)
            if config.task == 'classification':
                targets = targets.to(device)
            else:
                targets = {k: v.to(device) for k, v in targets.items()}
            outputs = model(points)
            loss = loss_func(outputs, targets)
            loss_meter.update(loss.item(), points.size(0))

            if config.task == 'classification':
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                preds = logits.argmax(dim=1)
                metric = preds.eq(targets).float().mean().item()
                logits_all.append(logits.cpu().numpy())
            else:
                seg_logits = outputs['seg_logits']
                seg_preds = seg_logits.argmax(dim=-1)
                metric = seg_preds.eq(targets['seg_label']).float().mean().item()
                logits_all.append(seg_logits.cpu().numpy())
            metric_meter.update(metric, points.size(0))

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')
    logger.info(f'Loss {loss_meter.avg:.4f} Metric {metric_meter.avg:.4f}')
    return np.concatenate(logits_all), loss_meter.avg, metric_meter.avg



def main():
    config = load_config()
    output_dir = pathlib.Path(config.test.output_dir or config.train.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = create_logger(name=__name__, distributed_rank=0, output_dir=output_dir)

    if config.test.checkpoint:
        model, config, _ = create_model_from_checkpoint(config, config.test.checkpoint, create_model)
    else:
        model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)

    test_loader = create_dataloader(config, is_train=False)
    _, test_loss = create_loss(config)
    preds, loss, metric = evaluate(config, model, test_loader, test_loss, logger)
    np.savez(output_dir / 'predictions.npz', preds=preds, loss=loss, metric=metric)


if __name__ == '__main__':
    main()
