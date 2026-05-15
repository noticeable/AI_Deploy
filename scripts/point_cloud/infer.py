#!/usr/bin/env python

import argparse
import pathlib

import numpy as np
import torch
from fvcore.common.checkpoint import Checkpointer

from pytorch_point_cloud import (
    apply_data_parallel_wrapper,
    create_model,
    create_model_from_checkpoint,
    get_default_config,
    update_config,
)



def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    config = update_config(config)
    config.freeze()
    return config, pathlib.Path(args.input), pathlib.Path(args.output)



def normalize(points):
    centroid = np.mean(points[:, :3], axis=0, keepdims=True)
    points[:, :3] -= centroid
    scale = np.max(np.linalg.norm(points[:, :3], axis=1))
    if scale > 0:
        points[:, :3] /= scale
    return points.astype(np.float32)



def main():
    config, input_path, output_path = load_config()
    if config.test.checkpoint:
        model, config, _ = create_model_from_checkpoint(config, config.test.checkpoint, create_model)
    else:
        model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)
    model.eval()

    points = np.load(input_path)
    if points.ndim == 2:
        points = points[None, ...]
    points = np.stack([normalize(sample.copy()) for sample in points], axis=0)
    tensor = torch.from_numpy(points).to(config.device)

    with torch.no_grad():
        outputs = model(tensor)

    if config.task == 'classification':
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        preds = logits.argmax(dim=1).cpu().numpy()
        np.savez(output_path, logits=logits.cpu().numpy(), preds=preds)
    else:
        seg_logits = outputs['seg_logits']
        seg_preds = seg_logits.argmax(dim=-1).cpu().numpy()
        np.savez(output_path, seg_logits=seg_logits.cpu().numpy(), seg_preds=seg_preds)


if __name__ == '__main__':
    main()
