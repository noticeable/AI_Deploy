#!/usr/bin/env python

import argparse
import pathlib

import numpy as np
import torch
from PIL import Image

from pytorch_object_detection import apply_data_parallel_wrapper, create_model, get_default_config, update_config
from pytorch_object_detection.utils.checkpoint import create_model_from_checkpoint
from pytorch_object_detection.utils.visualize import draw_detections


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


def main():
    config, input_path, output_path = load_config()
    if config.test.checkpoint:
        model, config, _ = create_model_from_checkpoint(config, config.test.checkpoint, create_model)
    else:
        model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)
    model.eval()

    image = Image.open(input_path).convert('RGB').resize((config.dataset.image_size, config.dataset.image_size))
    array = np.asarray(image).astype('float32') / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(config.device)
    detections = model(tensor)[0]
    draw_detections(input_path,
                    {
                        'boxes': detections['boxes'].detach().cpu().tolist(),
                        'scores': detections['scores'].detach().cpu().tolist(),
                        'labels': detections['labels'].detach().cpu().tolist(),
                        'candidate_boxes': detections.get('candidate_boxes', torch.empty(0, 4)).detach().cpu().tolist(),
                        'candidate_scores': detections.get('candidate_scores', torch.empty(0)).detach().cpu().tolist(),
                        'candidate_labels': detections.get('candidate_labels', torch.empty(0, dtype=torch.long)).detach().cpu().tolist(),
                    },
                    output_path,
                    class_names=config.dataset.class_names)


if __name__ == '__main__':
    main()
