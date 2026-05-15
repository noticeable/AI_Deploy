#!/usr/bin/env python

import argparse
import pathlib
import sys

import torch.nn as nn

try:
    import torch
except Exception as exc:
    raise RuntimeError(
        'Failed to import torch. Please install a working PyTorch environment before exporting ONNX. '
        f'Original error: {exc}') from exc

try:
    from fvcore.common.checkpoint import Checkpointer
except Exception as exc:
    raise RuntimeError(
        'Failed to import fvcore.common.checkpoint.Checkpointer. '
        'Please install fvcore before exporting ONNX. '
        f'Original error: {exc}') from exc

from pytorch_point_cloud import create_model, create_model_from_checkpoint, get_default_config, update_config
from pytorch_point_cloud.models.qat import convert_qat_model, is_qat_enabled, is_qat_supported_model, prepare_model_for_qat
from pytorch_image_classification.models.onnx_quant_export import make_float_onnx_path, patch_segmentation_qdq_input, quantize_onnx_model


class PointCloudExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, points):
        outputs = self.model(points)
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                return outputs['logits']
            if 'seg_logits' in outputs:
                return outputs['seg_logits']
            first_value = next(iter(outputs.values()))
            return first_value
        return outputs


def export_onnx_model(config, model, output_path):
    model.eval()
    export_model = PointCloudExportWrapper(model)
    dummy = torch.randn(1, config.dataset.num_points, config.dataset.n_channels,
                        device=next(model.parameters()).device)
    dynamic_axes = None
    if config.export.dynamic_axes:
        dynamic_axes = {
            'points': {0: 'batch_size', 1: 'num_points'},
            'outputs': {0: 'batch_size'},
        }
        model_outputs = export_model(dummy)
        if model_outputs.ndim >= 3:
            dynamic_axes['outputs'][1] = 'num_points'
    torch.onnx.export(export_model,
                      dummy,
                      output_path.as_posix(),
                      opset_version=config.export.opset,
                      input_names=['points'],
                      output_names=['outputs'],
                      dynamic_axes=dynamic_axes)


def load_float_model_from_qat_checkpoint(config, checkpoint_path):
    model = create_model(config)
    model, _ = prepare_model_for_qat(config, model)
    Checkpointer(model).load(str(checkpoint_path))
    if bool(getattr(config.qat, 'convert_before_export', True)):
        _ = convert_qat_model(config, model)
    float_model = create_model(config)
    Checkpointer(float_model).load(str(checkpoint_path))
    return float_model


def should_patch_segmentation_quantized_onnx(config):
    return (getattr(config, 'task', None) == 'segmentation' and
            ((config.model.type == 'dgcnn' and config.model.name == 'dgcnn_seg') or
             (config.model.type == 'pointnet2' and config.model.name == 'pointnet2_seg')))


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
    checkpoint = config.export.checkpoint or config.test.checkpoint or config.train.checkpoint
    checkpoint_path = pathlib.Path(checkpoint) if checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Export checkpoint not found: {checkpoint_path}. '
            'Please set export.checkpoint/test.checkpoint/train.checkpoint to a valid checkpoint path.')
    if checkpoint_path is None:
        print('WARNING: no export/test/train checkpoint configured, exporting model with current in-memory weights.',
              file=sys.stderr)

    quantized_onnx = bool(getattr(config.export, 'quantized_onnx', False))
    if quantized_onnx:
        if not is_qat_enabled(config):
            raise ValueError('export.quantized_onnx=True requires qat.enabled=True.')
        if not is_qat_supported_model(config):
            raise ValueError(f'QAT is not supported for point-cloud model {(config.model.type, config.model.name)}')
        if checkpoint_path is None:
            raise ValueError('export.quantized_onnx=True requires export.checkpoint/test.checkpoint/train.checkpoint to be set.')
        output_file = config.export.output_file
        if not output_file:
            output_file = pathlib.Path(config.test.output_dir or config.train.output_dir) / 'point_cloud_model.onnx'
        output_path = pathlib.Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        float_output_path = make_float_onnx_path(output_path)
        float_model = load_float_model_from_qat_checkpoint(config, checkpoint_path)
        export_onnx_model(config, float_model, float_output_path)
        quantize_onnx_model(float_output_path,
                            output_path,
                            getattr(config.export, 'quantized_onnx_backend', 'onnxruntime_dynamic'))
        if should_patch_segmentation_quantized_onnx(config):
            patch_segmentation_qdq_input(output_path, input_name='points')
        return

    model = create_model(config)
    if checkpoint_path is not None:
        model, config, _ = create_model_from_checkpoint(config, str(checkpoint_path), create_model)
    output_file = config.export.output_file
    if not output_file:
        output_file = pathlib.Path(config.test.output_dir or config.train.output_dir) / 'point_cloud_model.onnx'
    output_path = pathlib.Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    export_onnx_model(config, model, output_path)


if __name__ == '__main__':
    main()
