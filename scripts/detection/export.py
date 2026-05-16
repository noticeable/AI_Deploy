#!/usr/bin/env python

import argparse
import pathlib
import sys

import torch

try:
    from fvcore.common.checkpoint import Checkpointer
except Exception as exc:
    raise RuntimeError(
        'Failed to import fvcore.common.checkpoint.Checkpointer. '
        'Please install fvcore before exporting ONNX. '
        f'Original error: {exc}') from exc

from pytorch_image_classification.models.onnx_quant_export import make_float_onnx_path, quantize_onnx_model
from pytorch_object_detection import create_exporter, create_model, get_default_config, update_config
from pytorch_object_detection.models.qat import convert_qat_model, is_qat_enabled, prepare_model_for_qat
from pytorch_object_detection.utils.checkpoint import create_model_from_checkpoint


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


def load_model_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint.get('model', checkpoint)
    model.load_state_dict(model_state)


def load_export_model(config, checkpoint_path):
    if is_qat_enabled(config):
        model = create_model(config)
        model, _ = prepare_model_for_qat(config, model)
        if checkpoint_path is not None:
            Checkpointer(model).load(str(checkpoint_path))
        if config.qat.convert_before_export:
            model = convert_qat_model(config, model)
        return model, config

    if checkpoint_path is not None:
        model, config, _ = create_model_from_checkpoint(config, str(checkpoint_path), create_model)
        return model, config
    return create_model(config), config


def load_float_model_from_qat_checkpoint(config, checkpoint_path):
    float_model, loaded_config, _ = create_model_from_checkpoint(config, str(checkpoint_path), create_model)
    qat_model, _ = prepare_model_for_qat(loaded_config, float_model)
    Checkpointer(qat_model).load(str(checkpoint_path))
    restored_float_model, _, _ = create_model_from_checkpoint(loaded_config, str(checkpoint_path), create_model)
    restored_float_model.load_state_dict(qat_model.state_dict(), strict=False)
    return restored_float_model, loaded_config


def main():
    config = load_config()
    if str(getattr(config.eval, 'nms_type', 'hard')).lower() == 'soft':
        raise ValueError('scripts/detection/export.py requires eval.nms_type=hard because Soft-NMS is not supported in ONNX export.')
    checkpoint_path = pathlib.Path(config.export.checkpoint) if config.export.checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Export checkpoint not found: {checkpoint_path}. '
            'Please set export.checkpoint to a valid checkpoint path.')
    if checkpoint_path is None:
        print('WARNING: export.checkpoint is empty, exporting model with current in-memory weights.',
              file=sys.stderr)

    quantized_onnx = bool(getattr(config.export, 'quantized_onnx', False))
    if quantized_onnx and not is_qat_enabled(config):
        raise ValueError('export.quantized_onnx=True requires qat.enabled=True.')

    if quantized_onnx:
        if checkpoint_path is None:
            raise ValueError('export.quantized_onnx=True requires export.checkpoint to be set.')
        float_model, export_config = load_float_model_from_qat_checkpoint(config, checkpoint_path)
        output_file = export_config.export.output_file or 'detection_model.onnx'
        output_path = pathlib.Path(output_file)
        float_output_path = make_float_onnx_path(output_path)
        export_config.defrost()
        export_config.export.output_file = float_output_path.as_posix()
        export_config.freeze()
        exporter = create_exporter(export_config)
        exporter.export(float_model)
        quantize_onnx_model(float_output_path,
                            output_path,
                            getattr(config.export, 'quantized_onnx_backend', 'onnxruntime_dynamic'))
        return

    model, config = load_export_model(config, checkpoint_path)
    exporter = create_exporter(config)
    exporter.export(model)


if __name__ == '__main__':
    main()
