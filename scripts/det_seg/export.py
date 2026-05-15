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

from pytorch_det_seg import create_exporter, create_model, get_default_config, update_config
from pytorch_det_seg.models.qat import convert_qat_model, is_qat_enabled, prepare_model_for_qat
from pytorch_det_seg.utils import create_model_from_checkpoint
from pytorch_image_classification.models.onnx_quant_export import make_float_onnx_path, quantize_onnx_model


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
    model = create_model(config)
    model, _ = prepare_model_for_qat(config, model)
    Checkpointer(model).load(str(checkpoint_path))
    float_model = create_model(config)
    float_model.load_state_dict(model.state_dict(), strict=False)
    return float_model


def main():
    config = load_config()
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
        float_model = load_float_model_from_qat_checkpoint(config, checkpoint_path)
        output_file = config.export.output_file or 'det_seg_model.onnx'
        output_path = pathlib.Path(output_file)
        float_output_path = make_float_onnx_path(output_path)
        config.defrost()
        config.export.output_file = float_output_path.as_posix()
        config.freeze()
        exporter = create_exporter(config)
        exporter.export(float_model)
        quantize_onnx_model(float_output_path,
                            output_path,
                            getattr(config.export, 'quantized_onnx_backend', 'onnxruntime_dynamic'))
        print(f'OK | det-seg quantized onnx export | float={float_output_path} quantized={output_path}')
        return

    model, config = load_export_model(config, checkpoint_path)
    exporter = create_exporter(config)
    output_path = exporter.export(model)
    print(f'OK | det-seg export | output={output_path}')


if __name__ == '__main__':
    main()
