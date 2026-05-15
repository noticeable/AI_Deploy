#!/usr/bin/env python

import pathlib
import sys

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

from pytorch_image_classification import create_model
from pytorch_image_classification.models.onnx_quant_export import make_float_onnx_path, quantize_onnx_model
from pytorch_image_classification.models.qat import convert_qat_model, is_qat_enabled, prepare_model_for_qat
from pytorch_image_classification.tasks.classification import load_export_config, parse_export_args


def load_config():
    args = parse_export_args()
    return load_export_config(args)


def _get_model_device(model):
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device('cpu')


def _export_onnx_model(config, model, output_path):
    model.eval()
    dummy = torch.randn(1,
                        config.dataset.n_channels,
                        config.dataset.image_size,
                        config.dataset.image_size,
                        device=_get_model_device(model))
    dynamic_axes = None
    if config.export.dynamic_axes:
        dynamic_axes = {
            'images': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        }
    torch.onnx.export(model,
                      dummy,
                      output_path.as_posix(),
                      opset_version=config.export.opset,
                      input_names=['images'],
                      output_names=['logits'],
                      dynamic_axes=dynamic_axes)


def main():
    config = load_config()
    model = create_model(config)
    if is_qat_enabled(config):
        model, _ = prepare_model_for_qat(config, model)
    if config.export.checkpoint:
        checkpoint_path = pathlib.Path(config.export.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f'Export checkpoint not found: {checkpoint_path}. '
                'Please set export.checkpoint to a valid checkpoint path.')
        Checkpointer(model).load(config.export.checkpoint)
    else:
        print('WARNING: export.checkpoint is empty, exporting model with current in-memory weights.',
              file=sys.stderr)

    quantized_onnx = bool(getattr(config.export, 'quantized_onnx', False))
    if quantized_onnx and not is_qat_enabled(config):
        raise ValueError('export.quantized_onnx=True requires qat.enabled=True.')

    output_file = config.export.output_file or 'image_classification_model.onnx'
    output_path = pathlib.Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    if quantized_onnx:
        float_output_path = make_float_onnx_path(output_path)
        float_model = create_model(config)
        Checkpointer(float_model).load(config.export.checkpoint)
        _export_onnx_model(config, float_model, float_output_path)
        quantize_onnx_model(float_output_path,
                            output_path,
                            getattr(config.export, 'quantized_onnx_backend', 'onnxruntime_dynamic'))
        return

    if is_qat_enabled(config) and config.qat.convert_before_export:
        model = convert_qat_model(config, model)

    _export_onnx_model(config, model, output_path)


if __name__ == '__main__':
    main()
