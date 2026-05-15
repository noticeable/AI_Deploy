#!/usr/bin/env python

import argparse
import pathlib
import sys

try:
    import tensorrt as trt
except Exception as exc:
    raise RuntimeError(
        'Failed to import tensorrt. On Jetson Orin Nano, please install/use the JetPack-provided TensorRT runtime '
        'and Python bindings before building engines. '
        f'Original error: {exc}') from exc

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
except Exception as exc:
    raise RuntimeError(
        'Failed to import pycuda. Please install pycuda in the same environment used for TensorRT engine building. '
        f'Original error: {exc}') from exc

try:
    from PIL import Image
except Exception as exc:
    raise RuntimeError(
        'Failed to import Pillow. Please install pillow for INT8 calibration image loading. '
        f'Original error: {exc}') from exc

import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths, cache_path, batch_size, input_shape, input_layout):
        super().__init__()
        self.image_paths = image_paths
        self.cache_path = pathlib.Path(cache_path) if cache_path else None
        self.batch_size = batch_size
        self.channels, self.height, self.width = input_shape
        self.input_layout = input_layout
        self.current_index = 0
        self.device_input = cuda.mem_alloc(batch_size * self.channels * self.height * self.width * np.dtype(np.float32).itemsize)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[self.current_index:self.current_index + self.batch_size]
        if len(batch_paths) < self.batch_size:
            return None

        batch = np.stack([
            self._load_image(path) for path in batch_paths
        ], axis=0).astype(np.float32, copy=False)
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_path and self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(cache)

    def _load_image(self, path):
        image = Image.open(path).convert('RGB').resize((self.width, self.height))
        array = np.asarray(image).astype(np.float32) / 255.0
        if self.channels == 1:
            gray = image.convert('L').resize((self.width, self.height))
            array = np.asarray(gray).astype(np.float32) / 255.0
            array = np.expand_dims(array, axis=-1)

        if self.input_layout == 'chw':
            array = np.transpose(array, (2, 0, 1))
        return array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True)
    parser.add_argument('--engine', type=str, required=True)
    parser.add_argument('--precision', type=str, choices=['fp16', 'int8'], default='fp16')
    parser.add_argument('--workspace-mb', type=int, default=1024)
    parser.add_argument('--min-shape', nargs=4, type=int, metavar=('N', 'C', 'H', 'W'), default=None)
    parser.add_argument('--opt-shape', nargs=4, type=int, metavar=('N', 'C', 'H', 'W'), default=None)
    parser.add_argument('--max-shape', nargs=4, type=int, metavar=('N', 'C', 'H', 'W'), default=None)
    parser.add_argument('--calib-dir', type=str, default='')
    parser.add_argument('--calib-cache', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--input-name', type=str, default='images')
    parser.add_argument('--input-layout', type=str, choices=['chw', 'hwc'], default='chw')
    parser.add_argument('--input-size', nargs=2, type=int, metavar=('H', 'W'), default=None)
    parser.add_argument('--input-channels', type=int, default=3)
    return parser.parse_args()


def collect_image_paths(calib_dir):
    calib_path = pathlib.Path(calib_dir)
    if not calib_path.exists():
        raise FileNotFoundError(f'Calibration directory not found: {calib_path}')
    image_paths = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(sorted(calib_path.glob(pattern)))
    if not image_paths:
        raise RuntimeError(f'No calibration images found in: {calib_path}')
    return image_paths


def parse_onnx(onnx_path, builder):
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(pathlib.Path(onnx_path).read_bytes()):
        errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'Failed to parse ONNX: {onnx_path}\n{errors}')
    return network


def configure_profile(builder, config, network, args):
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    profile = builder.create_optimization_profile()

    if args.min_shape and args.opt_shape and args.max_shape:
        min_shape = tuple(args.min_shape)
        opt_shape = tuple(args.opt_shape)
        max_shape = tuple(args.max_shape)
    else:
        static_shape = tuple(input_tensor.shape)
        if any(dim <= 0 for dim in static_shape):
            raise RuntimeError(
                'Model uses dynamic input shape. Please provide --min-shape --opt-shape --max-shape explicitly.')
        min_shape = opt_shape = max_shape = static_shape

    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    return input_name, min_shape, opt_shape, max_shape


def resolve_calibrator_shape(network, args):
    input_tensor = network.get_input(0)
    shape = tuple(input_tensor.shape)
    if len(shape) != 4:
        raise RuntimeError(f'Expected 4D input tensor for image classification, got shape: {shape}')

    channels = args.input_channels if shape[1] <= 0 else shape[1]
    if args.input_size is not None:
        height, width = args.input_size
    else:
        if shape[2] <= 0 or shape[3] <= 0:
            raise RuntimeError('Please provide --input-size H W for dynamic ONNX inputs when using INT8 calibration.')
        height, width = shape[2], shape[3]
    return channels, height, width


def main():
    args = parse_args()
    onnx_path = pathlib.Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f'ONNX file not found: {onnx_path}')

    engine_path = pathlib.Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    builder = trt.Builder(TRT_LOGGER)
    network = parse_onnx(onnx_path, builder)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb * 1024 * 1024)

    input_name, min_shape, opt_shape, max_shape = configure_profile(builder, config, network, args)
    if args.input_name and args.input_name != input_name:
        print(
            f'WARNING: requested --input-name={args.input_name!r} but ONNX input is {input_name!r}; using ONNX input name.',
            file=sys.stderr)

    if args.precision == 'fp16':
        if not builder.platform_has_fast_fp16:
            print('WARNING: platform_has_fast_fp16 is false; TensorRT will still try to build.', file=sys.stderr)
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        if not builder.platform_has_fast_int8:
            print('WARNING: platform_has_fast_int8 is false; TensorRT INT8 may not be efficient on this platform.', file=sys.stderr)
        if not args.calib_dir:
            raise ValueError('INT8 build requires --calib-dir')
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator_shape = resolve_calibrator_shape(network, args)
        image_paths = collect_image_paths(args.calib_dir)
        if len(image_paths) < args.batch_size:
            raise RuntimeError(
                f'Calibration images ({len(image_paths)}) must be at least batch size ({args.batch_size}).')
        config.int8_calibrator = ImageEntropyCalibrator(
            image_paths=image_paths,
            cache_path=args.calib_cache,
            batch_size=args.batch_size,
            input_shape=calibrator_shape,
            input_layout=args.input_layout,
        )

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError('TensorRT engine build failed.')

    engine_path.write_bytes(serialized_engine)
    print(f'OK | saved TensorRT engine to {engine_path}')
    print(f'INFO | input_name={input_name} | precision={args.precision} | min_shape={min_shape} | opt_shape={opt_shape} | max_shape={max_shape}')
    if args.precision == 'int8' and args.calib_cache:
        print(f'INFO | calibration cache={args.calib_cache}')


if __name__ == '__main__':
    main()
