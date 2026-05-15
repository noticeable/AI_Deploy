#!/usr/bin/env python

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--onnx', type=str, required=True)
    parser.add_argument('--engine', type=str, required=True)
    parser.add_argument('--precision', type=str, choices=['fp16', 'int8'], default='int8')
    parser.add_argument('--calib-dir', type=str, default='')
    parser.add_argument('--calib-cache', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workspace-mb', type=int, default=1024)
    parser.add_argument('--min-shape', nargs=4, type=int, metavar=('N', 'C', 'H', 'W'), default=None)
    parser.add_argument('--opt-shape', nargs=4, type=int, metavar=('N', 'C', 'H', 'W'), default=None)
    parser.add_argument('--max-shape', nargs=4, type=int, metavar=('N', 'C', 'H', 'W'), default=None)
    parser.add_argument('options', nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def run(command):
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    args = parse_args()

    export_command = [
        PYTHON,
        'export.py',
        '--config',
        args.config,
        'export.output_file',
        args.onnx,
    ]
    if args.checkpoint:
        export_command.extend(['export.checkpoint', args.checkpoint])
    export_command.extend(args.options)
    run(export_command)

    build_command = [
        PYTHON,
        'tools/build_tensorrt_engine.py',
        '--onnx',
        args.onnx,
        '--engine',
        args.engine,
        '--precision',
        args.precision,
        '--workspace-mb',
        str(args.workspace_mb),
        '--input-name',
        'images',
        '--input-layout',
        'chw',
    ]

    if args.min_shape and args.opt_shape and args.max_shape:
        build_command.extend(['--min-shape', *[str(v) for v in args.min_shape]])
        build_command.extend(['--opt-shape', *[str(v) for v in args.opt_shape]])
        build_command.extend(['--max-shape', *[str(v) for v in args.max_shape]])

    if args.precision == 'int8':
        if not args.calib_dir:
            raise ValueError('INT8 build requires --calib-dir')
        build_command.extend([
            '--calib-dir',
            args.calib_dir,
            '--batch-size',
            str(args.batch_size),
        ])
        if args.calib_cache:
            build_command.extend(['--calib-cache', args.calib_cache])

    run(build_command)


if __name__ == '__main__':
    main()
