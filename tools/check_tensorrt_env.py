#!/usr/bin/env python

import importlib
import pathlib
import shutil
import subprocess
import sys


CHECKS = [
    ('tensorrt', 'TensorRT Python binding'),
    ('pycuda', 'PyCUDA package'),
    ('PIL', 'Pillow package'),
    ('onnx', 'ONNX package'),
]


def check_python_module(module_name, label):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        return 'OK', f'{label}: import ok (version={version})'
    except Exception as exc:
        return 'MISSING', f'{label}: {exc}'


def check_command(command):
    command_path = shutil.which(command)
    if not command_path:
        return 'MISSING', f'{command}: not found in PATH'
    try:
        result = subprocess.run([command, '--version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else 'version output unavailable'
        return 'OK', f'{command}: {command_path} | {first_line}'
    except Exception as exc:
        return 'WARN', f'{command}: found at {command_path}, but version check failed: {exc}'


def check_jetson_release():
    release_file = pathlib.Path('/etc/nv_tegra_release')
    if release_file.exists():
        try:
            return 'OK', f'Jetson release: {release_file.read_text().strip()}'
        except Exception as exc:
            return 'WARN', f'Jetson release file exists but could not be read: {exc}'
    return 'WARN', 'Jetson release file not found; this may not be a Jetson device or JetPack is not installed.'


def check_cuda_visible():
    try:
        import pycuda.driver as cuda
        cuda.init()
        device_count = cuda.Device.count()
        if device_count <= 0:
            return 'WARN', 'CUDA driver initialized but no visible GPU devices were found.'
        names = [cuda.Device(index).name() for index in range(device_count)]
        return 'OK', f'CUDA devices: {names}'
    except Exception as exc:
        return 'MISSING', f'CUDA device check failed: {exc}'


def main():
    statuses = []

    for module_name, label in CHECKS:
        statuses.append(check_python_module(module_name, label))

    statuses.append(check_command('trtexec'))
    statuses.append(check_jetson_release())
    statuses.append(check_cuda_visible())

    ok_count = sum(status == 'OK' for status, _ in statuses)
    warn_count = sum(status == 'WARN' for status, _ in statuses)
    missing_count = sum(status == 'MISSING' for status, _ in statuses)

    print(f'Python executable: {sys.executable}')
    print(f'Python version: {sys.version.split()[0]}')
    print('')
    for status, message in statuses:
        print(f'{status:<7} | {message}')

    print('')
    print(f'SUMMARY | ok={ok_count} | warn={warn_count} | missing={missing_count} | total={len(statuses)}')

    if missing_count:
        sys.exit(1)


if __name__ == '__main__':
    main()
