#!/usr/bin/env python

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = pathlib.Path.home() / '.conda' / 'envs' / 'pytorch310' / ('python.exe' if os.name == 'nt' else 'bin/python')
PYTHON = str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else pathlib.Path(sys.executable))
SMOKES = [
    'tools/smoke_test_qat_classification.py',
    'tools/smoke_test_qat_detection.py',
    'tools/smoke_test_qat_det_seg.py',
]


def run_smoke(script_path: str):
    command = [PYTHON, script_path]
    result = subprocess.run(command,
                            cwd=ROOT,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)
    return result.returncode, result.stdout


def main():
    passed = []
    failed = []

    for script in SMOKES:
        print(f'=== RUN {script} ===')
        returncode, output = run_smoke(script)
        print(output.rstrip())
        if returncode == 0:
            passed.append(script)
        else:
            failed.append(script)

    print('=== QAT SMOKE SUMMARY ===')
    print(f'passed={len(passed)} failed={len(failed)} total={len(SMOKES)}')
    if passed:
        print('passed_scripts=')
        for script in passed:
            print(f'  - {script}')
    if failed:
        print('failed_scripts=')
        for script in failed:
            print(f'  - {script}')
        sys.exit(1)

    print('OK | all qat smoke tests passed')


if __name__ == '__main__':
    main()
