# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Patched cho PyTorch 2.x: torch._six da bi xoa hoan toan tu PyTorch 1.13+.
# Python 3.7+ co importlib.util day du -> luon dung nhanh nay.
import importlib
import importlib.util
import sys


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
def import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module
