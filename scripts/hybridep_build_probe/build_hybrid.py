# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build only DeepEP's HybridEP extension for environment probing."""

import importlib.util
import os
from pathlib import Path

import pybind11
import setuptools
from torch.utils.cpp_extension import BuildExtension

source_root = Path.cwd()
spec = importlib.util.spec_from_file_location("deepep_setup", source_root / "setup.py")
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {source_root / 'setup.py'}")
deepep_setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepep_setup)

extension = deepep_setup.get_extension_hybrid_ep_cpp()
extension.include_dirs.insert(0, pybind11.get_include())
extension.include_dirs.insert(0, str(Path(os.environ["CUDA_HOME"]) / "include" / "cccl"))

setuptools.setup(
    name="hybrid_ep_probe",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)
