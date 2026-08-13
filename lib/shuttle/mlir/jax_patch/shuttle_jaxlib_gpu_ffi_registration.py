# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect the rebuilt CUDA plugin's exported Shuttle typed-FFI bundle."""

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

TARGET = "shuttle.gpu.executable_bundle.v1"
STAGES = ("instantiate", "prepare", "initialize", "execute")


def load_extension(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cuda_plugin_extension", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CUDA plugin extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-extension", type=Path, required=True)
    args = parser.parse_args()

    module = load_extension(args.plugin_extension.resolve())
    handlers = module.ffi_handlers()
    if TARGET not in handlers:
        raise RuntimeError(f"missing Shuttle CUDA typed-FFI target: {sorted(handlers)}")
    bundle = handlers[TARGET]
    if set(bundle) != {*STAGES, "api_version", "traits"}:
        raise RuntimeError("CUDA plugin exported an unknown Shuttle handler field")
    if bundle["api_version"] != 1 or bundle["traits"] != 0:
        raise RuntimeError("CUDA plugin exported the wrong Shuttle typed-FFI metadata")
    if any(bundle[stage] is None for stage in STAGES):
        raise RuntimeError("CUDA plugin did not export one complete Shuttle handler bundle")


if __name__ == "__main__":
    main()
