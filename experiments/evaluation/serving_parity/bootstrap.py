# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Instrument an isolated TPU environment; v4 adaptation never changes model config."""
import importlib.util
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if os.environ["PARITY_TPU_VARIANT"] == "v4-32":
    root = Path(importlib.util.find_spec("tpu_inference").origin).parent
    runpy.run_path(str(HERE / "patch_v4_tiles.py"))["patch"](root / "kernels/ragged_paged_attention/v3/kernel.py")
runpy.run_path(str(HERE / "trace_bootstrap.py"), run_name="parity_trace")
if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    if command == "python":
        path, *arguments = arguments
        sys.argv = [path, *arguments]
        runpy.run_path(path, run_name="__main__")
    elif command == "vllm":
        from vllm.entrypoints.cli.main import main

        sys.argv = ["vllm", *arguments]
        main()
    else:
        raise ValueError(f"Unsupported entrypoint {command}")
