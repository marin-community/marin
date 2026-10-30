# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit job-private instrumentation before launching the shipped TPU runtime."""

import argparse
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    preflight = runpy.run_path(str(HERE / "config_preflight.py"))["check_config"](config)
    patches = [runpy.run_path(str(HERE / "trace_bootstrap.py"))["patch_trace"]()]
    if config["diagnostic_v4_tiles"]:
        if config["hardware"] != "v4-32":
            raise ValueError("Experimental tiles apply only to a diagnostic v4 arm")
        root = Path(importlib.util.find_spec("tpu_inference").origin).parent
        patches.append(
            runpy.run_path(str(HERE / "patch_v4_tiles.py"))["patch"](
                root / "kernels/ragged_paged_attention/v3/kernel.py"
            )
        )
    Path(config["runtime_report_path"]).write_text(json.dumps(dict(preflight=preflight, patches=patches)))
    command, *arguments = args.command
    # Preflight imports backend modules. Exec a fresh interpreter so serving
    # reads the instrumented files instead of those cached module objects.
    if command == "python":
        os.execv(sys.executable, [sys.executable, *arguments])
    elif command == "vllm":
        os.execv(sys.executable, [sys.executable, str(Path(sys.executable).with_name("vllm")), *arguments])
    else:
        raise ValueError(f"Unsupported entrypoint {command}")


if __name__ == "__main__":
    main()
