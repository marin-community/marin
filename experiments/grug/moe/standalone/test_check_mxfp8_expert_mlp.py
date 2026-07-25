# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the MXFP8 accelerator diagnostic entry point."""

import json
import subprocess
import sys
from pathlib import Path


def test_environment_sentinel_is_machine_readable_before_numerics():
    script = Path(__file__).with_name("check_mxfp8_expert_mlp.py")
    result = subprocess.run(
        [sys.executable, str(script), "--sentinel-only"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    prefix = "CUTLASS_ENV_SENTINEL "
    assert lines[0].startswith(prefix)
    sentinel = json.loads(lines[0][len(prefix) :])
    assert "cuda_toolkit_path" in sentinel
    assert "cutlass_module_path" in sentinel
    assert "cutlass_payload" in sentinel
    assert "libnvvm_path" in sentinel
    assert "nvidia_cutlass_dsl_module_path" in sentinel
    assert "distributions" in sentinel
