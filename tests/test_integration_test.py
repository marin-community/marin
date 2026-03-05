# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import tempfile

import pytest


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip this test in CI, since we run it as a separate worflow.")
def test_integration_test_run():
    MARIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    """Test the dry runs of experiment scripts"""
    # Emulate running tests/integration_test.py on the cmdline
    # don't name dir `test` because tokenizer gets mad that you're trying to train on test
    with tempfile.TemporaryDirectory(prefix="executor-integration") as temp_dir:
        # Use -P to prevent the script's parent dir (tests/) from being prepended
        # to sys.path, which would cause tests/levanter/ to shadow src/levanter/.
        result = subprocess.run(
            [sys.executable, "-P", os.path.join(MARIN_ROOT, "tests/integration_test.py"), "--prefix", temp_dir],
            cwd=MARIN_ROOT,
            capture_output=False,
            text=True,
        )
    assert result.returncode == 0, f"Integration test run failed: {result.stderr}"
