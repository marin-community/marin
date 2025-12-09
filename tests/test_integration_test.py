# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import tempfile

import pytest


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip this test in CI, since we run it as a separate worflow.")
def test_integration_test_run():
    MARIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    """Test the dry runs of experiment scripts"""
    # Emulate running tests/integration_test.py on the cmdline
    # don't name dir `test` because tokenizer gets mad that you're trying to train on test
    with tempfile.TemporaryDirectory(prefix="executor-integration") as temp_dir:
        result = subprocess.run(
            ["python", os.path.join(MARIN_ROOT, "tests/integration_test.py"), "--prefix", temp_dir],
            capture_output=False,
            text=True,
        )
    assert result.returncode == 0, f"Integration test run failed: {result.stderr}"
