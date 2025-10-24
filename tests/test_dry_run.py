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

import logging
import os
import runpy
import sys
import tempfile
from pathlib import Path

import pytest
from gcsfs.retry import HttpError

from tests.test_utils import parameterize_with_configs

logger = logging.getLogger(__name__)


marin_root = Path(__file__).parent.parent
experiments_dir = marin_root / "experiments"

os.environ["RAY_LOCAL_CLUSTER"] = "1"


@parameterize_with_configs(pattern="**/*.py", config_path=str(experiments_dir))
def test_run_dry_runs(config_file, monkeypatch):
    """Test the dry runs of experiment scripts"""
    script = config_file
    # first get this script path
    # Check if the script contains the skip marker
    with open(script, "r") as file:
        content = file.read()

        # Hack: Skip files that don't seem to call executor_main
        if "executor_main(" not in content:
            pytest.skip(f"Skipping {script} (no executor_main)")

        if "nodryrun" in content:
            pytest.skip(f"Skipping {script} (contains nodryrun marker)")

    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        monkeypatch.setattr(
            sys, "argv", [script, "--dry_run", "True", "--executor_info_base_path", temp_dir, "--prefix", temp_dir]
        )
        try:
            runpy.run_path(script, run_name="__main__")
        except HttpError as e:
            # Skip if this experiment needs GCS access
            if "Anonymous caller does not have" in str(e) or "Permission" in str(e):
                pytest.skip(f"Skipping {script} (requires GCS access)")
            raise
        except OSError as e:
            # Hugging Face sometimes surfaces gated repo access as OSError
            msg = str(e)
            lower_msg = msg.lower()
            if (
                "gated repo" in lower_msg
                or "access to model" in lower_msg
                or "401 client error" in lower_msg
                or "you must have access" in lower_msg
            ):
                pytest.skip(f"Skipping {script} (requires access to a gated Hugging Face repo): {e}")
            raise
        except SystemExit as e:
            # Some scripts may call `sys.exit()`, so we catch it to treat `exit(0)` as success
            assert e.code == 0, f"Dry run failed with exit code {e.code} for {script}"
        except Exception as e:
            raise e
