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

"""Tests for smoke test logging infrastructure."""

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Import smoke-test.py as a module
_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
_SMOKE_TEST_PATH = _SCRIPTS_DIR / "smoke-test.py"


def _load_smoke_test_module():
    """Load smoke-test.py as a module."""
    spec = importlib.util.spec_from_file_location("smoke_test", _SMOKE_TEST_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load smoke-test.py from {_SMOKE_TEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["smoke_test"] = module
    spec.loader.exec_module(module)
    return module


smoke_test = _load_smoke_test_module()
SmokeTestConfig = smoke_test.SmokeTestConfig
SmokeTestRunner = smoke_test.SmokeTestRunner
SmokeTestLogger = smoke_test.SmokeTestLogger


def test_smoke_test_config_log_dir_creation():
    """Verify that SmokeTestConfig and SmokeTestRunner create proper log directory structure."""
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "test-logs"
        config = SmokeTestConfig(
            config_path=Path("dummy.yaml"),
            log_dir=log_dir,
            timeout_seconds=100,
            job_timeout_seconds=50,
            tpu_type="v5litepod-16",
            cleanup_on_failure=False,
            clean_start=False,
            build_images=False,
        )

        # Create runner which should initialize log directories
        runner = SmokeTestRunner(config)

        # Verify directory structure
        assert log_dir.exists(), "Log directory not created"
        assert (log_dir / "workers").exists(), "workers subdirectory not created"
        assert (log_dir / "scheduling").exists(), "scheduling subdirectory not created"
        assert (log_dir / "autoscaler").exists(), "autoscaler subdirectory not created"
        assert (log_dir / "summary.md").exists(), "summary.md not created"

        # Verify logger writes to summary.md
        runner.logger.log("Test message")
        runner.logger.close()

        summary_content = (log_dir / "summary.md").read_text()
        assert "Test message" in summary_content


def test_smoke_test_logger_writes_to_summary():
    """Verify that SmokeTestLogger writes to summary.md in the log directory."""
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "test-logs"
        log_dir.mkdir(parents=True)

        logger = SmokeTestLogger(log_dir)
        logger.log("INFO message", level="INFO")
        logger.log("ERROR message", level="ERROR")
        logger.section("Test Section")
        logger.close()

        summary_file = log_dir / "summary.md"
        assert summary_file.exists(), "summary.md not created"

        content = summary_file.read_text()
        assert "INFO message" in content
        assert "ERROR message" in content
        assert "Test Section" in content
        assert "[INFO]" in content
        assert "[ERROR]" in content
