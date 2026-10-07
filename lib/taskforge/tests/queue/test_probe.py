# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
SCRIPT = PACKAGE / "scripts" / "cluster_queue_probe.py"


def test_a_probe_that_fails_before_its_first_check_reports_the_error(tmp_path):
    config = json.loads((PACKAGE / "docs" / "policy.example.json").read_text())
    config |= {"host": "laptop", "root": str(tmp_path / "run"), "image_cache": str(tmp_path / "images")}
    config["glm"] = {"kind": "laptop", "base_url": "http://127.0.0.1:1/v1", "token_file": "/nonexistent", "pool": "high"}
    config["web"] = {"kind": "key_env", "env": "TASKFORGE_PROBE_TEST_KEY"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    env = {k: v for k, v in os.environ.items() if k != "TASKFORGE_PROBE_TEST_KEY"}

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--pool", "high", "--items", "1", "--image", "unused"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    report = json.loads((tmp_path / "run-queue_probe" / "probe.json").read_text())
    assert report["ok"] is False
    assert "TASKFORGE_PROBE_TEST_KEY" in report["error"]["error"]
    assert "host_secrets" in report["error"]["trace"]
