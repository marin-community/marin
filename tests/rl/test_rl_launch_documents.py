# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest
from marin.rl.skyrl import skyrl_cpu_command

from tests.rl.launch_document_census import render_launch_census


@pytest.mark.timeout(540)
def test_rl_launch_documents_load_with_the_vendored_launcher(tmp_path: Path) -> None:
    census = render_launch_census(tmp_path / "artifacts")
    assert not census.failures, census.failures
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"case": entry.case, "document": entry.document} for entry in census.documents]))
    output = tmp_path / "loaded.json"
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        environment.pop(name, None)
    result = subprocess.run(
        skyrl_cpu_command(
            "python",
            str(Path(__file__).with_name("launch_document_loader.py")),
            "--input",
            str(source),
            "--output",
            str(output),
        ),
        env=environment,
        capture_output=True,
        text=True,
        timeout=480,
    )
    assert output.exists(), result.stderr
    report = json.loads(output.read_text())
    assert not report["failures"], report["failures"]
    assert result.returncode == 0, result.stderr
    assert tuple(entry["case"] for entry in report["documents"]) == census.cases
