# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest
from marin.external_dependencies import MARIN_SKYRL

from tests.rl.launch_document_census import render_launch_census


@pytest.mark.timeout(540)
def test_rl_launch_documents_load_with_the_installed_launcher(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    project = root / "config/external" / MARIN_SKYRL.config_name
    lock = tomllib.loads((project / "uv.lock").read_text())
    package = next(package for package in lock["package"] if package["name"] == MARIN_SKYRL.distribution)
    assert package["source"]["git"].rsplit("#", 1)[1] == MARIN_SKYRL.commit
    census = render_launch_census(tmp_path / "artifacts")
    assert not census.failures, census.failures
    assert census.documents
    assert {entry.document["runtime"]["launcher_commit"] for entry in census.documents} == {MARIN_SKYRL.commit}
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"case": entry.case, "document": entry.document} for entry in census.documents]))
    output = tmp_path / "loaded.json"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--project",
            str(project),
            "--frozen",
            "python",
            str(Path(__file__).with_name("launch_document_loader.py")),
            "--input",
            str(source),
            "--output",
            str(output),
            "--expected-commit",
            MARIN_SKYRL.commit,
        ],
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
