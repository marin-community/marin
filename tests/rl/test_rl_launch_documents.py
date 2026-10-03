# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from marin import external_dependencies
from marin.external_dependencies import MARIN_SKYRL

from tests.rl.launch_document_census import render_launch_census

logger = logging.getLogger(__name__)


@pytest.mark.timeout(540)
def test_rl_launch_documents_load_with_the_installed_launcher(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    assert Path(external_dependencies.__file__).resolve() == root / "lib/marin/src/marin/external_dependencies.py"
    census = render_launch_census()
    (tmp_path / "manifest.json").write_text(json.dumps(asdict(census)))
    assert not census.failures, census.failures
    assert census.documents
    assert {entry.launcher_requirement for entry in census.documents} == {MARIN_SKYRL.requirement()}
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"case": entry.case, "document": entry.document} for entry in census.documents]))
    output = tmp_path / "loaded.json"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    start = time.monotonic()
    result = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--prerelease=allow",
            "--python",
            "3.12",
            "--with",
            MARIN_SKYRL.requirement(),
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
    logger.info(
        "Launcher %s at %s: %d loaded, %d failed; setup %.2fs, loading %.2fs",
        report["installed_commit"],
        report["launcher_source"],
        len(report["documents"]),
        len(report["failures"]),
        time.monotonic() - start - report["load_duration"],
        report["load_duration"],
    )
    assert not report["failures"], report["failures"]
    assert result.returncode == 0, result.stderr
    assert tuple(entry["case"] for entry in report["documents"]) == census.cases
