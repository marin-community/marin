# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import pytest
from marin.rl.skyrl import _run_launcher as run_launcher_for_test
from marin.rl.skyrl import skyrl_cpu_command

from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as snowball
from tests.rl.launch_document_census import render_launch_census


# A cold SkyRL environment sync and the full batch of compositions share this deadline.
@pytest.mark.timeout(540)
def test_rl_launch_documents_load_with_the_vendored_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snowball, "SELECTION_ROOT", str(tmp_path / "selections"))
    monkeypatch.setattr(snowball, "artifact_prefix", lambda cluster: str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        gate,
        "CONTROLS",
        {
            name: tuple(replace(pin, revision=snowball.PRESET_REVISION) for pin in pins)
            for name, pins in gate.CONTROLS.items()
        },
    )
    documents, failures = render_launch_census(tmp_path / "artifacts")
    assert not failures, failures
    source = tmp_path / "source.json"
    source.write_text(json.dumps(documents))
    output = tmp_path / "loaded.json"
    result = run_launcher_for_test(
        skyrl_cpu_command("python", str(Path(__file__).with_name("launch_document_loader.py")), str(source), str(output))
    )
    assert result.returncode == 0, result.stderr
    failures = json.loads(output.read_text())
    assert not failures, failures
