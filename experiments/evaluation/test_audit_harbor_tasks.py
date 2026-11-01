# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from experiments.evaluation.audit_harbor_tasks import inventory


def write_task(root: Path, name: str, compose: str | None = None, *, compose_name: str = "docker-compose.yaml") -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    if compose is not None:
        (task / "environment" / compose_name).write_text(compose)
    return task


@pytest.mark.parametrize("compose_name", ["docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"])
def test_inventory_distinguishes_wrappers_from_peers_without_exposing_values(tmp_path, compose_name):
    write_task(tmp_path, "plain")
    write_task(tmp_path, "wrapper", "services:\n  main:\n    image: example/image\n")
    peer = write_task(
        tmp_path,
        "peers",
        "services:\n  main:\n    depends_on: [runtime]\n  runtime:\n"
        "    image: example/runtime\n    privileged: true\n"
        "    environment:\n      TOKEN: secret-task-value\n"
        "    volumes: ['/var/run/docker.sock:/var/run/docker.sock']\n"
        "    custom_option: secret-option-value\n",
        compose_name=compose_name,
    )
    (peer / "solution.sh").write_text("docker run -e TOKEN=secret-command-value image\n")
    result = inventory(tmp_path, source="example/dataset", revision="snapshot")
    assert asdict(result.summary) == {
        "tasks": 3,
        "tasks_with_compose": 2,
        "tasks_with_multiple_services": 1,
        "tasks_with_docker_references": 1,
        "tasks_with_unscanned_files": 0,
    }
    peers = next(task for task in result.tasks if task.path == "peers")
    assert peers.compose[0].services == ("main", "runtime")
    assert peers.compose[0].review_fields == ("services.runtime.privileged", "services.runtime.volumes")
    assert peers.compose[0].other_fields == ("services.runtime.custom_option",)
    assert {reference.path for reference in peers.docker_references} == {
        f"environment/{compose_name}",
        "solution.sh",
    }
    encoded = json.dumps(asdict(result))
    assert "secret-task-value" not in encoded
    assert "secret-option-value" not in encoded
    assert "secret-command-value" not in encoded


def test_invalid_compose_does_not_report_a_single_sandbox(tmp_path):
    write_task(tmp_path, "broken", "services: [main]\n")
    with pytest.raises(ValueError, match="services mapping"):
        inventory(tmp_path, source="example/dataset", revision="snapshot")


def test_topology_uses_active_overlay_and_harbors_main_service(tmp_path):
    write_task(tmp_path, "target", "services:\n  victim:\n    image: example/victim\n")
    write_task(tmp_path, "empty-overlay", "{}\n")
    embedded = write_task(tmp_path, "embedded-development-files")
    nested = embedded / "environment" / "terminal"
    nested.mkdir()
    (nested / "docker-compose.yaml").write_text("services:\n  default: {}\n  mock_gist_server: {}\n")

    result = inventory(tmp_path, source="example/dataset", revision="snapshot")

    assert result.summary.tasks == 3
    assert result.summary.tasks_with_compose == 2
    assert result.summary.tasks_with_multiple_services == 1
    tasks = {task.path: task for task in result.tasks}
    assert tasks["target"].compose[0].services == ("victim",)
    assert tasks["target"].compose[0].effective_services == ("main", "victim")
    assert tasks["empty-overlay"].compose[0].effective_services == ("main",)
    assert tasks["embedded-development-files"].compose == ()
