# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Disposable database ownership survives failed and terminated test processes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import sqlalchemy
from marina.local_postgres import docker_database
from marina.testing import TEST_DATABASE_URL_ENV


def resources(port: int) -> tuple[str, list[str]]:
    container = subprocess.check_output(["docker", "ps", "-aq", "--filter", f"publish={port}"], text=True).strip()
    assert container
    mounts = json.loads(subprocess.check_output(["docker", "inspect", "--format", "{{json .Mounts}}", container]))
    volumes = [item["Name"] for item in mounts if item["Type"] == "volume"]
    assert volumes, "The regression must exercise an actual Docker data volume"
    return container, volumes


def assert_removed(container: str, volumes: list[str]) -> None:
    remaining = subprocess.check_output(["docker", "ps", "-aq", "--no-trunc"], text=True).splitlines()
    assert not any(item.startswith(container) for item in remaining)
    remaining_volumes = subprocess.check_output(["docker", "volume", "ls", "-q"], text=True).splitlines()
    assert not set(volumes).intersection(remaining_volumes)


@pytest.mark.docker
@pytest.mark.timeout(120)
def test_database_sigterm_removes_owned_container_and_volume(tmp_path: Path) -> None:
    script = tmp_path / "database_owner.py"
    script.write_text(
        "import json, signal, sqlalchemy\n"
        "from marina.local_postgres import docker_database\n"
        "with docker_database() as url:\n"
        " print(json.dumps({'port': sqlalchemy.make_url(url).port}), flush=True)\n"
        " signal.pause()\n"
    )
    child = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    container = None
    try:
        assert child.stdout is not None
        port = json.loads(child.stdout.readline())["port"]
        container, volumes = resources(port)
        child.terminate()
        child.wait(timeout=20)
        assert_removed(container, volumes)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        if container:
            subprocess.run(["docker", "rm", "--force", "--volumes", container], capture_output=True, check=False)


@pytest.mark.docker
@pytest.mark.parametrize("failed", [False, True])
def test_database_context_removes_only_its_container_and_volume(failed: bool) -> None:
    with docker_database() as other:
        other_container, other_volumes = resources(sqlalchemy.make_url(other).port)
        try:
            with docker_database() as url:
                container, volumes = resources(sqlalchemy.make_url(url).port)
                if failed:
                    raise ValueError("test body failed")
        except ValueError:
            assert failed
        assert_removed(container, volumes)
        engine = sqlalchemy.create_engine(other)
        try:
            with engine.connect() as connection:
                assert connection.execute(sqlalchemy.text("SELECT 42")).scalar_one() == 42
        finally:
            engine.dispose()
    assert_removed(other_container, other_volumes)


@pytest.mark.docker
@pytest.mark.timeout(120)
def test_journey_parent_reclaims_database_after_pytest_hard_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TEST_DATABASE_URL_ENV, raising=False)
    marker = tmp_path / "resources.json"
    journeys = tmp_path / "apps" / "crash" / "journeys"
    journeys.mkdir(parents=True)
    (journeys / "test_abort.py").write_text(
        "import json, os, pathlib, subprocess, sqlalchemy\n"
        "def test_abort(marina_database):\n"
        " port = sqlalchemy.make_url(marina_database.url).port\n"
        " container = subprocess.check_output(['docker','ps','-aq','--filter',f'publish={port}'], text=True).strip()\n"
        " mounts = json.loads(subprocess.check_output(['docker','inspect','--format','{{json .Mounts}}',container]))\n"
        " record = {'container':container,'volumes':[m['Name'] for m in mounts if m['Type']=='volume']}\n"
        f" pathlib.Path({str(marker)!r}).write_text(json.dumps(record))\n"
        " os._exit(17)\n"
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "marina.cli",
            "journey",
            "crash",
            "--apps-dir",
            str(tmp_path / "apps"),
            "--shots",
            str(tmp_path / "shots"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = child.communicate(timeout=45)
        assert child.returncode == 17, output
        record = json.loads(marker.read_text())
        assert record["volumes"]
        assert_removed(record["container"], record["volumes"])
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=20)
        if marker.exists():
            record = json.loads(marker.read_text())
            subprocess.run(
                ["docker", "rm", "--force", "--volumes", record["container"]], capture_output=True, check=False
            )


@pytest.mark.docker
def test_database_cleanup_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    original = subprocess.run
    container = None

    def fail_removal(command: list[str], **kwargs):
        if command[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(command, 125, stdout=b"", stderr=b"removal failed")
        return original(command, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(subprocess, "run", fail_removal)
            with pytest.raises(subprocess.CalledProcessError):
                with docker_database() as url:
                    container, volumes = resources(sqlalchemy.make_url(url).port)
    finally:
        if container:
            original(["docker", "rm", "--force", "--volumes", container], check=True, capture_output=True)
    assert_removed(container, volumes)
