# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise release state transitions through a fake ports subprocess boundary."""

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from marin.execution.lazy import run

from experiments.ports.release import ReleaseConfig, ports_release


@dataclass
class ReleaseFixture:
    config: ReleaseConfig
    events: Path
    fail_checks: bool = False


@pytest.fixture
def release_environment(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    (repository / "recipe.json").write_text('{"guest": "clang"}')
    subprocess.run(["git", "-C", str(repository), "add", "recipe.json"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-m",
            "Add guest recipe",
        ],
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    config = ReleaseConfig(
        source_revision=revision,
        version="2026.10.12",
        work_directory=str(tmp_path / "driver"),
        source_repository=str(repository),
        publication_prefix=str(tmp_path / "publication"),
        cache_prefix=str(tmp_path / "cache"),
        asset_url_prefix="https://ports.example.org",
    )
    original_run = subprocess.run
    events = tmp_path / "events.jsonl"
    fixture = ReleaseFixture(config, events)

    def fake_run(arguments, **kwargs):
        if arguments[1:3] != ["-m", "ports"]:
            return original_run(arguments, **kwargs)
        parser = argparse.ArgumentParser()
        parser.add_argument("operation")
        for name in (
            "state",
            "store",
            "backend",
            "iris-image",
            "max-workers",
            "worker-cpus",
            "worker-memory-gb",
            "worker-disk-gb",
            "url-prefix",
            "output",
        ):
            parser.add_argument("--" + name)
        for name in ("all", "copy-assets", "simple"):
            parser.add_argument("--" + name, action="store_true")
        args = parser.parse_args(arguments[3:])
        with events.open("a") as stream:
            stream.write(json.dumps({"operation": args.operation, "output": args.output}) + "\n")
        state = Path(args.state)
        asset = {
            "name": "clang",
            "path": "assets/immutable/clang.wasm",
            "url": f"{args.url_prefix}/assets/immutable/clang.wasm",
        }
        if args.operation == "build":
            assert args.all
            (state / "graph.json").write_text(json.dumps({"assets": [asset]}))
            (state / "results.json").write_text(json.dumps({"roots": ["toolchain/llvm:guest"], "prefix": args.store}))
            source = Path(args.store) / asset["path"]
            source.parent.mkdir(parents=True)
            source.write_bytes(b"guest clang")
        elif args.operation == "check":
            if fixture.fail_checks:
                raise subprocess.CalledProcessError(1, arguments)
            assert json.loads((state / "results.json").read_text())["roots"] == ["toolchain/llvm:guest"]
        else:
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            if args.copy_assets:
                target = output / asset["path"]
                if not target.exists():
                    target.parent.mkdir(parents=True)
                    target.write_bytes((Path(args.store) / asset["path"]).read_bytes())
            assert (Path(config.publication_prefix) / asset["path"]).read_bytes() == b"guest clang"
            (output / "catalog.json").write_text(json.dumps({"assets": [asset]}))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return fixture


def test_release_publishes_checked_snapshot_and_reuses_cached_output(release_environment):
    config, events = release_environment.config, release_environment.events
    handle = ports_release(config)
    run(handle)
    publication = Path(config.publication_prefix)
    stable = json.loads((publication / "catalog.json").read_text())
    assert stable == json.loads((publication / "releases" / config.version / "catalog.json").read_text())
    assert stable["assets"][0]["url"] == "https://ports.example.org/assets/immutable/clang.wasm"
    assert (publication / "assets/immutable/clang.wasm").read_bytes() == b"guest clang"
    before = events.read_bytes()
    shutil.rmtree(config.work_directory)
    shutil.rmtree(config.source_repository)
    run(handle)
    assert events.read_bytes() == before
    assert not Path(config.work_directory).exists()


def test_release_failed_guest_checks_preserve_published_catalog(release_environment):
    config, events = release_environment.config, release_environment.events
    publication = Path(config.publication_prefix)
    publication.mkdir()
    (publication / "catalog.json").write_bytes(b"existing catalog")
    release_environment.fail_checks = True
    with pytest.raises(RuntimeError):
        run(ports_release(config))
    assert (publication / "catalog.json").read_bytes() == b"existing catalog"
    assert not (publication / "releases").exists()
    assert [json.loads(line)["operation"] for line in events.read_text().splitlines()] == ["build", "check"]
