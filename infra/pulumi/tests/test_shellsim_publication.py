# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use a CLI boundary fake; guest execution belongs to Shellsim's own tests."""

import argparse
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from infra.shellsim.ports import ReleaseConfig, StageConfig, publish_ports


def test_failed_stable_publication_preserves_version_and_older_releases_cannot_replace_it(tmp_path, monkeypatch):
    publication = tmp_path / "public/shellsim/ports"
    publication.mkdir(parents=True)
    (publication / "release-version").write_text("1")
    (publication / "catalog.json").write_text("previous catalog")
    cache = tmp_path / "cache"
    (cache / "assets/build").mkdir(parents=True)
    (cache / "assets/build/package.whl").write_bytes(b"checked release")
    checked = tmp_path / "checked"
    checked.mkdir()
    (checked / "graph.json").write_text("{}")
    (checked / "results.json").write_text("{}")
    release = ReleaseConfig(
        source_revision="a" * 40,
        version="2",
        work_directory=str(tmp_path / "work"),
        worker_image="worker:checked",
        cache_prefix=str(cache),
        publication_prefix=str(publication),
        asset_url_prefix="https://public.example/shellsim/ports",
    )
    (tmp_path / "work/source" / release.source_revision / ".git").mkdir(parents=True)
    fail_stable = True

    def run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout=release.source_revision)
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--asset-output", type=Path)
        parser.add_argument("--url-prefix", required=True)
        parser.add_argument("--copy-assets", action="store_true")
        args, _ = parser.parse_known_args(command[4:])
        if args.output == publication and fail_stable:
            raise subprocess.CalledProcessError(1, command)
        args.output.mkdir(parents=True, exist_ok=True)
        if args.copy_assets:
            target = args.asset_output / "assets/build/package.whl"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache / "assets/build/package.whl", target)
        (args.output / "catalog.json").write_text(json.dumps({"url": args.url_prefix + "/assets/build/package.whl"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    config = StageConfig(release, str(tmp_path / "stages/published"), str(checked))
    with pytest.raises(subprocess.CalledProcessError):
        publish_ports(config)
    assert (publication / "release-version").read_text() == "1"
    assert (publication / "catalog.json").read_text() == "previous catalog"
    assert (publication / "assets/build/package.whl").read_bytes() == b"checked release"
    snapshot = (publication / "releases/2/catalog.json").read_bytes()
    fail_stable = False
    publish_ports(config)
    assert (publication / "catalog.json").read_bytes() == snapshot
    assert (publication / "release-version").read_text() == "2"
    publish_ports(replace(config, release=replace(release, version="1")))
    assert (publication / "catalog.json").read_bytes() == snapshot
    assert (publication / "release-version").read_text() == "2"
    assert (publication / "releases/1/catalog.json").is_file()
