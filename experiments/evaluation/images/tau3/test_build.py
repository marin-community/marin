# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepared tasks retain their source inputs and use the digest of their own build."""

import json
import shutil
import subprocess
from pathlib import Path

from experiments.evaluation.images.tau3 import build
from experiments.evaluation.images.tau3.build import prepare_tau3_task


def test_dockerfile_edits_build_a_new_image_instead_of_reusing_the_published_recipe(tmp_path, monkeypatch):
    for name in build.BUILD_INPUTS:
        shutil.copyfile(build.RECIPE / name, tmp_path / name)
    monkeypatch.setattr(build, "RECIPE", tmp_path)
    images: dict[str, str] = {}

    def docker(argv, **kwargs):
        if argv[1:4] == ["buildx", "imagetools", "inspect"]:
            image = argv[4]
            if image not in images:
                return subprocess.CompletedProcess(argv, 1, "", "manifest unknown")
            return subprocess.CompletedProcess(argv, 0, json.dumps({"digest": images[image]}), "")
        assert argv[1:3] == ["buildx", "build"]
        tag = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--tag="))
        images[tag] = "sha256:" + str(len(images) + 1) * 64
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", docker)
    first = build.tau3_image()
    assert build.tau3_image() == first
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(dockerfile.read_text() + '\nENV RECIPE_REGRESSION="updated"\n')
    updated = build.tau3_image()
    assert first == f"{build.DEFAULT_REPOSITORY}@sha256:{'1' * 64}"
    assert updated == f"{build.DEFAULT_REPOSITORY}@sha256:{'2' * 64}"


def test_prepare_tau3_task_reuses_main_and_pins_runtime_build_result(tmp_path, monkeypatch):
    task = tmp_path / "tau3-retail-0"
    runtime = task / "environment/runtime-server"
    runtime.mkdir(parents=True)
    configuration = '{"domain": "retail", "task": {"id": 0}}\n'
    (runtime / "task_config.json").write_text(configuration)
    (runtime / "server.py").write_text("# reviewed runtime\n")
    main_digest = "sha256:" + "a" * 64
    runtime_digest = "sha256:" + "b" * 64

    def docker(argv, **kwargs):
        if argv[1:4] == ["buildx", "imagetools", "inspect"]:
            # A later tag read could select a concurrent build; runtime pinning
            # must use this build's metadata instead.
            return subprocess.CompletedProcess(argv, 0, json.dumps({"digest": main_digest}), "")
        assert argv[1:3] == ["buildx", "build"]
        assert Path(argv[-1]) == runtime
        metadata = next(Path(arg.split("=", 1)[1]) for arg in argv if arg.startswith("--metadata-file="))
        metadata.write_text(json.dumps({"containerimage.digest": runtime_digest}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", docker)
    prepare_tau3_task(task)

    manifest = json.loads((task / "environment/shellbox-images.json").read_text())
    assert manifest["main"]["image"] == f"ghcr.io/marin-community/rollout-images@{main_digest}"
    assert manifest["tau3-runtime"]["image"] == f"ghcr.io/marin-community/rollout-images@{runtime_digest}"
    assert (runtime / "task_config.json").read_text() == configuration
