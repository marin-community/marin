# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepared tasks retain their source inputs and use the digest of their own build."""

import json
import subprocess
from pathlib import Path

from experiments.evaluation.images.tau3.build import prepare_tau3_task


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
