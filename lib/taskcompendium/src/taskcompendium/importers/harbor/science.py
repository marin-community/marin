# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower SWE-bench Science's Pier release while keeping its image-owned verifier."""

import json
import shlex
import shutil
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory

from taskcompendium.importers.harbor.package import HarborTask, JobSettings, PackageTests, lower_package
from taskcompendium.models import (
    ArtifactKind,
    FileReward,
    GraderMachine,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    VerifierArtifact,
)

PATCH = "/logs/artifacts/model.patch"


def lower_science_package(directory: Path, *, task_id: str, source: Source) -> HarborTask:
    """Translate Pier's collection hook and keep the native clean-rebuild entrypoint.

    The release's public test.sh is a loader stub. Only grader.py overlays the fresh verifier
    image; its own /tests/test.sh applies the patch and rebuilds before invoking that grader.
    """
    original = (directory / "task.toml").read_text()
    config = tomllib.loads(original)
    verifier = config["verifier"]
    if config["schema_version"] != "1.3" or verifier["environment_mode"] != "separate":
        raise ValueError("Expected SWE-bench Science's schema 1.3 separate-verifier release")
    for environment in (config["environment"], verifier["environment"]):
        if environment["allow_internet"] or "@sha256:" not in environment["docker_image"]:
            raise ValueError("SWE-bench Science requires offline digest-pinned images")
    if config["agent"]["network_mode"] != "no-network" or verifier["network_mode"] != "no-network":
        raise ValueError("SWE-bench Science requires the native no-network protocol")
    hook = (directory / "pre_artifacts.sh").read_text()
    command = f"cd {shlex.quote(config['environment']['workdir'])} && sh -c {shlex.quote(hook)}"
    with TemporaryDirectory(prefix="swe-science-package-") as temporary:
        prepared = Path(temporary) / directory.name
        (prepared / "environment").mkdir(parents=True)
        (prepared / "tests").mkdir()
        shutil.copyfile(directory / "instruction.md", prepared / "instruction.md")
        shutil.copyfile(directory / "tests/grader.py", prepared / "tests/grader.py")
        # This release's 1.3 fields have 1.2 equivalents: offline environment
        # declarations plus an explicit collection command replace Pier's hook.
        normalized = original.replace('schema_version = "1.3"', 'schema_version = "1.2"', 1)
        normalized += f"\n[[verifier.collect]]\ncommand = {json.dumps(command)}\ntimeout_sec = 300.0\n"
        (prepared / "task.toml").write_text(normalized)
        lowered = lower_package(
            prepared, task_id=task_id, source=source, rule=PackageTests(), image_digests={}, job=JobSettings()
        )
    grader = lowered.task.grader
    assert isinstance(grader, ScriptGrader)
    assert isinstance(grader.placement, GraderMachine)
    placement = grader.placement.model_copy(
        update={"artifacts": (VerifierArtifact(source=PATCH, target=PATCH, kind=ArtifactKind.FILE),)}
    )
    # JSON mode reads the native reward field without treating nested diagnostic
    # reports as numeric metrics. The original report is not rewritten.
    grader = grader.model_copy(
        update={
            "placement": placement,
            "reward": FileReward(files=(RewardFile(path="/logs/verifier/reward.json", format=RewardFileFormat.JSON),)),
        }
    )
    return HarborTask(lowered.task.model_copy(update={"grader": grader}), lowered.episode)
