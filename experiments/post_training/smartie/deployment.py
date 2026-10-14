# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checked package-lock to container-image deployment for admitted Smartie tasks."""

import hashlib
import json
import re
from dataclasses import dataclass, field

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import EnvironmentRequirements, ScriptGrader, TaskSpec, VerifyitGrader


@dataclass(frozen=True)
class LockImage:
    lock_sha256: str
    data: tuple[str, ...]
    image: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", self.image) is None:
            raise ValueError("Deployment requires a digest-pinned image")


@dataclass
class ImageDeployment:
    """Verify each source lock once per dataset build, retaining its deployment evidence."""

    images: tuple[LockImage, ...]
    evidence: dict[str, LockImage] = field(default_factory=dict, init=False)

    def environment(self, original: EnvironmentRequirements) -> EnvironmentRequirements:
        lock = original.packages_lock
        if lock is None:
            return original
        if lock not in self.evidence:
            # Admitted sources retain the artifact schema used to build their pinned images.
            record = json.loads((StoragePath(lock).parent / ".artifact.json").read_text())["result"]
            with StoragePath(lock).open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_hash != record["lock_sha256"]:
                raise ValueError(f"Source lock differs from its artifact record: {lock}")
            matches = [
                image
                for image in self.images
                if image.lock_sha256 == actual_hash and sorted(image.data) == sorted(record["data"])
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one image deployment for lock {lock}, found {len(matches)}")
            self.evidence[lock] = matches[0]
        return EnvironmentRequirements.model_validate(
            {**original.model_dump(), "packages_lock": None, "docker_image": self.evidence[lock].image}
        )

    def task(self, original: TaskSpec) -> TaskSpec:
        fields = original.model_dump()
        fields["environment_requirements"] = self.environment(original.environment_requirements).model_dump()
        grader = original.grader
        if isinstance(grader, VerifyitGrader | ScriptGrader) and grader.environment is not None:
            fields["grader"]["environment"] = self.environment(grader.environment).model_dump()
        return TaskSpec.model_validate(fields)
