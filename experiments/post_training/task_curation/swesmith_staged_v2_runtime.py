# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply public source handoff overlays before the agent begins its Stage3 session."""

import base64
import json
import time
from pathlib import Path

from shellbox.machine import Command
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.swesmith_staged import is_test_path, sha256
from experiments.post_training.task_curation.swesmith_staged_runtime import StagedSession

SESSION_NAME = "swesmith-staged-v2"


def handoff_program(files: dict[str, str], allowed: list[str], root: Path) -> str:
    """Reject hidden-test/path escapes before materializing actor-visible source."""
    for name, content in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name not in allowed or is_test_path(name):
            raise ValueError("Handoff overlay is outside permitted source")
        base64.b64decode(content, validate=True)
    return (
        "import base64,pathlib\nroot=pathlib.Path("
        + repr(str(root))
        + ")\n"
        + "files="
        + repr(files)
        + "\n"
        + "for name,content in files.items():\n"
        + " path=root/name\n"
        + " assert path.is_file() and not path.is_symlink()\n"
        + " assert path.resolve().is_relative_to(root)\n"
        + " path.write_bytes(base64.b64decode(content))\n"
    )


class HandoffSession(StagedSession):
    async def prepare(self):
        resources = self.context.lowered.task.resources.worker
        if resources:
            if len(resources) != 1 or resources[0].path != "handoff-files.json":
                raise ValueError("Unexpected actor resource")
            payload = resource_bytes(resources[0])
            if sha256(payload) != self.contract["handoff_sha256"]:
                raise ValueError("Handoff resource hash mismatch")
            program = handoff_program(json.loads(payload), self.contract["changed_source_paths"], Path("/testbed"))
            result = await self.machine.run(
                Command(("/usr/local/bin/python", "-"), cwd="/tmp", stdin=program.encode(), timeout=60)
            )
            if result.exit_code:
                raise RuntimeError("Could not materialize handoff source")
        return await super().prepare()


def session_factory(logs: Path):
    def create(context):
        return HandoffSession(context, logs / (str(time.time_ns()) + "-" + context.lowered.task.id.rsplit("/", 1)[-1]))

    return create
