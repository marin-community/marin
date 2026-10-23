# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""More tasks of the kind of an accepted task: its build program runs again with new task ids.

The program runs in a fresh machine of the workspace factory, on the workspace snapshot that ``submit`` took, so it
finds the files that it reads, for example research that the agent saved.
"""

import logging
import shlex
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError
from shellbox.machine import Command, MachineFactory, MachineSpec, PackageSpec
from taskcompendium.models import TaskSpec

from taskforge.agentic.tools import MAX_FILE_BYTES, WORKDIR, AcceptedTask, program_command

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = "/tmp/workspace.tgz"
INSTANCE_DIR = ".instances"


async def task_instances(
    accepted: AcceptedTask, task_ids: Sequence[str], factory: MachineFactory, *, timeout: float
) -> list[TaskSpec]:
    """The task that the program of ``accepted`` writes for each of ``task_ids``.

    An id whose run fails, or writes an invalid TaskSpec or one with another id, gives no task, and a warning names
    it.
    """
    machine = await factory.create(MachineSpec(source=PackageSpec(), workdir=str(WORKDIR)))
    try:
        with tempfile.TemporaryDirectory(prefix="taskforge-instances-") as directory:
            snapshot = Path(directory) / "workspace.tgz"
            snapshot.write_bytes(accepted.workspace)
            await machine.upload(snapshot, SNAPSHOT_PATH)
        extract = await machine.run(Command(("tar", "-xzf", SNAPSHOT_PATH, "-C", str(WORKDIR))))
        if extract.exit_code != 0:
            raise RuntimeError(f"The workspace snapshot does not extract: {extract.stderr.decode(errors='replace')}")
        tasks = []
        for task_id in task_ids:
            output = f"{INSTANCE_DIR}/{task_id}.json"
            command = shlex.join((*program_command(accepted.program), task_id, output))
            # The program's own output goes to stderr, so that stdout holds only the TaskSpec that it wrote.
            script = f"mkdir -p {INSTANCE_DIR} && {command} 1>&2 && cat {shlex.quote(output)}"
            result = await machine.run(
                Command(("sh", "-c", script), cwd=str(WORKDIR), timeout=timeout, output_limit_bytes=MAX_FILE_BYTES)
            )
            if result.exit_code != 0:
                logger.warning("%s: %s failed: %s", accepted.task.id, command, result.stderr.decode(errors="replace"))
                continue
            try:
                task = TaskSpec.model_validate_json(result.stdout)
            except ValidationError as error:
                logger.warning("%s: %s wrote an invalid TaskSpec: %s", accepted.task.id, command, error)
                continue
            if task.id != task_id:
                logger.warning("%s: %s wrote a task with the id %r", accepted.task.id, command, task.id)
                continue
            tasks.append(task)
        return tasks
    finally:
        await machine.close()
