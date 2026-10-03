# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private TaskCompendium grading over a Shellbox-owned Docker workspace."""

from shellbox.backends.docker.environment import DockerEnvironment
from shellbox.machine import InvalidWorkspaceFile, TerminalFileReader

from taskcompendium.submission import SubmissionFailure


class DockerWorkspaceEnvironment(DockerEnvironment):
    """Keep private host paths out of the machine and translate candidate errors."""

    async def read_workspace_file(self, path: str, max_bytes: int) -> bytes | None:
        if self.machine is None:
            raise RuntimeError("Docker trial machine is not available")
        if not isinstance(self.machine, TerminalFileReader):
            raise RuntimeError("Selected machine cannot acquire bounded terminal files")
        try:
            return await self.machine.read_file(path, max_bytes)
        except InvalidWorkspaceFile as error:
            raise SubmissionFailure(str(error)) from error
