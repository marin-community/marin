# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a user-facing command and environment into a RuntimeEntrypoint.

The RuntimeEntrypoint keeps setup separate from the user's command so each runtime
can handle it as needed (DockerRuntime runs setup in a build container;
ProcessRuntime skips it). Setup is the user's client-resolved
``EnvironmentConfig.setup_scripts`` followed by iris's own runtime-deps script.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol

from iris.cluster.setup_scripts import iris_runtime_setup_script
from iris.rpc import job_pb2


class CommandSource(Protocol):
    @property
    def command(self) -> Sequence[str]: ...

    @property
    def workdir_files(self) -> Mapping[str, bytes]: ...

    @property
    def workdir_file_refs(self) -> Mapping[str, str]: ...


def build_runtime_entrypoint(
    entrypoint: CommandSource,
    env_config: job_pb2.EnvironmentConfig,
) -> job_pb2.RuntimeEntrypoint:
    """Build a RuntimeEntrypoint from a user Entrypoint + env config.

    Assembles ``setup_commands`` as the user's resolved scripts followed by iris's
    runtime-deps script. The run_command is the user's original command, kept
    separate so runtimes that don't need setup can skip it cleanly.
    """
    rt = job_pb2.RuntimeEntrypoint()
    # Drop whitespace-only scripts; an empty user list means no setup at all, so the
    # build phase (iris script included) is skipped and the command runs as-is.
    user_scripts = [s for s in env_config.setup_scripts if s.strip()]
    if user_scripts:
        iris_script = iris_runtime_setup_script()
        rt.setup_commands[:] = [*user_scripts, iris_script]
    rt.run_command.argv[:] = entrypoint.command
    for k, v in entrypoint.workdir_files.items():
        rt.workdir_files[k] = v
    for k, v in entrypoint.workdir_file_refs.items():
        rt.workdir_file_refs[k] = v
    return rt
