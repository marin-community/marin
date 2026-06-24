# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a user-facing Entrypoint + EnvironmentConfig into a RuntimeEntrypoint.

The RuntimeEntrypoint keeps the setup script separate from the user's command so
each runtime can handle setup as it needs (DockerRuntime runs it in a build
container; ProcessRuntime skips it). Setup scripts are resolved client-side in
``EnvironmentConfig.setup_scripts`` and carried through verbatim.
"""

from iris.cluster.types import Entrypoint
from iris.rpc import job_pb2


def build_runtime_entrypoint(
    entrypoint: Entrypoint,
    env_config: job_pb2.EnvironmentConfig,
) -> job_pb2.RuntimeEntrypoint:
    """Build a RuntimeEntrypoint from a user Entrypoint + env config.

    Carries the resolved ``env_config.setup_scripts`` into ``setup_commands``. The
    run_command is the user's original command, kept separate so runtimes that
    don't need setup can skip it cleanly.
    """
    rt = job_pb2.RuntimeEntrypoint()
    # Drop whitespace-only scripts so an empty setup leaves setup_commands empty:
    # the build phase is skipped and the command runs in the image as-is.
    rt.setup_commands[:] = [s for s in env_config.setup_scripts if s.strip()]
    rt.run_command.argv[:] = entrypoint.command
    for k, v in entrypoint.workdir_files.items():
        rt.workdir_files[k] = v
    for k, v in entrypoint.workdir_file_refs.items():
        rt.workdir_file_refs[k] = v
    return rt
