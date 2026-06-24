# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a user-facing Entrypoint + EnvironmentConfig into a RuntimeEntrypoint.

The RuntimeEntrypoint separates the setup script (environment preparation) from
the user's actual command. This lets each runtime handle them appropriately:
- DockerRuntime runs the setup script in a build container to create the venv
- ProcessRuntime skips setup since the host env is already configured

The setup script is already resolved client-side (``EnvironmentConfig.setup_script``);
this only carries it into ``setup_commands`` as a single element (or empty for no
setup). Nothing here interprets it.
"""

from iris.cluster.types import Entrypoint
from iris.rpc import job_pb2


def build_runtime_entrypoint(
    entrypoint: Entrypoint,
    env_config: job_pb2.EnvironmentConfig,
) -> job_pb2.RuntimeEntrypoint:
    """Build a RuntimeEntrypoint from a user Entrypoint + env config.

    Carries the resolved ``env_config.setup_script`` into ``setup_commands``. The
    run_command is the user's original command, kept separate so runtimes that
    don't need setup can skip it cleanly.
    """
    rt = job_pb2.RuntimeEntrypoint()
    # A whitespace-only script means "no setup": leave setup_commands empty so the
    # build phase is skipped and the command runs in the image as-is.
    if env_config.setup_script.strip():
        rt.setup_commands[:] = [env_config.setup_script]
    rt.run_command.argv[:] = entrypoint.command
    for k, v in entrypoint.workdir_files.items():
        rt.workdir_files[k] = v
    for k, v in entrypoint.workdir_file_refs.items():
        rt.workdir_file_refs[k] = v
    return rt
