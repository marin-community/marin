# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a user-facing Entrypoint + EnvironmentConfig into a RuntimeEntrypoint.

The RuntimeEntrypoint keeps setup separate from the user's command so each runtime
can handle it as needed (DockerRuntime runs setup in a build container;
ProcessRuntime skips it). Setup is the user's client-resolved
resolved ``EnvironmentConfig.setup_layers`` and iris's own runtime-deps script.
"""

from iris.cluster.setup_scripts import EnvironmentLayer, iris_runtime_setup_script
from iris.cluster.types import Entrypoint
from iris.rpc import job_pb2


def build_runtime_entrypoint(
    entrypoint: Entrypoint,
    env_config: job_pb2.EnvironmentConfig,
) -> job_pb2.RuntimeEntrypoint:
    """Build a RuntimeEntrypoint from a user Entrypoint + env config.

    Assembles setup and activation commands from the resolved layer sequence.
    Iris's runtime-deps script follows user setup; activation is sourced after
    virtualenv activation by the runtime.
    """
    rt = job_pb2.RuntimeEntrypoint()
    layers = [EnvironmentLayer.from_proto(layer) for layer in env_config.setup_layers]
    setup_commands = [layer.setup for layer in layers if layer.setup.strip()]
    if setup_commands:
        iris_script = iris_runtime_setup_script()
        rt.setup_commands[:] = [*setup_commands, iris_script]
    rt.activation_commands[:] = [layer.activate for layer in layers if layer.activate.strip()]
    rt.run_command.argv[:] = entrypoint.command
    for k, v in entrypoint.workdir_files.items():
        rt.workdir_files[k] = v
    for k, v in entrypoint.workdir_file_refs.items():
        rt.workdir_file_refs[k] = v
    return rt
