# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Docker images with the host's registered gVisor runtime."""

from pathlib import Path

from shellbox.backends.docker.machine import DockerMachineFactory


class GvisorMachineFactory(DockerMachineFactory):
    """Require a Docker daemon with its ``runsc`` runtime registered."""

    def __init__(
        self,
        *,
        skopeo: Path | None = None,
        image_cache: Path | None = None,
        authfile: Path | None = None,
        policy: Path | None = None,
    ):
        super().__init__(skopeo=skopeo, image_cache=image_cache, authfile=authfile, policy=policy, runtime="runsc")
