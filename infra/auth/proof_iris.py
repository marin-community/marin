# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an isolated native Iris receiver for grant admission proofs."""

from contextlib import contextmanager
from pathlib import Path

from iris.cluster.config import AuthConfig
from iris.cluster.controller.auth import create_controller_auth
from iris.managed_thread import thread_container_scope
from iris.testing.controller import controller_factory
from rigging.shared_grants import SharedGrantConfig


@contextmanager
def iris(directory: Path, trust: SharedGrantConfig | None):
    directory.mkdir()
    with thread_container_scope():
        factory = controller_factory(directory)
        make = next(factory)
        try:
            auth = create_controller_auth(
                AuthConfig(shared_grants=trust) if trust else None, cluster_name="isolated-proof"
            )
            controller = make(auth=auth, host="127.0.0.1", port=0)
            controller.start()
            yield controller.url
        finally:
            next(factory, None)
