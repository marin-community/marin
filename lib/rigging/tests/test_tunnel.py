# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from unittest.mock import MagicMock

import pytest
from rigging.tunnel import K8sPortForwardTarget, open_tunnel


def test_open_tunnel_startup_timeout_raises_timeout_error() -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.stderr = None
    process.poll.return_value = 1
    target = K8sPortForwardTarget(namespace="iris", service="finelog", port=9090)

    with pytest.raises(TimeoutError):
        with open_tunnel(target, local_port=12345, timeout=0, spawn=lambda _argv: process):
            pass
