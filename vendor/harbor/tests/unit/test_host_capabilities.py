# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve host capabilities without optional backend dependencies."""

import subprocess
import sys
import textwrap


def test_host_capabilities_resolve_without_the_daytona_sdk() -> None:
    """Every CLI launch resolves capabilities; the Daytona extra stays optional."""
    script = textwrap.dedent(
        """
        import sys

        sys.modules["daytona"] = None  # the optional extra is absent
        from shellbox.backends.daytona.connection import DaytonaConnection
        from harbor.host_capabilities import host_capabilities_from_environment

        capabilities = host_capabilities_from_environment({}, model_authentication={})
        assert capabilities.daytona is not None
        connection = DaytonaConnection(
            api_key="synthetic-secret-key", api_url="https://fixture.invalid/api", target="fixture"
        )
        assert callable(connection.client_factory())
        print("resolved")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "resolved" in result.stdout
