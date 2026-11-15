# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared host requirements for tests that execute Linux guest behavior locally."""

import sys

import pytest

HOST_REQUIREMENTS = {
    "requires_host_setsid": "The local Linux guest fake needs the host setsid executable",
    "requires_linux_procfs": "CPU sampling exercises the Linux /proc implementation",
    "requires_linux_loopback": "DNS filters bind arbitrary addresses in the Linux 127.0.0.0/8 loopback range",
    "requires_linux_process_api": "The local process fake needs Linux /proc and waitid with WNOWAIT",
}


def pytest_configure(config: pytest.Config) -> None:
    for marker, reason in HOST_REQUIREMENTS.items():
        config.addinivalue_line("markers", f"{marker}: {reason} (skipped on macOS)")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if sys.platform != "darwin":
        return

    for item in items:
        for marker, reason in HOST_REQUIREMENTS.items():
            if item.get_closest_marker(marker) is not None:
                item.add_marker(pytest.mark.skip(reason=reason))
